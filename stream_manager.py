#!/usr/bin/env python3
"""
Stream Manager Module

This module handles video streaming to YouTube via RTMP.
It manages a continuous livestream with seamless transitions
between generated video clips.
"""

import os
import time
import math
import threading
import queue
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import ffmpeg
from PIL import Image, ImageDraw, ImageFilter, ImageStat, ImageEnhance, ImageChops
from loguru import logger


class StreamManager:
    """Manages a continuous video stream with dynamic content."""
    
    def __init__(self, rtmp_url: str, rtmp_key: str, 
                 default_video_path: str,
                 output_resolution: Tuple[int, int] = (1280, 720),
                 fps: int = 30,
                 bitrate: str = "4000k",
                 transition_type: str = "fade",
                 transition_duration: float = 1.0,
                 temp_dir: str = "./temp"):
        """
        Initialize stream manager.
        
        Args:
            rtmp_url: RTMP server URL
            rtmp_key: RTMP stream key
            default_video_path: Path to default video to play when no clips available
            output_resolution: Output resolution as (width, height)
            fps: Frames per second for the stream
            bitrate: Video bitrate
            transition_type: Transition type (fade, cut, dissolve)
            transition_duration: Duration of transitions in seconds
            temp_dir: Directory for temporary files
        """
        self.rtmp_url = rtmp_url
        self.rtmp_key = rtmp_key
        self.default_video_path = default_video_path
        self.output_resolution = output_resolution
        self.fps = fps
        self.bitrate = bitrate
        self.transition_type = transition_type
        self.transition_duration = transition_duration
        self.temp_dir = Path(temp_dir)
        
        # Create temp directory
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Stream state
        self.is_streaming = False
        self.stream_process = None
        self.stop_event = threading.Event()
        
        # Playlist management
        self.clip_queue = queue.Queue()
        self.current_clip = None
        self.playlist_thread = None
        
        # Playlist file for ffmpeg
        self.playlist_file = self.temp_dir / "playlist.txt"
        self.playlist_lock = threading.Lock()
    
    def add_clip(self, clip_path: str, duration: Optional[float] = None) -> bool:
        """
        Add a clip to the streaming queue.
        Create a smooth transition from previous clip if available.
        
        Args:
            clip_path: Path to video clip
            duration: Duration of clip in seconds (will be calculated if None)
            
        Returns:
            bool: True if added successfully
        """
        if not os.path.exists(clip_path):
            logger.error(f"Clip not found: {clip_path}")
            return False
        
        try:
            # Get clip duration if not provided
            if duration is None:
                duration = self._get_video_duration(clip_path)
            
            # Create a transition to this clip if there's a previous clip in the queue
            clips_in_queue = list(self.clip_queue.queue)
            if clips_in_queue and self.transition_type != 'cut':
                prev_clip = clips_in_queue[-1]['path']
                logger.debug(f"Creating transition from {os.path.basename(prev_clip)} to {os.path.basename(clip_path)}")
                
                # Make sure both clips exist
                if not os.path.exists(prev_clip):
                    logger.error(f"Previous clip not found for transition: {prev_clip}")
                else:
                    # Create a transition clip that blends from previous to new
                    transition_path = self._create_transition_clip(prev_clip, clip_path)
                    if transition_path:
                        # Make sure the transition file exists
                        if os.path.exists(transition_path):
                            # Add transition clip to queue
                            transition_duration = self._get_video_duration(transition_path)
                            self.clip_queue.put({
                                'path': transition_path,
                                'duration': transition_duration,
                                'added_time': time.time(),
                                'is_transition': True
                            })
                            logger.info(f"Added transition clip to queue: {transition_path} (duration: {transition_duration:.2f}s)")
                        else:
                            logger.error(f"Transition clip was created but file doesn't exist: {transition_path}")
                    else:
                        logger.warning("Failed to create transition clip, continuing without transition")
            
            # Add main clip to queue
            self.clip_queue.put({
                'path': clip_path,
                'duration': duration,
                'added_time': time.time(),
                'is_transition': False
            })
            
            logger.info(f"Added clip to stream queue: {clip_path} (duration: {duration:.2f}s)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add clip: {str(e)}")
            return False
    
    def _get_video_duration(self, video_path: str) -> float:
        """
        Get the duration of a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            float: Duration in seconds
        """
        try:
            probe = ffmpeg.probe(video_path)
            duration = float(probe['format']['duration'])
            return duration
        except Exception as e:
            logger.error(f"Failed to get video duration: {str(e)}")
            return 10.0  # Fallback duration
    
    def _update_playlist(self):
        """Update the playlist file for ffmpeg with current clips."""
        with self.playlist_lock:
            with open(self.playlist_file, 'w') as f:
                # Add all clips in queue
                clips = list(self.clip_queue.queue)
                
                if not clips:
                    # Use default video if no clips
                    f.write(f"file '{self.default_video_path}'\n")
                    logger.info("Using default video in playlist")
                else:
                    # Add all clips
                    for clip in clips:
                        f.write(f"file '{clip['path']}'\n")
                        
                        # Add duration for non-last clips
                        if clip != clips[-1]:
                            # If this is a transition clip, use its full duration
                            # Otherwise, trim a bit at the end to allow for smoother transitions
                            if clip.get('is_transition', False):
                                f.write(f"duration {clip['duration']}\n")
                            else:
                                # For main clips, trim the end slightly to make room for the transition
                                # but only if it's not the last clip and not followed by a transition
                                next_index = clips.index(clip) + 1
                                if next_index < len(clips) and not clips[next_index].get('is_transition', False):
                                    trimmed_duration = max(clip['duration'] - self.transition_duration, 0)
                                    f.write(f"duration {trimmed_duration:.3f}\n")
                                else:
                                    f.write(f"duration {clip['duration']}\n")
                    
                    # Count real clips (excluding transitions)
                    real_clips = sum(1 for clip in clips if not clip.get('is_transition', False))
                    logger.info(f"Updated playlist with {real_clips} main clips and {len(clips) - real_clips} transitions")
    
    def _playlist_manager(self):
        """Background thread that manages the playlist."""
        logger.info("Starting playlist manager thread")
        
        last_update_time = 0
        
        while not self.stop_event.is_set():
            try:
                # Check if we need to update the playlist
                current_time = time.time()
                
                # Update playlist at regular intervals or when new clips are added
                if current_time - last_update_time >= 5 or not self.clip_queue.empty():
                    self._update_playlist()
                    last_update_time = current_time
                
                # Sleep briefly
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in playlist manager: {str(e)}")
                time.sleep(5)  # Longer sleep on error
        
        logger.info("Playlist manager thread stopped")
    
    def start_stream(self) -> bool:
        """
        Start streaming to RTMP.
        
        Returns:
            bool: True if started successfully
        """
        if self.is_streaming:
            logger.warning("Stream already running")
            return True
        
        try:
            # Make sure the default video exists
            if not os.path.exists(self.default_video_path):
                logger.error(f"Default video not found: {self.default_video_path}")
                return False
                
            # Reset stop event
            self.stop_event.clear()
            
            # Start playlist manager thread
            self.playlist_thread = threading.Thread(target=self._playlist_manager, daemon=True)
            self.playlist_thread.start()
            
            # Initialize playlist with default video
            if self.clip_queue.empty():
                self.add_clip(self.default_video_path)
            
            # Use ffmpeg to stream to RTMP
            width, height = self.output_resolution
            stream_url = f"{self.rtmp_url}/{self.rtmp_key}"
            
            # Construct the ffmpeg command
            ffmpeg_cmd = [
                'ffmpeg',
                '-re',  # Read input at native frame rate
                '-stream_loop', '-1',  # Loop the playlist indefinitely
                '-f', 'concat',  # Use concat demuxer
                '-safe', '0',
                '-i', str(self.playlist_file),  # Input from playlist
                '-c:v', 'libx264',  # Video codec
                '-preset', 'medium',  # Encoding speed/quality balance
                '-b:v', self.bitrate,  # Video bitrate
                '-maxrate', self.bitrate,
                '-bufsize', str(int(self.bitrate.replace('k', '')) * 2) + 'k',
                '-g', str(self.fps * 2),  # GOP size
                '-keyint_min', str(self.fps),
                '-r', str(self.fps),  # Output frame rate
                '-s', f"{width}x{height}",  # Output resolution
                '-c:a', 'aac',  # Audio codec
                '-b:a', '128k',  # Audio bitrate
                '-ar', '44100',  # Audio sample rate
                '-f', 'flv',  # Output format
                stream_url  # RTMP URL
            ]
            
            # Start the ffmpeg process
            logger.info(f"Starting ffmpeg stream to {self.rtmp_url}")
            logger.debug(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
            
            self.stream_process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Check if process started successfully
            if self.stream_process.poll() is not None:
                error = self.stream_process.stderr.read()
                logger.error(f"FFmpeg process failed to start: {error}")
                return False
                
            self.is_streaming = True
            logger.info("Stream started successfully")
            
            # Start monitoring thread
            threading.Thread(target=self._monitor_stream, daemon=True).start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start stream: {str(e)}")
            self.stop_stream()
            return False
    
    def _monitor_stream(self):
        """Monitor the ffmpeg process and log its output."""
        logger.info("Starting stream monitor thread")
        
        try:
            while self.is_streaming and not self.stop_event.is_set():
                if self.stream_process.poll() is not None:
                    # Process ended
                    if self.stream_process.returncode != 0:
                        error = self.stream_process.stderr.read()
                        logger.error(f"FFmpeg process exited with code {self.stream_process.returncode}: {error}")
                    else:
                        logger.info("FFmpeg process ended normally")
                    
                    self.is_streaming = False
                    break
                
                # Check ffmpeg output for errors
                line = self.stream_process.stderr.readline().strip()
                if line:
                    if "error" in line.lower():
                        logger.error(f"FFmpeg error: {line}")
                    elif "warning" in line.lower():
                        logger.warning(f"FFmpeg warning: {line}")
                    else:
                        logger.debug(f"FFmpeg: {line}")
                
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in stream monitor: {str(e)}")
        
        logger.info("Stream monitor thread stopped")
    
    def stop_stream(self):
        """Stop the stream and clean up resources."""
        logger.info("Stopping stream...")
        
        # Signal threads to stop
        self.stop_event.set()
        
        # Terminate ffmpeg process
        if self.stream_process and self.stream_process.poll() is None:
            # Try graceful termination first
            self.stream_process.terminate()
            try:
                self.stream_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if not terminated
                self.stream_process.kill()
        
        # Wait for threads to finish
        if self.playlist_thread and self.playlist_thread.is_alive():
            self.playlist_thread.join(timeout=5)
        
        self.is_streaming = False
        logger.info("Stream stopped")
    
    def _create_transition_clip(self, from_clip: str, to_clip: str) -> Optional[str]:
        """
        Create a cinematic transition clip that perfectly blends between the last frame of one video
        and the first frame of the next video with professional-quality effects.
        
        Args:
            from_clip: Path to source clip
            to_clip: Path to destination clip
            
        Returns:
            str: Path to transition clip or None if failed
        """
        # Create a unique output path for this transition
        transition_dir = os.path.join(str(self.temp_dir), "transitions")
        os.makedirs(transition_dir, exist_ok=True)
        
        # Create a unique name for the transition file
        from_name = os.path.basename(from_clip).split('.')[0]
        to_name = os.path.basename(to_clip).split('.')[0]
        transition_name = f"transition_{from_name}_to_{to_name}.mp4"
        output_path = os.path.join(transition_dir, transition_name)
        
        # Set fixed 3-second transition duration (more cinematic feel)
        transition_duration = 3.0
        
        try:
            # More robust extraction of frames - get multiple frames to ensure quality
            # Extract last 5 frames from source video to handle potential decoding issues
            end_frames_dir = os.path.join(transition_dir, "end_frames")
            os.makedirs(end_frames_dir, exist_ok=True)
            
            (
                ffmpeg
                .input(from_clip)
                .filter('select', 'gte(n,n_frames-6)')  # Get the last 5 frames
                .output(os.path.join(end_frames_dir, 'frame_%03d.png'), start_number=1)
                .run(quiet=True, overwrite_output=True)
            )
            
            # Get the last frame (highest numbered file)
            end_frames = sorted([f for f in os.listdir(end_frames_dir) if f.endswith('.png')])
            if not end_frames:
                raise Exception("Failed to extract end frames from source video")
            last_frame = os.path.join(end_frames_dir, end_frames[-1])  # Use the very last frame
            
            # Extract first 5 frames from destination video
            start_frames_dir = os.path.join(transition_dir, "start_frames")
            os.makedirs(start_frames_dir, exist_ok=True)
            
            (
                ffmpeg
                .input(to_clip)
                .filter('select', 'lt(n,5)')  # Get the first 5 frames
                .output(os.path.join(start_frames_dir, 'frame_%03d.png'), start_number=1)
                .run(quiet=True, overwrite_output=True)
            )
            
            # Get the first frame
            start_frames = sorted([f for f in os.listdir(start_frames_dir) if f.endswith('.png')])
            if not start_frames:
                raise Exception("Failed to extract start frames from destination video")
            first_frame = os.path.join(start_frames_dir, start_frames[0])  # Use the very first frame
            
            # Find dimensions of both frames to ensure compatibility
            from_probe = ffmpeg.probe(last_frame)
            to_probe = ffmpeg.probe(first_frame)
            
            # Use the max dimensions to avoid scaling artifacts
            width = max(int(from_probe['streams'][0]['width']), int(to_probe['streams'][0]['width']))
            height = max(int(from_probe['streams'][0]['height']), int(to_probe['streams'][0]['height']))
            
            # Ensure dimensions are even (required for some codecs)
            width = width + (width % 2)
            height = height + (height % 2)
            
            # Select transition effect based on content analysis
            import random
            
            # Load the images
            img1 = Image.open(last_frame).convert('RGBA')
            img2 = Image.open(first_frame).convert('RGBA')
            
            # Analyze images to determine the best transition effect
            # Calculate average brightness, contrast, and color differences
            stat1 = ImageStat.Stat(img1.convert('RGB'))
            stat2 = ImageStat.Stat(img2.convert('RGB'))
            
            # Brightness calculation
            brightness1 = sum(stat1.mean[:3]) / 3
            brightness2 = sum(stat2.mean[:3]) / 3
            brightness_diff = abs(brightness1 - brightness2)
            
            # Color difference
            color_diff = sum(abs(m1 - m2) for m1, m2 in zip(stat1.mean, stat2.mean)) / 3
            
            # Texture difference (using a simple edge detection)
            edges1 = img1.convert('L').filter(ImageFilter.FIND_EDGES)
            edges2 = img2.convert('L').filter(ImageFilter.FIND_EDGES)
            edge_stat1 = ImageStat.Stat(edges1)
            edge_stat2 = ImageStat.Stat(edges2)
            texture_diff = abs(edge_stat1.mean[0] - edge_stat2.mean[0])
            
            # Motion analysis (determine likely scene movement from the names of clips)
            from_name_lower = os.path.basename(from_clip).lower()
            to_name_lower = os.path.basename(to_clip).lower()
            
            # Keywords that suggest certain transition types
            speed_keywords = ["car", "race", "run", "fast", "chase", "speed"]
            zoom_keywords = ["space", "stars", "galaxy", "universe", "cosmos"]
            wipe_keywords = ["slide", "swipe", "sweep", "pan"]
            rotate_keywords = ["spin", "twirl", "spiral", "twist", "dance"]
            fade_keywords = ["dream", "sleep", "night", "dark", "fog", "mist"]
            ripple_keywords = ["water", "ocean", "sea", "river", "lake", "rain", "wave"]
            
            # Analyze both filenames
            has_speed = any(kw in from_name_lower or kw in to_name_lower for kw in speed_keywords)
            has_zoom = any(kw in from_name_lower or kw in to_name_lower for kw in zoom_keywords)
            has_wipe = any(kw in from_name_lower or kw in to_name_lower for kw in wipe_keywords)
            has_rotate = any(kw in from_name_lower or kw in to_name_lower for kw in rotate_keywords)
            has_fade = any(kw in from_name_lower or kw in to_name_lower for kw in fade_keywords)
            has_ripple = any(kw in from_name_lower or kw in to_name_lower for kw in ripple_keywords)
            
            # Determine the most suitable transition based on both image analysis and content analysis
            if has_zoom:
                transition_effect = "zoom"
            elif has_ripple:
                transition_effect = "ripple"
            elif has_rotate:
                transition_effect = "rotate"
            elif has_wipe:
                transition_effect = "wipe"
            elif has_speed:
                transition_effect = "slide"
            elif has_fade:
                transition_effect = "fade"
            elif brightness_diff > 50:
                transition_effect = "fade"
            elif color_diff > 50:
                transition_effect = "dissolve"
            elif texture_diff > 20:
                transition_effect = "morph"
            else:
                # Choose a cinematic transition
                cinematic_transitions = ["zoom", "slide", "rotate", "blur", "wipe", "ripple"]
                transition_effect = random.choice(cinematic_transitions)
            
            logger.info(f"Selected {transition_effect} transition based on scene analysis")
            
            # Resize both images to consistent dimensions
            img1 = img1.resize((width, height), Image.LANCZOS)
            img2 = img2.resize((width, height), Image.LANCZOS)
            
            # Create a directory for individual transition frames
            frames_dir = os.path.join(transition_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            # Determine frame count (30fps for smooth transitions)
            fps = 30
            frame_count = int(transition_duration * fps)
            
            logger.info(f"Creating {frame_count} transition frames with {transition_effect} effect")
            
            # Generate the frames for the transition with the selected effect
            for i in range(frame_count):
                # Calculate blend factor (0.0 to 1.0)
                alpha = i / (frame_count - 1) if frame_count > 1 else 0.5
                
                # Apply easing function for smoother transitions
                # Cubic easing for more natural motion
                if alpha < 0.5:
                    # Ease in - slow start, faster middle
                    eased_alpha = 4 * alpha * alpha * alpha
                else:
                    # Ease out - faster middle, slow end
                    eased_alpha = 1 - pow(-2 * alpha + 2, 3) / 2
                
                if transition_effect == "fade":
                    # Enhanced crossfade with brightness adjustment
                    # This creates a more cinematic fade by slightly adjusting brightness
                    if eased_alpha < 0.5:
                        # First half - slightly darken img1
                        temp1 = ImageEnhance.Brightness(img1).enhance(1.0 - 0.2 * eased_alpha * 2)
                        temp2 = ImageEnhance.Brightness(img2).enhance(0.8 + 0.2 * eased_alpha * 2)
                        blended = Image.blend(temp1, temp2, eased_alpha * 2)
                    else:
                        # Second half - gradually return to normal brightness
                        adjusted_alpha = (eased_alpha - 0.5) * 2  # Rescale to 0-1
                        temp1 = ImageEnhance.Brightness(img1).enhance(0.9 - 0.1 * adjusted_alpha)
                        temp2 = ImageEnhance.Brightness(img2).enhance(0.9 + 0.1 * adjusted_alpha)
                        blended = Image.blend(temp1, temp2, 0.5 + adjusted_alpha/2)
                    
                elif transition_effect == "dissolve":
                    # Dissolve effect with granular random pixels
                    # Create a custom dissolve mask
                    mask = Image.new('L', (width, height), 0)
                    mask_pixels = mask.load()
                    noise_threshold = int(255 * eased_alpha)
                    
                    # Apply random noise pattern
                    for x in range(width):
                        for y in range(height):
                            if random.randint(0, 255) < noise_threshold:
                                mask_pixels[x, y] = 255
                    
                    # Blur the mask for smoother edges
                    mask = mask.filter(ImageFilter.GaussianBlur(radius=1))
                    
                    # Create composite image using the mask
                    blended = Image.composite(img2, img1, mask)
                    
                elif transition_effect == "zoom":
                    # Enhanced zoom transition with motion blur
                    if eased_alpha < 0.5:
                        # Zoom out from first image
                        zoom_factor = 1.0 + (0.5 - eased_alpha) * 0.8
                        zoom_img = img1.resize((int(width * zoom_factor), int(height * zoom_factor)), Image.LANCZOS)
                        
                        # Apply progressive motion blur
                        blur_radius = max(0, (0.5 - eased_alpha) * 3)
                        if blur_radius > 0:
                            zoom_img = zoom_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                            
                        # Crop to original size
                        left = (zoom_img.width - width) // 2
                        top = (zoom_img.height - height) // 2
                        blended = zoom_img.crop((left, top, left + width, top + height))
                    else:
                        # Zoom in to second image
                        adjusted_alpha = (eased_alpha - 0.5) * 2  # Rescale to 0-1
                        reverse_alpha = 1 - adjusted_alpha
                        
                        # Start with a slight zoom out, then zoom in
                        zoom_factor = 1.0 + reverse_alpha * 0.3
                        zoom_img = img2.resize((int(width * zoom_factor), int(height * zoom_factor)), Image.LANCZOS)
                        
                        # Apply progressive sharpening as we zoom in
                        sharpness = 1.0 + adjusted_alpha * 0.5
                        zoom_img = ImageEnhance.Sharpness(zoom_img).enhance(sharpness)
                        
                        # Crop to original size
                        left = (zoom_img.width - width) // 2
                        top = (zoom_img.height - height) // 2
                        blended = zoom_img.crop((left, top, left + width, top + height))
                        
                elif transition_effect == "slide":
                    # Sliding transition with slight parallax effect
                    direction = random.choice(["left", "right", "up", "down"])
                    
                    # Create blank canvas
                    blended = Image.new('RGBA', (width, height), (0, 0, 0, 0))
                    
                    if direction == "left":
                        offset = int(width * (1 - eased_alpha))
                        blended.paste(img1, (-width + offset, 0))
                        blended.paste(img2, (offset, 0))
                    elif direction == "right":
                        offset = int(width * (1 - eased_alpha))
                        blended.paste(img1, (offset, 0))
                        blended.paste(img2, (-width + offset, 0))
                    elif direction == "up":
                        offset = int(height * (1 - eased_alpha))
                        blended.paste(img1, (0, -height + offset))
                        blended.paste(img2, (0, offset))
                    elif direction == "down":
                        offset = int(height * (1 - eased_alpha))
                        blended.paste(img1, (0, offset))
                        blended.paste(img2, (0, -height + offset))
                    
                elif transition_effect == "rotate":
                    # Rotating transition with slight zoom
                    angle = eased_alpha * 10  # Gentle 10 degree rotation
                    
                    # Rotate and zoom first image
                    rotated1 = img1.rotate(angle, resample=Image.BICUBIC, expand=False)
                    zoom_factor1 = 1.0 + eased_alpha * 0.1
                    zoomed1 = rotated1.resize((int(width * zoom_factor1), int(height * zoom_factor1)), Image.LANCZOS)
                    
                    # Rotate and zoom second image in opposite direction
                    rotated2 = img2.rotate(-angle, resample=Image.BICUBIC, expand=False)
                    zoom_factor2 = 1.1 - eased_alpha * 0.1
                    zoomed2 = rotated2.resize((int(width * zoom_factor2), int(height * zoom_factor2)), Image.LANCZOS)
                    
                    # Crop to center
                    crop1 = self._center_crop(zoomed1, width, height)
                    crop2 = self._center_crop(zoomed2, width, height)
                    
                    # Blend the images
                    blended = Image.blend(crop1, crop2, eased_alpha)
                    
                elif transition_effect == "blur":
                    # Blur-through transition (blur first image, crossfade, then sharpen second)
                    if eased_alpha < 0.33:
                        # First third: progressively blur first image
                        blur_factor = eased_alpha * 3 * 10  # Scale to 0-10
                        blurred = img1.filter(ImageFilter.GaussianBlur(radius=blur_factor))
                        blended = blurred
                    elif eased_alpha < 0.66:
                        # Middle third: crossfade between blurred images
                        middle_alpha = (eased_alpha - 0.33) * 3  # Scale to 0-1
                        blurred1 = img1.filter(ImageFilter.GaussianBlur(radius=10))
                        blurred2 = img2.filter(ImageFilter.GaussianBlur(radius=10 * (1 - middle_alpha)))
                        blended = Image.blend(blurred1, blurred2, middle_alpha)
                    else:
                        # Last third: progressively sharpen second image
                        sharp_alpha = (eased_alpha - 0.66) * 3  # Scale to 0-1
                        blur_factor = 10 * (1 - sharp_alpha)
                        blended = img2.filter(ImageFilter.GaussianBlur(radius=blur_factor))
                
                elif transition_effect == "wipe":
                    try:
                        # Enhanced directional wipe with soft edge
                        direction = random.choice(["left", "right", "up", "down", "radial"])
                        logger.debug(f"Creating {direction} wipe transition")
                        
                        if direction == "radial":
                            # Radial wipe from center
                            mask = Image.new('L', (width, height), 0)
                            draw = ImageDraw.Draw(mask)
                            
                            # Calculate max radius needed to cover the image
                            max_radius = int(((width/2)**2 + (height/2)**2)**0.5)
                            current_radius = int(max_radius * eased_alpha)
                            
                            # Draw circle on mask
                            draw.ellipse((width/2 - current_radius, height/2 - current_radius, 
                                        width/2 + current_radius, height/2 + current_radius), fill=255)
                            
                            # Apply feathering to edge
                            mask = mask.filter(ImageFilter.GaussianBlur(radius=10))
                            
                            # Create composite
                            blended = Image.composite(img2, img1, mask)
                        else:
                            # Directional wipe with soft edge
                            mask = Image.new('L', (width, height), 0)
                            draw = ImageDraw.Draw(mask)
                            
                            if direction == "left":
                                wipe_pos = int(width * eased_alpha)
                                draw.rectangle((0, 0, wipe_pos, height), fill=255)
                            elif direction == "right":
                                wipe_pos = int(width * (1 - eased_alpha))
                                draw.rectangle((wipe_pos, 0, width, height), fill=255)
                            elif direction == "up":
                                wipe_pos = int(height * eased_alpha)
                                draw.rectangle((0, 0, width, wipe_pos), fill=255)
                            elif direction == "down":
                                wipe_pos = int(height * (1 - eased_alpha))
                                draw.rectangle((0, wipe_pos, width, height), fill=255)
                            
                            # Add feathered edge
                            mask = mask.filter(ImageFilter.GaussianBlur(radius=20))
                            
                            # Create composite
                            blended = Image.composite(img2, img1, mask)
                            
                        logger.debug(f"Wipe transition ({direction}) created successfully")
                    except Exception as e:
                        logger.warning(f"Error creating wipe transition: {str(e)}, falling back to simple crossfade")
                        # Fallback to simple crossfade
                        blended = Image.blend(img1, img2, eased_alpha)
                
                elif transition_effect == "ripple":
                    # Ripple transition effect
                    # Create displacement maps for ripple effect
                    displacement = eased_alpha * 30  # Max displacement in pixels
                    
                    # Create ripple displacement maps using numpy for better performance
                    displ_map = Image.new('RGB', (width, height), (128, 128, 128))
                    draw = ImageDraw.Draw(displ_map)
                    
                    # Use numpy for faster calculation
                    try:
                        import numpy as np
                        logger.debug("Creating ripple effect displacement map")
                        
                        # Create coordinate grids
                        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
                        center_x, center_y = width // 2, height // 2
                        
                        # Calculate distances from center (vectorized)
                        dx = x_coords - center_x
                        dy = y_coords - center_y
                        distances = np.sqrt(dx**2 + dy**2)
                        
                        # Calculate ripple effect (vectorized)
                        factor = 127 * np.sin(distances / 10 - eased_alpha * math.pi * 5)
                        
                        # Calculate displacement values (make sure to clip values to valid range)
                        r_values = np.clip(128 + factor * displacement / 30, 0, 255).astype(np.uint8)
                        g_values = r_values.copy()
                        
                        # Create RGB displacement map
                        displ_array = np.stack([r_values, g_values, np.full_like(r_values, 128)], axis=-1)
                        displ_map = Image.fromarray(displ_array, 'RGB')
                        logger.debug("Ripple displacement map created successfully")
                    except Exception as e:
                        # Fallback to simple displacement map
                        logger.warning(f"Error creating ripple displacement map: {str(e)}, using fallback")
                        displ_map = Image.new('RGB', (width, height), (128, 128, 128))
                    
                    # Apply displacement map
                    blended = img1.copy()
                    if eased_alpha < 0.5:
                        # First half: ripple first image
                        distorted = self._apply_displacement_map(img1, displ_map, displacement)
                        adjusted_alpha = eased_alpha * 2
                        blended = Image.blend(img1, distorted, adjusted_alpha)
                    else:
                        # Second half: blend to second image
                        distorted = self._apply_displacement_map(img1, displ_map, displacement * (1 - eased_alpha) * 2)
                        adjusted_alpha = (eased_alpha - 0.5) * 2
                        blended = Image.blend(distorted, img2, adjusted_alpha)
                
                elif transition_effect == "morph":
                    # Simple morphing-like effect
                    # First identify key features in both images
                    blend_alpha = eased_alpha
                    
                    # Create base blend
                    blended = Image.blend(img1, img2, blend_alpha)
                    
                    # Apply additional morphing-like effects
                    # Warp based on alpha
                    if blend_alpha < 0.5:
                        warp_strength = blend_alpha * 2
                        source_img = img1
                    else:
                        warp_strength = (1 - blend_alpha) * 2
                        source_img = img2
                    
                    # Create a slight wave distortion
                    if warp_strength > 0:
                        for y in range(height):
                            wave = int(math.sin(y / 20 + blend_alpha * math.pi * 2) * 5 * warp_strength)
                            if wave != 0:
                                row = blended.crop((0, y, width, y+1))
                                blended.paste(row, (wave, y))
                                
                else:
                    # Fallback to simple crossfade for any unimplemented effect
                    blended = Image.blend(img1, img2, eased_alpha)
                
                # Apply final processing for all transitions
                # Add subtle vignette effect for cinematic feel
                if i < frame_count // 3 or i > frame_count * 2 // 3:
                    # Only add vignette at beginning and end of transition
                    blended = self._add_vignette(blended.convert('RGB'), strength=0.2)
                
                # Apply slight film grain for cinematic feel
                blended = self._add_film_grain(blended.convert('RGB'), strength=0.05)
                
                # Save the frame
                frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
                blended.convert('RGB').save(frame_path, quality=95)
            
            # Use ffmpeg to create high-quality video from frames
            (
                ffmpeg
                .input(os.path.join(frames_dir, "frame_%04d.png"), framerate=fps)
                .output(output_path, 
                        vcodec='libx264', 
                        pix_fmt='yuv420p', 
                        preset='slow',  # Higher quality encoding
                        crf=18,         # Higher quality (lower is better)
                        tune='film',    # Optimize for film-like content
                        profile='high', # High profile for better quality
                        movflags='+faststart')  # Optimize for streaming
                .run(quiet=True, overwrite_output=True)
            )
            
            # Clean up temporary files
            import shutil
            for dir_path in [end_frames_dir, start_frames_dir, frames_dir]:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
            
            logger.info(f"Created cinematic transition clip: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create transition clip: {str(e)}")
            
            # Create a more reliable fallback transition
            try:
                logger.info("Creating fallback transition with direct ffmpeg crossfade")
                
                # Extract segments from videos using ffmpeg directly
                from_duration = self._get_video_duration(from_clip)
                
                # Set segment durations
                segment_duration = 1.5  # 1.5 seconds from each video
                
                # Create the transition using ffmpeg's xfade filter - corrected approach
                try:
                    # Create two input streams with ffmpeg-python
                    input1 = ffmpeg.input(from_clip).trim(start=max(0, from_duration - segment_duration), end=from_duration).setpts('PTS-STARTPTS')
                    input2 = ffmpeg.input(to_clip).trim(start=0, end=segment_duration).setpts('PTS-STARTPTS')
                    
                    # Apply xfade filter
                    joined = ffmpeg.filter([input1, input2], 'xfade', transition='fade', duration=1.5, offset=segment_duration - 1.5)
                    
                    # Output
                    out = ffmpeg.output(joined, output_path, vcodec='libx264', pix_fmt='yuv420p', preset='medium', crf=23)
                    out.run(quiet=True, overwrite_output=True)
                except Exception as e:
                    logger.error(f"Error in ffmpeg xfade: {str(e)}")
                    
                    # Even more basic fallback - direct command
                    import subprocess
                    try:
                        cmd = [
                            'ffmpeg', '-y',
                            '-i', from_clip,
                            '-i', to_clip,
                            '-filter_complex', 
                            f"[0:v]trim=start={max(0, from_duration - segment_duration)}:end={from_duration},setpts=PTS-STARTPTS[v0];" +
                            f"[1:v]trim=start=0:end={segment_duration},setpts=PTS-STARTPTS[v1];" +
                            f"[v0][v1]xfade=transition=fade:duration=1.5:offset={segment_duration - 1.5}[out]",
                            '-map', '[out]',
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'medium', '-crf', '23',
                            output_path
                        ]
                        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        logger.info("Created fallback transition using direct ffmpeg command")
                    except Exception as e2:
                        logger.error(f"Error in direct ffmpeg command: {str(e2)}")
                        raise
                
                logger.info(f"Created fallback transition clip: {output_path}")
                return output_path
            except Exception as e:
                logger.error(f"Failed to create fallback transition: {str(e)}")
                
                # Ultimate fallback: create a very simple transition using direct ffmpeg command
                try:
                    # This is the most direct approach possible - create a 1 second black video
                    import subprocess
                    
                    cmd = [
                        'ffmpeg', '-y',
                        '-f', 'lavfi',  # Use lavfi input format for synthetic sources
                        '-i', 'color=black:s=512x512:r=30:d=1',  # Black color source, 512x512, 30fps, 1 second duration
                        '-c:v', 'libx264',  # H.264 codec
                        '-pix_fmt', 'yuv420p',  # Standard pixel format for compatibility
                        '-preset', 'ultrafast',  # Fastest encoding
                        '-crf', '23',  # Reasonable quality
                        output_path  # Output file
                    ]
                    
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    logger.info(f"Created basic emergency fallback transition: {output_path}")
                    return output_path
                    
                except Exception as e:
                    logger.error(f"Failed to create even the basic fallback: {str(e)}")
                    return None
    
    def _center_crop(self, image, target_width, target_height):
        """Helper method to crop an image to center with target dimensions."""
        width, height = image.size
        left = (width - target_width) // 2
        top = (height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        return image.crop((left, top, right, bottom))
    
    def _add_vignette(self, image, strength=0.3):
        """Add a subtle vignette effect to an image."""
        width, height = image.size
        
        # Create a radial gradient mask
        mask = Image.new('L', (width, height), 255)
        draw = ImageDraw.Draw(mask)
        
        # Calculate parameters for the elliptical gradient
        cx, cy = width/2, height/2
        max_radius = max(width, height) / 2
        
        # Draw the gradient
        for y in range(height):
            for x in range(width):
                # Calculate normalized distance from center (0.0 to 1.0)
                dx, dy = (x - cx) / cx, (y - cy) / cy
                distance = (dx**2 + dy**2)**0.5
                
                # Apply quadratic falloff for more natural vignette
                # Adjust the 1.5 value to control vignette size
                distance = min(1.0, distance * 1.5)
                
                # Calculate brightness (darker at edges)
                brightness = int(255 * (1.0 - distance**2 * strength))
                mask.putpixel((x, y), brightness)
        
        # Apply the mask
        return ImageChops.multiply(image, mask.convert('RGB'))
    
    def _add_film_grain(self, image, strength=0.1):
        """Add subtle film grain effect (optimized)."""
        width, height = image.size
        
        # Create a numpy array for better performance
        import numpy as np
        # Generate random noise array (much faster than pixel-by-pixel)
        noise = np.random.randint(0, int(255 * strength), (height, width))
        # Make it grayscale (same value for all channels)
        noise_rgb = np.stack([noise, noise, noise], axis=2).astype(np.uint8)
        
        # Convert back to PIL image
        grain = Image.fromarray(noise_rgb, 'RGB')
        
        # Apply grain using screen blending
        return ImageChops.screen(image, grain)
    
    def _apply_displacement_map(self, image, displ_map, strength):
        """Apply a displacement map to an image."""
        try:
            logger.debug(f"Applying displacement map with strength {strength}")
            
            # Try using numpy for faster processing
            try:
                import numpy as np
                
                # Convert PIL images to numpy arrays
                image_array = np.array(image)
                displ_array = np.array(displ_map)
                
                # Get dimensions
                height, width = image_array.shape[:2]
                
                # Create coordinate arrays
                y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
                
                # Calculate displacement from the displacement map
                r_channel = displ_array[:, :, 0].astype(np.float32)
                g_channel = displ_array[:, :, 1].astype(np.float32)
                
                # Calculate the actual displacement in pixels
                dx = ((r_channel - 128) / 128.0 * strength).astype(np.int32)
                dy = ((g_channel - 128) / 128.0 * strength).astype(np.int32)
                
                # Apply displacement with bounds checking
                source_x = np.clip(x_coords + dx, 0, width - 1)
                source_y = np.clip(y_coords + dy, 0, height - 1)
                
                # Create output array
                result_array = np.zeros_like(image_array)
                
                # Map pixels
                for y in range(height):
                    for x in range(width):
                        src_x = source_x[y, x]
                        src_y = source_y[y, x]
                        result_array[y, x] = image_array[src_y, src_x]
                
                # Convert back to PIL
                result = Image.fromarray(result_array)
                logger.debug("Displacement map applied using numpy optimization")
                return result
                
            except (ImportError, Exception) as e:
                # Fallback to pixel-by-pixel approach if numpy fails
                logger.warning(f"Numpy displacement failed: {str(e)}, using pixel-by-pixel fallback")
                raise
                
        except Exception:
            # Pure PIL fallback method (slower but more reliable)
            logger.debug("Using pure PIL displacement method (slower)")
            result = image.copy()
            width, height = image.size
            
            # Apply displacement at lower resolution for speed (every 2 pixels)
            step = 2
            for y in range(0, height, step):
                for x in range(0, width, step):
                    try:
                        # Get displacement values from map
                        r, g, b = displ_map.getpixel((x, y))
                        
                        # Calculate displacement
                        dx = int((r - 128) / 128.0 * strength)
                        dy = int((g - 128) / 128.0 * strength)
                        
                        # Get source pixel with bounds checking
                        sx = max(0, min(width-1, x + dx))
                        sy = max(0, min(height-1, y + dy))
                        
                        # Copy pixel and fill a small block for speed
                        pixel = image.getpixel((sx, sy))
                        for by in range(min(step, height-y)):
                            for bx in range(min(step, width-x)):
                                result.putpixel((x+bx, y+by), pixel)
                    except Exception as e:
                        # Skip problematic pixels
                        continue
            
            logger.debug("PIL displacement map applied successfully")
            return result
    
    def close(self):
        """Clean up resources."""
        self.stop_stream()
        
        # Clean up temp files
        if os.path.exists(self.playlist_file):
            os.remove(self.playlist_file)
            
        # Clean up transition clips
        transition_dir = os.path.join(str(self.temp_dir), "transitions")
        if os.path.exists(transition_dir):
            try:
                import shutil
                shutil.rmtree(transition_dir)
                logger.info(f"Cleaned up transition directory: {transition_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up transition directory: {str(e)}")
        
        logger.info("Stream manager closed")


class SimulatedStreamManager(StreamManager):
    """Simulated stream manager for testing without actual streaming."""
    
    def __init__(self, default_video_path: str,
                 output_resolution: Tuple[int, int] = (1280, 720),
                 fps: int = 30,
                 transition_type: str = "fade",
                 transition_duration: float = 1.0,
                 temp_dir: str = "./temp"):
        """
        Initialize simulated stream manager.
        
        Args:
            default_video_path: Path to default video to play when no clips available
            output_resolution: Output resolution as (width, height)
            fps: Frames per second for the stream
            transition_type: Transition type (fade, cut, dissolve)
            transition_duration: Duration of transitions in seconds
            temp_dir: Directory for temporary files
        """
        super().__init__("rtmp://localhost/live", "simulatedkey", 
                        default_video_path, output_resolution, 
                        fps, "4000k", transition_type, 
                        transition_duration, temp_dir)
    
    def start_stream(self) -> bool:
        """
        Simulate starting a stream.
        
        Returns:
            bool: True if started successfully
        """
        if self.is_streaming:
            logger.warning("Simulated stream already running")
            return True
        
        try:
            # Make sure the default video exists
            if not os.path.exists(self.default_video_path):
                logger.error(f"Default video not found: {self.default_video_path}")
                return False
                
            # Reset stop event
            self.stop_event.clear()
            
            # Start playlist manager thread
            self.playlist_thread = threading.Thread(target=self._playlist_manager, daemon=True)
            self.playlist_thread.start()
            
            # Initialize playlist with default video
            if self.clip_queue.empty():
                self.add_clip(self.default_video_path)
            
            # Simulate ffmpeg process
            logger.info("Starting simulated RTMP stream")
            self.is_streaming = True
            
            # Start simulated monitor thread
            threading.Thread(target=self._simulated_monitor, daemon=True).start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start simulated stream: {str(e)}")
            self.stop_stream()
            return False
    
    def _simulated_monitor(self):
        """Simulate monitoring the stream process."""
        logger.info("Starting simulated stream monitor thread")
        
        try:
            while not self.stop_event.is_set():
                # Log current playlist state
                with self.playlist_lock:
                    if os.path.exists(self.playlist_file):
                        with open(self.playlist_file, 'r') as f:
                            playlist_content = f.read()
                        logger.debug(f"Current playlist: {len(playlist_content.splitlines())} entries")
                    
                # Sleep for a bit
                time.sleep(30)
                
        except Exception as e:
            logger.error(f"Error in simulated monitor: {str(e)}")
        
        logger.info("Simulated stream monitor thread stopped")
    
    def stop_stream(self):
        """Simulate stopping the stream."""
        logger.info("Stopping simulated stream...")
        
        # Signal threads to stop
        self.stop_event.set()
        
        # Wait for threads to finish
        if self.playlist_thread and self.playlist_thread.is_alive():
            self.playlist_thread.join(timeout=5)
        
        self.is_streaming = False
        logger.info("Simulated stream stopped")


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    import tempfile
    load_dotenv()
    
    # Create a temporary default video if needed
    default_video = os.environ.get("DEFAULT_VIDEO_PATH", "default_video.mp4")
    if not os.path.exists(default_video):
        # Create a simple default video
        temp_default = tempfile.gettempdir() + "/default_video.mp4"
        try:
            # Generate a simple test pattern video
            (
                ffmpeg
                .input('testsrc=size=1280x720:rate=30', f='lavfi', t=30)
                .output(temp_default, vcodec='libx264', pix_fmt='yuv420p', preset='medium')
                .run(quiet=True, overwrite_output=True)
            )
            default_video = temp_default
            print(f"Created temporary default video: {default_video}")
        except Exception as e:
            print(f"Failed to create default video: {str(e)}")
    
    # Test with simulated stream
    stream = SimulatedStreamManager(
        default_video_path=default_video,
        output_resolution=(1280, 720),
        fps=30,
        transition_type="fade",
        transition_duration=1.0,
        temp_dir="./temp"
    )
    
    # Start the stream
    stream.start_stream()
    
    try:
        # Simulate adding a few clips
        for i in range(3):
            print(f"Adding clip {i+1} to stream...")
            stream.add_clip(default_video)
            time.sleep(5)
        
        # Let it run for a bit
        print("Letting stream run for 30 seconds...")
        time.sleep(30)
        
    finally:
        stream.close()