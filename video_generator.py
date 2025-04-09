#!/usr/bin/env python3
"""
Video Generator Module

This module handles AI-powered video generation from text prompts.
It uses Hugging Face Diffusers models to generate video clips based on text prompts.
"""

import os
import time
import threading
import queue
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union, Callable

import torch
import numpy as np
from PIL import Image
from slugify import slugify
from tqdm import tqdm
from loguru import logger
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image


class VideoGenerator:
    """Handles AI-powered video generation from text prompts."""
    
    def __init__(self, model_repo: str, output_dir: str, 
                 video_duration: int = 10, fps: int = 24, 
                 resolution: Tuple[int, int] = (512, 512),
                 max_queue_size: int = 5,
                 callback: Callable = None,
                 device: str = None):
        """
        Initialize video generator.
        
        Args:
            model_repo: Hugging Face model repository ID
            output_dir: Directory to save generated videos
            video_duration: Duration of generated videos in seconds
            fps: Frames per second for generated videos
            resolution: Video resolution as (width, height)
            max_queue_size: Maximum number of pending generation tasks
            callback: Function to call when a video is generated
            device: Device to run model on ('cuda', 'mps', or 'cpu')
        """
        self.model_repo = model_repo
        self.output_dir = Path(output_dir)
        self.video_duration = video_duration
        self.fps = fps
        self.resolution = resolution
        self.num_frames = video_duration * fps
        self.max_queue_size = max_queue_size
        self.callback = callback
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"  # For Apple Silicon
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize model
        self.model = None
        
        # Task queue and worker thread
        self.task_queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.worker_thread = None
        self.is_initialized = False
    
    def initialize(self):
        """Download and initialize the model."""
        if self.is_initialized:
            return
            
        logger.info(f"Initializing video generation model from {self.model_repo}")
        
        try:
            # Load the model - this will download it if not already cached
            self.model = StableVideoDiffusionPipeline.from_pretrained(
                self.model_repo, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                variant="fp16" if self.device == "cuda" else None
            )
            
            if self.device == "cuda":
                self.model.to("cuda")
                # Enable memory optimization
                self.model.enable_model_cpu_offload()
            elif self.device == "mps":
                # MPS mode (Apple Silicon)
                self.model.to("mps")
            
            logger.info("Model initialized successfully")
            self.is_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise
    
    def generate_video(self, prompt: Dict) -> Optional[str]:
        """
        Generate a video based on a text prompt.
        
        Args:
            prompt: Prompt data with text and metadata
            
        Returns:
            str: Path to the generated video file or None if failed
        """
        if not self.is_initialized:
            logger.warning("Model not initialized, initializing now...")
            self.initialize()
        
        prompt_text = prompt['text']
        author = prompt.get('author', 'unknown')
        timestamp = prompt.get('timestamp', time.strftime('%Y%m%d-%H%M%S'))
        
        try:
            logger.info(f"Generating video for prompt: {prompt_text}")
            
            # Create a seed image (blank or based on prompt)
            # This is using a placeholder approach
            seed_image = self._create_seed_image()
            
            # SVD pipeline expects a conditioning image
            # We'll use a blank image for now as a placeholder
            conditioning_image = Image.new("RGB", self.resolution, color=(0, 0, 0))
            
            # Generate video frames
            logger.info("Running video generation pipeline...")
            outputs = self.model(
                prompt_text,
                video_length=self.video_duration,
                height=self.resolution[1],
                width=self.resolution[0],
                num_frames=self.num_frames,
                conditioning_image=seed_image,
                guidance_scale=7.5,  # Controls how closely to follow the prompt
                num_inference_steps=30  # Fewer steps for faster generation, more for quality
            )
            
            # Get video frames from the model output
            frames = outputs.frames[0]
            
            # Save video
            slug = slugify(prompt_text[:50])
            filename = f"{timestamp}_{slug}_{author}.mp4"
            output_path = str(self.output_dir / filename)
            
            # Convert frames to video using numpy and save
            self._save_frames_to_video(frames, output_path)
            
            logger.info(f"Video generation complete: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating video: {str(e)}")
            return None
    
    def _create_seed_image(self) -> Image.Image:
        """
        Create a seed image for video generation.
        
        Returns:
            PIL.Image: Seed image for video generation
        """
        # Simple placeholder implementation - create a gray image
        # In a production system, you could use a separate text-to-image model here
        img = Image.new("RGB", self.resolution, color=(127, 127, 127))
        return img
    
    def _save_frames_to_video(self, frames: List[np.ndarray], output_path: str):
        """
        Save frames to a video file using ffmpeg.
        
        Args:
            frames: List of numpy arrays representing video frames
            output_path: Path to save the video file
        """
        import ffmpeg
        
        # Create a temporary directory for frames
        import tempfile
        import shutil
        from pathlib import Path
        
        temp_dir = tempfile.mkdtemp()
        try:
            # Save frames as images
            frame_paths = []
            for i, frame in enumerate(frames):
                # Convert tensor to numpy if needed
                if isinstance(frame, torch.Tensor):
                    frame = frame.cpu().numpy()
                
                # Ensure uint8 format
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)
                
                # Save frame
                frame_path = os.path.join(temp_dir, f"frame_{i:05d}.png")
                Image.fromarray(frame).save(frame_path)
                frame_paths.append(frame_path)
            
            # Use ffmpeg to combine frames into video with proper codec settings
            (
                ffmpeg
                .input(os.path.join(temp_dir, "frame_%05d.png"), framerate=self.fps)
                .output(output_path, 
                        vcodec='libx264',       # Use H.264 codec
                        pix_fmt='yuv420p',      # Standard pixel format for compatibility
                        crf=23,                 # Quality setting
                        movflags='+faststart',  # Optimize for web streaming
                        preset='medium')        # Encoding speed/quality balance
                .run(quiet=True, overwrite_output=True)
            )
            
            # Store current scene information for continuity in future generations
            self.current_elements = elements
            self.scene_history.append({
                'type': scene_type,
                'bg_color': bg_color,
                'prompt': prompt_text
            })
            
            # Keep scene history manageable - only keep last 5 scenes
            if len(self.scene_history) > 5:
                self.scene_history = self.scene_history[-5:]
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
    
    def add_to_queue(self, prompt: Dict) -> bool:
        """
        Add a prompt to the generation queue.
        
        Args:
            prompt: Prompt data with text and metadata
            
        Returns:
            bool: True if added successfully, False otherwise
        """
        try:
            self.task_queue.put(prompt, block=False)
            logger.info(f"Added prompt to generation queue: {prompt['text'][:50]}...")
            return True
        except queue.Full:
            logger.warning("Generation queue is full, prompt rejected")
            return False
    
    def _worker_loop(self):
        """Background worker that processes the generation queue."""
        logger.info("Starting video generation worker thread")
        
        while not self.stop_event.is_set():
            try:
                # Get next prompt from queue with timeout
                prompt = self.task_queue.get(block=True, timeout=1)
                
                # Generate video
                video_path = self.generate_video(prompt)
                
                # Call callback if provided and generation was successful
                if video_path and self.callback:
                    self.callback(prompt, video_path)
                    
                # Mark task as done
                self.task_queue.task_done()
                
            except queue.Empty:
                # No task available, just continue
                continue
            except Exception as e:
                logger.error(f"Error in generation worker: {str(e)}")
                # Sleep briefly before continuing
                time.sleep(1)
        
        logger.info("Video generation worker thread stopped")
    
    def start_worker(self):
        """Start the background worker thread for video generation."""
        if not self.is_initialized:
            logger.info("Initializing model before starting worker...")
            self.initialize()
            
        # Reset stop event
        self.stop_event.clear()
        
        # Start worker thread
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        logger.info("Video generation worker started")
    
    def stop_worker(self):
        """Stop the background worker thread."""
        self.stop_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        logger.info("Video generation worker stopped")
    
    def close(self):
        """Clean up resources."""
        self.stop_worker()
        # Free up GPU memory
        self.model = None
        torch.cuda.empty_cache() if self.device == "cuda" else None
        logger.info("Video generator closed")


class SimulatedVideoGenerator(VideoGenerator):
    """Simulated video generator for testing without actual model."""
    
    def __init__(self, output_dir: str, 
                 video_duration: int = 10, fps: int = 24, 
                 resolution: Tuple[Union[int, str], Union[int, str]] = (512, 512),
                 max_queue_size: int = 5,
                 callback: Callable = None):
        """
        Initialize simulated video generator.
        
        Args:
            output_dir: Directory to save generated videos
            video_duration: Duration of generated videos in seconds
            fps: Frames per second for generated videos
            resolution: Video resolution as (width, height)
            max_queue_size: Maximum number of pending generation tasks
            callback: Function to call when a video is generated
        """
        # Ensure resolution values are integers
        if isinstance(resolution[0], str) or isinstance(resolution[1], str):
            width = int(resolution[0]) if isinstance(resolution[0], str) else resolution[0]
            height = int(resolution[1]) if isinstance(resolution[1], str) else resolution[1]
            resolution = (width, height)
            
        super().__init__("", output_dir, video_duration, fps, resolution, max_queue_size, callback, "cpu")
        self.is_initialized = True
        
        # Store the last frame and scene elements for continuity
        self.last_frame = None
        self.current_elements = []
        self.scene_history = []  # Store a history of previous scene types and elements
        self.continuity_factor = 0.7  # How much to preserve from previous scene (0-1)
    
    def initialize(self):
        """Simulate model initialization."""
        logger.info("Simulated video generator initialized")
        self.is_initialized = True
    
    def generate_video(self, prompt: Dict) -> Optional[str]:
        """
        Simulate video generation with continuity between clips.
        
        Args:
            prompt: Prompt data with text and metadata
            
        Returns:
            str: Path to the generated video file
        """
        prompt_text = prompt['text']
        author = prompt.get('author', 'unknown')
        timestamp = prompt.get('timestamp', time.strftime('%Y%m%d-%H%M%S'))
        
        try:
            logger.info(f"Simulating video generation for prompt: {prompt_text}")
            
            # Combine current prompt with previous scene information for continuity
            combined_prompt = prompt_text
            if self.scene_history:
                # Get the most recent scene type and key elements
                last_scene = self.scene_history[-1]
                logger.info(f"Building upon previous scene: {last_scene['type']}")
                
                # For better continuity, we could incorporate elements from the previous scene
                # in our interpretation of the new prompt
                combined_prompt = f"{prompt_text} (continuing from {last_scene['type']} scene)"
                logger.info(f"Combined prompt for continuity: {combined_prompt}")
            
            # Simulate processing time
            generation_time = min(5 + len(prompt_text) / 10, 30)  # 5-30 seconds
            logger.info(f"Simulating generation for {generation_time:.1f} seconds...")
            
            for i in range(10):
                if self.stop_event.is_set():
                    return None
                time.sleep(generation_time / 10)
                logger.info(f"Generation progress: {(i+1)*10}%")
            
            # Create a simulated output video that builds on previous scene
            slug = slugify(prompt_text[:50])
            
            # Format timestamp to be more filesystem-friendly (replace colons and periods)
            if isinstance(timestamp, str):
                safe_timestamp = timestamp.replace(':', '-').replace('.', '-')
            else:
                # If it's a datetime object, format it properly
                import datetime
                if isinstance(timestamp, datetime.datetime):
                    safe_timestamp = timestamp.strftime('%Y%m%d-%H%M%S')
                else:
                    # Default fallback
                    safe_timestamp = time.strftime('%Y%m%d-%H%M%S')
                
            filename = f"{safe_timestamp}_{slug}_{author}.mp4"
            output_path = str(self.output_dir / filename)
            
            # Generate video with continuity from previous frames
            self._create_dummy_video(prompt_text, output_path, preserve_elements=len(self.scene_history) > 0)
            
            logger.info(f"Simulated video generation complete: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error in simulated video generation: {str(e)}")
            return None
    
    def _create_dummy_video(self, prompt_text: str, output_path: str, preserve_elements=False):
        """
        Create a dummy video file for simulation that looks more like the prompt.
        Builds upon previous scene elements for continuity if preserve_elements=True.
        
        Args:
            prompt_text: Text prompt
            output_path: Path to save the video file
            preserve_elements: Whether to preserve elements from previous scene
        """
        import ffmpeg
        import numpy as np
        import tempfile
        import random
        import math
        import shutil
        from PIL import Image, ImageDraw, ImageFont, ImageFilter
        
        # Create a temporary directory for frames
        temp_dir = tempfile.mkdtemp()
        try:
            # Create frames with text
            width, height = self.resolution
            # Convert resolution tuple from strings to integers if needed
            if isinstance(width, str):
                width = int(width)
            if isinstance(height, str):
                height = int(height)
                
            # Choose a visualization style based on prompt keywords
            prompt_lower = prompt_text.lower()
            preserved_elements = []
            
            # Determine if we should preserve elements from previous scene
            if preserve_elements and self.current_elements:
                # Keep a percentage of elements from the previous scene for continuity
                preserve_count = int(len(self.current_elements) * self.continuity_factor)
                preserved_elements = random.sample(self.current_elements, min(preserve_count, len(self.current_elements)))
                logger.info(f"Preserving {len(preserved_elements)} elements from previous scene for continuity")
            
            # Create base scene elements based on prompt
            if "forest" in prompt_lower or "garden" in prompt_lower or "nature" in prompt_lower:
                # Forest/garden scene
                bg_color = (20, 80, 20)  # Dark green
                scene_type = "nature"
                elements = list(preserved_elements)  # Start with preserved elements
                
                # Calculate how many new elements to add
                new_tree_count = max(5, 15 - len([e for e in elements if e[0] == "tree"]))
                
                # Add trees
                for _ in range(new_tree_count):
                    x = random.randint(0, width)
                    y = random.randint(height//2, height)
                    size = random.randint(50, 150)
                    elements.append(("tree", x, y, size, (0, 100, 0)))
                
                # Add flowers or mushrooms
                if "mushroom" in prompt_lower:
                    new_mushroom_count = max(5, 20 - len([e for e in elements if e[0] == "mushroom"]))
                    for _ in range(new_mushroom_count):
                        x = random.randint(0, width)
                        y = random.randint(height//2, height)
                        size = random.randint(10, 30)
                        elements.append(("mushroom", x, y, size, (200, 50, 200)))
                else:
                    new_flower_count = max(5, 30 - len([e for e in elements if e[0] == "flower"]))
                    for _ in range(new_flower_count):
                        x = random.randint(0, width)
                        y = random.randint(height-100, height)
                        size = random.randint(5, 15)
                        elements.append(("flower", x, y, size, (random.randint(150, 255), random.randint(150, 255), 0)))
                
            elif "space" in prompt_lower or "star" in prompt_lower or "planet" in prompt_lower:
                # Space scene
                bg_color = (0, 0, 30)  # Dark blue
                scene_type = "space"
                elements = list(preserved_elements)  # Start with preserved elements
                
                # Add stars
                new_star_count = max(50, 200 - len([e for e in elements if e[0] == "star"]))
                for _ in range(new_star_count):
                    x = random.randint(0, width)
                    y = random.randint(0, height)
                    size = random.randint(1, 5)
                    elements.append(("star", x, y, size, (255, 255, 255)))
                
                # Add planets
                new_planet_count = max(1, 3 - len([e for e in elements if e[0] == "planet"]))
                for _ in range(new_planet_count):
                    x = random.randint(0, width)
                    y = random.randint(0, height)
                    size = random.randint(30, 80)
                    elements.append(("planet", x, y, size, (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))))
                
            elif "city" in prompt_lower or "cyberpunk" in prompt_lower or "futuristic" in prompt_lower:
                # City/cyberpunk scene
                bg_color = (30, 30, 50)  # Dark bluish
                scene_type = "city"
                elements = list(preserved_elements)  # Start with preserved elements
                
                # Add buildings
                new_building_count = max(5, 15 - len([e for e in elements if e[0] == "building"]))
                for _ in range(new_building_count):
                    x = random.randint(0, width)
                    y = height
                    size_w = random.randint(50, 100)
                    size_h = random.randint(100, 300)
                    elements.append(("building", x, y, (size_w, size_h), (random.randint(20, 50), random.randint(20, 50), random.randint(20, 50))))
                
                # Add lights
                new_light_count = max(10, 50 - len([e for e in elements if e[0] == "light"]))
                for _ in range(new_light_count):
                    x = random.randint(0, width)
                    y = random.randint(height//2, height)
                    size = random.randint(2, 8)
                    elements.append(("light", x, y, size, (random.randint(150, 255), random.randint(150, 255), 0)))
                
            elif "mountain" in prompt_lower or "landscape" in prompt_lower:
                # Mountain landscape
                bg_color = (100, 150, 200)  # Light blue
                scene_type = "landscape"
                elements = list(preserved_elements)  # Start with preserved elements
                
                # Add mountains
                new_mountain_count = max(2, 5 - len([e for e in elements if e[0] == "mountain"]))
                for _ in range(new_mountain_count):
                    x = random.randint(0, width)
                    y = height//2
                    size_w = random.randint(200, 400)
                    size_h = random.randint(100, 200)
                    elements.append(("mountain", x, y, (size_w, size_h), (100, 100, 100)))
                
                # Add snow if mentioned
                if "snow" in prompt_lower:
                    bg_color = (200, 220, 255)  # Light blue-white
                    new_snow_count = max(20, 100 - len([e for e in elements if e[0] == "snow"]))
                    for _ in range(new_snow_count):
                        x = random.randint(0, width)
                        y = random.randint(0, height)
                        size = random.randint(1, 3)
                        elements.append(("snow", x, y, size, (255, 255, 255)))
                        
                # Add northern lights if mentioned
                if "northern light" in prompt_lower or "aurora" in prompt_lower:
                    new_aurora_count = max(2, 10 - len([e for e in elements if e[0] == "aurora"]))
                    for _ in range(new_aurora_count):
                        x = random.randint(0, width)
                        y = random.randint(50, height//3)
                        size_w = random.randint(100, 300)
                        size_h = random.randint(50, 100)
                        elements.append(("aurora", x, y, (size_w, size_h), (0, 255, 100)))
                
            elif "dragon" in prompt_lower or "castle" in prompt_lower or "medieval" in prompt_lower:
                # Fantasy/medieval scene
                bg_color = (100, 120, 150)  # Muted blue
                scene_type = "fantasy"
                elements = list(preserved_elements)  # Start with preserved elements
                
                # Check if we need to add a castle
                if not any(e[0] == "castle" for e in elements):
                    # Add castle
                    elements.append(("castle", width//2, height//2, 200, (80, 80, 80)))
                
                # Add dragon if mentioned and not already present
                if "dragon" in prompt_lower and not any(e[0] == "dragon" for e in elements):
                    elements.append(("dragon", width//3, height//4, 100, (150, 0, 0)))
                
                # Add clouds
                new_cloud_count = max(3, 8 - len([e for e in elements if e[0] == "cloud"]))
                for _ in range(new_cloud_count):
                    x = random.randint(0, width)
                    y = random.randint(50, height//3)
                    size = random.randint(50, 100)
                    elements.append(("cloud", x, y, size, (220, 220, 220)))
                
            elif "robot" in prompt_lower or "chess" in prompt_lower:
                # Robot/chess scene
                bg_color = (240, 240, 240)  # Light gray
                scene_type = "robot_chess"
                elements = list(preserved_elements)  # Start with preserved elements
                
                # Check if we need core elements
                if not any(e[0] == "chessboard" for e in elements):
                    # Add chessboard
                    elements.append(("chessboard", width//2, height*2//3, 200, (0, 0, 0)))
                
                if not any(e[0] == "robot" for e in elements):
                    # Add robot
                    elements.append(("robot", width//3, height//2, 100, (100, 100, 100)))
                
                if not any(e[0] == "human" for e in elements):
                    # Add human
                    elements.append(("human", width*2//3, height//2, 100, (200, 150, 100)))
                
            elif "sunset" in prompt_lower or "ocean" in prompt_lower or "beach" in prompt_lower:
                # Sunset/ocean scene
                bg_color = (255, 150, 50)  # Orange
                scene_type = "sunset"
                elements = list(preserved_elements)  # Start with preserved elements
                
                # Check if we need core elements
                if not any(e[0] == "ocean" for e in elements):
                    # Add ocean
                    elements.append(("ocean", width//2, height*3//4, width, (0, 50, 200)))
                
                if not any(e[0] == "sun" for e in elements):
                    # Add sun
                    elements.append(("sun", width//2, height//3, 80, (255, 200, 0)))
                
                # Add palm trees if mentioned
                if "palm" in prompt_lower:
                    new_palm_count = max(2, 5 - len([e for e in elements if e[0] == "palm"]))
                    for _ in range(new_palm_count):
                        x = random.randint(width//4, width*3//4)
                        y = height*2//3
                        size = random.randint(100, 150)
                        elements.append(("palm", x, y, size, (0, 100, 0)))
                
            else:
                # Default abstract scene
                bg_color = (50, 50, 80)
                scene_type = "abstract"
                elements = list(preserved_elements)  # Start with preserved elements
                
                # Add abstract shapes
                new_shape_count = max(5, 20 - len([e for e in elements if e[0] == "shape"]))
                for _ in range(new_shape_count):
                    x = random.randint(0, width)
                    y = random.randint(0, height)
                    size = random.randint(20, 100)
                    color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                    elements.append(("shape", x, y, size, color))
                            
            # Create animation frames
            for frame in range(int(self.fps * self.video_duration)):
                # Background
                img = Image.new("RGB", (width, height), bg_color)
                draw = ImageDraw.Draw(img)
                
                time_factor = frame / (self.fps * self.video_duration)
                
                # Draw scene elements with animation
                for element_type, x, y, size, color in elements:
                    if element_type == "tree":
                        # Simple tree
                        trunk_height = size * 1.5
                        # Trunk
                        draw.rectangle((x-size//10, y-trunk_height, x+size//10, y), fill=(80, 50, 20))
                        # Leaves (with slight movement)
                        leaf_x = x + int(math.sin(time_factor * 6.28 + x) * 5)
                        draw.ellipse((leaf_x-size//2, y-trunk_height-size, leaf_x+size//2, y-trunk_height), fill=color)
                    
                    elif element_type == "mushroom":
                        # Simple mushroom with glow effect
                        glow_factor = 0.5 + 0.5 * math.sin(time_factor * 6.28 * 2)
                        glow_color = (int(color[0] * glow_factor), int(color[1] * glow_factor), int(color[2] * glow_factor))
                        # Stem
                        draw.rectangle((x-size//10, y-size*1.5, x+size//10, y), fill=(220, 220, 200))
                        # Cap
                        draw.ellipse((x-size, y-size*2, x+size, y-size), fill=glow_color)
                    
                    elif element_type == "flower":
                        # Simple flower
                        # Stem
                        draw.line((x, y, x, y-size*2), fill=(0, 100, 0), width=2)
                        # Petals (with slight movement)
                        petal_factor = 0.8 + 0.2 * math.sin(time_factor * 6.28 * 2 + x)
                        draw.ellipse((x-size*petal_factor, y-size*3, x+size*petal_factor, y-size), fill=color)
                    
                    elif element_type == "star":
                        # Twinkling star
                        twinkle = 0.5 + 0.5 * math.sin(time_factor * 6.28 * 3 + x * y)
                        star_color = (int(color[0] * twinkle), int(color[1] * twinkle), int(color[2] * twinkle))
                        draw.ellipse((x-size/2, y-size/2, x+size/2, y+size/2), fill=star_color)
                    
                    elif element_type == "planet":
                        # Moving planet
                        planet_x = x + int(math.sin(time_factor * 6.28) * 20)
                        draw.ellipse((planet_x-size, y-size, planet_x+size, y+size), fill=color)
                    
                    elif element_type == "building":
                        # Building with windows
                        size_w, size_h = size
                        draw.rectangle((x-size_w//2, y-size_h, x+size_w//2, y), fill=color)
                        # Add windows with lights that change
                        window_on = (frame % 10) < 5  # Blink every 5 frames
                        window_color = (255, 255, 100) if window_on else (100, 100, 100)
                        for wy in range(y-size_h+10, y-10, 20):
                            for wx in range(x-size_w//2+10, x+size_w//2-10, 20):
                                if random.random() > 0.3:  # Some windows are dark
                                    draw.rectangle((wx, wy, wx+10, wy+10), fill=window_color)
                    
                    elif element_type == "light":
                        # Blinking neon light
                        light_on = (frame + x + y) % 20 < 10  # Different blink patterns
                        if light_on:
                            # Draw with glow effect
                            for s in range(size*2, 0, -1):
                                alpha = int(255 * (1 - s/(size*2)))
                                glow_color = (min(255, color[0] + alpha), min(255, color[1] + alpha), min(255, color[2] + alpha))
                                draw.ellipse((x-s/2, y-s/2, x+s/2, y+s/2), fill=glow_color)
                    
                    elif element_type == "mountain":
                        # Simple mountain
                        size_w, size_h = size
                        # Draw triangular mountain
                        draw.polygon([
                            (x-size_w//2, y), 
                            (x, y-size_h), 
                            (x+size_w//2, y)
                        ], fill=color)
                        # Add snow cap if applicable
                        if "snow" in prompt_lower:
                            draw.polygon([
                                (x-size_w//4, y-size_h*0.7), 
                                (x, y-size_h), 
                                (x+size_w//4, y-size_h*0.7)
                            ], fill=(255, 255, 255))
                    
                    elif element_type == "snow":
                        # Falling snow
                        snow_y = (y + frame * 2) % height
                        draw.ellipse((x-size/2, snow_y-size/2, x+size/2, snow_y+size/2), fill=color)
                    
                    elif element_type == "aurora":
                        # Northern lights with movement
                        size_w, size_h = size
                        aurora_x = x + int(math.sin(time_factor * 6.28 * 0.5 + y/50) * 50)
                        aurora_points = []
                        for i in range(10):
                            px = aurora_x - size_w//2 + i * size_w//10
                            py = y + int(math.sin(time_factor * 6.28 + i/2) * size_h//2)
                            aurora_points.append((px, py))
                        
                        # Connect the points with wide lines
                        for i in range(len(aurora_points)-1):
                            x1, y1 = aurora_points[i]
                            x2, y2 = aurora_points[i+1]
                            # Vary the color
                            hue_shift = (i/10 + time_factor) % 1.0
                            r, g, b = self._hsv_to_rgb(120 * hue_shift, 1.0, 1.0)  # Green-blue hues
                            aurora_color = (int(r*255), int(g*255), int(b*255))
                            draw.line((x1, y1, x2, y2), fill=aurora_color, width=10)
                    
                    elif element_type == "castle":
                        # Draw castle
                        castle_size = size
                        # Main body
                        draw.rectangle((x-castle_size//2, y-castle_size, x+castle_size//2, y), fill=color)
                        # Towers
                        for tx in [x-castle_size//2, x+castle_size//2]:
                            draw.rectangle((tx-castle_size//8, y-castle_size*1.2, tx+castle_size//8, y), fill=color)
                            # Tower tops
                            draw.polygon([
                                (tx-castle_size//6, y-castle_size*1.2),
                                (tx, y-castle_size*1.4),
                                (tx+castle_size//6, y-castle_size*1.2),
                            ], fill=(150, 0, 0))  # Red roofs
                        
                    elif element_type == "dragon":
                        # Moving dragon
                        dragon_x = x + int(math.sin(time_factor * 6.28) * 50)
                        dragon_y = y + int(math.cos(time_factor * 6.28) * 30)
                        # Body
                        draw.ellipse((dragon_x-size//2, dragon_y-size//4, dragon_x+size//2, dragon_y+size//4), fill=color)
                        # Wings
                        wing_extension = 0.7 + 0.3 * math.sin(time_factor * 6.28 * 3)
                        draw.polygon([
                            (dragon_x, dragon_y),
                            (dragon_x-size, dragon_y-size*wing_extension),
                            (dragon_x-size//2, dragon_y),
                        ], fill=color)
                        draw.polygon([
                            (dragon_x, dragon_y),
                            (dragon_x+size, dragon_y-size*wing_extension),
                            (dragon_x+size//2, dragon_y),
                        ], fill=color)
                        
                    elif element_type == "cloud":
                        # Moving cloud
                        cloud_x = (x + frame) % (width + size) - size
                        # Draw cloud as a series of overlapping circles
                        for cx in range(cloud_x-size//2, cloud_x+size//2, size//4):
                            draw.ellipse((cx-size//4, y-size//4, cx+size//4, y+size//4), fill=color)
                    
                    elif element_type == "chessboard":
                        # Chessboard
                        board_size = size
                        square_size = board_size // 8
                        for row in range(8):
                            for col in range(8):
                                square_color = (255, 255, 255) if (row + col) % 2 == 0 else (0, 0, 0)
                                draw.rectangle((
                                    x - board_size//2 + col * square_size,
                                    y - board_size//2 + row * square_size,
                                    x - board_size//2 + (col+1) * square_size,
                                    y - board_size//2 + (row+1) * square_size
                                ), fill=square_color)
                    
                    elif element_type == "robot":
                        # Robot with simple animation
                        robot_x = x + int(math.sin(time_factor * 6.28) * 5)
                        # Body
                        draw.rectangle((robot_x-size//3, y-size//2, robot_x+size//3, y+size//2), fill=color)
                        # Head
                        draw.rectangle((robot_x-size//4, y-size, robot_x+size//4, y-size//2), fill=color)
                        # Eyes
                        eye_color = (0, 255, 255)  # Cyan
                        draw.ellipse((robot_x-size//8-5, y-size*0.85, robot_x-size//8+5, y-size*0.75), fill=eye_color)
                        draw.ellipse((robot_x+size//8-5, y-size*0.85, robot_x+size//8+5, y-size*0.75), fill=eye_color)
                        
                    elif element_type == "human":
                        # Human figure
                        # Body
                        draw.ellipse((x-size//4, y-size//2, x+size//4, y+size//2), fill=color)
                        # Head
                        draw.ellipse((x-size//6, y-size*0.9, x+size//6, y-size*0.6), fill=color)
                        
                    elif element_type == "ocean":
                        # Ocean with waves
                        sea_height = size
                        for wx in range(0, width, 20):
                            wave_y = y + int(math.sin(time_factor * 6.28 * 2 + wx/30) * 10)
                            draw.rectangle((wx, wave_y, wx+30, height), fill=color)
                            
                    elif element_type == "sun":
                        # Sun with rays
                        sun_size = size + int(math.sin(time_factor * 6.28 * 2) * 5)
                        draw.ellipse((x-sun_size//2, y-sun_size//2, x+sun_size//2, y+sun_size//2), fill=color)
                        # Rays
                        for angle in range(0, 360, 30):
                            rad = math.radians(angle)
                            ray_length = sun_size//2 + 20 + int(math.sin(time_factor * 6.28 * 3 + angle) * 10)
                            draw.line((
                                x + int(math.cos(rad) * sun_size//2),
                                y + int(math.sin(rad) * sun_size//2),
                                x + int(math.cos(rad) * ray_length),
                                y + int(math.sin(rad) * ray_length)
                            ), fill=color, width=3)
                            
                    elif element_type == "palm":
                        # Palm tree
                        # Trunk with slight movement
                        trunk_x = x + int(math.sin(time_factor * 6.28) * 5)
                        draw.line((trunk_x, y, trunk_x + int(math.sin(time_factor * 6.28 + x) * 20), y-size), fill=(100, 60, 20), width=10)
                        # Leaves
                        for angle in range(0, 360, 45):
                            rad = math.radians(angle)
                            leaf_x = trunk_x + int(math.sin(time_factor * 6.28 + x) * 20) + int(math.cos(rad) * size//2)
                            leaf_y = y-size + int(math.sin(rad) * size//3)
                            # Draw a leaf
                            draw.ellipse((leaf_x-15, leaf_y-40, leaf_x+15, leaf_y), fill=color)
                            
                    elif element_type == "shape":
                        # Abstract shapes
                        shape_x = x + int(math.sin(time_factor * 6.28 + y/50) * 20)
                        shape_y = y + int(math.cos(time_factor * 6.28 + x/50) * 20)
                        shape_type = hash(str(x) + str(y)) % 3
                        if shape_type == 0:
                            draw.ellipse((shape_x-size//2, shape_y-size//2, shape_x+size//2, shape_y+size//2), fill=color)
                        elif shape_type == 1:
                            draw.rectangle((shape_x-size//2, shape_y-size//2, shape_x+size//2, shape_y+size//2), fill=color)
                        else:
                            draw.polygon([
                                (shape_x, shape_y-size//2),
                                (shape_x+size//2, shape_y+size//2),
                                (shape_x-size//2, shape_y+size//2),
                            ], fill=color)
                
                # Apply a slight blur for smoothness
                img = img.filter(ImageFilter.GaussianBlur(radius=1))
                
                # Add text overlay with prompt
                try:
                    font = ImageFont.truetype("Arial", 18)
                except:
                    font = ImageFont.load_default()
                
                # Draw semi-transparent background for text
                text_bg = Image.new('RGBA', (width, 40), (0, 0, 0, 180))
                img.paste(Image.composite(text_bg, img.crop((0, 0, width, 40)), text_bg), (0, 0))
                
                draw = ImageDraw.Draw(img)
                draw.text((10, 10), prompt_text, fill=(255, 255, 255), font=font)
                
                # Save the last frame for potential use in future generations
                if frame == int(self.fps * self.video_duration) - 1:
                    self.last_frame = img.copy()
                
                # Save frame
                frame_path = os.path.join(temp_dir, f"frame_{frame:05d}.png")
                img.save(frame_path)
            
            # Use ffmpeg to combine frames into video with proper codec settings
            (
                ffmpeg
                .input(os.path.join(temp_dir, "frame_%05d.png"), framerate=self.fps)
                .output(output_path, 
                        vcodec='libx264',       # Use H.264 codec
                        pix_fmt='yuv420p',      # Standard pixel format for compatibility
                        crf=23,                 # Quality setting
                        movflags='+faststart',  # Optimize for web streaming
                        preset='medium')        # Encoding speed/quality balance
                .run(quiet=True, overwrite_output=True)
            )
            
            # Store current scene information for continuity in future generations
            self.current_elements = elements
            self.scene_history.append({
                'type': scene_type,
                'bg_color': bg_color,
                'prompt': prompt_text
            })
            
            # Keep scene history manageable - only keep last 5 scenes
            if len(self.scene_history) > 5:
                self.scene_history = self.scene_history[-5:]
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
    
    def _hsv_to_rgb(self, h, s, v):
        """
        Convert HSV color to RGB.
        
        Args:
            h: Hue (0-360)
            s: Saturation (0-1)
            v: Value (0-1)
            
        Returns:
            tuple: (r, g, b) each in 0-1 range
        """
        h = h % 360
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        
        if h < 60:
            r, g, b = c, x, 0
        elif h < 120:
            r, g, b = x, c, 0
        elif h < 180:
            r, g, b = 0, c, x
        elif h < 240:
            r, g, b = 0, x, c
        elif h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
            
        return (r + m, g + m, b + m)


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Example callback function
    def on_video_generated(prompt, video_path):
        print(f"Video generated for prompt: {prompt['text']}")
        print(f"Video saved to: {video_path}")
    
    # Testing with simulated generator
    output_dir = os.environ.get("OUTPUT_DIR", "./generated_clips")
    generator = SimulatedVideoGenerator(
        output_dir=output_dir,
        video_duration=5,
        fps=24,
        resolution=(512, 512),
        callback=on_video_generated
    )
    
    # Start worker
    generator.start_worker()
    
    try:
        # Add some test prompts
        test_prompts = [
            {
                'text': "A spaceship launching from Earth into the stars",
                'author': "TestUser1",
                'timestamp': time.strftime('%Y%m%d-%H%M%S')
            },
            {
                'text': "A magical forest with glowing mushrooms and fairies",
                'author': "TestUser2",
                'timestamp': time.strftime('%Y%m%d-%H%M%S')
            }
        ]
        
        for prompt in test_prompts:
            generator.add_to_queue(prompt)
        
        # Wait for all tasks to complete
        generator.task_queue.join()
        
    finally:
        generator.close()