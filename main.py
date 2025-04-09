#!/usr/bin/env python3
"""
AI-Powered YouTube Livestream System

This application creates a continuous YouTube livestream that generates
videos based on viewer prompts in the live chat. It uses AI models to
generate video clips from text prompts and seamlessly integrates them
into the livestream.

Features:
- Real-time YouTube Live Chat monitoring
- AI-powered text-to-video generation
- Continuous streaming with smooth transitions
- Prompt queueing system to prevent spam
- Simulation mode for testing without a real livestream
"""

import os
import sys
import time
import argparse
import threading
import json
import tempfile
import shutil
import signal
from typing import Dict, Optional
from pathlib import Path

import ffmpeg
from dotenv import load_dotenv
from loguru import logger

# Import local modules
from youtube_chat import YouTubeLiveChat, SimulatedYouTubeLiveChat
from video_generator import VideoGenerator, SimulatedVideoGenerator
from stream_manager import StreamManager, SimulatedStreamManager


def setup_logger(log_level: str = "INFO"):
    """
    Set up the logger configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logger.remove()  # Remove default handler
    
    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level
    )
    
    # Add file handler
    log_file = Path("./logs/ytvideo.log")
    os.makedirs(log_file.parent, exist_ok=True)
    
    logger.add(
        log_file,
        rotation="10 MB",
        retention="1 week",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level
    )
    
    logger.info(f"Logger initialized with level {log_level}")


def ensure_default_video(default_video_path: str) -> str:
    """
    Ensure the default video exists, create if it doesn't.
    
    Args:
        default_video_path: Path to the default video
        
    Returns:
        str: Path to the default video (created if needed)
    """
    if os.path.exists(default_video_path):
        logger.info(f"Using existing default video: {default_video_path}")
        return default_video_path
    
    logger.warning(f"Default video not found: {default_video_path}")
    logger.info("Creating a temporary default video...")
    
    # Create a default video pattern
    temp_dir = Path("./temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_video = str(temp_dir / "default_video.mp4")
    
    try:
        # Generate a simple test pattern video
        (
            ffmpeg
            .input('testsrc=size=1280x720:rate=30', f='lavfi', t=60)
            .drawtext(
                text='Waiting for viewer prompts...', 
                fontsize=40, 
                fontcolor='white', 
                x='(w-text_w)/2', 
                y='(h-text_h)/2'
            )
            .output(temp_video, vcodec='libx264', pix_fmt='yuv420p', preset='medium')
            .run(quiet=True, overwrite_output=True)
        )
        logger.info(f"Created temporary default video: {temp_video}")
        
        # Save to user-specified location if possible
        if os.access(os.path.dirname(default_video_path), os.W_OK):
            shutil.copy(temp_video, default_video_path)
            logger.info(f"Saved default video to: {default_video_path}")
            return default_video_path
        else:
            logger.warning(f"Cannot write to {os.path.dirname(default_video_path)}, using temporary video")
            return temp_video
        
    except Exception as e:
        logger.error(f"Failed to create default video: {str(e)}")
        raise


class YTVideoApp:
    """Main application that coordinates all components."""
    
    def __init__(self, config: Dict):
        """
        Initialize the application.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.simulate = config.get('simulate', False)
        
        # Paths
        self.output_dir = Path(config.get('output_dir', './generated_clips'))
        self.temp_dir = Path(config.get('temp_dir', './temp'))
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Ensure default video exists
        self.default_video_path = ensure_default_video(config.get('default_video_path', 'default_video.mp4'))
        
        # Create components
        self._init_chat()
        self._init_generator()
        self._init_streamer()
        
        # Shutdown flag
        self.shutdown_event = threading.Event()
        
        logger.info("YTVideoApp initialized successfully")
    
    def _init_chat(self):
        """Initialize the chat component."""
        if self.simulate:
            logger.info("Using simulated YouTube chat")
            self.chat = SimulatedYouTubeLiveChat(
                max_queue_size=self.config.get('max_prompt_queue', 10),
                callback=self.on_new_prompt
            )
        else:
            logger.info("Initializing YouTube chat client")
            self.chat = YouTubeLiveChat(
                api_key=self.config.get('youtube_api_key', ''),
                livestream_id=self.config.get('youtube_livestream_id', ''),
                max_queue_size=self.config.get('max_prompt_queue', 10),
                callback=self.on_new_prompt
            )
    
    def _init_generator(self):
        """Initialize the video generator component."""
        if self.simulate:
            logger.info("Using simulated video generator")
            self.generator = SimulatedVideoGenerator(
                output_dir=str(self.output_dir),
                video_duration=self.config.get('video_duration', 10),
                fps=self.config.get('fps', 24),
                resolution=(
                    self.config.get('video_resolution', '512x512').split('x')[0],
                    self.config.get('video_resolution', '512x512').split('x')[1]
                ),
                max_queue_size=self.config.get('max_prompt_queue', 10),
                callback=self.on_video_generated
            )
        else:
            logger.info("Initializing AI video generator")
            self.generator = VideoGenerator(
                model_repo=self.config.get('model_repo', 'stabilityai/stable-video-diffusion-img2vid-xt'),
                output_dir=str(self.output_dir),
                video_duration=self.config.get('video_duration', 10),
                fps=self.config.get('fps', 24),
                resolution=(
                    int(self.config.get('video_resolution', '512x512').split('x')[0]),
                    int(self.config.get('video_resolution', '512x512').split('x')[1])
                ),
                max_queue_size=self.config.get('max_prompt_queue', 10),
                callback=self.on_video_generated
            )
    
    def _init_streamer(self):
        """Initialize the stream manager component."""
        if self.simulate:
            logger.info("Using simulated stream manager")
            self.streamer = SimulatedStreamManager(
                default_video_path=self.default_video_path,
                output_resolution=(1280, 720),
                fps=self.config.get('fps', 30),
                transition_type=self.config.get('transition_type', 'fade'),
                transition_duration=float(self.config.get('transition_duration', 1.0)),
                temp_dir=str(self.temp_dir)
            )
        else:
            logger.info("Initializing stream manager")
            self.streamer = StreamManager(
                rtmp_url=self.config.get('rtmp_url', 'rtmp://a.rtmp.youtube.com/live2'),
                rtmp_key=self.config.get('rtmp_key', ''),
                default_video_path=self.default_video_path,
                output_resolution=(1280, 720),
                fps=self.config.get('fps', 30),
                bitrate=self.config.get('bitrate', '4000k'),
                transition_type=self.config.get('transition_type', 'fade'),
                transition_duration=float(self.config.get('transition_duration', 1.0)),
                temp_dir=str(self.temp_dir)
            )
    
    def on_new_prompt(self, prompt: Dict):
        """
        Callback for when a new prompt is received.
        
        Args:
            prompt: Prompt data
        """
        logger.info(f"New prompt from {prompt['author']}: {prompt['text']}")
        
        # Add to generator queue
        self.generator.add_to_queue(prompt)
    
    def on_video_generated(self, prompt: Dict, video_path: str):
        """
        Callback for when a video is generated.
        
        Args:
            prompt: Prompt data
            video_path: Path to generated video
        """
        logger.info(f"Generated video for prompt: {prompt['text'][:50]}...")
        logger.info(f"Video path: {video_path}")
        
        # Add to stream
        self.streamer.add_clip(video_path)
    
    def start(self):
        """Start all components."""
        logger.info("Starting YTVideoApp...")
        
        try:
            # Start video generator
            logger.info("Starting video generator...")
            self.generator.start_worker()
            
            # Start stream
            logger.info("Starting stream...")
            if not self.streamer.start_stream():
                logger.error("Failed to start stream")
                return False
            
            # Start chat listener
            logger.info("Connecting to YouTube chat...")
            
            if not self.simulate:
                # Authenticate with YouTube API
                if not self.chat.authenticate():
                    logger.error("Failed to authenticate with YouTube API")
                    return False
                
                # Get active livestreams if no ID provided
                if not self.config.get('youtube_livestream_id'):
                    livestreams = self.chat.get_active_livestreams(
                        self.config.get('youtube_channel_id')
                    )
                    
                    if not livestreams:
                        logger.error("No active livestreams found")
                        return False
                    
                    # Use the first livestream
                    livestream_id = list(livestreams.keys())[0]
                    logger.info(f"Using livestream: {livestreams[livestream_id]} ({livestream_id})")
                else:
                    livestream_id = self.config.get('youtube_livestream_id')
                
                # Connect to chat
                if not self.chat.connect_to_chat(livestream_id):
                    logger.error("Failed to connect to YouTube chat")
                    return False
            
            # Start chat listener
            logger.info("Starting chat listener...")
            self.chat.start_listener()
            
            logger.info("YTVideoApp started successfully")
            
            # Register signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._handle_shutdown)
            signal.signal(signal.SIGTERM, self._handle_shutdown)
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting YTVideoApp: {str(e)}")
            self.shutdown()
            return False
    
    def _handle_shutdown(self, sig, frame):
        """Handle shutdown signals."""
        logger.info(f"Received shutdown signal {sig}")
        self.shutdown()
        sys.exit(0)
    
    def wait(self):
        """Wait until shutdown is requested."""
        try:
            while not self.shutdown_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            self.shutdown()
    
    def shutdown(self):
        """Shut down all components."""
        logger.info("Shutting down YTVideoApp...")
        
        # Stop components in reverse order
        logger.info("Stopping chat listener...")
        self.chat.stop_listener()
        self.chat.close()
        
        logger.info("Stopping video generator...")
        self.generator.stop_worker()
        self.generator.close()
        
        logger.info("Stopping stream...")
        self.streamer.stop_stream()
        self.streamer.close()
        
        # Set shutdown event
        self.shutdown_event.set()
        
        logger.info("YTVideoApp shutdown complete")


def load_config() -> Dict:
    """
    Load configuration from environment variables.
    
    Returns:
        Dict: Configuration dictionary
    """
    # Load .env file if it exists
    load_dotenv()
    
    # Create configuration dictionary
    config = {
        # YouTube API settings
        'youtube_api_key': os.environ.get('YOUTUBE_API_KEY', ''),
        'youtube_client_id': os.environ.get('YOUTUBE_CLIENT_ID', ''),
        'youtube_client_secret': os.environ.get('YOUTUBE_CLIENT_SECRET', ''),
        'youtube_channel_id': os.environ.get('YOUTUBE_CHANNEL_ID', ''),
        'youtube_livestream_id': os.environ.get('YOUTUBE_LIVESTREAM_ID', ''),
        
        # RTMP settings
        'rtmp_url': os.environ.get('RTMP_URL', 'rtmp://a.rtmp.youtube.com/live2'),
        'rtmp_key': os.environ.get('RTMP_KEY', ''),
        
        # AI model settings
        'model_repo': os.environ.get('MODEL_REPO', 'stabilityai/stable-video-diffusion-img2vid-xt'),
        'default_prompt': os.environ.get('DEFAULT_PROMPT', 'A serene landscape with mountains'),
        'max_prompt_queue': int(os.environ.get('MAX_PROMPT_QUEUE', '10')),
        'video_duration': int(os.environ.get('VIDEO_DURATION', '10')),
        'video_resolution': os.environ.get('VIDEO_RESOLUTION', '512x512'),
        'fps': int(os.environ.get('FPS', '24')),
        
        # Streaming settings
        'default_video_path': os.environ.get('DEFAULT_VIDEO_PATH', 'default_video.mp4'),
        'transition_type': os.environ.get('TRANSITION_TYPE', 'fade'),
        'transition_duration': float(os.environ.get('TRANSITION_DURATION', '1.0')),
        'bitrate': os.environ.get('BITRATE', '4000k'),
        
        # System settings
        'log_level': os.environ.get('LOG_LEVEL', 'INFO'),
        'temp_dir': os.environ.get('TEMP_DIR', './temp'),
        'output_dir': os.environ.get('OUTPUT_DIR', './generated_clips'),
        'simulate': os.environ.get('SIMULATE_CHAT', 'false').lower() in ('true', '1', 'yes')
    }
    
    return config


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='AI-Powered YouTube Livestream System')
    
    parser.add_argument('--simulate', action='store_true',
                        help='Run in simulation mode without actual YouTube connection')
    
    parser.add_argument('--config', type=str,
                        help='Path to JSON configuration file (overrides .env)')
    
    parser.add_argument('--rtmp-url', type=str,
                        help='RTMP server URL')
    
    parser.add_argument('--rtmp-key', type=str,
                        help='RTMP stream key')
    
    parser.add_argument('--api-key', type=str,
                        help='YouTube API key')
    
    parser.add_argument('--livestream-id', type=str,
                        help='YouTube livestream ID')
    
    parser.add_argument('--model', type=str,
                        help='Hugging Face model repository ID')
    
    parser.add_argument('--default-video', type=str,
                        help='Path to default video')
    
    parser.add_argument('--output-dir', type=str,
                        help='Directory to save generated videos')
    
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    
    return parser.parse_args()


def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config()
    
    # Override with command line arguments
    if args.simulate:
        config['simulate'] = True
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    if args.rtmp_url:
        config['rtmp_url'] = args.rtmp_url
    
    if args.rtmp_key:
        config['rtmp_key'] = args.rtmp_key
    
    if args.api_key:
        config['youtube_api_key'] = args.api_key
    
    if args.livestream_id:
        config['youtube_livestream_id'] = args.livestream_id
    
    if args.model:
        config['model_repo'] = args.model
    
    if args.default_video:
        config['default_video_path'] = args.default_video
    
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    if args.log_level:
        config['log_level'] = args.log_level
    
    # Set up logger
    setup_logger(config.get('log_level', 'INFO'))
    
    # Create application
    app = YTVideoApp(config)
    
    # Start application
    if app.start():
        logger.info("Application started successfully")
        
        # Wait for shutdown signal
        app.wait()
    else:
        logger.error("Failed to start application")
        app.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()