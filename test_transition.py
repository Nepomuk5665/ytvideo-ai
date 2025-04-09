#!/usr/bin/env python3
"""
Test script for transitions
"""

import os
import sys
import time
from pathlib import Path
from loguru import logger

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the StreamManager
from stream_manager import StreamManager


def setup_logger():
    """Set up the logger with DEBUG level"""
    logger.remove()  # Remove default handler
    
    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="DEBUG"
    )
    
    # Add file handler
    log_file = Path("./logs/test_transition.log")
    os.makedirs(log_file.parent, exist_ok=True)
    
    logger.add(
        log_file,
        rotation="10 MB",
        retention="1 week",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG"
    )


def create_test_videos():
    """Create two test videos with different colors"""
    import subprocess
    from PIL import Image
    
    output_dir = Path("./generated_clips")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a blue test frame
    img1 = Image.new('RGB', (512, 512), color=(0, 0, 255))
    img1.save('/tmp/test_frame1.png')
    
    # Create a red test frame
    img2 = Image.new('RGB', (512, 512), color=(255, 0, 0))
    img2.save('/tmp/test_frame2.png')
    
    # Create test videos
    video1_path = output_dir / "test_video1.mp4"
    video2_path = output_dir / "test_video2.mp4"
    
    # Create first video (blue)
    subprocess.run([
        'ffmpeg', '-y', '-loop', '1', '-i', '/tmp/test_frame1.png',
        '-c:v', 'libx264', '-t', '5', '-pix_fmt', 'yuv420p', str(video1_path)
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Create second video (red)
    subprocess.run([
        'ffmpeg', '-y', '-loop', '1', '-i', '/tmp/test_frame2.png',
        '-c:v', 'libx264', '-t', '5', '-pix_fmt', 'yuv420p', str(video2_path)
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    logger.info(f"Created test videos: {video1_path} and {video2_path}")
    return str(video1_path), str(video2_path)


def test_transition_creation():
    """Test the transition creation feature"""
    # Create test videos
    video1_path, video2_path = create_test_videos()
    
    # Initialize Stream Manager
    stream_manager = StreamManager(
        rtmp_url="rtmp://localhost/live",
        rtmp_key="test_key",
        default_video_path=video1_path,
        output_resolution=(512, 512),
        fps=30,
        bitrate="2000k",
        transition_type="fade",  # Try different transition types
        transition_duration=2.0,
        temp_dir="./temp/test_transitions"
    )
    
    try:
        # Add first video
        logger.info(f"Adding first clip: {video1_path}")
        stream_manager.add_clip(video1_path)
        
        # Wait a moment
        time.sleep(1)
        
        # Add second video (should create a transition)
        logger.info(f"Adding second clip: {video2_path}")
        result = stream_manager.add_clip(video2_path)
        
        # Wait for processing
        time.sleep(5)
        
        # Check transition directory
        transition_dir = Path("./temp/test_transitions/transitions")
        if transition_dir.exists():
            transitions = list(transition_dir.glob("*.mp4"))
            logger.info(f"Found {len(transitions)} transitions:")
            for t in transitions:
                logger.info(f"  - {t}")
        else:
            logger.warning(f"Transition directory not found: {transition_dir}")
        
        logger.info("Test completed")
        
    finally:
        # Clean up
        stream_manager.close()


if __name__ == "__main__":
    setup_logger()
    logger.info("Starting transition test")
    test_transition_creation()