# AI-Powered YouTube Livestream System

A Python-based livestream system that generates videos based on YouTube live chat prompts and seamlessly integrates them into an ongoing livestream.

## Features

- 🎥 **Real-time YouTube Live Chat Monitoring** - Listens to your livestream chat in real-time
- 🤖 **AI Video Generation** - Uses Stable Video Diffusion models to generate videos from text prompts
- 🔄 **Seamless Streaming** - Continuously streams to YouTube RTMP with smooth transitions between clips
- 🎬 **Professional Transitions** - Content-aware transitions automatically chosen based on scene analysis
- 🧠 **Prompt Queue System** - Handles multiple requests in order, preventing spam
- 🧪 **Simulation Mode** - Test without a real livestream using simulated chat and video generation
- 📊 **Robust Logging** - Comprehensive logging for monitoring and debugging

## Requirements

- macOS (Monterey or later recommended)
- Python 3.10 or higher
- FFmpeg (install via Homebrew: `brew install ffmpeg`)
- YouTube API credentials
- GPU recommended for faster video generation (Apple Silicon MPS supported)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ytvideo.git
   cd ytvideo
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy the example environment file and fill in your credentials:
   ```bash
   cp .env.example .env
   # Edit .env with your YouTube API keys and other settings
   ```

## YouTube API Setup

1. Create a project in the [Google Cloud Console](https://console.cloud.google.com/)
2. Enable the YouTube Data API v3
3. Create OAuth credentials (Client ID and Client Secret)
4. Create an API Key
5. Add these credentials to your `.env` file

## Configuration

Edit the `.env` file with your settings:

```
# YouTube API Configuration
YOUTUBE_API_KEY=your_youtube_api_key_here
YOUTUBE_CLIENT_ID=your_youtube_client_id_here
YOUTUBE_CLIENT_SECRET=your_youtube_client_secret_here
YOUTUBE_CHANNEL_ID=your_youtube_channel_id_here
YOUTUBE_LIVESTREAM_ID=your_youtube_livestream_id_here

# RTMP Streaming Settings
RTMP_URL=rtmp://a.rtmp.youtube.com/live2
RTMP_KEY=your_rtmp_stream_key_here

# AI Model Settings
MODEL_REPO=stabilityai/stable-video-diffusion-img2vid-xt
DEFAULT_PROMPT="A serene landscape with mountains"
MAX_PROMPT_QUEUE=10
VIDEO_DURATION=10
VIDEO_RESOLUTION=512x512
FPS=24

# Streaming Settings
DEFAULT_VIDEO_PATH=default_video.mp4
TRANSITION_TYPE=fade  # Options: fade, cut, dissolve
TRANSITION_DURATION=1.0  # In seconds

# System Settings
LOG_LEVEL=INFO
TEMP_DIR=./temp
OUTPUT_DIR=./generated_clips
SIMULATE_CHAT=false  # Set to true for testing without real YouTube chat
```

## Usage

### Normal Mode

Run the system connecting to a real YouTube livestream:

```bash
python main.py
```

### Simulation Mode

Test the system without a real YouTube connection:

```bash
python main.py --simulate
```

### Command Line Options

```
usage: main.py [-h] [--simulate] [--config CONFIG] [--rtmp-url RTMP_URL] [--rtmp-key RTMP_KEY] [--api-key API_KEY]
               [--livestream-id LIVESTREAM_ID] [--model MODEL] [--default-video DEFAULT_VIDEO]
               [--output-dir OUTPUT_DIR] [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]

AI-Powered YouTube Livestream System

options:
  -h, --help            show this help message and exit
  --simulate            Run in simulation mode without actual YouTube connection
  --config CONFIG       Path to JSON configuration file (overrides .env)
  --rtmp-url RTMP_URL   RTMP server URL
  --rtmp-key RTMP_KEY   RTMP stream key
  --api-key API_KEY     YouTube API key
  --livestream-id LIVESTREAM_ID
                        YouTube livestream ID
  --model MODEL         Hugging Face model repository ID
  --default-video DEFAULT_VIDEO
                        Path to default video
  --output-dir OUTPUT_DIR
                        Directory to save generated videos
  --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Logging level
```

## Architecture

The system consists of three main components:

1. **YouTube Chat Handler** (`youtube_chat.py`) - Connects to YouTube's API to retrieve live chat messages
2. **Video Generator** (`video_generator.py`) - Uses AI models to generate video clips from text prompts
3. **Stream Manager** (`stream_manager.py`) - Manages the FFmpeg process that streams to YouTube RTMP

These components are orchestrated by the main application (`main.py`).

### Professional Transitions

The system features sophisticated, content-aware transitions between video clips:

- **Scene Analysis** - Examines brightness, color, and texture differences between scenes
- **Content Detection** - Analyzes video filenames to determine appropriate transition effects
- **Dynamic Selection** - Automatically chooses from fade, dissolve, zoom, slide, rotate, wipe, and ripple transitions
- **GPU Acceleration** - Uses numpy for fast image processing when available
- **Multiple Fallbacks** - Guarantees smooth transitions even in error conditions
- **Professional Effects** - Includes sophisticated film-like effects like grain, vignettes, and easing functions

For detailed information about the transition system, see [IMPROVEMENTS.md](./IMPROVEMENTS.md)

## AI Models

By default, the system uses Stability AI's Stable Video Diffusion model for text-to-video generation. You can change the model by setting the `MODEL_REPO` environment variable to a different Hugging Face model ID.

## Troubleshooting

- **FFmpeg errors**: Make sure FFmpeg is installed (`brew install ffmpeg`)
- **YouTube API errors**: Verify your API credentials in the `.env` file
- **Model download issues**: Ensure you have a stable internet connection
- **Performance issues**: For faster video generation, use a computer with a GPU

Check the logs in the `logs` directory for detailed error messages.

## License

[MIT License](LICENSE)

## Acknowledgements

- [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) by Stability AI
- [FFmpeg](https://ffmpeg.org/) for video processing
- [YouTube Data API](https://developers.google.com/youtube/v3) for live chat integration