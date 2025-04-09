# Contributing to YTVideo

Thank you for your interest in contributing to YTVideo! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project.

## How to Contribute

1. **Fork the repository**
2. **Clone your fork**
   ```
   git clone https://github.com/your-username/ytvideo-ai.git
   cd ytvideo-ai
   ```

3. **Create a branch**
   ```
   git checkout -b feature/your-feature-name
   ```

4. **Make your changes**
   - Follow the code style of the project
   - Add or update tests as necessary
   - Update documentation as needed

5. **Test your changes**
   ```
   python test_transition.py
   ```

6. **Commit your changes**
   ```
   git commit -m "Add a descriptive commit message"
   ```

7. **Push to your fork**
   ```
   git push origin feature/your-feature-name
   ```

8. **Create a pull request**
   - Go to the GitHub page of your fork
   - Click on "Pull Request"
   - Follow the prompts to create your pull request

## Development Setup

1. **Set up a virtual environment**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Install FFmpeg**
   - macOS: `brew install ffmpeg`
   - Ubuntu: `sudo apt-get install ffmpeg`
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

4. **Create a .env file**
   ```
   cp .env.example .env
   # Edit .env with your settings
   ```

## Testing

- Run the transition test: `python test_transition.py`
- Run the application in simulation mode: `python main.py --simulate`

## Coding Style

- Follow PEP 8 guidelines
- Use clear, descriptive variable and function names
- Add docstrings for functions and classes
- Include type hints when possible

## Pull Request Process

1. Ensure your code follows the style guidelines
2. Update the README.md with details of changes if applicable
3. The PR should work in both simulation mode and real mode
4. Your PR will be reviewed by a maintainer

## Feature Requests and Bug Reports

Please use the GitHub Issues section to submit feature requests and bug reports.

Thank you for contributing!