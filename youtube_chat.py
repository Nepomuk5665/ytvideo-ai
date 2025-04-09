#!/usr/bin/env python3
"""
YouTube Live Chat Handler Module

This module handles interactions with the YouTube Live Chat API.
It connects to a livestream, retrieves chat messages in real-time,
and provides a queue system for processing viewer prompts.
"""

import os
import time
import queue
import threading
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any

import google.oauth2.credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from loguru import logger


class YouTubeLiveChat:
    """Handles interaction with YouTube Live Chat API."""
    
    def __init__(self, api_key: str, livestream_id: str = None, 
                 max_queue_size: int = 10, callback: Callable = None):
        """
        Initialize YouTube Live Chat handler.
        
        Args:
            api_key: YouTube API key for authentication
            livestream_id: ID of the active livestream
            max_queue_size: Maximum number of prompts to queue
            callback: Function to call when a new prompt is received
        """
        self.api_key = api_key
        self.livestream_id = livestream_id
        self.max_queue_size = max_queue_size
        self.callback = callback
        self.prompt_queue = queue.Queue(maxsize=max_queue_size)
        self.active_livestreams = {}
        self.next_page_token = None
        self.youtube = None
        self.is_connected = False
        self.stop_event = threading.Event()
        self.chat_thread = None
        
    def authenticate(self) -> bool:
        """
        Authenticate with YouTube API.
        
        Returns:
            bool: True if authentication successful, False otherwise
        """
        try:
            self.youtube = build('youtube', 'v3', developerKey=self.api_key)
            logger.info("Successfully authenticated with YouTube API")
            return True
        except Exception as e:
            logger.error(f"Failed to authenticate with YouTube API: {str(e)}")
            return False
    
    def get_active_livestreams(self, channel_id: str = None) -> Dict:
        """
        Get all active livestreams for the authenticated channel.
        
        Args:
            channel_id: Optional channel ID to check
            
        Returns:
            Dict: Dictionary of active livestreams {id: title}
        """
        try:
            if not self.youtube:
                if not self.authenticate():
                    return {}
            
            # If no channel_id provided, get the authenticated user's channel
            if not channel_id:
                channel_response = self.youtube.channels().list(
                    part="id",
                    mine=True
                ).execute()
                
                if not channel_response.get('items'):
                    logger.error("No channel found for authenticated user")
                    return {}
                
                channel_id = channel_response['items'][0]['id']
            
            # Get active broadcasts
            broadcasts = self.youtube.liveBroadcasts().list(
                part="id,snippet,status",
                broadcastStatus="active",
                broadcastType="all",
                channelId=channel_id
            ).execute()
            
            # Store livestream info
            self.active_livestreams = {
                item['id']: item['snippet']['title']
                for item in broadcasts.get('items', [])
                if item['status']['lifeCycleStatus'] == 'live'
            }
            
            if not self.active_livestreams:
                logger.warning(f"No active livestreams found for channel {channel_id}")
            else:
                logger.info(f"Found {len(self.active_livestreams)} active livestreams")
            
            return self.active_livestreams
            
        except HttpError as e:
            logger.error(f"YouTube API error: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Error getting active livestreams: {str(e)}")
            return {}
    
    def connect_to_chat(self, livestream_id: str = None) -> bool:
        """
        Connect to the live chat of a specific livestream.
        
        Args:
            livestream_id: ID of the livestream to connect to
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Use provided livestream_id or the one set during initialization
            self.livestream_id = livestream_id or self.livestream_id
            
            if not self.livestream_id:
                logger.error("No livestream ID provided")
                return False
            
            if not self.youtube:
                if not self.authenticate():
                    return False
            
            # Get live chat ID for the livestream
            video_response = self.youtube.videos().list(
                part="liveStreamingDetails",
                id=self.livestream_id
            ).execute()
            
            if not video_response.get('items'):
                logger.error(f"No video found with ID {self.livestream_id}")
                return False
            
            video_details = video_response['items'][0]
            if 'liveStreamingDetails' not in video_details or 'activeLiveChatId' not in video_details['liveStreamingDetails']:
                logger.error(f"Video {self.livestream_id} does not have an active live chat")
                return False
            
            self.live_chat_id = video_details['liveStreamingDetails']['activeLiveChatId']
            logger.info(f"Connected to live chat {self.live_chat_id} for livestream {self.livestream_id}")
            
            self.is_connected = True
            self.next_page_token = None
            return True
            
        except HttpError as e:
            logger.error(f"YouTube API error when connecting to chat: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error connecting to live chat: {str(e)}")
            return False
    
    def _process_chat_message(self, message: Dict) -> Optional[str]:
        """
        Process a chat message and extract prompt if valid.
        
        Args:
            message: Chat message data from YouTube API
            
        Returns:
            str: Extracted prompt or None if not a valid prompt
        """
        try:
            snippet = message.get('snippet', {})
            if not snippet:
                return None
                
            # Extract message text and author details
            display_message = snippet.get('displayMessage', '')
            author_name = snippet.get('authorChannelId', {}).get('value', 'Unknown')
            author_display_name = snippet.get('displayName', 'Unknown')
            
            # Simple parsing - any message is treated as a prompt for now
            # You can implement more complex filtering/parsing here
            if display_message and len(display_message) > 3:
                # Log the message
                logger.info(f"Chat message from {author_display_name}: {display_message}")
                
                # Return as prompt
                return {
                    'text': display_message,
                    'author': author_display_name,
                    'author_id': author_name,
                    'timestamp': datetime.now().isoformat()
                }
                
            return None
            
        except Exception as e:
            logger.error(f"Error processing chat message: {str(e)}")
            return None
    
    def _fetch_chat_messages(self) -> List[Dict]:
        """
        Fetch new chat messages from YouTube API.
        
        Returns:
            List[Dict]: List of new chat messages
        """
        try:
            if not self.is_connected or not self.live_chat_id:
                logger.error("Not connected to a live chat")
                return []
            
            # Fetch chat messages with pagination
            chat_response = self.youtube.liveChatMessages().list(
                liveChatId=self.live_chat_id,
                part="snippet,authorDetails",
                pageToken=self.next_page_token
            ).execute()
            
            # Update pagination token for next request
            self.next_page_token = chat_response.get('nextPageToken')
            
            # Extract messages
            messages = chat_response.get('items', [])
            
            # Update polling interval based on API response
            polling_interval = chat_response.get('pollingIntervalMillis', 5000) / 1000
            
            return messages, polling_interval
            
        except HttpError as e:
            if e.resp.status == 404:
                logger.error("Live chat not found or ended")
                self.is_connected = False
            else:
                logger.error(f"YouTube API error when fetching messages: {str(e)}")
            return [], 5
        except Exception as e:
            logger.error(f"Error fetching chat messages: {str(e)}")
            return [], 5
    
    def _chat_listener(self):
        """Background thread that listens for new chat messages."""
        logger.info("Starting chat listener thread")
        
        # Initial polling interval (will be updated by API)
        polling_interval = 5
        
        while not self.stop_event.is_set() and self.is_connected:
            try:
                # Fetch new messages
                messages, polling_interval = self._fetch_chat_messages()
                
                # Process each message
                for message in messages:
                    prompt = self._process_chat_message(message)
                    if prompt:
                        try:
                            # Add to queue if there's space
                            if self.prompt_queue.qsize() < self.max_queue_size:
                                self.prompt_queue.put(prompt, block=False)
                                logger.info(f"Added prompt to queue: {prompt['text'][:50]}...")
                                
                                # Call callback function if provided
                                if self.callback:
                                    self.callback(prompt)
                            else:
                                logger.warning("Prompt queue full, skipping message")
                        except queue.Full:
                            logger.warning("Prompt queue full, skipping message")
                
                # Wait for next polling interval
                time.sleep(polling_interval)
                
            except Exception as e:
                logger.error(f"Error in chat listener thread: {str(e)}")
                time.sleep(5)  # Wait before retrying
        
        logger.info("Chat listener thread stopped")
    
    def start_listener(self) -> bool:
        """
        Start the background thread that listens for chat messages.
        
        Returns:
            bool: True if started successfully
        """
        if not self.is_connected:
            logger.error("Cannot start listener: not connected to a live chat")
            return False
        
        # Reset stop event
        self.stop_event.clear()
        
        # Start listener thread
        self.chat_thread = threading.Thread(target=self._chat_listener, daemon=True)
        self.chat_thread.start()
        
        logger.info("Chat listener started")
        return True
    
    def stop_listener(self):
        """Stop the chat listener thread."""
        self.stop_event.set()
        if self.chat_thread and self.chat_thread.is_alive():
            self.chat_thread.join(timeout=5)
        logger.info("Chat listener stopped")
    
    def get_next_prompt(self, block: bool = True, timeout: float = None) -> Optional[Dict]:
        """
        Get the next prompt from the queue.
        
        Args:
            block: If True, block until a prompt is available
            timeout: How long to wait for a prompt if blocking
            
        Returns:
            Dict: Prompt data or None if queue empty/timeout
        """
        try:
            return self.prompt_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
    
    def close(self):
        """Clean up resources."""
        self.stop_listener()
        self.is_connected = False
        logger.info("YouTube Live Chat handler closed")


class SimulatedYouTubeLiveChat(YouTubeLiveChat):
    """Simulated chat handler for testing without actual YouTube connection."""
    
    def __init__(self, max_queue_size: int = 10, callback: Callable = None):
        """
        Initialize simulated chat for testing.
        
        Args:
            max_queue_size: Maximum number of prompts to queue
            callback: Function to call when a new prompt is received
        """
        super().__init__("", "", max_queue_size, callback)
        self.is_connected = True
        self.sample_prompts = [
            "A sunset over a calm ocean with palm trees",
            "A spaceship launching from Earth into the stars",
            "A futuristic city with flying cars and neon lights",
            "A magical forest with glowing mushrooms and fairies",
            "A snowy mountain landscape with northern lights",
            "A robot playing chess with a human in a park",
            "A dragon soaring through clouds over a medieval castle",
            "An underwater city with mermaids and colorful fish",
            "A cyberpunk street scene with rain and reflections",
            "A peaceful garden with butterflies and cherry blossoms",
        ]
    
    def authenticate(self) -> bool:
        """Simulate authentication."""
        logger.info("Simulated YouTube authentication successful")
        return True
    
    def get_active_livestreams(self, channel_id: str = None) -> Dict:
        """Simulate getting active livestreams."""
        self.active_livestreams = {"sim-livestream-id": "Simulated Livestream"}
        logger.info("Found 1 simulated active livestream")
        return self.active_livestreams
    
    def connect_to_chat(self, livestream_id: str = None) -> bool:
        """Simulate connecting to a livestream."""
        self.livestream_id = livestream_id or "sim-livestream-id"
        self.is_connected = True
        logger.info(f"Connected to simulated live chat for livestream {self.livestream_id}")
        return True
    
    def _chat_listener(self):
        """Simulate chat messages arriving periodically."""
        logger.info("Starting simulated chat listener thread")
        
        import random
        
        while not self.stop_event.is_set() and self.is_connected:
            try:
                # Wait a random time between 5-15 seconds
                time.sleep(random.uniform(5, 15))
                
                # Generate a simulated prompt
                prompt_text = random.choice(self.sample_prompts)
                author_name = f"SimUser{random.randint(1000, 9999)}"
                
                prompt = {
                    'text': prompt_text,
                    'author': author_name,
                    'author_id': f"sim-{author_name}",
                    'timestamp': datetime.now().isoformat()
                }
                
                # Add to queue
                try:
                    if self.prompt_queue.qsize() < self.max_queue_size:
                        self.prompt_queue.put(prompt, block=False)
                        logger.info(f"Simulated chat: {author_name}: {prompt_text}")
                        
                        # Call callback if provided
                        if self.callback:
                            self.callback(prompt)
                except queue.Full:
                    logger.warning("Prompt queue full, skipping simulated message")
                    
            except Exception as e:
                logger.error(f"Error in simulated chat listener: {str(e)}")
                time.sleep(5)
        
        logger.info("Simulated chat listener thread stopped")


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Example callback function
    def on_new_prompt(prompt):
        print(f"New prompt received: {prompt['text']}")
    
    # Testing with simulated chat
    chat = SimulatedYouTubeLiveChat(max_queue_size=5, callback=on_new_prompt)
    chat.start_listener()
    
    try:
        # Run for 1 minute
        time.sleep(60)
        
        # Process prompts
        while not chat.prompt_queue.empty():
            prompt = chat.get_next_prompt(block=False)
            if prompt:
                print(f"Processing: {prompt['text']}")
    
    finally:
        chat.close()