# -*- coding: utf-8 -*-
"""
Configuration settings for the MidiMaker project.
"""

import os

# Configuration dictionary
CONFIG = {
    "gemini_model": "gemini-2.5-flash-preview-04-17", #"gemini-2.0-flash-thinking-exp-01-21" 
    "api_key": os.environ.get("GEMINI_KEY"),
    "openai_model": "o4-mini", 
    "openai_api_key": os.getenv("OPENAI_KEY"), # Recommended: Load from environment
    "use_openai": False,  # Set to True to use OpenAI instead of Gemini
    "initial_description": "Ambient soundscape with evolving textures and ethereal pads. (Ambient, atmospheric, subtle)",
    "output_dir": "output",
    "default_tempo": 120,
    "default_timesig": (4, 4),
    "default_key": "Cmaj",  # Will likely be overridden by enrichment
    "default_program": 0,  # Default GM Program (Acoustic Grand Piano)
    "default_instrument_name": "Piano",  # Default instrument name
    "generation_retries": 3,
    "generation_delay": 65,  # Seconds between retries (Gemini has rate limits)
    "max_total_bars": 128,  # Limit total length for safety/cost
    "min_section_bars": 8,  # Minimum bars per generated section
    "max_section_bars": 32,  # Maximum bars per generated section
    "temperature": 0.7,  # LLM Temperature for creativity vs predictability
    "safety_settings": {  # Configure content safety settings for Gemini
        # Options: BLOCK_NONE, BLOCK_ONLY_HIGH, BLOCK_MEDIUM_AND_ABOVE, BLOCK_LOW_AND_ABOVE
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    },
}