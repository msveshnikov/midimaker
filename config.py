# -*- coding: utf-8 -*-
"""
Configuration settings for the MidiMaker project.
"""

import os

# Configuration dictionary
CONFIG = {
    "gemini_model": "gemini-2.5-flash-preview-04-17", #"gemini-2.5-pro-exp-03-25", #"gemini-2.5-pro-preview-05-06" 
    "api_key": os.environ.get("GEMINI_KEY"),
    "openai_model": "o4-mini", 
    "openai_api_key": os.getenv("OPENAI_KEY"), # Recommended: Load from environment
    "anthropic_api_key": os.getenv("CLAUDE_KEY"), # Recommended: Load from environment
    "grok_api_key": os.getenv("GROK_KEY"), # Recommended: Load from environment
    "anthropic_model": "claude-3-7-sonnet-20250219", # Default Anthropic model
    "use_openai": False,  # Set to True to use OpenAI instead of Gemini
    "use_anthropic": False,  # Set to True to use Anthropic instead of Gemini
    "use_grok": False,  # Set to True to use Anthropic instead of Gemini
    "initial_description": "Adrenochrome",
    "output_dir": "output",
    "default_tempo": 120,
    "default_timesig": (4, 4),
    "default_key": "Cmaj",  # Will likely be overridden by enrichment
    "default_program": 0,  # Default GM Program (Acoustic Grand Piano)
    "default_instrument_name": "Piano",  # Default instrument name
    "generation_retries": 2,
    "generation_delay": 65,  # Seconds between retries (Gemini has rate limits)
    "max_total_bars": 128,  # Limit total length for safety/cost
    "min_section_bars": 8,  # Minimum bars per generated section
    "max_section_bars": 32,  # Maximum bars per generated section
    "temperature": 1.2,  # LLM Temperature for creativity vs predictability
    "thinking_budget": 8192
}