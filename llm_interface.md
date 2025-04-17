# File: llm_interface.py

## Overview

`llm_interface.py` is a core utility module responsible for managing interactions with Large Language Models (LLMs), specifically Google Gemini and OpenAI models. It provides a standardized interface (`call_llm`) for generating text or structured data from these models, abstracting away the specifics of each API. The module handles API client configuration, selects the active model based on configuration, and implements robust retry logic for API calls.

This file is crucial for any part of the project that requires generative text capabilities, likely being imported and used by modules involved in the main processing pipeline (`pipeline.py`) or specific generation tasks.

## Project Context

Within the project structure, `llm_interface.py` acts as a dedicated layer for all LLM communication. It depends on `config.py` for API keys, model names, and generation parameters (retries, delay, temperature, etc.). Other modules that need to interact with an LLM (e.g., `pipeline.py`, or potentially parts of `symbolic_parser.py` if it involved LLM calls) would import `llm_interface` and use its public functions (`configure_llm_clients`, `call_llm`).
