# -*- coding: utf-8 -*-
"""
Handles interactions with Large Language Models (LLMs) like Gemini, OpenAI, Anthropic, and Grok.
Includes API client configuration and generic call functions with retry logic.
"""

import os
import json
import re
import time
import traceback
import sys

from google import genai
from google.genai import types
import openai

try:
    import anthropic
except ImportError:
    anthropic = None

import config

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

_openai_client = None
_gemini_client = None
_anthropic_client = None


def configure_llm_clients():
    global _openai_client, _gemini_client, _anthropic_client
    gemini_api_key = config.CONFIG.get("api_key")
    if gemini_api_key:
        try:
            _gemini_client = genai.Client(api_key=gemini_api_key)
            print(f"Google Generative AI configured using model: {config.CONFIG['gemini_model']}")
        except Exception as e:
            print(f"Error configuring Generative AI: {e}")
    else:
        print("WARNING: GEMINI_KEY not set; Gemini features will be disabled.")

    openai_api_key = config.CONFIG.get("openai_api_key")
    if openai_api_key:
        try:
            _openai_client = openai.OpenAI(api_key=openai_api_key)
            print(f"OpenAI SDK configured using model: {config.CONFIG['openai_model']}")
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            _openai_client = None
    else:
        print("WARNING: OPENAI_KEY not set; OpenAI features will be disabled.")
        _openai_client = None

    anthropic_api_key = config.CONFIG.get("anthropic_api_key")
    if config.CONFIG.get("use_anthropic", False):
        if not anthropic_api_key:
            print("WARNING: ANTHROPIC_KEY not set; Anthropic features will be disabled.")
            _anthropic_client = None
        elif anthropic is None:
            print("WARNING: anthropic library not installed; Anthropic features will be disabled.")
            _anthropic_client = None
        else:
            try:
                _anthropic_client = anthropic.Client(api_key=anthropic_api_key)
                print(f"Anthropic client configured using model: {config.CONFIG['anthropic_model']}")
            except Exception as e:
                print(f"Error initializing Anthropic client: {e}")
                _anthropic_client = None


def call_llm(prompt, retries=None, delay=None, output_format="text"):
    if _anthropic_client is None and _openai_client is None and _gemini_client is None:
        configure_llm_clients()
    if config.CONFIG.get("use_grok", False):
        return _call_grok(prompt, retries, delay, output_format)
    if config.CONFIG.get("use_anthropic", False):
        if _anthropic_client is None:
            print("Error: Attempted to call Anthropic, but client is not configured.")
            return None
        return _call_anthropic(prompt, retries, delay, output_format)
    if config.CONFIG.get("use_openai", False):
        if _openai_client is None:
            print("Error: Attempted to call OpenAI, but client is not configured.")
            return None
        return _call_openai(prompt, retries, delay, output_format)
    return _call_gemini(prompt, retries, delay, output_format)


def _call_gemini(prompt, retries=None, delay=None, output_format="text"):
    retries = retries if retries is not None else config.CONFIG.get("generation_retries", 3)
    delay = delay if delay is not None else config.CONFIG.get("generation_delay", 5)
    model = config.CONFIG.get("gemini_model")
    for attempt in range(retries):
        try:
            response = _gemini_client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=config.CONFIG.get("temperature", 1.0),
                    safety_settings=[
                        types.SafetySetting(
                            category="HARM_CATEGORY_HATE_SPEECH",
                            threshold="BLOCK_ONLY_HIGH",
                        )
                    ],
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=config.CONFIG.get("thinking_budget")
                    ),
                ),
            )
            content = None
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    content = candidate.content.parts[0].text.strip()
            elif hasattr(response, "parts") and response.parts:
                content = response.text.strip()
            if content is not None:
                if output_format == "json":
                    try:
                        cleaned = re.sub(r"^```json\n?", "", content, flags=re.IGNORECASE | re.MULTILINE)
                        cleaned = re.sub(r"\n?```$", "", cleaned)
                        return json.loads(cleaned)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON from Gemini response: {e}")
                else:
                    cleaned = re.sub(r"^```[a-z]*\n?", "", content, flags=re.IGNORECASE | re.MULTILINE)
                    cleaned = re.sub(r"\n?```$", "", cleaned)
                    return cleaned.strip()
            print(f"Warning: No valid content from Gemini (Attempt {attempt + 1})")
        except Exception as e:
            print(f"Error calling Gemini API (Attempt {attempt + 1}/{retries}): {e}")
            traceback.print_exc()
        if attempt < retries - 1:
            time.sleep(delay)
    return None


def _call_openai(prompt, retries=None, delay=None, output_format="text"):
    retries = retries if retries is not None else config.CONFIG.get("generation_retries", 3)
    delay = delay if delay is not None else config.CONFIG.get("generation_delay", 5)
    model_name = config.CONFIG.get("openai_model", "gpt-3.5-turbo-0613")
    messages = [{"role": "user", "content": prompt}]
    api_args = {"model": model_name, "messages": messages}
    if output_format == "json":
        api_args["response_format"] = {"type": "json_object"}
    for attempt in range(retries):
        try:
            response = _openai_client.chat.completions.create(**api_args)
            if not response.choices:
                print(f"Warning: No choices from OpenAI (Attempt {attempt + 1})")
                continue
            choice = response.choices[0]
            content = (
                choice.message.content.strip()
                if hasattr(choice.message, "content") and choice.message.content
                else choice.message.get("content", "").strip()
            )
            if content:
                if output_format == "json":
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON from OpenAI response: {e}")
                else:
                    cleaned = re.sub(
                        r"^```[a-z]*\n?", "", content, flags=re.IGNORECASE | re.MULTILINE
                    )
                    cleaned = re.sub(r"\n?```$", "", cleaned)
                    return cleaned.strip()
            print(f"Warning: Empty content from OpenAI (Attempt {attempt + 1})")
        except Exception as e:
            print(f"Error calling OpenAI API (Attempt {attempt + 1}/{retries}): {e}")
            traceback.print_exc()
        if attempt < retries - 1:
            time.sleep(delay)
    return None


def _call_anthropic(prompt, retries=None, delay=None, output_format="text"):
    retries = retries if retries is not None else config.CONFIG.get("generation_retries", 3)
    delay = delay if delay is not None else config.CONFIG.get("generation_delay", 5)
    model = config.CONFIG.get("anthropic_model")
    max_tokens = config.CONFIG.get("thinking_budget")
    temperature = config.CONFIG.get("temperature", 0.7)
    for attempt in range(retries):
        try:
            response = _anthropic_client.messages.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.content[0].text if response.content else None
            if content:
                if output_format == "json":
                    try:
                        cleaned = re.sub(
                            r"^```json\n?", "", content, flags=re.IGNORECASE | re.MULTILINE
                        )
                        cleaned = re.sub(r"\n?```$", "", cleaned)
                        return json.loads(cleaned)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON from Anthropic response: {e}")
                else:
                    cleaned = re.sub(
                        r"^```[a-z]*\n?", "", content, flags=re.IGNORECASE | re.MULTILINE
                    )
                    cleaned = re.sub(r"\n?```$", "", cleaned)
                    return cleaned.strip()
            print(f"Warning: No valid content from Anthropic (Attempt {attempt + 1})")
        except Exception as e:
            print(f"Error calling Anthropic API (Attempt {attempt + 1}/{retries}): {e}")
            traceback.print_exc()
        if attempt < retries - 1:
            time.sleep(delay)
    return None


def _call_grok(prompt, retries=None, delay=None, output_format="text"):
    retries = retries if retries is not None else config.CONFIG.get("generation_retries", 3)
    delay = delay if delay is not None else config.CONFIG.get("generation_delay", 5)
    grok_model = config.CONFIG.get("grok_model", "grok-3-mini-beta")
    temperature = config.CONFIG.get("temperature", 0.7)
    grok_api_key = config.CONFIG.get("grok_api_key")
    if not grok_api_key:
        print("Error: GROK_KEY not set; Grok features will be disabled.")
        return None
    grok_api_base = config.CONFIG.get("grok_base_url", "https://api.x.ai/v1")
    _openai_client = openai.OpenAI(api_key=grok_api_key, base_url=grok_api_base)
    messages = [{"role": "user", "content": prompt}]
    api_args = {"model": grok_model, "temperature": temperature, "messages": messages}
    if output_format == "json":
        api_args["response_format"] = {"type": "json_object"}
    for attempt in range(retries):
        try:
            response = _openai_client.chat.completions.create(**api_args)
            if not response.choices:
                print(f"Warning: No choices from OpenAI (Attempt {attempt + 1})")
                continue
            choice = response.choices[0]
            content = (
                choice.message.content.strip()
                if hasattr(choice.message, "content") and choice.message.content
                else choice.message.get("content", "").strip()
            )
            if content:
                if output_format == "json":
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON from OpenAI response: {e}")
                else:
                    cleaned = re.sub(
                        r"^```[a-z]*\n?", "", content, flags=re.IGNORECASE | re.MULTILINE
                    )
                    cleaned = re.sub(r"\n?```$", "", cleaned)
                    return cleaned.strip()
            print(f"Warning: Empty content from OpenAI (Attempt {attempt + 1})")
        except Exception as e:
            print(f"Error calling OpenAI API (Attempt {attempt + 1}/{retries}): {e}")
            traceback.print_exc()
        if attempt < retries - 1:
            time.sleep(delay)
    return None