# -*- coding: utf-8 -*-
"""
Handles interactions with Large Language Models (LLMs) like Gemini and OpenAI.
Includes API client configuration and generic call functions with retry logic.
"""

import os
import json
import re
import time
import traceback

import google.generativeai as genai
import openai

import config # Import the configuration

# Global clients, initialized by configure_llm_clients
_openai_client = None


def configure_llm_clients():
    """Configures the Google Generative AI and OpenAI libraries based on config."""
    global _openai_client

    # --- Configure Gemini ---
    gemini_api_key = config.CONFIG.get("api_key")
    if not gemini_api_key:
        print(
            "ERROR: GEMINI_KEY environment variable is not set or empty in config."
            " Please set it or add it to the CONFIG dictionary."
        )
        # Decide if this is fatal or if OpenAI can still be used
        # exit(1) # Option: Exit if Gemini is essential
    else:
        try:
            genai.configure(api_key=gemini_api_key)
            print(f"Google Generative AI configured using model: {config.CONFIG['gemini_model']}")
        except Exception as e:
            print(f"Error configuring Generative AI: {e}")
            print("Please ensure your GEMINI_KEY is set correctly and valid.")
            # exit(1) # Option: Exit if Gemini fails configuration

    # --- Configure OpenAI ---
    openai_api_key = config.CONFIG.get("openai_api_key")
    if not openai_api_key:
        print(
            "Warning: OPENAI_KEY not found in environment or CONFIG."
            " OpenAI features will be disabled."
        )
        _openai_client = None
    else:
        try:
            _openai_client = openai.OpenAI(api_key=openai_api_key)
            print(f"OpenAI SDK configured using model: {config.CONFIG['openai_model']}")
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            _openai_client = None


def call_llm(prompt, retries=None, delay=None, output_format="text"):
    """
    Calls the configured LLM (Gemini or OpenAI) based on CONFIG['use_openai'].

    Args:
        prompt (str): The prompt to send to the LLM.
        retries (int, optional): Maximum number of retry attempts. Defaults to CONFIG['generation_retries'].
        delay (int, optional): Delay in seconds between retries. Defaults to CONFIG['generation_delay'].
        output_format (str): Expected output format ('text' or 'json').

    Returns:
        str, dict, or None: The generated content, or None if generation failed after retries.
    """
    use_openai = config.CONFIG.get("use_openai", False)

    if use_openai:
        if _openai_client is None:
            print("Error: Attempted to call OpenAI, but client is not configured.")
            return None
        return _call_openai(prompt, retries, delay, output_format)
    else:
        # Assuming Gemini is configured if use_openai is False
        # Add check if Gemini configuration failed earlier if needed
        return _call_gemini(prompt, retries, delay, output_format)


def _call_gemini(prompt, retries=None, delay=None, output_format="text"):
    """
    Calls the Gemini API with the specified prompt and handles retries.

    Args:
        prompt (str): The prompt to send to the LLM.
        retries (int, optional): Maximum number of retry attempts. Defaults to CONFIG['generation_retries'].
        delay (int, optional): Delay in seconds between retries. Defaults to CONFIG['generation_delay'].
        output_format (str): Expected output format ('text' or 'json').

    Returns:
        str, dict, or None: The generated content, or None if generation failed after retries.
    """
    # print(f"Prompt: {prompt}") # Uncomment for debugging prompts

    retries = retries if retries is not None else config.CONFIG["generation_retries"]
    delay = delay if delay is not None else config.CONFIG["generation_delay"]
    model = genai.GenerativeModel(config.CONFIG["gemini_model"])
    gen_config_args = {"temperature": config.CONFIG["temperature"]}

    generation_config = genai.types.GenerationConfig(**gen_config_args)
    safety_settings = config.CONFIG.get("safety_settings")

    for attempt in range(retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings,
            )

            # Debug: Print raw response structure if needed
            # print(f"DEBUG: Gemini Response (Attempt {attempt + 1}): {response}")

            if (
                hasattr(response, "prompt_feedback")
                and response.prompt_feedback.block_reason
            ):
                print(
                    f"Prompt blocked (Attempt {attempt + 1}):"
                    f" {response.prompt_feedback.block_reason}"
                )
                return None

            content = None
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                # Check finish reason at the top level and candidate level
                top_level_finish_reason = getattr(response, "finish_reason", None) # May not exist
                candidate_finish_reason = getattr(candidate, "finish_reason", None)

                # Prioritize candidate finish reason if available
                finish_reason = candidate_finish_reason if candidate_finish_reason else top_level_finish_reason

                if finish_reason not in [None, "STOP", 1]: # 1 is often STOP for Gemini
                     print(
                        f"Generation stopped for reason: {finish_reason} (Code: {candidate.finish_reason if hasattr(candidate, 'finish_reason') else 'N/A'}) on attempt {attempt + 1}"
                     )
                     # Try to get content even if stopped early, but check safety ratings
                     if hasattr(candidate, 'safety_ratings') and any(rating.probability != 'NEGLIGIBLE' for rating in candidate.safety_ratings):
                         print(f"Content potentially unsafe, blocked by API. Ratings: {candidate.safety_ratings}")
                         # Fall through to retry logic if possible, but likely blocked

                # Check for content within the candidate
                if candidate.content and candidate.content.parts:
                    content = candidate.content.parts[0].text.strip()
                elif not content: # If we didn't get content yet
                     print(f"Warning: Received response with no usable candidate content (Attempt {attempt + 1}). Finish Reason: {finish_reason}")
                     # Fall through to retry logic

            # Fallback or primary check via response.text (if parts exist)
            elif hasattr(response, "parts") and response.parts:
                 content = response.text.strip()
                 finish_reason = getattr(response, "finish_reason", None)
                 if finish_reason not in [None, "STOP", 1]:
                     print(f"Generation stopped for reason: {finish_reason} (Response Level) on attempt {attempt + 1}")

            elif not content: # If still no content
                print(f"Warning: Received response with no parts or candidates (Attempt {attempt + 1}).")
                # Fall through to retry logic

            # Process the content based on expected format
            if content is not None:
                if output_format == "json":
                    try:
                        content_cleaned = re.sub(
                            r"^```json\n?", "", content, flags=re.IGNORECASE | re.MULTILINE
                        )
                        content_cleaned = re.sub(r"\n?```$", "", content_cleaned)
                        return json.loads(content_cleaned)
                    except json.JSONDecodeError as json_e:
                        print(
                            f"Error decoding JSON response (Attempt {attempt + 1}): {json_e}"
                        )
                        print(f"Received text: {content[:500]}...")
                else:
                    content_cleaned = re.sub(
                        r"^```[a-z]*\n?",
                        "",
                        content,
                        flags=re.MULTILINE | re.IGNORECASE,
                    )
                    content_cleaned = re.sub(r"\n?```$", "", content_cleaned)
                    return content_cleaned.strip()

            print(f"Warning: Could not extract valid content (Attempt {attempt + 1}).")

        except Exception as e:
            print(f"Error calling Gemini API (Attempt {attempt + 1}/{retries}): {e}")
            traceback.print_exc()

        if attempt < retries - 1:
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
        else:
            print("Max retries reached. Failing.")
            return None
    return None



def _call_openai(prompt, retries=None, delay=None, output_format="text"):
    """
    Calls the OpenAI API with the specified prompt and handles retries.

    Args:
        prompt (str): The prompt to send to the LLM.
        retries (int, optional): Maximum number of retry attempts. Defaults to CONFIG['generation_retries'].
        delay (int, optional): Delay in seconds between retries. Defaults to CONFIG['generation_delay'].
        output_format (str): Expected output format ('text' or 'json').

    Returns:
        str, dict, or None: The generated content, or None if generation failed after retries.
    """
    if _openai_client is None:
        print("Error: OpenAI client not initialized. Cannot make API call.")
        return None

    # print(f"Prompt: {prompt}") # Uncomment for debugging prompts
    retries = retries if retries is not None else config.CONFIG.get("generation_retries", 3)
    delay = delay if delay is not None else config.CONFIG.get("generation_delay", 5)
    model_name = config.CONFIG.get("openai_model", "gpt-3.5-turbo-0125") # Default fallback

    # Prepare messages for OpenAI ChatCompletion format
    messages = [{"role": "user", "content": prompt}]

    # Prepare API call arguments
    api_args = {
        "model": model_name,
        "messages": messages,
        "reasoning_effort": "high"  # This can be set to low, medium or high.
    }

    # Use OpenAI's JSON mode if requested
    if output_format == "json":
        # Note: For JSON mode to work reliably, your prompt should explicitly
        # instruct the model to output JSON.
        # Models like gpt-3.5-turbo-0125 and later support this well.
        api_args["response_format"] = {"type": "json_object"}
        # print("DEBUG: Requesting JSON format from OpenAI.") # Uncomment for debugging

    for attempt in range(retries):
        try:
            response = _openai_client.chat.completions.create(**api_args)

            # Debug: Print raw response structure if needed
            # print(f"DEBUG: OpenAI Response (Attempt {attempt + 1}): {response.model_dump_json(indent=2)}")

            if not response.choices:
                print(f"Warning: Received response with no choices (Attempt {attempt + 1}).")
                # Fall through to retry logic

            choice = response.choices[0]
            finish_reason = choice.finish_reason

            # Check finish reason
            if finish_reason == "stop":
                content = choice.message.content
                if content:
                    content = content.strip()
                else:
                     print(f"Warning: Received 'stop' finish reason but no content (Attempt {attempt + 1}).")
                     # Fall through to retry logic
                     continue # Go to next retry attempt

            elif finish_reason == "length":
                print(f"Warning: Generation stopped due to length (max_tokens reached) on attempt {attempt + 1}.")
                # Return partial content, as it might still be useful
                content = choice.message.content.strip() if choice.message.content else None
                # Decide if you want to return partial or retry. Let's return partial for now.

            elif finish_reason == "content_filter":
                print(f"Generation stopped due to OpenAI content filter (Attempt {attempt + 1}).")
                # This is similar to Gemini's block reason
                return None # Content is blocked, unlikely to succeed on retry

            else: # Other reasons like 'tool_calls' (if applicable) or unexpected ones
                print(f"Generation stopped for reason: {finish_reason} on attempt {attempt + 1}.")
                content = choice.message.content.strip() if choice.message.content else None
                # Potentially retry if content is None, otherwise process what we have

            # Process the content based on expected format
            if content is not None:
                if output_format == "json":
                    try:
                        # If using response_format={"type": "json_object"},
                        # OpenAI should guarantee valid JSON string.
                        return json.loads(content)
                    except json.JSONDecodeError as json_e:
                        print(
                            f"Error decoding JSON response even with JSON mode (Attempt {attempt + 1}): {json_e}"
                        )
                        print(f"Received text: {content[:500]}...")
                        # Fall through to retry logic as the model failed to produce valid JSON
                else: # output_format == "text"
                    # Clean potential markdown code blocks if not expecting JSON
                    content_cleaned = re.sub(
                        r"^```[a-z]*\n?",
                        "",
                        content,
                        flags=re.MULTILINE | re.IGNORECASE,
                    )
                    content_cleaned = re.sub(r"\n?```$", "", content_cleaned)
                    return content_cleaned.strip()

            # If we reached here, content was None or JSON decoding failed
            print(f"Warning: Could not extract or process valid content (Attempt {attempt + 1}). Finish Reason: {finish_reason}")


        except openai.APIError as e:
            # Handle API errors (e.g., server issues)
            print(f"OpenAI API Error (Attempt {attempt + 1}/{retries}): {e}")
        except openai.RateLimitError as e:
            # Handle rate limit errors
            print(f"OpenAI Rate Limit Error (Attempt {attempt + 1}/{retries}): {e}")
            # Rate limit errors often benefit most from delays
        except openai.AuthenticationError as e:
             print(f"OpenAI Authentication Error: {e}. Check your API key.")
             return None # No point retrying if auth fails
        except Exception as e:
            # Handle other potential errors (network issues, etc.)
            print(f"Error calling OpenAI API (Attempt {attempt + 1}/{retries}): {e}")
            traceback.print_exc()

        # Wait before retrying if it wasn't the last attempt
        if attempt < retries - 1:
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
        else:
            print("Max retries reached. Failing.")
            return None

    # Should technically not be reached if loop completes, but as a safeguard:
    print("Exited retry loop unexpectedly. Failing.")
    return None