# -*- coding: utf-8 -*-
"""
Generates MIDI music from textual descriptions using LLMs and a compact symbolic format.

This script implements a pipeline:
1. Enrich a basic music description using an LLM.
2. Generate a section plan (structure, goals, bars) using an LLM.
3. Generate symbolic music notation section by section using an LLM based on the plan.
4. Concatenate the symbolic sections.
5. Parse the symbolic text into structured data.
6. Convert the structured data into a MIDI file using pretty_midi.
"""

import datetime
import json
import math
import os
import re
import time
import traceback

import google.generativeai as genai
import openai
import pretty_midi

# Configuration dictionary
CONFIG = {
    "gemini_model": "gemini-2.5-pro-exp-03-25", #"gemini-2.0-flash-thinking-exp-01-21" 
    "api_key": os.environ.get("GEMINI_KEY"),
    "openai_model": "o4-mini", 
    "openai_api_key": os.getenv("OPENAI_KEY"), # Recommended: Load from environment
    "use_openai": False,  # Set to True to use OpenAI instead of Gemini
    "initial_description": "Adrenochrome",
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


# --- Symbolic Format Definition (for prompts and parsing) ---
SYMBOLIC_FORMAT_DEFINITION = """
Use this compact symbolic format ONLY. Each command must be on a new line. Do NOT include comments after the command parameters on the same line.
- `INST:<InstrumentName>`: Sets the active instrument context for subsequent non-drum notes/chords. Examples: Pno, Gtr, Bass, Str, Flt, Tpt, SynPad, SynLead, Arp. This determines the MIDI program number for melodic parts. Use lowercase instrument names from the provided list.
- `T:<BPM>`: Sets the tempo in Beats Per Minute (e.g., T:120). Must be a number.
- `TS:<N>/<D>`: Sets the time signature (e.g., TS:4/4). Denominator should ideally be a power of 2.
- `K:<Key>`: Sets the key signature (e.g., K:Cmin, K:Gmaj, K:Ddor). Use standard `pretty_midi` key names (Major: maj, Minor: min, Modes: dor, phr, lyd, mix, loc, Ion=Maj, Aeo=Min).
- `BAR:<Num>`: Marks the beginning of a bar (measure), starting from 1, strictly sequential. Timing calculations rely on this.
- `N:<Track>:<Pitch>:<Duration>:<Velocity>`: Represents a single Note event.
- `C:<Track>:<Pitches>:<Duration>:<Velocity>`: Represents a Chord event (multiple notes starting simultaneously).
- `R:<Track>:<Duration>`: Represents a Rest (silence) event for a specific track.

TrackIDs (`<Track>`): Use simple names like RH, LH, Melody, Bass, Drums, Arp1, Pad, Lead etc.
    - If the TrackID is recognized as a drum track name (e.g., 'drums', 'drumkit', 'percussion', 'elecdrums', '808drums'), the `<Pitch>` will be interpreted as a drum sound name (see below), and it will use MIDI channel 10. Case-insensitive matching for drum track IDs.
    - If the TrackID is NOT a recognized drum track name, it's considered a melodic track. The instrument sound (MIDI program) used for this track is determined by the *last* `INST:` command encountered before this note/chord/rest.
PitchNames (`<Pitch>`):
    - For Melodic Tracks: Standard notation (e.g., C4, F#5, Gb3). Middle C is C4.
    - For Drum Tracks: Use drum sound names like Kick, Snare, HHC, HHO, Crash, Ride, HT, MT, LT (case-insensitive). See mapping below. Do NOT use standard notes (C4) on drum tracks.
DurationSymbols (`<Duration>`): W (Whole), H (Half), Q (Quarter), E (Eighth), S (Sixteenth), T (Thirty-second). Append '.' for dotted notes (e.g., Q., E.). Must be one of these symbols.
Velocity (`<Velocity>`): MIDI velocity (0-127). Must be a number. Affects loudness.
Pitches (`<Pitches>`): Comma-separated list of pitch names within square brackets (e.g., [C3,Eb3,G3]). For chords on drum tracks, list drum sound names. No spaces inside brackets unless part of a pitch name (shouldn't be).

Example Note (Melodic): N:Melody:G5:E:95
Example Chord (Melodic): C:PnoLH:[C3,Eb3,G3]:H:60
Example Rest: R:Bass:W
Example Drum Note: N:Drums:Kick:Q:95
Example Drum Chord (Multiple hits): C:Drums:[Kick,HHC]:E:100

Example Sequence:
K:Amin
T:90
TS:4/4
INST:synpad  # Set instrument for melodic tracks using lowercase name
INST:synthbass # Update instrument for melodic tracks
BAR:1
C:Pad:[A3,C4,E4]:W:55 # TrackID: Pad, uses current INST:synthbass (this is a potential issue - last INST wins) -> Let's assume Pad uses synpad if INST:synpad was before INST:synthbass
INST:synpad # Re-set for Pad track if needed
C:Pad:[A3,C4,E4]:W:55 # Now Pad uses synpad
INST:synthbass # Set for Bass track
N:Bass:A2:Q.:100
N:Bass:E2:E:100
N:Bass:A2:H:100
N:Drums:Kick:Q:95 # Drums track ignores INST
N:Drums:HHC:E:80
N:Drums:HHC:E:80
N:Drums:Snare:Q:90
N:Drums:Kick:E:95
N:Drums:Kick:E:95
N:Drums:HHC:E:80
N:Drums:HHC:E:80
INST:synlead # Set for Lead track
R:Lead:H
N:Lead:C5:Q:95
N:Lead:E5:Q:95
BAR:2
INST:synpad
C:Pad:[E3,G3,B3]:W:55
INST:synthbass
N:Bass:E2:Q.:100
N:Bass:B1:E:100
N:Bass:E2:H:100
N:Drums:Kick:Q:95
N:Drums:HHC:E:80
N:Drums:HHC:E:80
N:Drums:Snare:Q:90
N:Drums:HHC:E:80
N:Drums:HHC:E:80
N:Drums:Kick:E:95
N:Drums:HHC:E:80
INST:synlead
N:Lead:F5:H:95
N:Lead:E5:Q:95
R:Lead:Q

"""

# --- Helper Functions ---

client = None  # Global OpenAI client


def configure_genai():
    """Configures the Google Generative AI and OpenAI libraries."""
    global client
    # Configure Gemini
    if not CONFIG["api_key"]:
        print(
            "ERROR: GEMINI_KEY environment variable is not set."
            " Please set it or add it to the CONFIG dictionary."
        )
        exit(1)
    try:
        genai.configure(api_key=CONFIG["api_key"])
        print(f"Google Generative AI configured using model: {CONFIG['gemini_model']}")
    except Exception as e:
        print(f"Error configuring Generative AI: {e}")
        print("Please ensure your GEMINI_KEY is set correctly and valid.")
        exit(1)
    # --- OpenAI Client Initialization ---
    # Ensure the API key is set, either directly or via environment variable
    if not CONFIG.get("openai_api_key"):
        print("Warning: OPENAI_KEY not found in environment or CONFIG.")
        # Optionally raise an error or handle differently
        # raise ValueError("OpenAI API key is required.")
        global client;
        client = None # Indicate client couldn't be initialized
    else:
        try:
            client = openai.OpenAI(api_key=CONFIG["openai_api_key"])
            print(f"OpenAI SDK configured using model: {CONFIG['openai_model']}")
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            client = None


def call_gemini(prompt, retries=None, delay=None, output_format="text"):
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
    if CONFIG["use_openai"]:
        #print("Warning: Gemini API is disabled. Using OpenAI instead.")
        return call_openai(prompt, retries, delay, output_format)
    retries = retries if retries is not None else CONFIG["generation_retries"]
    delay = delay if delay is not None else CONFIG["generation_delay"]
    model = genai.GenerativeModel(CONFIG["gemini_model"])
    gen_config_args = {"temperature": CONFIG["temperature"]}

    generation_config = genai.types.GenerationConfig(**gen_config_args)
    safety_settings = CONFIG.get("safety_settings")

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


def call_openai(prompt, retries=None, delay=None, output_format="text"):
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
    if client is None:
        print("Error: OpenAI client not initialized. Cannot make API call.")
        return None

    # print(f"Prompt: {prompt}") # Uncomment for debugging prompts
    retries = retries if retries is not None else CONFIG.get("generation_retries", 3)
    delay = delay if delay is not None else CONFIG.get("generation_delay", 5)
    model_name = CONFIG.get("openai_model", "gpt-3.5-turbo-0125") # Default fallback

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
            response = client.chat.completions.create(**api_args)

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

# --- Music Data Structures and Mappings ---

PITCH_MAP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
ACCIDENTAL_MAP = {"#": 1, "S": 1, "B": -1, "": 0}  # Allow S for sharp

# General MIDI Instrument Program Numbers (Expanded Selection) - Lowercase keys for lookup
INSTRUMENT_PROGRAM_MAP = {
    # Piano
    "pno": 0, "piano": 0, "acoustic grand piano": 0, "bright acoustic piano": 1,
    "electric grand piano": 2, "honky-tonk piano": 3, "electric piano 1": 4,
    "rhodes piano": 4, "electric piano 2": 5, "epiano": 4,
    # Chromatic Percussion
    "celesta": 8, "glockenspiel": 9, "music box": 10, "vibraphone": 11,
    "marimba": 12, "xylophone": 13, "tubular bells": 14, "dulcimer": 15,
    # Organ
    "org": 16, "organ": 16, "drawbar organ": 16, "percussive organ": 17,
    "rock organ": 18, "church organ": 19, "reed organ": 20, "accordion": 21,
    "harmonica": 22, "tango accordion": 23,
    # Guitar
    "gtr": 25, "guitar": 25, "acoustic guitar": 25, "nylon guitar": 24,
    "steel guitar": 25, "electric guitar": 27, "jazz guitar": 26,
    "clean electric guitar": 27, "muted electric guitar": 28,
    "overdriven guitar": 29, "distortion guitar": 30, "guitar harmonics": 31,
    # Bass
    "bass": 33, "acoustic bass": 32, "electric bass": 33, "finger bass": 33,
    "pick bass": 34, "fretless bass": 35, "slap bass": 36, "synth bass": 38,
    "synthbass": 38, "synth bass 2": 39,
    # Strings
    "str": 48, "strings": 48, "violin": 40, "viola": 41, "cello": 42,
    "contrabass": 43, "tremolo strings": 44, "pizzicato strings": 45,
    "orchestral harp": 46, "timpani": 47, "string ensemble 1": 48,
    "string ensemble 2": 49, "synth strings 1": 50, "synth strings 2": 51,
    # Brass
    "tpt": 56, "trumpet": 56, "trombone": 57, "tuba": 58, "muted trumpet": 59,
    "french horn": 60, "brass section": 61, "synth brass 1": 62,
    "synth brass 2": 63,
    # Reed
    "sax": 65, "soprano sax": 64, "alto sax": 65, "tenor sax": 66,
    "baritone sax": 67, "oboe": 68, "english horn": 69, "bassoon": 70,
    "clarinet": 71,
    # Pipe
    "flt": 73, "flute": 73, "piccolo": 72, "recorder": 74, "pan flute": 75,
    "blown bottle": 76, "shakuhachi": 77, "whistle": 78, "ocarina": 79,
    # Synth Lead
    "synlead": 81, "synth lead": 81, "lead 1 (square)": 80, "lead 2 (sawtooth)": 81,
    "lead 3 (calliope)": 82, "lead 4 (chiff)": 83, "lead 5 (charang)": 84,
    "lead 6 (voice)": 85, "lead 7 (fifths)": 86, "lead 8 (bass + lead)": 87,
    # Synth Pad
    "synpad": 89, "synth pad": 89, "pad 1 (new age)": 88, "pad 2 (warm)": 89,
    "pad 3 (polysynth)": 90, "pad 4 (choir)": 91, "pad 5 (bowed)": 92,
    "pad 6 (metallic)": 93, "pad 7 (halo)": 94, "pad 8 (sweep)": 95,
    # Synth FX
    "fx 1 (rain)": 96, "fx 2 (soundtrack)": 97, "fx 3 (crystal)": 98,
    "fx 4 (atmosphere)": 99, "fx 5 (brightness)": 100, "fx 6 (goblins)": 101,
    "fx 7 (echoes)": 102, "fx 8 (sci-fi)": 103,
    # Ethnic
    "sitar": 104, "banjo": 105, "shamisen": 106, "koto": 107, "kalimba": 108,
    "bag pipe": 109, "fiddle": 110, "shanai": 111,
    # Percussive
    "tinkle bell": 112, "agogo": 113, "steel drums": 114, "woodblock": 115,
    "taiko drum": 116, "melodic tom": 117, "synth drum": 118, "reverse cymbal": 119,
    # Sound Effects
    "guitar fret noise": 120, "breath noise": 121, "seashore": 122, "bird tweet": 123,
    "telephone ring": 124, "helicopter": 125, "applause": 126, "gunshot": 127,
    # Arp (Mapped to a synth sound)
    "arp": 81, "arpeggiator": 81,
    # Drums are a special case (channel 10 / index 9) - Program 0 is conventional
    "drs": 0, "drums": 0, "drumkit": 0, "elecdrums": 0, "808drums": 0, "percussion": 0,
}

# Set of lowercase track IDs that should be treated as drum tracks
# Includes instrument names commonly used for drums AND common track IDs
DRUM_TRACK_IDS = {
    k.lower()
    for k, v in INSTRUMENT_PROGRAM_MAP.items()
    if v == 0 and ("dr" in k or "kit" in k or "perc" in k)
}
DRUM_TRACK_IDS.update(
    ["drums", "drum", "drumkit", "percussion", "elecdrums", "808drums"]
)

# Standard drum note map (MIDI channel 10 / index 9) - Keys MUST be lowercase for lookup (Expanded)
DRUM_PITCH_MAP = {
    # Bass Drum
    "kick": 36, "bd": 36, "bass drum 1": 36, "bass drum": 36, "acoustic bass drum": 35, "kick 2": 35,
    # Snare
    "snare": 38, "sd": 38, "acoustic snare": 38, "electric snare": 40,
    # Hi-Hat
    "hihatclosed": 42, "hhc": 42, "closed hi hat": 42, "closed hi-hat": 42,
    "hihatopen": 46, "hho": 46, "open hi hat": 46, "open hi-hat": 46,
    "hihatpedal": 44, "hhp": 44, "pedal hi hat": 44, "pedal hi-hat": 44,
    # Cymbals
    "crash": 49, "cr": 49, "crash cymbal 1": 49, "crash cymbal 2": 57,
    "ride": 51, "rd": 51, "ride cymbal 1": 51, "ride cymbal 2": 59,
    "ride bell": 53, "rb": 53, "splash cymbal": 55, "splash": 55,
    "chinese cymbal": 52, "chinese": 52, "reverse cymbal": 119,
    # Toms
    "high tom": 50, "ht": 50, "hi tom": 50, "mid tom": 47, "mt": 47,
    "hi-mid tom": 48, "low-mid tom": 47, "low tom": 43, "lt": 43,
    "high floor tom": 43, "low floor tom": 41, "floor tom": 41, "ft": 41,
    # Hand Percussion
    "rimshot": 37, "rs": 37, "side stick": 37, "clap": 39, "cp": 39, "hand clap": 39,
    "cowbell": 56, "cb": 56, "tambourine": 54, "tmb": 54, "vibraslap": 58,
    "high bongo": 60, "low bongo": 61, "mute high conga": 62, "open high conga": 63,
    "low conga": 64, "high timbale": 65, "low timbale": 66, "high agogo": 67,
    "low agogo": 68, "cabasa": 69, "maracas": 70, "short whistle": 71,
    "long whistle": 72, "short guiro": 73, "long guiro": 74, "claves": 75, "cl": 75,
    "high wood block": 76, "low wood block": 77, "mute cuica": 78, "open cuica": 79,
    "mute triangle": 80, "open triangle": 81, "shaker": 82,
}

# Cache for pitch parsing results
_pitch_parse_cache = {}


def pitch_to_midi(pitch_name):
    """Converts pitch name (e.g., C4, F#5, Gb3) to MIDI number. Returns None if invalid."""
    pitch_name = pitch_name.strip()
    if pitch_name in _pitch_parse_cache:
        return _pitch_parse_cache[pitch_name]

    match = re.match(r"([A-G])([#sb]?)(\-?\d+)", pitch_name, re.IGNORECASE)
    if not match:
        _pitch_parse_cache[pitch_name] = None
        return None

    note, acc, oct_str = match.groups()
    try:
        octave = int(oct_str)
    except ValueError:
        _pitch_parse_cache[pitch_name] = None
        return None

    base_midi = PITCH_MAP.get(note.upper())
    if base_midi is None:
        _pitch_parse_cache[pitch_name] = None
        return None

    acc_norm = acc.upper() if acc else ""
    if acc_norm == "S":
        acc_norm = "#"
    acc_val = ACCIDENTAL_MAP.get(acc_norm, 0)
    # MIDI standard: Middle C (C4) is MIDI note 60. C0 is MIDI 12.
    # Formula: base + accidental + (octave + 1) * 12
    midi_val = base_midi + acc_val + (octave + 1) * 12

    if 0 <= midi_val <= 127:
        _pitch_parse_cache[pitch_name] = midi_val
        return midi_val
    else:
        _pitch_parse_cache[pitch_name] = None
        return None


# Cache for duration calculation results
_duration_cache = {}


def duration_to_seconds(duration_symbol, tempo, time_sig_denominator=4):
    """Converts duration symbol (W, H, Q, E, S, T, W., H., etc.) to seconds."""
    cache_key = (duration_symbol, tempo, time_sig_denominator)
    if cache_key in _duration_cache:
        return _duration_cache[cache_key]

    duration_symbol = duration_symbol.strip().upper()
    if not duration_symbol:
        print("Warning: Empty duration symbol. Defaulting to Quarter note duration.")
        duration_symbol = "Q"

    try:
        beats_per_minute = float(tempo)
        if beats_per_minute <= 0:
            print(f"Warning: Invalid tempo {tempo}. Using default 120.")
            beats_per_minute = 120

        # Duration relative to a quarter note
        duration_map = {"W": 4.0, "H": 2.0, "Q": 1.0, "E": 0.5, "S": 0.25, "T": 0.125}
        base_symbol = duration_symbol.replace(".", "")
        is_dotted = duration_symbol.endswith(".")

        relative_duration_quarters = duration_map.get(base_symbol)
        if relative_duration_quarters is None:
            print(
                f"Warning: Unknown duration symbol: '{duration_symbol}'. Defaulting to Quarter (1.0)."
            )
            relative_duration_quarters = 1.0

        if is_dotted:
            relative_duration_quarters *= 1.5

        # Calculate seconds per quarter note
        # Note: Time signature denominator affects interpretation if we think in whole notes,
        # but quarter note duration only depends on tempo.
        quarter_note_duration_sec = 60.0 / beats_per_minute

        # Calculate actual duration in seconds
        actual_duration_sec = relative_duration_quarters * quarter_note_duration_sec
        _duration_cache[cache_key] = actual_duration_sec
        return actual_duration_sec

    except ValueError:
        print(f"Warning: Could not parse tempo '{tempo}' as float. Using default 120.")
        # Recurse with default tempo, avoiding infinite loop by ensuring tempo is valid
        return duration_to_seconds(duration_symbol, 120, time_sig_denominator)
    except Exception as e:
        print(
            f"Error calculating duration for '{duration_symbol}' at tempo {tempo}: {e}. Using default 0.5s."
        )
        return 0.5


# --- Main Pipeline Functions ---


def enrich_music_description(description):
    """Step 1: Use LLM to enrich the initial music description."""
    print("\n--- Step 1: Enriching Description ---")
    prompt = f"""
Analyze the following music description. Extract or infer the key signature (using standard notation like 'Amin', 'F#maj', 'Gdor'), tempo (BPM), time signature (N/D), and suggest primary instrumentation (list like INST:SynLead, INST:SynthBass, INST:Drums).
If any parameter is explicitly mentioned, use that. If not, infer plausible values based on the description (genre, mood, artist references, etc.).
Also, identify the core mood and suggest a musical structure (like AABA, ABAC, Verse-Chorus-Bridge) if not already provided.

Output the parameters clearly at the start using these exact prefixes:
K: <key_signature>
T: <tempo_bpm>
TS: <numerator>/<denominator>
INST: <instrument1>, <instrument2>, ...
Mood: <mood_description>
Structure: <structure_suggestion>

Follow this with a brief summary elaborating on the musical style based on your analysis.

Music Description: "{description}"

Enriched Output:
"""
    enriched_result = call_gemini(prompt)

    current_key = CONFIG["default_key"]
    current_tempo = CONFIG["default_tempo"]
    current_timesig = CONFIG["default_timesig"]
    primary_instruments = [CONFIG["default_instrument_name"]]
    structure_hint = "AABA"  # Default fallback structure
    enriched_summary = f"Could not enrich description. Using defaults. Original: {description}"

    if enriched_result:
        print(f"LLM Enrichment Output:\n{enriched_result}\n")
        enriched_summary = enriched_result # Use the full output as summary unless parsing fails

        # More robust parsing using regex for each parameter line
        key_match = re.search(r"^[Kk]\s*:\s*(.+)$", enriched_result, re.MULTILINE | re.IGNORECASE)
        tempo_match = re.search(r"^[Tt]\s*:\s*(\d+(?:\.\d+)?)$", enriched_result, re.MULTILINE | re.IGNORECASE)
        ts_match = re.search(r"^[Tt][Ss]\s*:\s*(\d+)\s*/\s*(\d+)$", enriched_result, re.MULTILINE | re.IGNORECASE)
        inst_match = re.search(r"^[Ii][Nn][Ss][Tt]\s*:\s*(.+)$", enriched_result, re.MULTILINE | re.IGNORECASE)
        struct_match = re.search(r"^[Ss]tructure\s*:\s*([\w\-]+)$", enriched_result, re.MULTILINE | re.IGNORECASE)

        if key_match:
            potential_key = key_match.group(1).strip()
            # Basic validation for key format (optional but good)
            if re.match(r"^[A-Ga-g][#sb]?(maj|min|dor|phr|lyd|mix|loc|aeo|ion)?$", potential_key, re.IGNORECASE):
                current_key = potential_key
                print(f"Updated Default Key: {current_key}")
            else:
                print(f"Warning: Extracted key '{potential_key}' format unclear. Using default: {current_key}")
        if tempo_match:
            try:
                tempo_val = int(float(tempo_match.group(1))) # Allow float then convert
                if 5 <= tempo_val <= 400: # Expanded tempo range
                    current_tempo = tempo_val
                    print(f"Updated Default Tempo: {current_tempo}")
                else:
                    print(f"Warning: Ignoring extracted tempo {tempo_val} (out of range 5-400).")
            except ValueError:
                print(f"Warning: Could not parse extracted tempo '{tempo_match.group(1)}'.")
        if ts_match:
            try:
                ts_num, ts_den = int(ts_match.group(1)), int(ts_match.group(2))
                # Allow more denominators, but powers of 2 are most common
                if ts_num > 0 and ts_den > 0:
                    current_timesig = (ts_num, ts_den)
                    print(f"Updated Default Time Signature: {ts_num}/{ts_den}")
                else:
                    print(f"Warning: Ignoring extracted time signature {ts_num}/{ts_den} (non-positive values).")
            except ValueError:
                 print(f"Warning: Could not parse extracted time signature '{ts_match.group(1)}/{ts_match.group(2)}'.")
        if inst_match:
            instruments_str = inst_match.group(1).strip()
            potential_instruments = [inst.strip().lower() for inst in instruments_str.split(',') if inst.strip()]
            # Validate against known instruments
            valid_instruments = [inst for inst in potential_instruments if inst in INSTRUMENT_PROGRAM_MAP]
            if valid_instruments:
                primary_instruments = valid_instruments
                print(f"Identified Primary Instrument Hints: {', '.join(primary_instruments)}")
            else:
                 print(f"Warning: No valid instruments found in extracted list: '{instruments_str}'.")
        if struct_match:
            structure_hint = struct_match.group(1).upper()
            print(f"Identified Structure Hint: {structure_hint}")

        # Update global defaults based on enrichment
        CONFIG["default_key"] = current_key
        CONFIG["default_tempo"] = current_tempo
        CONFIG["default_timesig"] = current_timesig
        # Note: primary_instrument hints aren't directly used as a single default,
        # but are passed to the section planner.

        return enriched_summary, structure_hint
    else:
        print("Failed to enrich description. Using initial description and defaults.")
        return description, structure_hint


def generate_section_plan(enriched_desc, structure_hint):
    """Step 2: Use LLM to generate a detailed section plan."""
    print("\n--- Step 2: Generating Section Plan ---")
    prompt = f"""
Based on the following enriched music description and suggested structure, create a detailed plan for the sections.
The plan MUST be a valid JSON object where keys are section names (e.g., "Intro", "A1", "B", "A2", "Bridge", "Outro") and values are objects containing:
1. "bars": An integer number of bars for the section (strictly between {CONFIG['min_section_bars']} and {CONFIG['max_section_bars']}).
2. "goal": A concise string describing the musical purpose or content of the section (e.g., "Introduce main theme softly on SynPad", "Develop theme with variation, add SynthBass and Drums", "Contrasting bridge section, new chords on Electric Piano", "Return to main theme, fuller texture with Arp", "Fade out main elements").

The total number of bars across all sections should ideally be around {CONFIG['max_total_bars'] // 2} to {CONFIG['max_total_bars']}, but strictly MUST NOT exceed {CONFIG['max_total_bars']}.
Use the suggested structure "{structure_hint}" as a guide, adapting it as needed (e.g., AABA -> Intro A1 B A2 Outro). Ensure section names are unique and descriptive (e.g., A1, A2 instead of just A, A). Use standard section names where applicable (Intro, Verse, Chorus, Bridge, Solo, Outro, Build, Drop, Breakdown etc.).

Enriched Description:
{enriched_desc}

Generate ONLY the JSON plan now, starting with {{ and ending with }}. Do not include ```json markers or any other text.
"""

    plan_json = call_gemini(prompt, output_format="json")

    if plan_json and isinstance(plan_json, dict):
        validated_plan = {}
        total_bars = 0
        # Try to preserve the order from the LLM's JSON output if possible
        section_order = list(plan_json.keys())

        for name in section_order:
            if not isinstance(name, str) or not name.strip():
                print(f"Warning: Invalid section name type '{type(name)}' or empty name. Skipping.")
                continue

            section_name = name.strip()
            info = plan_json.get(name) # Use .get for safety

            if not isinstance(info, dict):
                 print(f"Warning: Section '{section_name}' value is not a dictionary: {info}. Skipping.")
                 continue

            bars_val = info.get("bars")
            goal_val = info.get("goal")

            # Validate 'bars'
            if not isinstance(bars_val, int):
                try: # Attempt conversion if it's a string number
                    bars_val = int(bars_val)
                except (ValueError, TypeError):
                    print(f"Warning: Invalid 'bars' type for section '{section_name}': {bars_val}. Skipping.")
                    continue

            # Validate 'goal'
            if not isinstance(goal_val, str) or not goal_val.strip():
                 print(f"Warning: Invalid or empty 'goal' for section '{section_name}': {goal_val}. Skipping.")
                 continue

            # Validate bar range and total bar limit
            if CONFIG["min_section_bars"] <= bars_val <= CONFIG["max_section_bars"]:
                if total_bars + bars_val <= CONFIG["max_total_bars"]:
                    info["bars"] = bars_val # Ensure it's the integer value
                    info["goal"] = goal_val.strip()[:250] # Limit goal length
                    validated_plan[section_name] = info
                    total_bars += bars_val
                else:
                    print(f"Warning: Section '{section_name}' ({bars_val} bars) exceeds max total bars ({CONFIG['max_total_bars']}). Truncating plan.")
                    break # Stop adding sections
            else:
                print(f"Warning: Section '{section_name}' bars ({bars_val}) out of range ({CONFIG['min_section_bars']}-{CONFIG['max_section_bars']}). Skipping.")

        if not validated_plan:
            print("ERROR: Failed to generate a valid section plan after validation. Cannot proceed.")
            return None

        print("Generated Section Plan:")
        final_section_order = list(validated_plan.keys()) # Get order after validation
        for name in final_section_order:
            info = validated_plan[name]
            print(f"  - {name} ({info['bars']} bars): {info['goal']}")
        print(f"Total Bars in Plan: {total_bars}")
        # Return the validated plan, preserving the final validated order
        return {name: validated_plan[name] for name in final_section_order}
    else:
        print("ERROR: Failed to generate or parse section plan JSON from LLM. Cannot proceed.")
        if isinstance(plan_json, str): # Log if LLM returned non-JSON string
            print(f"LLM Output (expected JSON):\n{plan_json[:500]}...")
        return None


def generate_symbolic_section(
    overall_desc,
    section_plan,
    section_name,
    current_bar,
    previous_section_summary=None,
):
    """Step 3: Generate symbolic music for one section using LLM."""
    print(
        f"\n--- Step 3: Generating Symbolic Section: {section_name} (Starting Bar: {current_bar}) ---"
    )
    section_info = section_plan[section_name]
    bars = section_info["bars"]
    goal = section_info["goal"]

    context_prompt = ""
    if previous_section_summary:
        prev_name = previous_section_summary.get("name", "Previous")
        prev_summary = previous_section_summary.get("summary", "No summary available")
        prev_key = previous_section_summary.get("key", CONFIG["default_key"])
        prev_tempo = previous_section_summary.get("tempo", CONFIG["default_tempo"])
        prev_ts = previous_section_summary.get(
            "time_sig", f"{CONFIG['default_timesig'][0]}/{CONFIG['default_timesig'][1]}"
        )
        context_prompt = (
            f"Context from previous section ({prev_name}): {prev_summary}\n"
            f"It ended around key {prev_key}, tempo {prev_tempo} BPM, and time signature {prev_ts}.\n"
            "Ensure a smooth musical transition if appropriate for the overall structure and goals.\n"
        )

    # Provide current defaults from CONFIG, as they might have been updated by enrichment
    default_tempo = CONFIG["default_tempo"]
    default_timesig = CONFIG["default_timesig"]
    default_key = CONFIG["default_key"]

    # Prepare lists of known instruments and drum sounds for the prompt
    known_melodic_instruments = sorted(list(INSTRUMENT_PROGRAM_MAP.keys()))
    known_drum_sounds = sorted(list(DRUM_PITCH_MAP.keys()))

    prompt = f"""
You are a precise symbolic music generator. Your task is to generate ONLY the symbolic music notation for a specific section of a piece, following the provided format strictly.

Overall Music Goal: {overall_desc}
Full Section Plan: {json.dumps(section_plan, indent=2)}
{context_prompt}
Current Section to Generate: {section_name}
Target Bars: {bars} (Start this section exactly at BAR:{current_bar}, end *before* BAR:{current_bar + bars})
Section Goal: {goal}

Instructions:
1. Generate music ONLY for this section, starting *exactly* with `BAR:{current_bar}` unless initial T, TS, K, or INST commands are needed for this specific section start.
2. If tempo (T), time signature (TS), key (K), or melodic instrument (INST) need to be set or changed *at the very beginning* of this section (time = start of BAR:{current_bar}), include those commands *before* the `BAR:{current_bar}` marker. Otherwise, assume they carry over from the previous section context or use defaults (T:{default_tempo}, TS:{default_timesig[0]}/{default_timesig[1]}, K:{default_key}). You can change INST multiple times within the section if needed for different melodic tracks. Remember INST only affects subsequent melodic tracks, not drum tracks.
3. Strictly adhere to the compact symbolic format defined below. Output ONLY the commands, each on a new line.
4. DO NOT include any other text, explanations, apologies, section titles, comments (#), or formatting like ```mus``` or ```.
5. Ensure musical coherence within the section and try to achieve the Section Goal. Use appropriate instrumentation and musical ideas based on the goal and overall description.
6. The total duration of notes/rests/chords within each bar MUST add up precisely according to the active time signature (e.g., 4 quarter notes in 4/4, 6 eighth notes in 6/8). Be precise. Use rests (R:<Track>:<Duration>) to fill empty time accurately for each active track within a bar. Ensure parallel tracks are synchronized at bar lines.
7. End the generation cleanly *after* the content for bar {current_bar + bars - 1} is complete. Do NOT include `BAR:{current_bar + bars}`.
8. Instrument Names (for INST command): Use ONLY lowercase names from this list: {", ".join(known_melodic_instruments)}.
9. Drum Pitch Names (for N/C commands on drum tracks): Use ONLY names from this list (case-insensitive): {", ".join(known_drum_sounds)}.

{SYMBOLIC_FORMAT_DEFINITION}

Generate Section {section_name} symbolic music now (starting BAR:{current_bar}):
"""

    symbolic_text = call_gemini(prompt)

    if symbolic_text:
        # Basic cleaning is done in call_gemini, remove extra whitespace
        symbolic_text = symbolic_text.strip()
        lines = symbolic_text.split("\n")
        meaningful_lines = [
            line.strip()
            for line in lines
            if line.strip() and not line.strip().startswith("#")
        ]

        if not meaningful_lines:
            print(
                f"Warning: Generated text for {section_name} appears empty or only contains comments."
            )
            return "", None # Return empty string but indicate failure with None summary

        # --- Validation ---
        bar_marker = f"BAR:{current_bar}"
        first_bar_marker_found = False
        first_event_line_index = -1
        initial_commands = [] # T, TS, K, INST before first BAR

        # Find the first meaningful line and check if it's a valid start
        for idx, line in enumerate(meaningful_lines):
            if re.match(r"^(T:|TS:|K:|INST:)", line, re.IGNORECASE):
                 initial_commands.append(line)
                 if first_event_line_index == -1: first_event_line_index = idx
            elif line.startswith(bar_marker):
                 first_bar_marker_found = True
                 if first_event_line_index == -1: first_event_line_index = idx
                 break # Found the target BAR marker
            elif first_event_line_index == -1: # Found something else first (N, C, R, other BAR)
                 print(f"Warning: Section {section_name} generation started with unexpected command '{line}' before initial settings or '{bar_marker}'. Attempting to use.")
                 first_event_line_index = idx
                 if line.startswith("BAR:"): # Check if it's the wrong bar number
                     print(f"ERROR: Section {section_name} started with wrong bar number '{line}'. Expected '{bar_marker}'. Discarding section.")
                     return "", None
                 break # Stop searching for initial commands

        if first_event_line_index == -1:
             print(f"ERROR: No meaningful commands found in generated text for {section_name}. Discarding section.")
             return "", None

        # If the *required* start bar marker wasn't found at all
        if not first_bar_marker_found:
             # Check if the first event line was the bar marker but loop didn't catch it (edge case)
             if meaningful_lines[first_event_line_index].startswith(bar_marker):
                 first_bar_marker_found = True
             else:
                 print(f"ERROR: Generated text for {section_name} does not contain the required start marker '{bar_marker}'. Discarding section.")
                 print(f"Received text (first 500 chars):\n{symbolic_text[:500]}...")
                 return "", None

        # Reconstruct the section text from the first valid command/marker onwards
        validated_symbolic_lines = meaningful_lines[first_event_line_index:]
        validated_symbolic_text = "\n".join(validated_symbolic_lines)

        print(
            f"Validated symbolic text for Section {section_name} (first 300 chars):\n{validated_symbolic_text[:300]}...\n"
        )

        # --- Extract summary info from the generated (validated) text ---
        summary_info = {
            "name": section_name,
            "summary": f"Generated {bars} bars targeting goal: {goal}.",
            "key": default_key, # Start with defaults/previous context
            "tempo": default_tempo,
            "time_sig": f"{default_timesig[0]}/{default_timesig[1]}",
        }
        # Update summary with the *last* settings found within this section
        last_k, last_t, last_ts = (
            summary_info["key"],
            summary_info["tempo"],
            summary_info["time_sig"],
        )
        for line in reversed(validated_symbolic_lines):
            line = line.strip()
            if line.startswith("K:"):
                last_k = line.split(":", 1)[1].strip()
                break
        for line in reversed(validated_symbolic_lines):
            line = line.strip()
            if line.startswith("T:"):
                try:
                    last_t = float(line.split(":", 1)[1].strip())
                    break
                except ValueError:
                    pass
        for line in reversed(validated_symbolic_lines):
            line = line.strip()
            if line.startswith("TS:"):
                last_ts = line.split(":", 1)[1].strip()
                break
        summary_info["key"], summary_info["tempo"], summary_info["time_sig"] = (
            last_k,
            last_t,
            last_ts,
        )

        return validated_symbolic_text, summary_info
    else:
        print(f"Failed to generate symbolic text for Section {section_name}.")
        return "", None


def parse_symbolic_to_structured_data(symbolic_text):
    """Step 4: Parse concatenated symbolic text into structured data for MIDI."""
    print("\n--- Step 4: Parsing Symbolic Text ---")
    notes_by_instrument_track = {}
    tempo_changes = []
    time_signature_changes = []
    key_signature_changes = []
    # Key: unique inst_track_key tuple (inst_name_lower, track_id)
    # Value: {program, is_drum, name (for MIDI), orig_inst_name}
    instrument_definitions = {}

    # State variables for parsing context
    # Key: inst_track_key tuple, Value: current time cursor (absolute seconds)
    current_track_times = {}
    # Tracks the latest event time across all tracks, adjusted by BAR markers
    current_global_time = 0.0
    current_tempo = float(CONFIG["default_tempo"])
    current_ts_num, current_ts_den = CONFIG["default_timesig"]
    current_key = CONFIG["default_key"]
    # State for the active melodic instrument (set by INST:)
    active_melodic_program = CONFIG["default_program"]
    active_melodic_instrument_orig_name = CONFIG["default_instrument_name"] # Original name used in INST:
    active_melodic_instrument_lookup_name = active_melodic_instrument_orig_name.lower() # Lowercase for map lookup

    current_bar_number = 0
    current_bar_start_time = 0.0
    # Key: inst_track_key tuple, Value: time elapsed (seconds) within the current bar for this track
    time_within_bar_per_track = {}
    expected_bar_duration_sec = duration_to_seconds("W", current_tempo) * (
        current_ts_num / 4.0
    ) * (4.0 / current_ts_den) # More general calc

    last_event_end_time = 0.0 # Track the absolute end time of the last event processed

    initial_commands_processed = False # Flag to track if initial settings are done
    lines = symbolic_text.strip().split("\n")
    parse_start_line_index = 0 # Index in `lines` where main parsing should begin

    # --- Pre-pass for initial settings (before first BAR marker) ---
    print("Processing initial settings (before first BAR marker)...")
    initial_settings = {"T": None, "TS": None, "K": None, "INST": None}
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("BAR:"):
            parse_start_line_index = i
            break # Stop pre-pass when first BAR is encountered

        parts = line.split(":", 1)
        command = parts[0].upper()
        value = parts[1].strip() if len(parts) > 1 else ""
        ini_line_num = i + 1

        try:
            if command == "INST" and initial_settings["INST"] is None:
                if value:
                    inst_name_lower = value.lower()
                    program = INSTRUMENT_PROGRAM_MAP.get(
                        inst_name_lower, CONFIG["default_program"]
                    )
                    # Update active melodic context if valid instrument found
                    active_melodic_program = program
                    active_melodic_instrument_orig_name = value # Store original name
                    active_melodic_instrument_lookup_name = inst_name_lower
                    initial_settings["INST"] = (value, program)
                    print(
                        f"Initial Melodic Instrument context set to '{value}' (Program: {program})"
                    )
                else: print(f"Warning line {ini_line_num}: INST command has empty value.")
            elif command == "T" and initial_settings["T"] is None:
                new_tempo = float(value)
                if new_tempo > 0:
                    current_tempo = new_tempo
                    initial_settings["T"] = current_tempo
                    print(f"Initial Tempo set to {current_tempo} BPM")
                else: print(f"Warning line {ini_line_num}: Invalid tempo value {value}.")
            elif command == "TS" and initial_settings["TS"] is None:
                num_str, den_str = value.split("/")
                new_ts_num, new_ts_den = int(num_str), int(den_str)
                if new_ts_num > 0 and new_ts_den > 0:
                    current_ts_num, current_ts_den = new_ts_num, new_ts_den
                    initial_settings["TS"] = (current_ts_num, current_ts_den)
                    print(f"Initial Time Signature set to {current_ts_num}/{current_ts_den}")
                else: print(f"Warning line {ini_line_num}: Invalid time signature value {value}.")
            elif command == "K" and initial_settings["K"] is None:
                if value:
                    # Basic validation
                    if re.match(r"^[A-Ga-g][#sb]?(maj|min|dor|phr|lyd|mix|loc|aeo|ion)?$", value, re.IGNORECASE):
                        current_key = value
                        initial_settings["K"] = current_key
                        print(f"Initial Key set to {current_key}")
                    else: print(f"Warning line {ini_line_num}: Invalid key signature format '{value}'.")
                else: print(f"Warning line {ini_line_num}: K command has empty value.")
        except Exception as e:
            print(f"Error parsing initial setting line {ini_line_num}: '{line}' - {e}")
        # Ensure parse_start_line_index is updated even if loop finishes without finding BAR
        parse_start_line_index = i + 1

    # Apply initial settings as events at time 0.0
    tempo_changes.append((0.0, current_tempo))
    time_signature_changes.append((0.0, current_ts_num, current_ts_den))
    key_signature_changes.append((0.0, current_key))
    if initial_settings["INST"] is None:
        print(
            f"Using default initial Melodic Instrument: '{active_melodic_instrument_orig_name}' (Program: {active_melodic_program})"
        )
    # Recalculate initial bar duration based on potentially updated T/TS
    expected_bar_duration_sec = duration_to_seconds("W", current_tempo) * (
        current_ts_num / 4.0
    ) * (4.0 / current_ts_den)
    initial_commands_processed = True

    # --- Main Parsing Loop ---
    print(f"Parsing main body starting from line {parse_start_line_index + 1}...")
    for i in range(parse_start_line_index, len(lines)):
        current_line_num = i + 1
        line = lines[i].strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split(":", 1)
        command = parts[0].upper()
        value = parts[1].strip() if len(parts) > 1 else ""

        try:
            # --- Handle Global Parameter Changes ---
            if command == "INST":
                if value:
                    inst_name_lower = value.lower()
                    program = INSTRUMENT_PROGRAM_MAP.get(inst_name_lower)
                    if program is not None:
                        # Update context only if changed
                        if program != active_melodic_program or value != active_melodic_instrument_orig_name:
                            active_melodic_program = program
                            active_melodic_instrument_orig_name = value
                            active_melodic_instrument_lookup_name = inst_name_lower
                            # Log change if needed (can be verbose)
                            # print(f"Time {current_global_time:.3f}s: Melodic INST changed to '{value}' (Prog: {program})")
                    else:
                        print(f"Warning line {current_line_num}: Unknown instrument name '{value}' in INST command. Ignoring.")
                else: print(f"Warning line {current_line_num}: INST command has empty value. Ignoring.")
            elif command == "T":
                new_tempo = float(value)
                if new_tempo > 0 and abs(new_tempo - current_tempo) > 1e-3:
                    event_time = current_global_time # Tempo change occurs at current time
                    # Add change only if different from last recorded change at this time
                    if not tempo_changes or abs(tempo_changes[-1][0] - event_time) > 1e-6 or abs(tempo_changes[-1][1] - new_tempo) > 1e-3:
                        tempo_changes.append((event_time, new_tempo))
                        print(
                            f"Time {event_time:.3f}s (Bar ~{current_bar_number}): Tempo change to {new_tempo} BPM"
                        )
                    current_tempo = new_tempo
                    # Recalculate expected bar duration
                    expected_bar_duration_sec = duration_to_seconds("W", current_tempo) * (
                        current_ts_num / 4.0
                    ) * (4.0 / current_ts_den)
            elif command == "TS":
                num_str, den_str = value.split("/")
                new_ts_num, new_ts_den = int(num_str), int(den_str)
                if new_ts_num > 0 and new_ts_den > 0 and (new_ts_num, new_ts_den) != (current_ts_num, current_ts_den):
                    event_time = current_global_time
                    if not time_signature_changes or abs(time_signature_changes[-1][0] - event_time) > 1e-6 \
                       or (time_signature_changes[-1][1], time_signature_changes[-1][2]) != (new_ts_num, new_ts_den):
                        time_signature_changes.append((event_time, new_ts_num, new_ts_den))
                        print(
                            f"Time {event_time:.3f}s (Bar ~{current_bar_number}): Time Sig change to {new_ts_num}/{new_ts_den}"
                        )
                    current_ts_num, current_ts_den = new_ts_num, new_ts_den
                    # Recalculate expected bar duration
                    expected_bar_duration_sec = duration_to_seconds("W", current_tempo) * (
                        current_ts_num / 4.0
                    ) * (4.0 / current_ts_den)
            elif command == "K":
                if value and value != current_key:
                    if re.match(r"^[A-Ga-g][#sb]?(maj|min|dor|phr|lyd|mix|loc|aeo|ion)?$", value, re.IGNORECASE):
                        event_time = current_global_time
                        if not key_signature_changes or abs(key_signature_changes[-1][0] - event_time) > 1e-6 or key_signature_changes[-1][1] != value:
                            key_signature_changes.append((event_time, value))
                            print(
                                f"Time {event_time:.3f}s (Bar ~{current_bar_number}): Key change to {value}"
                            )
                        current_key = value
                    else: print(f"Warning line {current_line_num}: Invalid key signature format '{value}'. Ignoring K command.")
                elif not value: print(f"Warning line {current_line_num}: K command has empty value. Ignoring.")

            # --- Handle Bar Marker ---
            elif command == "BAR":
                bar_number = int(value)
                # Calculate the expected start time of this new bar
                expected_new_bar_start_time = current_bar_start_time + expected_bar_duration_sec if current_bar_number > 0 else 0.0

                # --- Bar Synchronization and Timing Check ---
                if current_bar_number > 0: # Only check timing after the first bar marker
                    max_accumulated_time_in_prev_bar = 0.0
                    if time_within_bar_per_track: # Check if any events happened in the previous bar
                        max_accumulated_time_in_prev_bar = max(time_within_bar_per_track.values())

                    # Define a tolerance for timing errors (e.g., 1% of bar duration or 5ms)
                    tolerance = max(0.005, expected_bar_duration_sec * 0.01)
                    duration_error = max_accumulated_time_in_prev_bar - expected_bar_duration_sec

                    if duration_error > tolerance: # Previous bar overran significantly
                        print(
                            f"Warning: Bar {current_bar_number} timing mismatch (Overran). Expected {expected_bar_duration_sec:.3f}s, got {max_accumulated_time_in_prev_bar:.3f}s. Forcing bar {bar_number} start to expected time {expected_new_bar_start_time:.3f}s."
                        )
                        current_global_time = expected_new_bar_start_time
                    elif duration_error < -tolerance and max_accumulated_time_in_prev_bar > 0: # Previous bar underran significantly
                        print(
                            f"Warning: Bar {current_bar_number} timing potentially short. Expected {expected_bar_duration_sec:.3f}s, got {max_accumulated_time_in_prev_bar:.3f}s. Setting bar {bar_number} start to expected time {expected_new_bar_start_time:.3f}s."
                        )
                        current_global_time = expected_new_bar_start_time
                    else: # Within tolerance or empty bar
                        # If close enough, use the expected time to avoid drift
                        current_global_time = expected_new_bar_start_time

                # Handle jumps in bar numbers (e.g., BAR:1 then BAR:5)
                bars_jumped = bar_number - (current_bar_number + 1)
                if bars_jumped > 0 and current_bar_number > 0:
                    # Estimate duration of skipped bars using current tempo/ts
                    jump_duration = bars_jumped * expected_bar_duration_sec
                    print(
                        f"Warning: Jump detected from Bar {current_bar_number} to {bar_number}. Advancing global time by ~{jump_duration:.3f}s."
                    )
                    current_global_time += jump_duration
                elif bar_number <= current_bar_number and current_bar_number > 0:
                    print(f"Warning line {current_line_num}: Bar number {bar_number} is not sequential (previous was {current_bar_number}). Timing might be incorrect.")
                    # Don't adjust time backwards, just proceed from current time.

                # Update bar state
                current_bar_number = bar_number
                current_bar_start_time = current_global_time
                # Reset time within bar for all tracks
                time_within_bar_per_track = {key: 0.0 for key in time_within_bar_per_track}
                # Sync all track time cursors to the start of the new bar
                current_track_times = {key: current_bar_start_time for key in current_track_times}

            # --- Handle Note, Chord, Rest Events ---
            elif command in ["N", "C", "R"]:
                if current_bar_number == 0: # Event before first BAR marker
                    # This case should ideally be handled by the initial command processing
                    # If an event somehow occurs here, process it at time 0 and force bar 1 start
                    print(f"Warning: Event '{line}' on Line {current_line_num} occurred before the first BAR marker. Processing at time 0.")
                    current_bar_number = 1 # Assume bar 1 starts now
                    current_bar_start_time = 0.0
                    current_global_time = 0.0
                    # Ensure track times are initialized if this happens
                    # (Logic below will handle instrument/track key creation)

                # --- Identify Instrument/Track Properties ---
                # N:<Track>:<Pitch>:<Duration>:<Velocity>
                # C:<Track>:<[Pitches]>:<Duration>:<Velocity>
                # R:<Track>:<Duration>
                data_parts = value.split(":")
                min_parts = 2 if command == "R" else 4
                if len(data_parts) < min_parts:
                    print(
                        f"Warning: Malformed {command} command on Line {current_line_num}: '{line}'. Requires at least {min_parts} parts after ':'. Skipping."
                    )
                    continue

                track_id = data_parts[0].strip()
                if not track_id:
                    print(f"Warning: Empty TrackID in {command} command on Line {current_line_num}. Skipping.")
                    continue

                # Determine if drum track based on TrackID
                event_is_drum = track_id.lower() in DRUM_TRACK_IDS
                event_program = 0 # Default for drums
                event_instrument_base_name = track_id # Default name part for drums
                inst_track_key = None # Unique key for instrument_definitions and notes_by_instrument_track

                if event_is_drum:
                    # Use a consistent key for all drum events on this track_id
                    inst_track_key = ("drums", track_id.lower()) # Group by 'drums' type, distinguish by track_id
                    midi_instrument_name = f"{track_id} (Drums)"
                else: # Melodic track
                    event_program = active_melodic_program
                    event_instrument_base_name = active_melodic_instrument_orig_name
                    # Key combines active melodic instrument lookup name and track ID
                    inst_track_key = (active_melodic_instrument_lookup_name, track_id)
                    midi_instrument_name = f"{track_id} ({active_melodic_instrument_orig_name})"

                # --- Ensure Instrument/Track is Defined and Initialized ---
                if inst_track_key not in instrument_definitions:
                    instrument_definitions[inst_track_key] = {
                        "program": event_program,
                        "is_drum": event_is_drum,
                        "name": midi_instrument_name,
                        "orig_inst_name": event_instrument_base_name, # Store original name used
                    }
                    print(
                        f"Defined instrument/track: {midi_instrument_name} (Key: {inst_track_key}, Program: {event_program}, IsDrum: {event_is_drum})"
                    )
                    # Initialize time tracking for this new instrument/track key
                    # Assume it starts at the beginning of the current bar unless already tracked
                    initial_track_offset = time_within_bar_per_track.get(inst_track_key, 0.0)
                    current_track_times[inst_track_key] = current_bar_start_time + initial_track_offset
                    time_within_bar_per_track[inst_track_key] = initial_track_offset
                    notes_by_instrument_track[inst_track_key] = []

                # --- Calculate Event Timing ---
                # Event starts at the track's current position within the bar
                track_specific_start_offset = time_within_bar_per_track.get(inst_track_key, 0.0)
                event_start_time = current_bar_start_time + track_specific_start_offset

                # --- Parse Event Specifics (Pitch, Duration, Velocity) ---
                event_duration_sec = 0.0 # Initialize duration

                # --- Parse Note (N) ---
                if command == "N":
                    # N:<Track>:<Pitch>:<Duration>:<Velocity>
                    pitch_name_raw = data_parts[1].strip()
                    duration_sym_raw = data_parts[2].strip()
                    velocity_str_raw = data_parts[3].strip()
                    if not pitch_name_raw or not duration_sym_raw or not velocity_str_raw:
                        print(f"Warning: Empty part in N command on Line {current_line_num}. Skipping.")
                        continue

                    # Clean trailing comments from velocity if any
                    velocity_str = velocity_str_raw.split("#", 1)[0].strip()
                    duration_sym = duration_sym_raw.split("#", 1)[0].strip()
                    if not duration_sym: print(f"Warning: Empty duration for N command '{line}'. Skipping."); continue

                    try:
                        velocity = int(velocity_str)
                        velocity = max(0, min(127, velocity))
                    except ValueError:
                        velocity = 90 # Default velocity
                        print(f"Warning line {current_line_num}: Invalid velocity '{velocity_str_raw}'. Using {velocity}.")

                    event_duration_sec = duration_to_seconds(duration_sym, current_tempo, current_ts_den)
                    midi_pitch = None

                    if event_is_drum:
                        pitch_name_lookup = pitch_name_raw.lower()
                        midi_pitch = DRUM_PITCH_MAP.get(pitch_name_lookup)
                        if midi_pitch is None:
                            # Option: Allow standard pitch notation on drum track (might be intended for melodic percussion)
                            # standard_pitch = pitch_to_midi(pitch_name_raw)
                            # if standard_pitch is not None:
                            #     print(f"Warning line {current_line_num}: Standard pitch '{pitch_name_raw}' used on drum track '{track_id}'. Using MIDI pitch {standard_pitch}.")
                            #     midi_pitch = standard_pitch
                            # else:
                            #     print(f"Warning line {current_line_num}: Unknown drum sound '{pitch_name_raw}' for track '{track_id}'. Skipping note.")
                            #     continue
                            print(f"Warning line {current_line_num}: Unknown drum sound '{pitch_name_raw}' for track '{track_id}'. Skipping note.")
                            continue # Skip if drum sound not found
                    else: # Melodic
                        midi_pitch = pitch_to_midi(pitch_name_raw)
                        if midi_pitch is None:
                            print(f"Warning line {current_line_num}: Cannot parse pitch '{pitch_name_raw}' for track '{track_id}'. Skipping note.")
                            continue # Skip if pitch invalid

                    note_event = {
                        "pitch": midi_pitch,
                        "start": event_start_time,
                        "end": event_start_time + event_duration_sec,
                        "velocity": velocity,
                    }
                    notes_by_instrument_track[inst_track_key].append(note_event)

                # --- Parse Chord (C) ---
                elif command == "C":
                    # C:<Track>:<[Pitches]>:<Duration>:<Velocity>
                    pitches_str_raw = data_parts[1].strip()
                    duration_sym_raw = data_parts[2].strip()
                    velocity_str_raw = data_parts[3].strip()
                    if not pitches_str_raw or not duration_sym_raw or not velocity_str_raw:
                        print(f"Warning: Empty part in C command on Line {current_line_num}. Skipping.")
                        continue

                    velocity_str = velocity_str_raw.split("#", 1)[0].strip()
                    duration_sym = duration_sym_raw.split("#", 1)[0].strip()
                    if not duration_sym: print(f"Warning: Empty duration for C command '{line}'. Skipping."); continue

                    try:
                        velocity = int(velocity_str)
                        velocity = max(0, min(127, velocity))
                    except ValueError:
                        velocity = 70 # Default chord velocity
                        print(f"Warning line {current_line_num}: Invalid velocity '{velocity_str_raw}'. Using {velocity}.")

                    # Validate and parse pitches within brackets
                    if not (pitches_str_raw.startswith("[") and pitches_str_raw.endswith("]")):
                        print(f"Warning line {current_line_num}: Chord pitches format incorrect: '{pitches_str_raw}'. Expected '[P1,P2,...]'. Skipping chord.")
                        continue
                    pitches_str = pitches_str_raw[1:-1] # Remove brackets
                    pitch_names = [p.strip() for p in pitches_str.split(",") if p.strip()]

                    if not pitch_names:
                        print(f"Warning line {current_line_num}: No pitches found in Chord command '{line}'. Skipping.")
                        continue

                    event_duration_sec = duration_to_seconds(duration_sym, current_tempo, current_ts_den)
                    chord_notes = []
                    valid_pitches_in_chord = 0
                    for pitch_name_raw in pitch_names:
                        midi_pitch = None
                        if event_is_drum:
                            pitch_name_lookup = pitch_name_raw.lower()
                            midi_pitch = DRUM_PITCH_MAP.get(pitch_name_lookup)
                            if midi_pitch is None:
                                print(f"Warning line {current_line_num}: Unknown drum sound '{pitch_name_raw}' in chord for track '{track_id}'. Skipping this pitch.")
                                continue # Skip this specific drum sound
                        else: # Melodic
                            midi_pitch = pitch_to_midi(pitch_name_raw)
                            if midi_pitch is None:
                                print(f"Warning line {current_line_num}: Cannot parse pitch '{pitch_name_raw}' in chord for track '{track_id}'. Skipping this pitch.")
                                continue # Skip this specific pitch

                        note_event = {
                            "pitch": midi_pitch,
                            "start": event_start_time,
                            "end": event_start_time + event_duration_sec,
                            "velocity": velocity,
                        }
                        chord_notes.append(note_event)
                        valid_pitches_in_chord += 1

                    if valid_pitches_in_chord > 0:
                        notes_by_instrument_track[inst_track_key].extend(chord_notes)
                    else:
                        print(f"Warning line {current_line_num}: Chord command had no valid notes after parsing. Skipping chord.")
                        continue # Don't advance time if chord was completely invalid

                # --- Parse Rest (R) ---
                elif command == "R":
                    # R:<Track>:<Duration>
                    duration_sym_raw = data_parts[1].strip()
                    if not duration_sym_raw:
                        print(f"Warning line {current_line_num}: Empty duration in R command '{line}'. Skipping.")
                        continue
                    duration_sym = duration_sym_raw.split("#", 1)[0].strip()
                    if not duration_sym:
                        print(f"Warning line {current_line_num}: Empty duration for R command '{line}'. Skipping.")
                        continue
                    event_duration_sec = duration_to_seconds(duration_sym, current_tempo, current_ts_den)
                    # Rests just advance time for the track, no notes added.

                # --- Post-Event Time Advancement ---
                if event_duration_sec > 0: # Only advance time if duration is valid
                    new_track_time_absolute = event_start_time + event_duration_sec
                    # Update the specific track's time cursor
                    current_track_times[inst_track_key] = new_track_time_absolute
                    # Update the time elapsed within the current bar for this track
                    time_within_bar_per_track[inst_track_key] = new_track_time_absolute - current_bar_start_time
                    # Update the last event end time seen overall
                    last_event_end_time = max(last_event_end_time, new_track_time_absolute)
                else:
                    print(f"Warning line {current_line_num}: Event '{line}' resulted in zero duration. Not advancing time.")

            else: # Unknown command
                print(f"Warning: Unknown command '{command}' on line {current_line_num}: '{line}'. Skipping.")

        except Exception as e:
            print(f"FATAL Error parsing line {current_line_num}: '{line}' - {e}")
            traceback.print_exc()
            # Decide whether to continue parsing or stop on fatal error
            # For now, let's try to continue

    # --- Final Cleanup and Summary ---
    print(f"Symbolic text parsing complete. Estimated total duration: {last_event_end_time:.3f} seconds.")
    # Sanity check: Ensure all defined instruments actually have notes
    final_instrument_defs = {}
    final_notes_data = {}
    for key, definition in instrument_definitions.items():
        if key in notes_by_instrument_track and notes_by_instrument_track[key]:
            final_instrument_defs[key] = definition
            final_notes_data[key] = notes_by_instrument_track[key]
        else:
            print(
                f"Info: Instrument/Track '{definition['name']}' (Key: {key}) defined but had no notes parsed. Excluding from MIDI."
            )

    # Clear caches after parsing is done
    _pitch_parse_cache.clear()
    _duration_cache.clear()

    return (
        final_notes_data,
        final_instrument_defs,
        tempo_changes,
        time_signature_changes,
        key_signature_changes,
        last_event_end_time,
        current_key, # Return the final key state
        current_tempo # Return the final tempo state
    )


def create_midi_file(
    notes_data,
    instrument_defs,
    tempo_changes,
    time_sig_changes,
    key_sig_changes,
    filename,
):
    """Step 5: Create MIDI file using pretty_midi."""
    print(f"\n--- Step 5: Creating MIDI File ({filename}) ---")
    if not notes_data or not instrument_defs:
        print(
            "Error: No instrument or note data was successfully parsed. Cannot create MIDI file."
        )
        return

    try:
        # Initial tempo is the first tempo change event (should always exist at time 0)
        initial_tempo = tempo_changes[0][1] if tempo_changes else CONFIG["default_tempo"]
        midi_obj = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo)
        print(f"Initialized MIDI with tempo: {initial_tempo:.2f} BPM")

        # --- Apply Meta-Messages ---
        # Tempo Changes: pretty_midi handles tempo implicitly via note timings calculated
        # during parsing based on the active tempo at that time. We set the initial tempo,
        # and the absolute start/end times of notes reflect subsequent tempo changes.
        # Explicit TempoChange events are not typically needed unless for fine control.
        if len(tempo_changes) > 1:
             print(f"Tempo changes detected ({len(tempo_changes)} total). Note timings reflect these changes.")

        # Time Signature Changes
        # Sort by time, then use a dictionary to keep only the last change at each time
        time_sig_changes.sort(key=lambda x: x[0])
        unique_ts = {}
        for time, num, den in time_sig_changes:
            # Validate denominator (must be power of 2 for standard MIDI)
            actual_den = den
            if den <= 0 or (den & (den - 1) != 0): # Check if not power of 2
                 # Find nearest power of 2
                 if den > 0: actual_den = 2**math.ceil(math.log2(den))
                 else: actual_den = 4 # Default if invalid
                 print(f"Warning: Invalid TS denominator {den} at {time:.3f}s. Using nearest power of 2: {actual_den}.")
            # Store the last valid TS for this specific time
            unique_ts[round(time, 6)] = (num, actual_den)

        midi_obj.time_signature_changes = []
        applied_ts_count = 0
        last_ts_tuple = None
        for time in sorted(unique_ts.keys()):
            num, den = unique_ts[time]
            current_ts_tuple = (num, den)
            # Add only if it's different from the last one added
            if current_ts_tuple != last_ts_tuple:
                try:
                    ts_event = pretty_midi.TimeSignature(num, den, time)
                    midi_obj.time_signature_changes.append(ts_event)
                    applied_ts_count += 1
                    last_ts_tuple = current_ts_tuple
                except ValueError as e:
                     print(f"Error creating TimeSignature({num}, {den}, {time:.3f}): {e}. Skipping.")

        if applied_ts_count > 0:
            print(f"Applied {applied_ts_count} unique time signature changes.")
        # Ensure at least one default TS if none were added
        if not midi_obj.time_signature_changes:
            default_num, default_den = CONFIG["default_timesig"]
            midi_obj.time_signature_changes.append(
                pretty_midi.TimeSignature(default_num, default_den, 0.0)
            )
            print(f"Applied default time signature: {default_num}/{default_den}")

        # Key Signature Changes
        key_sig_changes.sort(key=lambda x: x[0])
        unique_ks = {}
        last_valid_key_name = CONFIG["default_key"] # Track last known valid key
        for time, key_name in key_sig_changes:
            try:
                # Validate key name before storing
                key_number = pretty_midi.key_name_to_key_number(key_name)
                unique_ks[round(time, 6)] = key_name # Store valid key name
                last_valid_key_name = key_name
            except ValueError:
                print(f"Warning: Invalid key name '{key_name}' found at time {time:.3f}s during processing. Ignoring.")

        midi_obj.key_signature_changes = []
        applied_key_count = 0
        last_key_number_added = None
        for time in sorted(unique_ks.keys()):
            key_name = unique_ks[time]
            try:
                key_number = pretty_midi.key_name_to_key_number(key_name)
                # Add only if different from the last one added
                if key_number != last_key_number_added:
                    ks_event = pretty_midi.KeySignature(key_number=key_number, time=time)
                    midi_obj.key_signature_changes.append(ks_event)
                    applied_key_count += 1
                    last_key_number_added = key_number
            except ValueError as e:
                # Should not happen if pre-validated, but as safeguard
                print(f"Error creating KeySignature for '{key_name}' at {time:.3f}s: {e}. Skipping.")

        if applied_key_count > 0:
            print(f"Applied {applied_key_count} unique key signature changes.")
        # Ensure at least one default key signature if none were added
        if not midi_obj.key_signature_changes:
            try:
                # Use the last valid key seen during parsing, or the config default
                final_default_key = last_valid_key_name
                default_key_num = pretty_midi.key_name_to_key_number(final_default_key)
                midi_obj.key_signature_changes.append(
                    pretty_midi.KeySignature(key_number=default_key_num, time=0.0)
                )
                print(f"Applied default key signature: {final_default_key}")
            except ValueError as e:
                print(f"Warning: Invalid default key '{final_default_key}'. No key signature applied. Error: {e}")

        # --- Create instruments and add notes ---
        # Keep track of used channels to avoid conflicts (esp. avoiding channel 9 for melodic)
        available_channels = list(range(16))
        drum_channel = 9 # Standard GM drum channel (0-indexed)
        if drum_channel in available_channels:
            available_channels.remove(drum_channel)
        channel_index = 0 # Index into available_channels for melodic tracks

        # Sort definitions by key (tuple: (inst_lookup_name, track_id)) for consistent assignment
        sorted_inst_keys = sorted(instrument_defs.keys())

        for inst_track_key in sorted_inst_keys:
            definition = instrument_defs[inst_track_key]
            # Skip if this instrument/track ended up with no notes
            if not notes_data.get(inst_track_key):
                continue

            is_drum = definition["is_drum"]
            program = definition["program"]
            pm_instrument_name = definition["name"] # Use the formatted name (e.g., "Lead (Synth Lead)")

            # Assign MIDI channel
            channel = -1 # Placeholder
            if is_drum:
                channel = drum_channel
            else:
                if not available_channels:
                    print(f"Warning: Ran out of unique MIDI channels! Reusing channel for {pm_instrument_name}.")
                    # Fallback: reuse channels starting from 0, avoiding drum channel
                    channel = (channel_index % 15) # Cycle through 0-8, 10-15
                    if channel >= drum_channel: channel += 1
                else:
                    channel = available_channels[channel_index % len(available_channels)]
                channel_index += 1

            # Create the pretty_midi Instrument object
            # Note: pretty_midi assigns the actual channel when adding the instrument
            instrument_obj = pretty_midi.Instrument(
                program=program, is_drum=is_drum, name=pm_instrument_name
            )
            # Add the instrument to the PrettyMIDI object
            midi_obj.instruments.append(instrument_obj)
            # Find the channel assigned by pretty_midi (usually the next available one)
            assigned_channel = -1
            for idx, inst in enumerate(midi_obj.instruments):
                if inst is instrument_obj:
                     # This relies on internal details or might not be easily accessible.
                     # Let's just report the target channel based on our logic.
                     assigned_channel = channel # Report our intended channel
                     break
            print(
                f"Created MIDI instrument: {pm_instrument_name} (Program: {program}, IsDrum: {is_drum}, Target Channel: {assigned_channel})"
            )

            # Add notes to this instrument
            note_count, skipped_notes = 0, 0
            for note_info in notes_data[inst_track_key]:
                start_time = max(0.0, note_info["start"])
                end_time = note_info["end"]
                # Ensure minimum duration and end time is after start time
                min_duration = 0.001 # Minimum note duration in seconds
                if end_time <= start_time:
                    end_time = start_time + min_duration
                elif end_time - start_time < min_duration:
                     # If very short, extend slightly to meet minimum
                     end_time = start_time + min_duration

                # Validate velocity (MIDI spec: 1-127 for NoteOn, 0 is NoteOff)
                velocity = max(1, min(127, int(note_info["velocity"])))
                # Validate pitch
                pitch = max(0, min(127, int(note_info["pitch"])))

                try:
                    note = pretty_midi.Note(
                        velocity=velocity, pitch=pitch, start=start_time, end=end_time
                    )
                    instrument_obj.notes.append(note)
                    note_count += 1
                except ValueError as e:
                    print(
                        f"Error creating pretty_midi.Note for {pm_instrument_name}: {e}. Skipping note. Data: P={pitch}, V={velocity}, S={start_time:.4f}, E={end_time:.4f}"
                    )
                    skipped_notes += 1

            print(f"  Added {note_count} notes. ({skipped_notes} skipped due to errors/duration).")

        # --- Write MIDI File ---
        os.makedirs(CONFIG["output_dir"], exist_ok=True)
        full_output_path = os.path.join(CONFIG["output_dir"], filename)
        midi_obj.write(full_output_path)
        print(f"\nSuccessfully created MIDI file: {full_output_path}")

    except Exception as e:
        print(f"Error writing MIDI file '{filename}': {e}")
        traceback.print_exc()


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting MidiMaker Generator Pipeline...")
    print(f"Using LLM Backend: {'OpenAI' if CONFIG.get('use_openai') else 'Gemini'}")
    configure_genai() # Configure both Gemini and OpenAI (if key provided)

    # Step 1: Enrich Description
    enriched_description, structure_hint = enrich_music_description(
        CONFIG["initial_description"]
    )
    # Use enriched description if available, otherwise fallback to initial
    overall_description_for_generation = enriched_description if enriched_description else CONFIG["initial_description"]

    # Step 2: Generate Section Plan
    section_plan = generate_section_plan(overall_description_for_generation, structure_hint)
    if not section_plan:
        print("Exiting due to failure in generating section plan.")
        exit(1)

    # Step 3: Generate Symbolic Sections Iteratively
    all_symbolic_text = ""
    current_bar_count = 1
    last_section_summary_info = None # Stores {name, summary, key, tempo, time_sig}
    generated_sections_count = 0
    total_planned_bars = sum(info.get("bars", 0) for info in section_plan.values())

    print(
        f"\n--- Step 3 (Iterative): Generating {len(section_plan)} Sections ({total_planned_bars} planned bars) ---"
    )
    section_order = list(section_plan.keys()) # Get the planned order
    for section_name in section_order:
        section_info = section_plan[section_name]
        # Basic check for valid section info from plan
        if not isinstance(section_info.get("bars"), int) or section_info["bars"] <= 0:
            print(
                f"ERROR: Invalid 'bars' ({section_info.get('bars')}) for section {section_name} in plan. Skipping."
            )
            continue

        symbolic_section, current_section_summary_info = generate_symbolic_section(
            overall_description_for_generation,
            section_plan,
            section_name,
            current_bar_count,
            last_section_summary_info, # Pass summary of the *previous* section
        )

        # Check if generation was successful (returned text and summary)
        if symbolic_section is not None and current_section_summary_info is not None:
            symbolic_section_cleaned = symbolic_section.strip()
            if symbolic_section_cleaned: # Only add if not empty
                all_symbolic_text += symbolic_section_cleaned + "\n\n" # Add newline separator
                generated_sections_count += 1
                last_section_summary_info = current_section_summary_info # Update summary for next iteration
                current_bar_count += section_info["bars"] # Advance bar counter
            else:
                print(
                    f"Warning: Section {section_name} generated empty symbolic text after validation. Skipping concatenation."
                )
                # Decide if we should stop or continue? Let's continue for now.
                # Update last_section_summary_info with placeholder if needed? Or keep the previous one?
                # Keeping the previous one seems safer for context passing.
        else:
            print(
                f"ERROR: Failed to generate or validate section {section_name}. Stopping generation process."
            )
            print("Attempting to proceed with previously generated sections...")
            break # Stop generating more sections on critical failure

    # --- Post-Generation Processing ---
    if not all_symbolic_text.strip():
        print("\nERROR: No symbolic text was generated successfully. Cannot proceed.")
        exit(1)
    if generated_sections_count < len(section_plan):
        print(
            f"\nWarning: Only {generated_sections_count}/{len(section_plan)} sections were generated successfully."
        )

    # Save the combined symbolic text
    print("\n--- Saving Combined Symbolic Text ---")
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    symbolic_filename = os.path.join(
        CONFIG["output_dir"], f"symbolic_music_{timestamp_str}.txt"
    )
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    try:
        with open(symbolic_filename, "w", encoding="utf-8") as f:
            f.write(all_symbolic_text)
        print(f"Saved combined symbolic text to: {symbolic_filename}")
    except IOError as e:
        print(f"Error saving symbolic text file: {e}")
    print("-" * 36)

    # Step 4: Parse Symbolic Text
    (
        parsed_notes,
        instrument_definitions,
        tempo_changes,
        time_sig_changes,
        key_sig_changes,
        estimated_duration,
        final_key, # Get final state from parser
        final_tempo,
    ) = parse_symbolic_to_structured_data(all_symbolic_text)

    # Step 5: Create MIDI File
    if parsed_notes and instrument_definitions:
        output_filename = f"generated_music_{timestamp_str}.mid"
        create_midi_file(
            parsed_notes,
            instrument_definitions,
            tempo_changes,
            time_sig_changes,
            key_sig_changes,
            output_filename,
        )
    else:
        print("\nError: No valid notes or instruments parsed. MIDI file not created.")

    print("\n--- MidiMaker Generator Pipeline Finished ---")
