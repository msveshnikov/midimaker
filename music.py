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
import pretty_midi

# --- Configuration ---
GEMINI_KEY = os.environ.get("GEMINI_KEY", "")  # Placeholder - Replace or use env var

# Configure the Gemini model to use
GEMINI_MODEL = "gemini-2.5-pro-exp-03-25" #"gemini-2.0-flash-exp-image-generation" 

# Configuration dictionary
CONFIG = {
    "api_key": GEMINI_KEY,
    "gemini_model": GEMINI_MODEL,
    "initial_description": "disco pop song with a catchy melody and upbeat tempo",
    "output_dir": "output",
    "default_tempo": 120,
    "default_timesig": (4, 4),
    "default_key": "Cmin",  # Will likely be overridden by enrichment
    "default_program": 0, # Default GM Program (Acoustic Grand Piano)
    "default_instrument_name": "Piano", # Default instrument name
    "generation_retries": 3,
    "generation_delay": 65,  # Seconds between retries
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
- `INST:<InstrumentName>`: Sets the active instrument context for subsequent non-drum notes/chords. Examples: Pno, Gtr, Bass, Str, Flt, Tpt, SynPad, SynLead, Arp. This determines the MIDI program number for melodic parts.
- `T:<BPM>`: Sets the tempo in Beats Per Minute (e.g., T:120).
- `TS:<N>/<D>`: Sets the time signature (e.g., TS:4/4). Denominator should ideally be a power of 2.
- `K:<Key>`: Sets the key signature (e.g., K:Cmin, K:Gmaj, K:Ddor). Use standard `pretty_midi` key names (Major: maj, Minor: min, Modes: dor, phr, lyd, mix, loc).
- `BAR:<Num>`: Marks the beginning of a bar (measure), starting from 1, strictly sequential. Timing calculations rely on this.
- `N:<Track>:<Pitch>:<Duration>:<Velocity>`: Represents a single Note event.
- `C:<Track>:<Pitches>:<Duration>:<Velocity>`: Represents a Chord event (multiple notes starting simultaneously).
- `R:<Track>:<Duration>`: Represents a Rest (silence) event for a specific track.

TrackIDs (`<Track>`): Use simple names like RH, LH, Melody, Bass, Drums, Arp1, Pad, Lead etc.
    - If the TrackID is recognized as a drum track name (e.g., 'Drums', 'Drumkit', 'Percussion', 'ElecDrums', '808Drums'), the `<Pitch>` will be interpreted as a drum sound name (see below), and it will use MIDI channel 10.
    - If the TrackID is NOT a recognized drum track name, it's considered a melodic track. The instrument sound (MIDI program) used for this track is determined by the *last* `INST:` command encountered before this note/chord/rest.
PitchNames (`<Pitch>`):
    - For Melodic Tracks: Standard notation (e.g., C4, F#5, Gb3). Middle C is C4.
    - For Drum Tracks: Use drum sound names like Kick, Snare, HHC, HHO, Crash, Ride, HT, MT, LT (case-insensitive). See mapping below. Do NOT use standard notes (C4) on drum tracks.
DurationSymbols (`<Duration>`): W (Whole), H (Half), Q (Quarter), E (Eighth), S (Sixteenth), T (Thirty-second). Append '.' for dotted notes (e.g., Q., E.).
Velocity (`<Velocity>`): MIDI velocity (0-127). Must be a number. Affects loudness.
Pitches (`<Pitches>`): Comma-separated list of pitch names within square brackets (e.g., [C3,Eb3,G3]). For chords on drum tracks, list drum sound names.

Example Note (Melodic): N:Melody:G5:E:95
Example Chord (Melodic): C:PnoLH:[C3,Eb3,G3]:H:60
Example Rest: R:Bass:W
Example Drum Note: N:Drums:Kick:Q:95
Example Drum Chord (Multiple hits): C:Drums:[Kick,HHC]:E:100

Example Sequence:
K:Amin
T:90
TS:4/4
INST:SynPad  # Set instrument for melodic tracks
INST:SynthBass # Update instrument for melodic tracks
BAR:1
C:SynthPad:[A3,C#4,E4,G#4]:W:55
N:SynthBass:A2:Q.:100
N:SynthBass:E2:E:100
N:SynthBass:A2:H:100
N:Drums:Kick:Q:95
N:Drums:HHC:E:80
N:Drums:HHC:E:80
N:Drums:Snare:Q:90
N:Drums:Kick:E:95
N:Drums:Kick:E:95
N:Drums:HHC:E:80
N:Drums:HHC:E:80
R:SynthLead:H
N:SynthLead:C#5:Q:95
N:SynthLead:E5:Q:95
BAR:2
C:SynthPad:[E3,G#3,B3,D#4]:W:55
N:SynthBass:E2:Q.:100
N:SynthBass:B2:E:100
N:SynthBass:E2:H:100
N:Drums:Kick:Q:95
N:Drums:HHC:E:80
N:Drums:HHC:E:80
N:Drums:Snare:Q:90
N:Drums:HHC:E:80
N:Drums:HHC:E:80
N:Drums:Kick:E:95
N:Drums:HHC:E:80
N:SynthLead:F#5:H:95
N:SynthLead:E5:Q:95
R:SynthLead:Q

"""

# --- Helper Functions ---


def configure_genai():
    """Configures the Google Generative AI library."""
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
    retries = retries if retries is not None else CONFIG["generation_retries"]
    delay = delay if delay is not None else CONFIG["generation_delay"]
    model = genai.GenerativeModel(CONFIG["gemini_model"])
    gen_config_args = {"temperature": CONFIG["temperature"]}
    if output_format == "json":
        gen_config_args["response_mime_type"] = "application/json"

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


# --- Music Data Structures and Mappings ---

PITCH_MAP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
ACCIDENTAL_MAP = {"#": 1, "S": 1, "B": -1, "": 0}  # Allow S for sharp

# General MIDI Instrument Program Numbers (Expanded Selection)
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
    k.lower() for k, v in INSTRUMENT_PROGRAM_MAP.items()
    if v == 0 and ("dr" in k or "kit" in k or "perc" in k)
}
DRUM_TRACK_IDS.update(["drums", "drum", "drumkit", "percussion", "elecdrums", "808drums"])

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

# Store the result of pitch_to_midi to avoid re-parsing invalid names repeatedly
_pitch_parse_cache = {}

def pitch_to_midi(pitch_name):
    """Converts pitch name (e.g., C4, F#5, Gb3) to MIDI number. Returns None if invalid."""
    pitch_name = pitch_name.strip()
    if pitch_name in _pitch_parse_cache:
        return _pitch_parse_cache[pitch_name]

    match = re.match(
        r"([A-G])([#sb]?)(\-?\d+)", pitch_name, re.IGNORECASE
    )
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
    if acc_norm == "S": acc_norm = "#"
    acc_val = ACCIDENTAL_MAP.get(acc_norm, 0)
    midi_val = base_midi + acc_val + (octave + 1) * 12

    if 0 <= midi_val <= 127:
        _pitch_parse_cache[pitch_name] = midi_val
        return midi_val
    else:
        _pitch_parse_cache[pitch_name] = None
        return None


def duration_to_seconds(duration_symbol, tempo, time_sig_denominator=4):
    """Converts duration symbol (W, H, Q, E, S, T, W., H., etc.) to seconds."""
    duration_symbol = duration_symbol.strip().upper()
    if not duration_symbol:
        print("Warning: Empty duration symbol. Defaulting to Quarter note duration.")
        duration_symbol = "Q"

    try:
        beats_per_minute = float(tempo)
        if beats_per_minute <= 0:
            print(f"Warning: Invalid tempo {tempo}. Using default 120.")
            beats_per_minute = 120

        quarter_note_duration_sec = 60.0 / beats_per_minute
        duration_map = {"W": 4.0, "H": 2.0, "Q": 1.0, "E": 0.5, "S": 0.25, "T": 0.125}
        base_symbol = duration_symbol.replace(".", "")
        is_dotted = duration_symbol.endswith(".")

        relative_duration_quarters = duration_map.get(base_symbol)
        if relative_duration_quarters is None:
            print(f"Warning: Unknown duration symbol: '{duration_symbol}'. Defaulting to Quarter (1.0).")
            relative_duration_quarters = 1.0

        if is_dotted:
            relative_duration_quarters *= 1.5

        actual_duration_sec = relative_duration_quarters * quarter_note_duration_sec
        return actual_duration_sec

    except ValueError:
        print(f"Warning: Could not parse tempo '{tempo}' as float. Using default 120.")
        return duration_to_seconds(duration_symbol, 120, time_sig_denominator)
    except Exception as e:
        print(f"Error calculating duration for '{duration_symbol}' at tempo {tempo}: {e}. Using default 0.5s.")
        return 0.5


# --- Main Pipeline Functions ---


def enrich_music_description(description):
    """Step 1: Use LLM to enrich the initial music description."""
    print("\n--- Step 1: Enriching Description ---")
    prompt = f"""
    Analyze the following music description. Extract or infer the key signature (K:), tempo (T:), time signature (TS:), and suggest primary instrumentation (INST:).
    If any parameter is explicitly mentioned, use that. If not, infer plausible values based on the description (genre, mood, etc.).
    Also, identify the core mood and suggest a musical structure (like AABA, ABAC, Verse-Chorus-Bridge) if not already provided.
    Output the parameters clearly, preferably at the start, followed by a brief summary.

    Music Description: "{description}"

    Enriched Summary (Example Format: K:Amin T:70 TS:4/4 INST:Pno Mood: Melancholic, sparse. Structure: A-B-A-Coda):
    """
    enriched = call_gemini(prompt)

    current_key = CONFIG["default_key"]
    current_tempo = CONFIG["default_tempo"]
    current_timesig = CONFIG["default_timesig"]
    primary_instrument = CONFIG["default_instrument_name"]
    structure_hint = "AABA"

    if enriched:
        print(f"Enriched Description:\n{enriched}\n")
        key_match = re.search(r"[Kk](?:ey)?\s*:\s*([A-Ga-g][#sb]?(?:maj|min|dor|phr|lyd|mix|loc|aeo|ion)?)", enriched, re.IGNORECASE)
        tempo_match = re.search(r"[Tt](?:empo)?\s*:\s*(\d+)", enriched)
        ts_match = re.search(r"[Tt](?:ime)?\s*[Ss](?:ig)?\s*:\s*(\d+)\s*/\s*(\d+)", enriched)
        inst_match = re.search(r"INST\s*:\s*(\w+)", enriched, re.IGNORECASE)
        if not inst_match: inst_match = re.search(r"(?:instrument(?:s|ation)?|primary inst)\s*:\s*([A-Za-z]+(?:[ ][A-Za-z]+)*)", enriched, re.IGNORECASE)
        struct_match = re.search(r"[Ss]tructure\s*:\s*([\w\-]+)", enriched, re.IGNORECASE)

        if key_match: current_key = key_match.group(1); print(f"Updated Default Key: {current_key}")
        if tempo_match:
            try:
                tempo_val = int(tempo_match.group(1))
                if 5 <= tempo_val <= 300: current_tempo = tempo_val; print(f"Updated Default Tempo: {current_tempo}")
                else: print(f"Warning: Ignoring extracted tempo {tempo_val} (out of range 5-300).")
            except ValueError: pass
        if ts_match:
            try:
                ts_num, ts_den = int(ts_match.group(1)), int(ts_match.group(2))
                if ts_num > 0 and ts_den > 0 and (ts_den & (ts_den - 1) == 0 or ts_den == 1):
                    current_timesig = (ts_num, ts_den); print(f"Updated Default Time Signature: {ts_num}/{ts_den}")
                else: print(f"Warning: Ignoring extracted time signature {ts_num}/{ts_den}.")
            except ValueError: pass
        if inst_match: primary_instrument = inst_match.group(1).strip(); print(f"Identified Primary Instrument Hint: {primary_instrument}")
        if struct_match: structure_hint = struct_match.group(1).upper(); print(f"Identified Structure Hint: {structure_hint}")

        CONFIG["default_key"] = current_key
        CONFIG["default_tempo"] = current_tempo
        CONFIG["default_timesig"] = current_timesig
        # Note: primary_instrument hint isn't directly used as default, INST commands handle it.

        return enriched, structure_hint
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
2. "goal": A concise string describing the musical purpose or content of the section (e.g., "Introduce main theme softly", "Develop theme with variation and drums", "Contrasting bridge section, new chords", "Return to main theme, fuller texture", "Fade out").

The total number of bars across all sections should ideally be around {CONFIG['max_total_bars'] // 2} to {CONFIG['max_total_bars']}, but strictly MUST NOT exceed {CONFIG['max_total_bars']}.
Use the suggested structure "{structure_hint}" as a guide, adapting it as needed (e.g., AABA -> Intro A1 B A2 Outro). Ensure section names are unique and descriptive (e.g., A1, A2 instead of just A, A).

Enriched Description:
{enriched_desc}

Generate ONLY the JSON plan now, starting with {{ and ending with }}. Do not include ```json markers or any other text.
"""

    plan_json = call_gemini(prompt, output_format="json")

    if plan_json and isinstance(plan_json, dict):
        validated_plan = {}
        total_bars = 0
        section_order = list(plan_json.keys()) # Preserve order

        for name in section_order:
            info = plan_json[name]
            if not isinstance(name, str) or not name.strip():
                print(f"Warning: Invalid section name '{name}'. Skipping.")
                continue

            section_name = name.strip()
            if (isinstance(info, dict) and "bars" in info and isinstance(info["bars"], int)
                    and "goal" in info and isinstance(info["goal"], str) and info["goal"].strip()
                    and CONFIG["min_section_bars"] <= info["bars"] <= CONFIG["max_section_bars"]):
                if total_bars + info["bars"] <= CONFIG["max_total_bars"]:
                    info["goal"] = info["goal"].strip()[:250] # Limit goal length
                    validated_plan[section_name] = info
                    total_bars += info["bars"]
                else:
                    print(f"Warning: Section '{section_name}' ({info['bars']} bars) exceeds max total bars ({CONFIG['max_total_bars']}). Truncating plan.")
                    break
            else:
                print(f"Warning: Invalid format/bars/goal for section '{section_name}': {info}. Skipping.")

        if not validated_plan:
            print("ERROR: Failed to generate a valid section plan. Cannot proceed.")
            return None

        print("Generated Section Plan:")
        final_section_order = list(validated_plan.keys())
        for name in final_section_order:
            info = validated_plan[name]
            print(f"  - {name} ({info['bars']} bars): {info['goal']}")
        print(f"Total Bars in Plan: {total_bars}")
        return {name: validated_plan[name] for name in final_section_order}
    else:
        print("ERROR: Failed to generate or parse section plan JSON from LLM. Cannot proceed.")
        if isinstance(plan_json, str): print(f"LLM Output (expected JSON):\n{plan_json[:500]}...")
        return None


def generate_symbolic_section(
    overall_desc,
    section_plan,
    section_name,
    current_bar,
    previous_section_summary=None,
):
    """Step 3: Generate symbolic music for one section using LLM."""
    print(f"--- Step 3: Generating Symbolic Section {section_name} (Starting Bar: {current_bar}) ---")
    section_info = section_plan[section_name]
    bars = section_info["bars"]
    goal = section_info["goal"]

    context_prompt = ""
    if previous_section_summary:
        prev_name = previous_section_summary.get("name", "Previous")
        prev_summary = previous_section_summary.get("summary", "No summary")
        prev_key = previous_section_summary.get("key", "Unknown")
        prev_tempo = previous_section_summary.get("tempo", "Unknown")
        prev_ts = previous_section_summary.get("time_sig", "Unknown")
        context_prompt = (
            f"Context from previous section ({prev_name}): {prev_summary}\n"
            f"It ended around key {prev_key}, tempo {prev_tempo} BPM, and time signature {prev_ts}.\n"
            "Ensure a smooth musical transition if appropriate for the overall structure.\n"
        )

    default_tempo = CONFIG["default_tempo"]
    default_timesig = CONFIG["default_timesig"]
    default_key = CONFIG["default_key"]

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
2. If tempo (T), time signature (TS), key (K), or melodic instrument (INST) need to be set or changed *at the very beginning* of this section (time = start of BAR:{current_bar}), include those commands *before* the `BAR:{current_bar}` marker. Otherwise, assume they carry over from the previous section or use defaults (T:{default_tempo}, TS:{default_timesig[0]}/{default_timesig[1]}, K:{default_key}). You can change INST multiple times within the section if needed for melodic tracks. Drum tracks (e.g., TrackID 'Drums') are handled automatically.
3. Strictly adhere to the compact symbolic format defined below. Output ONLY the commands, each on a new line.
4. DO NOT include any other text, explanations, apologies, section titles, comments (#), or formatting like ```mus``` or ```.
5. Ensure musical coherence within the section and try to achieve the Section Goal.
6. The total duration of notes/rests/chords within each bar MUST add up precisely according to the active time signature (e.g., 4 quarter notes in 4/4, 6 eighth notes in 6/8). Be precise. Use rests (R:<Track>:<Duration>) to fill empty time accurately for each active track within a bar. Ensure parallel tracks are synchronized at bar lines.
7. End the generation cleanly *after* the content for bar {current_bar + bars - 1} is complete. Do NOT include `BAR:{current_bar + bars}`.

{SYMBOLIC_FORMAT_DEFINITION}

Generate Section {section_name} symbolic music now (starting BAR:{current_bar}):
"""

    symbolic_text = call_gemini(prompt)

    if symbolic_text:
        symbolic_text = symbolic_text.strip() # Basic cleaning done in call_gemini
        lines = symbolic_text.split('\n')
        meaningful_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]

        if not meaningful_lines:
            print(f"Warning: Generated text for {section_name} appears empty or only contains comments.")
            return "", None

        bar_marker = f"BAR:{current_bar}"
        bar_marker_found = any(line.strip().startswith(bar_marker) for line in meaningful_lines)

        if not bar_marker_found:
            print(f"ERROR: Generated text for {section_name} does not contain the expected start marker '{bar_marker}'. Discarding section.")
            print(f"Received text (first 500 chars):\n{symbolic_text[:500]}...")
            return "", None

        start_index = -1
        for idx, line in enumerate(lines):
            line_content = line.strip()
            if not line_content or line_content.startswith("#"): continue
            if re.match(r"^(T:|TS:|K:|INST:)", line_content, re.IGNORECASE) or line_content.startswith(bar_marker):
                start_index = idx
                break
            else: # Found unexpected content before a valid start
                print(f"Warning: Generated text for {section_name} had unexpected content before first valid command or '{bar_marker}'. Trimming preamble.")
                break

        if start_index == -1: # If preamble trimming stopped early, find the first actual valid line
             for idx, line in enumerate(lines):
                 line_content = line.strip()
                 if re.match(r"^(T:|TS:|K:|INST:)", line_content, re.IGNORECASE) or line_content.startswith(bar_marker):
                     start_index = idx
                     print(f"Adjusted start index for {section_name} to line {start_index + 1}.")
                     break

        if start_index == -1:
            print(f"ERROR: Could not find any valid starting line (T/TS/K/INST/BAR) in {section_name}. Discarding section.")
            return "", None

        symbolic_text = "\n".join(lines[start_index:])
        print(f"Generated symbolic text for Section {section_name} (first 300 chars):\n{symbolic_text[:300]}...\n")

        # Extract summary info
        summary_info = {
            "name": section_name,
            "summary": f"Generated {bars} bars targeting goal: {goal}.",
            "key": default_key, "tempo": default_tempo, "time_sig": f"{default_timesig[0]}/{default_timesig[1]}",
        }
        last_k, last_t, last_ts = default_key, default_tempo, f"{default_timesig[0]}/{default_timesig[1]}"
        for line in reversed(symbolic_text.split("\n")):
            line = line.strip()
            if line.startswith("K:"): last_k = line.split(":", 1)[1].strip(); break
        for line in reversed(symbolic_text.split("\n")):
            line = line.strip()
            if line.startswith("T:"):
                try: last_t = float(line.split(":", 1)[1].strip()); break
                except ValueError: pass
        for line in reversed(symbolic_text.split("\n")):
            line = line.strip()
            if line.startswith("TS:"): last_ts = line.split(":", 1)[1].strip(); break
        summary_info["key"], summary_info["tempo"], summary_info["time_sig"] = last_k, last_t, last_ts

        return symbolic_text, summary_info
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
    instrument_definitions = {} # Key: inst_track_key tuple, Value: {program, is_drum, name, orig_inst_name}

    # State variables for parsing context
    current_track_times = {} # Key: inst_track_key tuple, Value: current time cursor for this track
    current_global_time = 0.0 # Tracks the latest event time across all tracks, adjusted by BAR markers
    current_tempo = float(CONFIG["default_tempo"])
    current_ts_num, current_ts_den = CONFIG["default_timesig"]
    current_key = CONFIG["default_key"]
    # State for the active melodic instrument (set by INST:)
    active_melodic_program = CONFIG["default_program"]
    active_melodic_instrument_orig_name = CONFIG["default_instrument_name"]

    current_bar_number = 0
    current_bar_start_time = 0.0
    time_within_bar_per_track = {} # Key: inst_track_key tuple, Value: time elapsed within the current bar
    expected_bar_duration_sec = (60.0 / current_tempo) * current_ts_num * (4.0 / current_ts_den)
    last_event_end_time = 0.0

    initial_commands_set = {"T": False, "TS": False, "K": False, "INST": False}
    lines = symbolic_text.strip().split("\n")
    parse_start_line_index = 0

    # --- Pre-pass for initial settings (before first BAR marker) ---
    print("Processing initial settings (before first BAR marker)...")
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith("#"): continue
        if line.startswith("BAR:"):
            parse_start_line_index = i
            break

        parts = line.split(":", 1)
        command = parts[0].upper()
        value = parts[1].strip() if len(parts) > 1 else ""
        ini_line_num = i + 1

        try:
            if command == "INST":
                if value:
                    inst_name_lower = value.lower()
                    program = INSTRUMENT_PROGRAM_MAP.get(inst_name_lower, CONFIG["default_program"])
                    if program != active_melodic_program or value != active_melodic_instrument_orig_name:
                        active_melodic_program = program
                        active_melodic_instrument_orig_name = value
                        initial_commands_set["INST"] = True
                        print(f"Initial Melodic Instrument context set to '{value}' (Program: {program})")
            elif command == "T" and not initial_commands_set["T"]:
                new_tempo = float(value)
                if new_tempo > 0:
                    current_tempo = new_tempo
                    tempo_changes.append((0.0, current_tempo))
                    expected_bar_duration_sec = (60.0 / current_tempo) * current_ts_num * (4.0 / current_ts_den)
                    initial_commands_set["T"] = True
                    print(f"Initial Tempo set to {current_tempo} BPM")
            elif command == "TS" and not initial_commands_set["TS"]:
                num_str, den_str = value.split("/")
                new_ts_num, new_ts_den = int(num_str), int(den_str)
                if new_ts_num > 0 and new_ts_den > 0 and (new_ts_den & (new_ts_den - 1) == 0 or new_ts_den == 1):
                    current_ts_num, current_ts_den = new_ts_num, new_ts_den
                    time_signature_changes.append((0.0, current_ts_num, current_ts_den))
                    expected_bar_duration_sec = (60.0 / current_tempo) * current_ts_num * (4.0 / current_ts_den)
                    initial_commands_set["TS"] = True
                    print(f"Initial Time Signature set to {current_ts_num}/{current_ts_den}")
            elif command == "K" and not initial_commands_set["K"]:
                if value:
                    current_key = value
                    key_signature_changes.append((0.0, current_key))
                    initial_commands_set["K"] = True
                    print(f"Initial Key set to {current_key}")
        except Exception as e:
            print(f"Error parsing initial setting line {ini_line_num}: '{line}' - {e}")
        parse_start_line_index = i + 1

    # Set defaults if not specified initially
    if not initial_commands_set["T"]: tempo_changes.append((0.0, current_tempo))
    if not initial_commands_set["TS"]: time_signature_changes.append((0.0, current_ts_num, current_ts_den))
    if not initial_commands_set["K"]: key_signature_changes.append((0.0, current_key))
    if not initial_commands_set["INST"]: print(f"Using default initial Melodic Instrument: '{active_melodic_instrument_orig_name}' (Program: {active_melodic_program})")

    # --- Main Parsing Loop ---
    print(f"Parsing main body starting from line {parse_start_line_index + 1}...")
    for i in range(parse_start_line_index, len(lines)):
        current_line_num = i + 1
        line = lines[i].strip()
        if not line or line.startswith("#"): continue

        parts = line.split(":", 1)
        command = parts[0].upper()
        value = parts[1].strip() if len(parts) > 1 else ""

        try:
            if command == "INST":
                if value:
                    inst_name_lower = value.lower()
                    program = INSTRUMENT_PROGRAM_MAP.get(inst_name_lower, CONFIG["default_program"])
                    # Update context only if changed
                    if program != active_melodic_program or value != active_melodic_instrument_orig_name:
                        active_melodic_program = program
                        active_melodic_instrument_orig_name = value
                        # Don't print every change, can be verbose
            elif command == "T":
                new_tempo = float(value)
                if new_tempo > 0 and abs(new_tempo - current_tempo) > 1e-3:
                    event_time = current_global_time
                    if not tempo_changes or abs(tempo_changes[-1][0] - event_time) > 1e-6 or abs(tempo_changes[-1][1] - new_tempo) > 1e-3:
                        tempo_changes.append((event_time, new_tempo))
                        print(f"Time {event_time:.3f}s (Bar ~{current_bar_number}): Tempo change to {new_tempo} BPM")
                    current_tempo = new_tempo
                    expected_bar_duration_sec = (60.0 / current_tempo) * current_ts_num * (4.0 / current_ts_den)
            elif command == "TS":
                num_str, den_str = value.split("/")
                new_ts_num, new_ts_den = int(num_str), int(den_str)
                if new_ts_num > 0 and new_ts_den > 0 and (new_ts_den & (new_ts_den - 1) == 0 or new_ts_den == 1) \
                   and (new_ts_num, new_ts_den) != (current_ts_num, current_ts_den):
                    event_time = current_global_time
                    if not time_signature_changes or abs(time_signature_changes[-1][0] - event_time) > 1e-6 \
                       or (time_signature_changes[-1][1], time_signature_changes[-1][2]) != (new_ts_num, new_ts_den):
                        time_signature_changes.append((event_time, new_ts_num, new_ts_den))
                        print(f"Time {event_time:.3f}s (Bar ~{current_bar_number}): Time Sig change to {new_ts_num}/{new_ts_den}")
                    current_ts_num, current_ts_den = new_ts_num, new_ts_den
                    expected_bar_duration_sec = (60.0 / current_tempo) * current_ts_num * (4.0 / current_ts_den)
            elif command == "K":
                if value and value != current_key:
                    event_time = current_global_time
                    if not key_signature_changes or abs(key_signature_changes[-1][0] - event_time) > 1e-6 or key_signature_changes[-1][1] != value:
                        key_signature_changes.append((event_time, value))
                        print(f"Time {event_time:.3f}s (Bar ~{current_bar_number}): Key change to {value}")
                    current_key = value
            elif command == "BAR":
                bar_number = int(value)
                expected_new_bar_start_time = current_bar_start_time + expected_bar_duration_sec if current_bar_number > 0 else 0.0

                # Check timing accuracy of the previous bar
                if current_bar_number > 0:
                    max_accumulated_time_in_prev_bar = 0.0
                    for accumulated_time in time_within_bar_per_track.values():
                        max_accumulated_time_in_prev_bar = max(max_accumulated_time_in_prev_bar, accumulated_time)

                    tolerance = max(0.005, expected_bar_duration_sec * 0.01)
                    duration_error = max_accumulated_time_in_prev_bar - expected_bar_duration_sec

                    if duration_error > tolerance: # Overran
                         print(f"Warning: Bar {current_bar_number} timing mismatch (Overran). Expected {expected_bar_duration_sec:.3f}s, got {max_accumulated_time_in_prev_bar:.3f}s. Forcing bar {bar_number} start to expected time {expected_new_bar_start_time:.3f}s.")
                         current_global_time = expected_new_bar_start_time
                    # Optional: Check for underrun (might be intentional silence)
                    # elif duration_error < -tolerance and max_accumulated_time_in_prev_bar > 0: # Underran
                    #     print(f"Warning: Bar {current_bar_number} timing potentially short. Expected {expected_bar_duration_sec:.3f}s, got {max_accumulated_time_in_prev_bar:.3f}s. Setting bar {bar_number} start to expected time {expected_new_bar_start_time:.3f}s.")
                    #     current_global_time = expected_new_bar_start_time
                    else: # Within tolerance or correctly filled
                         current_global_time = expected_new_bar_start_time

                # Handle jumps in bar numbers
                bars_jumped = bar_number - (current_bar_number + 1)
                if bars_jumped > 0 and current_bar_number > 0:
                    jump_duration = bars_jumped * expected_bar_duration_sec
                    print(f"Warning: Jump detected from Bar {current_bar_number} to {bar_number}. Advancing global time by ~{jump_duration:.3f}s.")
                    current_global_time += jump_duration

                # Update bar state
                current_bar_number = bar_number
                current_bar_start_time = current_global_time
                time_within_bar_per_track = {key: 0.0 for key in time_within_bar_per_track} # Reset time within bar
                current_track_times = {key: current_bar_start_time for key in current_track_times} # Sync track cursors

            elif command in ["N", "C", "R"]:
                if current_bar_number == 0: # Event before first BAR marker
                    if not any(l.strip().startswith("BAR:") for l in lines[:i]):
                        print(f"Warning: Event '{line}' on Line {current_line_num} before first BAR marker. Processing at time 0.")
                        current_bar_number = 1; current_bar_start_time = 0.0; current_global_time = 0.0
                    else:
                        print(f"Internal Warning: Event processing while current_bar_number is 0. Line {current_line_num}: '{line}'")

                # Determine instrument properties based on TrackID and active INST context
                data_parts = value.split(":")
                if len(data_parts) < (2 if command == "R" else 4):
                    print(f"Warning: Malformed {command} command on Line {current_line_num}: '{line}'. Skipping.")
                    continue

                track_id = data_parts[0].strip()
                if not track_id: print(f"Warning: Empty TrackID in {command} command on Line {current_line_num}. Skipping."); continue

                event_is_drum = track_id.lower() in DRUM_TRACK_IDS
                event_program = 0
                event_instrument_base_name = track_id # Default for drums
                midi_instrument_name = f"{track_id} (Drums)"
                inst_track_key = (track_id.lower(), track_id) # Key for drum tracks

                if not event_is_drum:
                    event_program = active_melodic_program
                    event_instrument_base_name = active_melodic_instrument_orig_name
                    midi_instrument_name = f"{track_id} ({active_melodic_instrument_orig_name})"
                    # Key combines active melodic instrument and track ID
                    inst_track_key = (active_melodic_instrument_orig_name.lower(), track_id)

                # Define instrument in definitions if first time seeing this key
                if inst_track_key not in instrument_definitions:
                    instrument_definitions[inst_track_key] = {
                        "program": event_program,
                        "is_drum": event_is_drum,
                        "name": midi_instrument_name,
                        "orig_inst_name": event_instrument_base_name,
                    }
                    print(f"Defined instrument: {midi_instrument_name} (Key: {inst_track_key}, Program: {event_program}, IsDrum: {event_is_drum})")
                    # Initialize time tracking for this new instrument/track key
                    initial_track_offset = time_within_bar_per_track.get(inst_track_key, 0.0)
                    current_track_times[inst_track_key] = current_bar_start_time + initial_track_offset
                    time_within_bar_per_track[inst_track_key] = initial_track_offset
                    notes_by_instrument_track[inst_track_key] = []

                # Calculate event start time based on track's progress within the bar
                track_specific_start_offset = time_within_bar_per_track.get(inst_track_key, 0.0)
                event_start_time = current_bar_start_time + track_specific_start_offset

                # --- Parse Note (N) ---
                if command == "N":
                    # N:<Track>:<Pitch>:<Duration>:<Velocity>
                    pitch_name_raw = data_parts[1].strip()
                    duration_sym_raw = data_parts[2].strip()
                    velocity_str_raw = data_parts[3].strip()
                    if not pitch_name_raw or not duration_sym_raw or not velocity_str_raw:
                        print(f"Warning: Empty part in N command on Line {current_line_num}. Skipping."); continue

                    velocity_str = velocity_str_raw.split("#", 1)[0].strip()
                    duration_sym = duration_sym_raw
                    try:
                        velocity = int(velocity_str); velocity = max(0, min(127, velocity))
                    except ValueError: velocity = 90; print(f"Warning: Invalid velocity '{velocity_str_raw}'. Using {velocity}.")

                    event_duration_sec = duration_to_seconds(duration_sym, current_tempo, current_ts_den)
                    midi_pitch = None

                    if event_is_drum:
                        pitch_name_lookup = pitch_name_raw.lower()
                        midi_pitch = DRUM_PITCH_MAP.get(pitch_name_lookup)
                        if midi_pitch is None:
                             standard_pitch = pitch_to_midi(pitch_name_raw) # Allow standard pitch on drum track?
                             if standard_pitch is not None:
                                 print(f"Warning: Standard pitch '{pitch_name_raw}' used on drum track '{track_id}'. Using MIDI pitch {standard_pitch}.")
                                 midi_pitch = standard_pitch
                             else:
                                 print(f"Warning: Unknown drum sound '{pitch_name_raw}' for track '{track_id}'. Using Kick (36)."); midi_pitch = 36
                    else: # Melodic
                        midi_pitch = pitch_to_midi(pitch_name_raw)
                        if midi_pitch is None: print(f"Warning: Cannot parse pitch '{pitch_name_raw}' for track '{track_id}'. Skipping note."); continue

                    note_event = {"pitch": midi_pitch, "start": event_start_time, "end": event_start_time + event_duration_sec, "velocity": velocity}
                    notes_by_instrument_track[inst_track_key].append(note_event)

                # --- Parse Chord (C) ---
                elif command == "C":
                    # C:<Track>:<[Pitches]>:<Duration>:<Velocity>
                    pitches_str_raw = data_parts[1].strip()
                    duration_sym_raw = data_parts[2].strip()
                    velocity_str_raw = data_parts[3].strip()
                    if not pitches_str_raw or not duration_sym_raw or not velocity_str_raw:
                        print(f"Warning: Empty part in C command on Line {current_line_num}. Skipping."); continue

                    velocity_str = velocity_str_raw.split("#", 1)[0].strip()
                    duration_sym = duration_sym_raw
                    try:
                        velocity = int(velocity_str); velocity = max(0, min(127, velocity))
                    except ValueError: velocity = 70; print(f"Warning: Invalid velocity '{velocity_str_raw}'. Using {velocity}.")

                    if pitches_str_raw.startswith("[") and pitches_str_raw.endswith("]"): pitches_str = pitches_str_raw[1:-1]
                    else: print(f"Warning: Chord pitches format '{pitches_str_raw}' incorrect. Expected '[P1,P2,...]'. Attempting parse."); pitches_str = pitches_str_raw

                    pitch_names = [p.strip() for p in pitches_str.split(",") if p.strip()]
                    if not pitch_names: print(f"Warning: No pitches in Chord command '{line}'. Skipping."); continue

                    event_duration_sec = duration_to_seconds(duration_sym, current_tempo, current_ts_den)
                    chord_notes = []
                    valid_pitches_in_chord = 0
                    for pitch_name_raw in pitch_names:
                        midi_pitch = None
                        if event_is_drum:
                            pitch_name_lookup = pitch_name_raw.lower()
                            midi_pitch = DRUM_PITCH_MAP.get(pitch_name_lookup)
                            if midi_pitch is None:
                                standard_pitch = pitch_to_midi(pitch_name_raw)
                                if standard_pitch is not None: print(f"Warning: Standard pitch '{pitch_name_raw}' in chord on drum track '{track_id}'. Using MIDI pitch {standard_pitch}."); midi_pitch = standard_pitch
                                else: print(f"Warning: Unknown drum sound '{pitch_name_raw}' in chord for track '{track_id}'. Using Kick (36)."); midi_pitch = 36
                        else: # Melodic
                            midi_pitch = pitch_to_midi(pitch_name_raw)
                            if midi_pitch is None: print(f"Warning: Cannot parse pitch '{pitch_name_raw}' in chord for track '{track_id}'. Skipping pitch."); continue

                        note_event = {"pitch": midi_pitch, "start": event_start_time, "end": event_start_time + event_duration_sec, "velocity": velocity}
                        chord_notes.append(note_event)
                        valid_pitches_in_chord += 1

                    if valid_pitches_in_chord > 0: notes_by_instrument_track[inst_track_key].extend(chord_notes)
                    else: print(f"Warning: Chord command on line {current_line_num} had no valid notes. Skipping."); continue # Don't advance time if chord empty

                # --- Parse Rest (R) ---
                elif command == "R":
                    # R:<Track>:<Duration>
                    duration_sym_raw = data_parts[1].strip()
                    if not duration_sym_raw: print(f"Warning: Empty duration in R command '{line}'. Skipping."); continue
                    duration_sym = duration_sym_raw.split("#", 1)[0].strip()
                    if not duration_sym: print(f"Warning: Empty duration for R command '{line}'. Skipping."); continue
                    event_duration_sec = duration_to_seconds(duration_sym, current_tempo, current_ts_den)
                    # Rests just advance time, no notes added.

                # --- Post-Event Time Advancement ---
                new_track_time_absolute = event_start_time + event_duration_sec
                current_track_times[inst_track_key] = new_track_time_absolute
                time_within_bar_per_track[inst_track_key] = new_track_time_absolute - current_bar_start_time
                last_event_end_time = max(last_event_end_time, new_track_time_absolute)

            else:
                print(f"Warning: Unknown command '{command}' on line {current_line_num}: '{line}'. Skipping.")

        except Exception as e:
            print(f"FATAL Error parsing line {current_line_num}: '{line}' - {e}")
            traceback.print_exc()

    print(f"Symbolic text parsing complete. Estimated total duration: {last_event_end_time:.3f} seconds.")
    # Sanity check: Ensure all defined instruments actually have notes
    final_instrument_defs = {}
    final_notes_data = {}
    for key, definition in instrument_definitions.items():
        if key in notes_by_instrument_track and notes_by_instrument_track[key]:
            final_instrument_defs[key] = definition
            final_notes_data[key] = notes_by_instrument_track[key]
        else:
            print(f"Info: Instrument '{definition['name']}' (Key: {key}) defined but had no notes parsed. Excluding from MIDI.")

    return (final_notes_data, final_instrument_defs, tempo_changes, time_signature_changes, key_signature_changes,
            last_event_end_time, current_key, current_tempo)


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
        print("Error: No instrument or note data was successfully parsed. Cannot create MIDI file.")
        return

    try:
        initial_tempo = tempo_changes[0][1] if tempo_changes else CONFIG["default_tempo"]
        midi_obj = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo)
        print(f"Initialized MIDI with tempo: {initial_tempo:.2f} BPM")

        # --- Apply Meta-Messages ---
        # Tempo Changes
        if len(tempo_changes) > 1: # Already handled initial tempo
            print("Applying tempo changes...")
            unique_tempo_times = {}
            for t, bpm in tempo_changes: unique_tempo_times[round(t, 6)] = bpm
            sorted_tempo_times = sorted(unique_tempo_times.keys())
            final_tempo_times, final_tempo_bpm = [], []
            for t in sorted_tempo_times:
                if t > 1e-6: # Apply changes after time 0
                    final_tempo_times.append(t)
                    final_tempo_bpm.append(unique_tempo_times[t])

        # Time Signature Changes
        time_sig_changes.sort(key=lambda x: x[0])
        unique_ts = {}
        for time, num, den in time_sig_changes:
            actual_den = den
            if den <= 0 or (den & (den - 1) != 0 and den != 1):
                actual_den = 2**math.ceil(math.log2(den)) if den > 0 else 4
                print(f"Warning: Invalid TS denominator {den} at {time:.3f}s. Using {actual_den}.")
            unique_ts[round(time, 6)] = (num, actual_den)

        midi_obj.time_signature_changes = []
        applied_ts_count = 0; last_ts = None
        for time in sorted(unique_ts.keys()):
            if time >= 0:
                num, den = unique_ts[time]
                if last_ts != (num, den):
                    midi_obj.time_signature_changes.append(pretty_midi.TimeSignature(num, den, time))
                    applied_ts_count += 1; last_ts = (num, den)
        if applied_ts_count > 0: print(f"Applied {applied_ts_count} unique time signature changes.")
        if not midi_obj.time_signature_changes:
            default_num, default_den = CONFIG["default_timesig"]
            midi_obj.time_signature_changes.append(pretty_midi.TimeSignature(default_num, default_den, 0.0))
            print(f"Applied default time signature: {default_num}/{default_den}")

        # Key Signature Changes
        key_sig_changes.sort(key=lambda x: x[0])
        unique_ks = {}
        last_valid_key_name = CONFIG["default_key"]
        for time, key_name in key_sig_changes: unique_ks[round(time, 6)] = key_name

        midi_obj.key_signature_changes = []
        applied_key_count = 0; last_key_number = None
        for time in sorted(unique_ks.keys()):
            if time >= 0:
                key_name = unique_ks[time]
                try:
                    key_number = pretty_midi.key_name_to_key_number(key_name)
                    if key_number != last_key_number:
                        midi_obj.key_signature_changes.append(pretty_midi.KeySignature(key_number=key_number, time=time))
                        applied_key_count += 1; last_key_number = key_number; last_valid_key_name = key_name
                except ValueError as e: print(f"Warning: Invalid key '{key_name}' at {time:.3f}s. Skipping. Error: {e}")
        if applied_key_count > 0: print(f"Applied {applied_key_count} unique key signature changes.")
        if not midi_obj.key_signature_changes:
            try:
                final_default_key = last_valid_key_name if last_key_number is not None else CONFIG["default_key"]
                default_key_num = pretty_midi.key_name_to_key_number(final_default_key)
                midi_obj.key_signature_changes.append(pretty_midi.KeySignature(key_number=default_key_num, time=0.0))
                print(f"Applied default key signature: {final_default_key}")
            except ValueError as e: print(f"Warning: Invalid default key '{final_default_key}'. No key signature applied. Error: {e}")

        # --- Create instruments and add notes ---
        available_channels = list(range(16))
        drum_channel = 9 # Standard GM drum channel
        if drum_channel in available_channels: available_channels.remove(drum_channel)
        channel_index = 0

        # Sort definitions by key (tuple: (inst_lower, track_id)) for consistent assignment
        sorted_inst_keys = sorted(instrument_defs.keys())

        for inst_track_key in sorted_inst_keys:
            definition = instrument_defs[inst_track_key]
            if not notes_data.get(inst_track_key): continue # Skip if no notes

            is_drum = definition["is_drum"]
            program = definition["program"]
            pm_instrument_name = definition["name"]
            channel = drum_channel if is_drum else available_channels[channel_index % len(available_channels)]

            if not is_drum:
                if channel_index >= len(available_channels): print(f"Warning: Reusing channel {channel} for {pm_instrument_name}.")
                channel_index += 1

            instrument_obj = pretty_midi.Instrument(program=program, is_drum=is_drum, name=pm_instrument_name)
            midi_obj.instruments.append(instrument_obj) # pretty_midi assigns channel here
            print(f"Created MIDI instrument: {pm_instrument_name} (Program: {program}, IsDrum: {is_drum}, Target Channel: {channel})")

            note_count, skipped_notes = 0, 0
            for note_info in notes_data[inst_track_key]:
                start_time = max(0.0, note_info["start"])
                min_duration = 0.001
                end_time = max(start_time + min_duration, note_info["end"])
                velocity = max(1, min(127, int(note_info["velocity"]))) # Velocity 0 is NoteOff
                pitch = max(0, min(127, int(note_info["pitch"])))

                if end_time - start_time < min_duration / 2:
                    print(f"Warning: Skipping note for '{pm_instrument_name}' with near-zero duration (Start: {start_time:.4f}, End: {end_time:.4f}, Pitch: {pitch}).")
                    skipped_notes += 1; continue
                try:
                    instrument_obj.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=end_time))
                    note_count += 1
                except ValueError as e:
                    print(f"Error creating pretty_midi.Note for {pm_instrument_name}: {e}. Skipping note. Data: P={pitch}, V={velocity}, S={start_time:.4f}, E={end_time:.4f}")
                    skipped_notes += 1

            print(f"  Added {note_count} notes. ({skipped_notes} skipped).")

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
    print(f"Using Model: {CONFIG['gemini_model']}")
    configure_genai()

    enriched_description, structure_hint = enrich_music_description(CONFIG["initial_description"])
    if not enriched_description: enriched_description = CONFIG["initial_description"] # Fallback

    section_plan = generate_section_plan(enriched_description, structure_hint)
    if not section_plan: print("Exiting due to failure in generating section plan."); exit(1)

    all_symbolic_text = ""
    current_bar_count = 1
    last_section_summary_info = None
    generated_sections_count = 0
    total_planned_bars = sum(info.get("bars", 0) for info in section_plan.values())

    print(f"\n--- Step 3: Generating {len(section_plan)} Sections ({total_planned_bars} planned bars) ---")
    section_order = list(section_plan.keys())
    for section_name in section_order:
        section_info = section_plan[section_name]
        if not isinstance(section_info.get("bars"), int) or section_info["bars"] <= 0:
            print(f"ERROR: Invalid 'bars' ({section_info.get('bars')}) for section {section_name}. Skipping."); continue

        symbolic_section, current_section_summary_info = generate_symbolic_section(
            enriched_description, section_plan, section_name, current_bar_count, last_section_summary_info
        )

        if symbolic_section is not None and current_section_summary_info is not None:
            symbolic_section_cleaned = symbolic_section.strip()
            if symbolic_section_cleaned:
                all_symbolic_text += symbolic_section_cleaned + "\n"
                generated_sections_count += 1
                last_section_summary_info = current_section_summary_info
                current_bar_count += section_info["bars"]
            else:
                print(f"Warning: Section {section_name} generated empty symbolic text. Skipping concatenation.")
        else:
            print(f"Failed to generate or validate section {section_name}. Stopping generation.")
            print("Attempting to proceed with previously generated sections...")
            break # Stop on critical failure

    if not all_symbolic_text.strip():
        print("\nERROR: No symbolic text was generated successfully. Cannot proceed.")
        exit(1)
    if generated_sections_count < len(section_plan):
        print(f"\nWarning: Only {generated_sections_count}/{len(section_plan)} sections were generated successfully.")

    print("\n--- Saving Combined Symbolic Text ---")
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    symbolic_filename = os.path.join(CONFIG["output_dir"], f"symbolic_music_{timestamp_str}.txt")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    try:
        with open(symbolic_filename, "w", encoding="utf-8") as f: f.write(all_symbolic_text)
        print(f"Saved combined symbolic text to: {symbolic_filename}")
    except IOError as e: print(f"Error saving symbolic text file: {e}")
    print("------------------------------------")

    _pitch_parse_cache.clear() # Clear cache before parsing
    (parsed_notes, instrument_definitions, tempo_changes, time_sig_changes, key_sig_changes,
     estimated_duration, final_key, final_tempo) = parse_symbolic_to_structured_data(all_symbolic_text)

    if parsed_notes and instrument_definitions:
        output_filename = f"generated_music_{timestamp_str}.mid"
        create_midi_file(
            parsed_notes, instrument_definitions, tempo_changes, time_sig_changes, key_sig_changes, output_filename
        )
    else:
        print("\nError: No valid notes or instruments parsed. MIDI file not created.")

    print("\n--- MidiMaker Generator Pipeline Finished ---")