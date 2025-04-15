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
GEMINI_MODEL = "gemini-2.5-pro-exp-03-25" #"gemini-2.0-flash-thinking-exp-01-21" 

# Configuration dictionary
CONFIG = {
    "api_key": GEMINI_KEY,
    "gemini_model": GEMINI_MODEL,
    "initial_description": "trans electronic hiphop cosmic ambient music fast 160 bpm optimistic",
    "output_dir": "output",
    "default_tempo": 120,
    "default_timesig": (4, 4),
    "default_key": "Cmin",  # Will likely be overridden by enrichment
    "generation_retries": 3,
    "generation_delay": 5,  # Seconds between retries
    "max_total_bars": 64,  # Limit total length for safety/cost
    "min_section_bars": 8,  # Minimum bars per generated section
    "max_section_bars": 16,  # Maximum bars per generated section
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
- `INST:<InstrumentName>` (Instrument: e.g., Pno, Gtr, Bass, Drs, Str, Flt, Tpt, SynPad, SynLead, Arp). Use 'Drs' or 'Drums' for drum kits.
- `T:<BPM>` (Tempo: e.g., T:120)
- `TS:<N>/<D>` (Time Signature: e.g., TS:4/4)
- `K:<Key>` (Key Signature: e.g., K:Cmin, K:Gmaj, K:Ddor). Use standard `pretty_midi` key names (Major: maj, Minor: min, Modes: dor, phr, lyd, mix, loc).
- `BAR:<Num>` (Bar marker, starting from 1, strictly sequential)
- `N:<Track>:<Pitch>:<Duration>:<Velocity>` (Note: TrackID: PitchName: DurationSymbol: Velocity[0-127])
- `C:<Track>:<Pitches>:<Duration>:<Velocity>` (Chord: TrackID: [Pitch1,Pitch2,...]: DurationSymbol: Velocity)
- `R:<Track>:<Duration>` (Rest: TrackID: DurationSymbol)

TrackIDs: Use simple names like RH, LH, Melody, Bass, Drums, Arp1, Pad, Lead etc. A TrackID combined with the current INST defines a unique part.
PitchNames: Standard notation (e.g., C4, F#5, Gb3). Middle C is C4. For Drums (when INST is Drs/Drums), use names like Kick, Snare, HHC, HHO, Crash, Ride, HT, MT, LT (case-insensitive).
DurationSymbols: W (Whole), H (Half), Q (Quarter), E (Eighth), S (Sixteenth), T (Thirty-second). Append '.' for dotted notes (e.g., Q., E.).
Velocity: MIDI velocity (0-127). Must be a number.
Do not use standard notes on drum track!
Do not use drum names for instrument tracks!


Example Note: N:SynthBass:G5:E:95
Example Chord: C:PnoLH:[C3,Eb3,G3]:H:60
Example Rest: R:Bass:W
Example Drum: N:Drums:Kick:Q:95
Example output:

K:Amaj
T:160
TS:4/4
INST:SynthPad
INST:SynthBass
INST:ElecDrums
INST:SynthLead
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
    print(f"Prompt: {prompt}")
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

            # Handle potential API response variations and errors
            if (
                hasattr(response, "prompt_feedback")
                and response.prompt_feedback.block_reason
            ):
                print(
                    f"Prompt blocked (Attempt {attempt + 1}):"
                    f" {response.prompt_feedback.block_reason}"
                )
                # If blocked, retrying likely won't help
                return None

            content = None
            # Check candidates for content, especially if parts is empty or finish reason isn't STOP
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if candidate.finish_reason != "STOP":
                    print(
                        f"Generation stopped for reason: {candidate.finish_reason} (Candidate Level) on attempt {attempt + 1}"
                    )
                    # Try to get content even if stopped early
                    if candidate.content and candidate.content.parts:
                        print("Attempting to use partial content from candidate.")
                        content = candidate.content.parts[0].text.strip()
                    else:
                        print("No content available from early-stopped candidate.")
                        # Fall through to retry logic if possible
                elif candidate.content and candidate.content.parts:
                    # Standard case via candidate
                    content = candidate.content.parts[0].text.strip()
                else:
                    print(
                        f"Warning: Received response with no usable candidate content (Attempt {attempt + 1})."
                    )
                    # Fall through to retry logic

            # Fallback or primary check via response.parts / response.text
            elif hasattr(response, "parts") and response.parts:
                content = response.text.strip()
            else:
                print(
                    f"Warning: Received response with no parts or candidates (Attempt {attempt + 1})."
                )
                # Fall through to retry logic

            # Process the content based on expected format
            if content is not None:
                if output_format == "json":
                    try:
                        # Remove potential markdown fences before parsing JSON
                        content_cleaned = re.sub(
                            r"^```json\n", "", content, flags=re.IGNORECASE
                        )
                        content_cleaned = re.sub(r"\n```$", "", content_cleaned)
                        return json.loads(content_cleaned)
                    except json.JSONDecodeError as json_e:
                        print(
                            f"Error decoding JSON response (Attempt {attempt + 1}): {json_e}"
                        )
                        print(f"Received text: {content[:500]}...")
                        # Fall through to retry logic
                else:
                    # Remove potential markdown fences from text output
                    content_cleaned = re.sub(
                        r"^```[a-z]*\n",
                        "",
                        content,
                        flags=re.MULTILINE | re.IGNORECASE,
                    )
                    content_cleaned = re.sub(r"\n```$", "", content_cleaned)
                    return content_cleaned.strip()  # Return cleaned text

            # If content is None or JSON parsing failed, log and potentially retry
            print(f"Warning: Could not extract valid content (Attempt {attempt + 1}).")

        except Exception as e:
            print(
                f"Error calling Gemini API (Attempt {attempt + 1}/{retries}): {e}"
            )
            traceback.print_exc()  # Print full traceback for debugging

        if attempt < retries - 1:
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
        else:
            print("Max retries reached. Failing.")
            return None  # Indicate failure after all retries
    return None  # Fallback


# --- Music Data Structures and Mappings ---

PITCH_MAP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
ACCIDENTAL_MAP = {"#": 1, "S": 1, "B": -1, "": 0}  # Allow S for sharp
# General MIDI Instrument Program Numbers (Expanded Selection)
INSTRUMENT_PROGRAM_MAP = {
    # Piano
    "pno": 0,
    "Piano": 0,
    "Acoustic grand piano": 0,
    "Bright acoustic piano": 1,
    "Electric grand piano": 2,
    "Honky-tonk piano": 3,
    "Electric piano 1": 4,
    "Rhodes piano": 4,
    "Electric piano 2": 5,
    # Chromatic Percussion
    "Celesta": 8,
    "Glockenspiel": 9,
    "Music box": 10,
    "Vibraphone": 11,
    "Marimba": 12,
    "Xylophone": 13,
    "Tubular bells": 14,
    "Dulcimer": 15,
    # Organ
    "org": 16,
    "Organ": 16,
    "Drawbar organ": 16,
    "Percussive organ": 17,
    "Rock organ": 18,
    "Church organ": 19,
    "Reed organ": 20,
    "Accordion": 21,
    "Harmonica": 22,
    "Tango accordion": 23,
    # Guitar
    "gtr": 25,
    "Acoustic guitar": 25,
    "Nylon guitar": 24,
    "Steel guitar": 25,
    "Electric guitar": 27,
    "Jazz guitar": 26,
    "Clean electric guitar": 27,
    "Muted electric guitar": 28,
    "Overdriven guitar": 29,
    "Distortion guitar": 30,
    "Guitar harmonics": 31,
    # Bass
    "Bass": 33,
    "Acoustic bass": 32,
    "Electric bass": 33,
    "Finger bass": 33,
    "Pick bass": 34,
    "Fretless bass": 35,
    "Slap bass": 36,
    "Synth bass": 38,
    "Synthbass": 38,
    "Synth bass 2": 39,
    # Strings
    "str": 48,
    "Strings": 48,
    "Violin": 40,
    "Viola": 41,
    "Cello": 42,
    "Contrabass": 43,
    "Tremolo strings": 44,
    "Pizzicato strings": 45,
    "Orchestral harp": 46,
    "Timpani": 47,
    "String ensemble 1": 48,
    "String ensemble 2": 49,
    "Synth strings 1": 50,
    "Synth strings 2": 51,
    # Brass
    "tpt": 56,
    "Trumpet": 56,
    "Trombone": 57,
    "Tuba": 58,
    "Muted trumpet": 59,
    "French horn": 60,
    "Brass section": 61,
    "Synth brass 1": 62,
    "Synth brass 2": 63,
    # Reed
    "sax": 65,
    "Soprano sax": 64,
    "Alto sax": 65,
    "Tenor sax": 66,
    "Baritone sax": 67,
    "Oboe": 68,
    "English horn": 69,
    "Bassoon": 70,
    "Clarinet": 71,
    # Pipe
    "flt": 73,
    "Flute": 73,
    "Piccolo": 72,
    "Recorder": 74,
    "Pan flute": 75,
    "Blown bottle": 76,
    "Shakuhachi": 77,
    "Whistle": 78,
    "Ocarina": 79,
    # Synth Lead
    "synlead": 81,
    "Synth lead": 81,
    "Lead 1 (square)": 80,
    "Lead 2 (sawtooth)": 81,
    "Lead 3 (calliope)": 82,
    "Lead 4 (chiff)": 83,
    "Lead 5 (charang)": 84,
    "Lead 6 (voice)": 85,
    "Lead 7 (fifths)": 86,
    "Lead 8 (bass + lead)": 87,
    # Synth Pad
    "synpad": 89,
    "Synth pad": 89,
    "Pad 1 (new age)": 88,
    "Pad 2 (warm)": 89,
    "Pad 3 (polysynth)": 90,
    "Pad 4 (choir)": 91,
    "Pad 5 (bowed)": 92,
    "Pad 6 (metallic)": 93,
    "Pad 7 (halo)": 94,
    "Pad 8 (sweep)": 95,
    # Synth FX
    "Fx 1 (rain)": 96,
    "Fx 2 (soundtrack)": 97,
    "Fx 3 (crystal)": 98,
    "Fx 4 (atmosphere)": 99,
    "Fx 5 (brightness)": 100,
    "Fx 6 (goblins)": 101,
    "Fx 7 (echoes)": 102,
    "Fx 8 (sci-fi)": 103,
    # Ethnic
    "Sitar": 104,
    "Banjo": 105,
    "Shamisen": 106,
    "Koto": 107,
    "Kalimba": 108,
    "Bag pipe": 109,
    "Fiddle": 110,
    "Shanai": 111,
    # Percussive
    "Tinkle bell": 112,
    "Agogo": 113,
    "Steel drums": 114,
    "Woodblock": 115, # Note: GM has High/Low on drum channel
    "Taiko drum": 116,
    "Melodic tom": 117,
    "Synth drum": 118,
    "Reverse cymbal": 119,
    # Sound Effects
    "Guitar fret noise": 120,
    "Breath noise": 121,
    "Seashore": 122,
    "Bird tweet": 123,
    "Telephone ring": 124,
    "Helicopter": 125,
    "Applause": 126,
    "Gunshot": 127,
    # Arp (Mapped to a synth sound)
    "arp": 81,  # Map Arp to Sawtooth Lead by default
    # Drums are a special case (channel 10 / index 9) - Program 0 is conventional
    "drs": 0,
    "Drums": 0,
    "808drums": 0,
    "Drumkit": 0,
}
# Set of lowercase names for checking INST if it's a drum track
DRUM_INSTRUMENT_NAMES = {
    k
    for k, v in INSTRUMENT_PROGRAM_MAP.items()
    if v == 0 and ("dr" in k or "kit" in k)
}

# Standard drum note map (MIDI channel 10 / index 9) - Keys MUST be lowercase for lookup (Expanded)
DRUM_PITCH_MAP = {
    # Bass Drum
    "kick": 36,
    "bd": 36,
    "bass drum 1": 36,
    "bass drum": 36,
    "acoustic bass drum": 35,
    "kick 2": 35,
    # Snare
    "snare": 38,
    "sd": 38,
    "acoustic snare": 38,
    "electric snare": 40,
    # Hi-Hat
    "hihatclosed": 42,
    "hhc": 42,
    "closed hi hat": 42,
    "closed hi-hat": 42,
    "hihatopen": 46,
    "hho": 46,
    "open hi hat": 46,
    "open hi-hat": 46,
    "hihatpedal": 44,
    "hhp": 44,
    "pedal hi hat": 44,
    "pedal hi-hat": 44,
    # Cymbals
    "crash": 49,
    "cr": 49,
    "crash cymbal 1": 49,
    "crash cymbal 2": 57,
    "ride": 51,
    "rd": 51,
    "ride cymbal 1": 51,
    "ride cymbal 2": 59,
    "ride bell": 53,
    "rb": 53,
    "splash cymbal": 55,
    "splash": 55,
    "chinese cymbal": 52,
    "chinese": 52,
    "reverse cymbal": 119, # Also available as effect
    # Toms
    "high tom": 50,
    "ht": 50,
    "hi tom": 50,
    "mid tom": 47,
    "mt": 47,
    "hi-mid tom": 48,
    "low-mid tom": 47,
    "low tom": 43,
    "lt": 43,
    "high floor tom": 43,
    "low floor tom": 41,
    "floor tom": 41,
    "ft": 41,
    # Hand Percussion
    "rimshot": 37,
    "rs": 37,
    "side stick": 37,
    "clap": 39,
    "cp": 39,
    "hand clap": 39,
    "cowbell": 56,
    "cb": 56,
    "tambourine": 54,
    "tmb": 54,
    "vibraslap": 58,
    "high bongo": 60,
    "low bongo": 61,
    "mute high conga": 62,
    "open high conga": 63,
    "low conga": 64,
    "high timbale": 65,
    "low timbale": 66,
    "high agogo": 67,
    "low agogo": 68,
    "cabasa": 69,
    "maracas": 70,
    "short whistle": 71,
    "long whistle": 72,
    "short guiro": 73,
    "long guiro": 74,
    "claves": 75,
    "cl": 75,
    "high wood block": 76,
    "low wood block": 77,
    "mute cuica": 78,
    "open cuica": 79,
    "mute triangle": 80,
    "open triangle": 81,
    "shaker": 82, # Common mapping
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
    )  # Allow 's' for sharp
    if not match:
        # print(f"Debug: Could not parse pitch name: '{pitch_name}'") # Too verbose
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
        # print(f"Warning: Invalid note name: '{note}'.") # Too verbose
        _pitch_parse_cache[pitch_name] = None
        return None

    # Normalize accidentals: 's' becomes '#'
    acc_norm = acc.upper() if acc else ""
    if acc_norm == "S":
        acc_norm = "#"

    acc_val = ACCIDENTAL_MAP.get(acc_norm, 0)
    midi_val = base_midi + acc_val + (octave + 1) * 12

    if 0 <= midi_val <= 127:
        _pitch_parse_cache[pitch_name] = midi_val
        return midi_val
    else:
        # print(f"Warning: Calculated MIDI pitch {midi_val} out of range (0-127) for '{pitch_name}'.")
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

        duration_map = {
            "W": 4.0,
            "H": 2.0,
            "Q": 1.0,
            "E": 0.5,
            "S": 0.25,
            "T": 0.125,
        }

        base_symbol = duration_symbol.replace(".", "")
        is_dotted = duration_symbol.endswith(".")

        relative_duration_quarters = duration_map.get(base_symbol)
        if relative_duration_quarters is None:
            print(
                f"Warning: Unknown duration symbol: '{duration_symbol}'. Defaulting to Quarter (1.0)."
            )
            relative_duration_quarters = 1.0

        # Duration is relative to the beat defined by tempo (quarter notes/min)
        # Time signature denominator is primarily for bar duration calculation.

        if is_dotted:
            relative_duration_quarters *= 1.5

        actual_duration_sec = relative_duration_quarters * quarter_note_duration_sec
        return actual_duration_sec

    except ValueError:
        print(f"Warning: Could not parse tempo '{tempo}' as float. Using default 120.")
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
    Analyze the following music description. Extract or infer the key signature (K:), tempo (T:), time signature (TS:), and suggest primary instrumentation (INST:).
    If any parameter is explicitly mentioned, use that. If not, infer plausible values based on the description (genre, mood, etc.).
    Also, identify the core mood and suggest a musical structure (like AABA, ABAC, Verse-Chorus-Bridge) if not already provided.
    Output the parameters clearly, preferably at the start, followed by a brief summary.

    Music Description: "{description}"

    Enriched Summary (Example Format: K:Amin T:70 TS:4/4 INST:Pno Mood: Melancholic, sparse. Structure: A-B-A-Coda):
    """
    enriched = call_gemini(prompt)

    # Update global config defaults based on enrichment
    current_key = CONFIG["default_key"]
    current_tempo = CONFIG["default_tempo"]
    current_timesig = CONFIG["default_timesig"]
    primary_instrument = "Pno"  # Default if not found
    structure_hint = "AABA"  # Default if not found

    if enriched:
        print(f"Enriched Description:\n{enriched}\n")
        # Use more robust regex to find parameters anywhere in the text
        key_match = re.search(
            r"[Kk](?:ey)?\s*:\s*([A-Ga-g][#sb]?(?:maj|min|dor|phr|lyd|mix|loc|aeo|ion)?)",
            enriched,
            re.IGNORECASE,
        )
        tempo_match = re.search(r"[Tt](?:empo)?\s*:\s*(\d+)", enriched)
        ts_match = re.search(
            r"[Tt](?:ime)?\s*[Ss](?:ig)?\s*:\s*(\d+)\s*/\s*(\d+)", enriched
        )
        inst_match = re.search(
            r"INST\s*:\s*(\w+)", enriched, re.IGNORECASE
        )  # Look for INST specifically if possible
        if not inst_match:
            # Look for common instrument names if INST: tag is missing
            inst_match = re.search(
                r"(?:instrument(?:s|ation)?|primary inst)\s*:\s*([A-Za-z]+(?:[ ][A-Za-z]+)*)",
                enriched,
                re.IGNORECASE,
            )
        struct_match = re.search(
            r"[Ss]tructure\s*:\s*([\w\-]+)", enriched, re.IGNORECASE
        )

        if key_match:
            current_key = key_match.group(1)
            # Validate key using pretty_midi later, just store the string for now
            print(f"Updated Default Key: {current_key}")
        if tempo_match:
            try:
                tempo_val = int(tempo_match.group(1))
                if 5 <= tempo_val <= 300:  # Plausible tempo range
                    current_tempo = tempo_val
                    print(f"Updated Default Tempo: {current_tempo}")
                else:
                    print(
                        f"Warning: Ignoring extracted tempo {tempo_val} (out of range 5-300)."
                    )
            except ValueError:
                pass
        if ts_match:
            try:
                ts_num, ts_den = int(ts_match.group(1)), int(ts_match.group(2))
                # Basic validation for common time signatures
                if ts_num > 0 and ts_den > 0 and (ts_den & (ts_den - 1) == 0 or ts_den == 1):  # Power of 2 or 1
                    current_timesig = (ts_num, ts_den)
                    print(
                        f"Updated Default Time Signature: {current_timesig[0]}/{current_timesig[1]}"
                    )
                else:
                    print(
                        f"Warning: Ignoring extracted time signature {ts_num}/{ts_den} (invalid denominator)."
                    )
            except ValueError:
                pass
        if inst_match:
            primary_instrument = inst_match.group(1).strip()
            print(f"Identified Primary Instrument Hint: {primary_instrument}")
        if struct_match:
            structure_hint = struct_match.group(1).upper()  # Standardize structure hint
            print(f"Identified Structure Hint: {structure_hint}")

        # Update CONFIG with potentially changed defaults for subsequent steps
        CONFIG["default_key"] = current_key
        CONFIG["default_tempo"] = current_tempo
        CONFIG["default_timesig"] = current_timesig

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

Generate ONLY the JSON plan now, starting with {{ and ending with }}. Do not include ```json markers.
"""

    plan_json = call_gemini(prompt, output_format="json")

    if plan_json and isinstance(plan_json, dict):
        # Validate the plan
        validated_plan = {}
        total_bars = 0
        # Attempt to preserve order if OrderedDict or Python 3.7+ dict
        section_order = list(plan_json.keys())

        for name in section_order:
            info = plan_json[name]
            if not isinstance(name, str) or not name.strip():
                print(f"Warning: Invalid section name '{name}'. Skipping.")
                continue

            section_name = name.strip()

            if (
                isinstance(info, dict)
                and "bars" in info
                and isinstance(info["bars"], int)
                and "goal" in info
                and isinstance(info["goal"], str)
                and info["goal"].strip()
                and CONFIG["min_section_bars"]
                <= info["bars"]
                <= CONFIG["max_section_bars"]
            ):
                if total_bars + info["bars"] <= CONFIG["max_total_bars"]:
                    # Ensure goal is not excessively long
                    info["goal"] = info["goal"].strip()[:200]  # Limit goal length
                    validated_plan[section_name] = info
                    total_bars += info["bars"]
                else:
                    print(
                        f"Warning: Section '{section_name}' ({info['bars']} bars) exceeds max total bars ({CONFIG['max_total_bars']}) when added to current total {total_bars}. Truncating plan."
                    )
                    is_valid = False
                    break  # Stop adding sections
            else:
                print(
                    f"Warning: Invalid format, bar count, or goal for section '{section_name}' in generated plan: {info}. Skipping."
                )
                is_valid = False
                # Continue validating other sections if possible

        if not validated_plan:
            print(
                "ERROR: Failed to generate a valid section plan from LLM response. Cannot proceed."
            )
            return None

        print("Generated Section Plan:")
        final_section_order = list(validated_plan.keys())
        for name in final_section_order:
            info = validated_plan[name]
            print(f"  - {name} ({info['bars']} bars): {info['goal']}")
        print(f"Total Bars in Plan: {total_bars}")
        # Return validated plan with potentially preserved order
        return {name: validated_plan[name] for name in final_section_order}
    else:
        print(
            "ERROR: Failed to generate or parse section plan JSON from LLM. Cannot proceed."
        )
        if isinstance(plan_json, str):  # Log if we got string instead of JSON
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
        f"--- Step 3: Generating Symbolic Section {section_name} (Starting Bar: {current_bar}) ---"
    )
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

    # Use the potentially updated defaults from CONFIG
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
2. If tempo (T), time signature (TS), key (K), or instrument (INST) need to be set or changed *at the very beginning* of this section (time = start of BAR:{current_bar}), include those commands *before* the `BAR:{current_bar}` marker. Otherwise, assume they carry over from the previous section or use defaults (T:{default_tempo}, TS:{default_timesig[0]}/{default_timesig[1]}, K:{default_key}). You can change INST multiple times within the section if needed.
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
        # Basic cleaning (already done in call_gemini, but good practice)
        symbolic_text = re.sub(
            r"^```[a-z]*\n", "", symbolic_text, flags=re.MULTILINE | re.IGNORECASE
        )
        symbolic_text = re.sub(r"\n```$", "", symbolic_text)
        symbolic_text = symbolic_text.strip()

        # Validate start and content
        lines = symbolic_text.split("\n")
        meaningful_lines = [
            line.strip() for line in lines if line.strip() and not line.strip().startswith("#")
        ]

        if not meaningful_lines:
            print(
                f"Warning: Generated text for {section_name} appears empty or only contains comments."
            )
            return "", None  # Treat as failure

        first_meaningful_line = meaningful_lines[0]
        bar_marker = f"BAR:{current_bar}"

        # Check if the expected BAR marker is present *somewhere*
        bar_marker_found = any(
            line.strip().startswith(bar_marker) for line in meaningful_lines
        )

        if not bar_marker_found:
            print(
                f"ERROR: Generated text for {section_name} does not contain the expected start marker '{bar_marker}'. Discarding section."
            )
            print(f"Received text (first 500 chars):\n{symbolic_text[:500]}...")
            return "", None

        # Find the index of the first valid starting line (T, TS, K, INST, or the correct BAR)
        start_index = -1
        for idx, line in enumerate(lines):
            line_content = line.strip()
            if not line_content or line_content.startswith("#"):
                continue
            if re.match(r"^(T:|TS:|K:|INST:)", line_content, re.IGNORECASE) or line_content.startswith(bar_marker):
                start_index = idx
                break
            else:
                # Found unexpected content before a valid start command or BAR marker
                print(
                    f"Warning: Generated text for {section_name} had unexpected content before first valid command or '{bar_marker}'. Trimming preamble."
                )
                # We will start from the first valid command or BAR marker found later
                break  # Stop searching here

        # If we didn't find a valid start line immediately, find the first occurrence
        if start_index == -1:
            for idx, line in enumerate(lines):
                line_content = line.strip()
                if re.match(r"^(T:|TS:|K:|INST:)", line_content, re.IGNORECASE) or line_content.startswith(bar_marker):
                    start_index = idx
                    print(
                        f"Adjusted start index for {section_name} to line {start_index + 1}."
                    )
                    break

        if start_index == -1:
            # Should be unreachable due to bar_marker_found check, but defensive coding
            print(
                f"ERROR: Could not find any valid starting line (T/TS/K/INST/BAR) in {section_name}. Discarding section."
            )
            return "", None

        # Reconstruct the text from the valid start index
        symbolic_text = "\n".join(lines[start_index:])

        print(
            f"Generated symbolic text for Section {section_name} (first 300 chars):\n{symbolic_text[:300]}...\n"
        )

        # Extract summary info (simple version, relies on parser for accuracy)
        # TODO: Could potentially parse the section here to get more accurate end state
        summary_info = {
            "name": section_name,
            "summary": f"Generated {bars} bars targeting goal: {goal}.",
            "key": default_key,  # Placeholder - actual key might change
            "tempo": default_tempo,  # Placeholder
            "time_sig": f"{default_timesig[0]}/{default_timesig[1]}",  # Placeholder
        }
        # Attempt to find last specified K, T, TS in this section for better context
        last_k = default_key
        last_t = default_tempo
        last_ts = f"{default_timesig[0]}/{default_timesig[1]}"
        for line in reversed(symbolic_text.split("\n")):
            line = line.strip()
            if line.startswith("K:"):
                last_k = line.split(":", 1)[1].strip()
                break  # Found most recent K
        for line in reversed(symbolic_text.split("\n")):
            line = line.strip()
            if line.startswith("T:"):
                try:
                    last_t = float(line.split(":", 1)[1].strip())
                    break
                except ValueError:
                    pass
        for line in reversed(symbolic_text.split("\n")):
            line = line.strip()
            if line.startswith("TS:"):
                last_ts = line.split(":", 1)[1].strip()
                break
        summary_info["key"] = last_k
        summary_info["tempo"] = last_t
        summary_info["time_sig"] = last_ts

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
    instrument_definitions = (
        {}
    )  # Key: (inst_name_lower, track_id), Value: {program, is_drum, name, orig_inst_name}

    # State variables
    current_track_times = (
        {}
    )  # Key: (inst_name_lower, track_id), Value: current time cursor for this track
    current_global_time = (
        0.0  # Tracks the latest event time across all tracks, adjusted by BAR markers
    )

    current_tempo = float(CONFIG["default_tempo"])
    current_ts_num, current_ts_den = CONFIG["default_timesig"]
    current_key = CONFIG["default_key"]
    active_instrument_name_orig = "Pno"  # Original case name
    active_instrument_name_lower = "pno"  # Lowercase for lookups
    active_instrument_is_drum = False

    current_bar_number = 0
    current_bar_start_time = 0.0  # Global time when the current bar started
    time_within_bar_per_track = (
        {}
    )  # Key: (inst_name_lower, track_id), Value: time elapsed within the current bar for this track
    expected_bar_duration_sec = (
        (60.0 / current_tempo) * current_ts_num * (4.0 / current_ts_den)
    )
    last_event_end_time = 0.0  # Tracks the absolute end time of the last note/rest

    initial_commands_set = {"T": False, "TS": False, "K": False}
    lines = symbolic_text.strip().split("\n")
    parse_start_line_index = 0

    # --- Pre-pass for initial settings (before first BAR marker) ---
    print("Processing initial settings (before first BAR marker)...")
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("BAR:"):
            parse_start_line_index = i
            break  # Stop pre-pass when BAR is encountered

        parts = line.split(":", 1)
        command = parts[0].upper()
        value = parts[1].strip() if len(parts) > 1 else ""
        ini_line_num = i + 1

        try:
            if command == "INST":
                if value:
                    active_instrument_name_orig = value
                    active_instrument_name_lower = value.lower()
                    active_instrument_is_drum = (
                        active_instrument_name_lower in DRUM_INSTRUMENT_NAMES
                    )
                    print(
                        f"Initial Instrument context set to '{active_instrument_name_orig}' (Is Drum: {active_instrument_is_drum})"
                    )
            elif command == "T" and not initial_commands_set["T"]:
                new_tempo = float(value)
                if new_tempo > 0:
                    current_tempo = new_tempo
                    tempo_changes.append((0.0, current_tempo))
                    expected_bar_duration_sec = (
                        (60.0 / current_tempo)
                        * current_ts_num
                        * (4.0 / current_ts_den)
                    )
                    initial_commands_set["T"] = True
                    print(f"Initial Tempo set to {current_tempo} BPM")
            elif command == "TS" and not initial_commands_set["TS"]:
                num_str, den_str = value.split("/")
                new_ts_num, new_ts_den = int(num_str), int(den_str)
                if (
                    new_ts_num > 0
                    and new_ts_den > 0 and (new_ts_den & (new_ts_den - 1) == 0 or new_ts_den == 1) # Power of 2 or 1
                    and (new_ts_num, new_ts_den) != (current_ts_num, current_ts_den)
                ):
                    current_ts_num, current_ts_den = new_ts_num, new_ts_den
                    time_signature_changes.append(
                        (0.0, current_ts_num, current_ts_den)
                    )
                    expected_bar_duration_sec = (
                        (60.0 / current_tempo)
                        * current_ts_num
                        * (4.0 / current_ts_den)
                    )
                    initial_commands_set["TS"] = True
                    print(
                        f"Initial Time Signature set to {current_ts_num}/{current_ts_den}"
                    )
            elif command == "K" and not initial_commands_set["K"]:
                if value:
                    # Validate key using pretty_midi later
                    current_key = value
                    key_signature_changes.append((0.0, current_key))
                    initial_commands_set["K"] = True
                    print(f"Initial Key set to {current_key}")

        except Exception as e:
            print(
                f"Error parsing initial setting line {ini_line_num}: '{line}' - {e}"
            )
        # Keep track of the line index to start main parsing after pre-pass
        parse_start_line_index = i + 1

    # Set defaults if not specified initially
    if not initial_commands_set["T"]:
        tempo_changes.append((0.0, current_tempo))
    if not initial_commands_set["TS"]:
        time_signature_changes.append((0.0, current_ts_num, current_ts_den))
    if not initial_commands_set["K"]:
        key_signature_changes.append((0.0, current_key))

    # --- Main Parsing Loop ---
    print(f"Parsing main body starting from line {parse_start_line_index + 1}...")
    for i in range(parse_start_line_index, len(lines)):
        current_line_num = i + 1
        line = lines[i].strip()
        if not line or line.startswith("#"):
            continue

        # Split command from value
        parts = line.split(":", 1)
        command = parts[0].upper()
        value = parts[1].strip() if len(parts) > 1 else ""

        try:
            if command == "INST":
                if value:
                    active_instrument_name_orig = value
                    active_instrument_name_lower = value.lower()
                    active_instrument_is_drum = (
                        active_instrument_name_lower in DRUM_INSTRUMENT_NAMES
                    )
                    # Don't print every INST change, can be verbose
            elif command == "T":
                new_tempo = float(value)
                if new_tempo > 0 and abs(new_tempo - current_tempo) > 1e-3:  # Avoid tiny float diffs
                    # Use current_global_time, which reflects bar starts
                    event_time = current_global_time
                    # Add only if time or value differs from the last entry
                    if (
                        not tempo_changes
                        or abs(tempo_changes[-1][0] - event_time) > 1e-6
                        or abs(tempo_changes[-1][1] - new_tempo) > 1e-3
                    ):
                        tempo_changes.append((event_time, new_tempo))
                        current_tempo = new_tempo
                        expected_bar_duration_sec = (
                            (60.0 / current_tempo)
                            * current_ts_num
                            * (4.0 / current_ts_den)
                        )
                        print(
                            f"Time {event_time:.3f}s (Bar ~{current_bar_number}): Tempo change to {new_tempo} BPM"
                        )
                    current_tempo = new_tempo  # Update state even if not added (e.g., same time)

            elif command == "TS":
                num_str, den_str = value.split("/")
                new_ts_num, new_ts_den = int(num_str), int(den_str)
                if (
                    new_ts_num > 0
                    and new_ts_den > 0 and (new_ts_den & (new_ts_den - 1) == 0 or new_ts_den == 1)
                    and (new_ts_num, new_ts_den) != (current_ts_num, current_ts_den)
                ):
                    event_time = current_global_time
                    if (
                        not time_signature_changes
                        or abs(time_signature_changes[-1][0] - event_time) > 1e-6
                        or (time_signature_changes[-1][1], time_signature_changes[-1][2])
                        != (new_ts_num, new_ts_den)
                    ):
                        time_signature_changes.append(
                            (event_time, new_ts_num, new_ts_den)
                        )
                        current_ts_num, current_ts_den = new_ts_num, new_ts_den
                        expected_bar_duration_sec = (
                            (60.0 / current_tempo)
                            * current_ts_num
                            * (4.0 / current_ts_den)
                        )
                        print(
                            f"Time {event_time:.3f}s (Bar ~{current_bar_number}): Time Sig change to {new_ts_num}/{new_ts_den}"
                        )
                    current_ts_num, current_ts_den = (
                        new_ts_num,
                        new_ts_den,
                    )  # Update state

            elif command == "K":
                if value and value != current_key:
                    event_time = current_global_time
                    if (
                        not key_signature_changes
                        or abs(key_signature_changes[-1][0] - event_time) > 1e-6
                        or key_signature_changes[-1][1] != value
                    ):
                        key_signature_changes.append((event_time, value))
                        current_key = value
                        print(
                            f"Time {event_time:.3f}s (Bar ~{current_bar_number}): Key change to {current_key}"
                        )
                    current_key = value  # Update state

            elif command == "BAR":
                bar_number = int(value)
                # Calculate expected start time of this new bar based on previous bar's end
                expected_new_bar_start_time = (
                    current_bar_start_time + expected_bar_duration_sec
                    if current_bar_number > 0
                    else 0.0
                )

                # Check timing accuracy of the previous bar before moving to the new one
                if current_bar_number > 0:
                    max_accumulated_time_in_prev_bar = 0.0
                    for (
                        track_key,
                        accumulated_time,
                    ) in time_within_bar_per_track.items():
                        max_accumulated_time_in_prev_bar = max(
                            max_accumulated_time_in_prev_bar, accumulated_time
                        )

                    # Allow a small tolerance for floating point comparisons
                    # Tolerance: 1% of bar duration or 5ms, whichever is larger
                    tolerance = max(0.005, expected_bar_duration_sec * 0.01)
                    duration_error = (
                        max_accumulated_time_in_prev_bar - expected_bar_duration_sec
                    )

                    # Check if ANY track significantly overran the bar
                    overran = duration_error > tolerance
                    # Check if ALL tracks significantly underran the bar (might be intentional silence at end)
                    all_underran = True
                    for (
                        track_key,
                        accumulated_time,
                    ) in time_within_bar_per_track.items():
                        if accumulated_time > expected_bar_duration_sec - tolerance:
                            all_underran = False
                            break

                    if overran:
                        print(
                            f"Warning: Bar {current_bar_number} timing mismatch on Line {current_line_num}. "
                            f"Expected duration {expected_bar_duration_sec:.3f}s, max accumulated {max_accumulated_time_in_prev_bar:.3f}s "
                            f"(Error: {duration_error:.3f}s > tolerance {tolerance:.3f}s). Forcing bar {bar_number} start to expected time {expected_new_bar_start_time:.3f}s."
                        )
                        # Force global time to the expected start time
                        current_global_time = expected_new_bar_start_time
                    elif (
                        all_underran and max_accumulated_time_in_prev_bar > 0
                    ):  # Avoid warning if bar was empty
                        print(
                            f"Warning: Bar {current_bar_number} timing potentially short on Line {current_line_num}. "
                            f"Expected duration {expected_bar_duration_sec:.3f}s, max accumulated {max_accumulated_time_in_prev_bar:.3f}s "
                            f"(Short by: {-duration_error:.3f}s > tolerance {tolerance:.3f}s). Setting bar {bar_number} start to expected time {expected_new_bar_start_time:.3f}s."
                        )
                        # Force global time to the expected start time
                        current_global_time = expected_new_bar_start_time
                    else:
                        # If within tolerance or correctly filled, use the expected time to avoid drift
                        current_global_time = expected_new_bar_start_time

                # Handle jumps in bar numbers (more than 1 bar increment)
                bars_jumped = bar_number - (current_bar_number + 1)
                if bars_jumped > 0 and current_bar_number > 0:
                    # Estimate jump duration based on current tempo/TS
                    jump_duration = bars_jumped * expected_bar_duration_sec
                    print(
                        f"Warning: Jump detected from Bar {current_bar_number} to {bar_number} (Line {current_line_num}). "
                        f"Advancing global time by ~{jump_duration:.3f}s ({bars_jumped} bars)."
                    )
                    current_global_time += jump_duration  # Add estimated time for skipped bars

                # Update bar state
                current_bar_number = bar_number
                current_bar_start_time = current_global_time
                # Reset time within the new bar for all tracks
                time_within_bar_per_track = {
                    key: 0.0 for key in time_within_bar_per_track
                }
                # Sync individual track cursors to the new bar start time
                current_track_times = {
                    key: current_bar_start_time for key in current_track_times
                }

            elif command in ["N", "C", "R"]:
                if current_bar_number == 0:
                    # This happens if events occur before BAR:1
                    if not any(line.strip().startswith("BAR:") for line in lines[:i]):
                        print(
                            f"Warning: Event '{line}' on Line {current_line_num} found before first BAR marker. Processing at time 0."
                        )
                        # Ensure bar state is initialized if events occur before BAR:1
                        current_bar_number = 1  # Assume Bar 1 implicitly starts
                        current_bar_start_time = 0.0
                        current_global_time = 0.0
                    else:
                        # This case should theoretically not happen if BAR handling is correct
                        print(
                            f"Internal Warning: Event processing while current_bar_number is 0 despite BAR markers existing. Line {current_line_num}: '{line}'"
                        )

                inst_name_for_event_lower = active_instrument_name_lower
                inst_name_for_event_orig = active_instrument_name_orig
                is_drum_track_for_event = active_instrument_is_drum
                inst_track_key = (
                    None  # Will be set below based on command (inst_name_lower, track_id)
                )

                event_duration_sec = 0.0
                event_start_time = 0.0

                # --- Parse Note (N) ---
                if command == "N":
                    # N:<Track>:<Pitch>:<Duration>:<Velocity>
                    data_parts = value.split(":")
                    if len(data_parts) < 4:
                        print(
                            f"Warning: Malformed N command on Line {current_line_num}: '{line}'. Requires 4 parts. Skipping."
                        )
                        continue

                    track_id = data_parts[0].strip()
                    pitch_name_raw = data_parts[1].strip()
                    duration_sym_raw = data_parts[2].strip()
                    velocity_str_raw = data_parts[3].strip()

                    if (
                        not track_id
                        or not pitch_name_raw
                        or not duration_sym_raw
                        or not velocity_str_raw
                    ):
                        print(
                            f"Warning: Empty part in N command on Line {current_line_num}: '{line}'. Skipping."
                        )
                        continue

                    # Clean potential comments from last part (velocity)
                    velocity_str = velocity_str_raw.split("#", 1)[0].strip()
                    duration_sym = duration_sym_raw  # Assume no comments in duration

                    inst_track_key = (inst_name_for_event_lower, track_id)

                    try:
                        velocity = int(velocity_str)
                        if not 0 <= velocity <= 127:
                            print(
                                f"Warning: Velocity {velocity} out of range (0-127) on Line {current_line_num}. Clamping."
                            )
                            velocity = max(0, min(127, velocity))
                    except ValueError:
                        velocity = 90  # Default velocity for notes
                        print(
                            f"Warning: Invalid velocity '{velocity_str_raw}' on Line {current_line_num}. Using {velocity}."
                        )

                    event_duration_sec = duration_to_seconds(
                        duration_sym, current_tempo, current_ts_den
                    )
                    midi_pitch = None

                    if is_drum_track_for_event:
                        pitch_name_lookup = pitch_name_raw.lower()
                        midi_pitch = DRUM_PITCH_MAP.get(pitch_name_lookup)
                        if midi_pitch is None:
                            # Try parsing as standard pitch if not a known drum name
                            standard_pitch = pitch_to_midi(pitch_name_raw)
                            if standard_pitch is not None:
                                print(
                                    f"Warning: Standard pitch notation '{pitch_name_raw}' used on drum track (INST:{inst_name_for_event_orig}) line {current_line_num}. Using MIDI pitch {standard_pitch}."
                                )
                                midi_pitch = standard_pitch
                            else:
                                print(
                                    f"Warning: Unknown drum sound '{pitch_name_raw}' on drum track (INST:{inst_name_for_event_orig}) line {current_line_num}. Using Kick (36)."
                                )
                                midi_pitch = 36  # Default to Kick
                    else:
                        midi_pitch = pitch_to_midi(pitch_name_raw)
                        if midi_pitch is None:
                            print(
                                f"Warning: Could not parse pitch name '{pitch_name_raw}' on non-drum track (INST:{inst_name_for_event_orig}) line {current_line_num}. Skipping note."
                            )
                            continue  # Skip this note if pitch is invalid

                    # Get current time for this specific track/instrument combination
                    # Start time is the later of the global bar start time or the track's last event end time
                    track_specific_start_offset = time_within_bar_per_track.get(
                        inst_track_key, 0.0
                    )
                    event_start_time = (
                        current_bar_start_time + track_specific_start_offset
                    )

                    note_event = {
                        "pitch": midi_pitch,
                        "start": event_start_time,
                        "end": event_start_time + event_duration_sec,
                        "velocity": velocity,
                    }
                    # Defer adding note until instrument is defined

                # --- Parse Chord (C) ---
                elif command == "C":
                    # C:<Track>:<[Pitches]>:<Duration>:<Velocity>
                    data_parts = value.split(":")
                    if len(data_parts) < 4:
                        print(
                            f"Warning: Malformed C command on Line {current_line_num}: '{line}'. Requires 4 parts. Skipping."
                        )
                        continue

                    track_id = data_parts[0].strip()
                    pitches_str_raw = data_parts[1].strip()
                    duration_sym_raw = data_parts[2].strip()
                    velocity_str_raw = data_parts[3].strip()

                    if (
                        not track_id
                        or not pitches_str_raw
                        or not duration_sym_raw
                        or not velocity_str_raw
                    ):
                        print(
                            f"Warning: Empty part in C command on Line {current_line_num}: '{line}'. Skipping."
                        )
                        continue

                    # Clean potential comments from last part (velocity)
                    velocity_str = velocity_str_raw.split("#", 1)[0].strip()
                    duration_sym = duration_sym_raw

                    inst_track_key = (inst_name_for_event_lower, track_id)

                    try:
                        velocity = int(velocity_str)
                        if not 0 <= velocity <= 127:
                            print(
                                f"Warning: Velocity {velocity} out of range (0-127) on Line {current_line_num}. Clamping."
                            )
                            velocity = max(0, min(127, velocity))
                    except ValueError:
                        velocity = 70  # Default velocity for chords
                        print(
                            f"Warning: Invalid velocity '{velocity_str_raw}' on Line {current_line_num}. Using {velocity}."
                        )

                    # Tolerate missing brackets but warn
                    if pitches_str_raw.startswith("[") and pitches_str_raw.endswith(
                        "]"
                    ):
                        pitches_str = pitches_str_raw[1:-1]
                    else:
                        print(
                            f"Warning: Chord pitches format incorrect on Line {current_line_num}: '{pitches_str_raw}'. Expected '[P1,P2,...]'. Attempting parse."
                        )
                        pitches_str = pitches_str_raw

                    pitch_names = [
                        p.strip() for p in pitches_str.split(",") if p.strip()
                    ]

                    if not pitch_names:
                        print(
                            f"Warning: No valid pitches found in Chord command on Line {current_line_num}: '{line}'. Skipping."
                        )
                        continue

                    event_duration_sec = duration_to_seconds(
                        duration_sym, current_tempo, current_ts_den
                    )

                    # Get current time for this specific track/instrument combination
                    track_specific_start_offset = time_within_bar_per_track.get(
                        inst_track_key, 0.0
                    )
                    event_start_time = (
                        current_bar_start_time + track_specific_start_offset
                    )

                    if is_drum_track_for_event:
                        print(
                            f"Info: Chord command 'C:' used for drum instrument '{inst_name_for_event_orig}' on Line {current_line_num}. Treating pitches as individual drum sounds."
                        )

                    # Create multiple note events for the chord
                    chord_notes = []
                    valid_pitches_in_chord = 0
                    for pitch_name_raw in pitch_names:
                        midi_pitch = None
                        if is_drum_track_for_event:
                            pitch_name_lookup = pitch_name_raw.lower()
                            midi_pitch = DRUM_PITCH_MAP.get(pitch_name_lookup)
                            if midi_pitch is None:
                                # Try parsing as standard pitch if not a known drum name
                                standard_pitch = pitch_to_midi(pitch_name_raw)
                                if standard_pitch is not None:
                                    print(
                                        f"Warning: Standard pitch notation '{pitch_name_raw}' used in chord on drum track (INST:{inst_name_for_event_orig}) line {current_line_num}. Using MIDI pitch {standard_pitch}."
                                    )
                                    midi_pitch = standard_pitch
                                else:
                                    print(
                                        f"Warning: Unknown drum sound '{pitch_name_raw}' in chord on drum track (INST:{inst_name_for_event_orig}) line {current_line_num}. Using Kick (36)."
                                    )
                                    midi_pitch = 36  # Default to Kick
                        else:
                            midi_pitch = pitch_to_midi(pitch_name_raw)
                            if midi_pitch is None:
                                print(
                                    f"Warning: Could not parse pitch '{pitch_name_raw}' in chord on non-drum track (INST:{inst_name_for_event_orig}) line {current_line_num}. Skipping this pitch."
                                )
                                continue  # Skip this specific pitch in the chord

                        note_event = {
                            "pitch": midi_pitch,
                            "start": event_start_time,
                            "end": event_start_time + event_duration_sec,
                            "velocity": velocity,
                        }
                        chord_notes.append(note_event)
                        valid_pitches_in_chord += 1

                    if valid_pitches_in_chord == 0:
                        print(
                            f"Warning: Chord command on line {current_line_num} resulted in no valid notes. Skipping chord."
                        )
                        continue  # Skip advancing time if chord was empty
                    # Defer adding notes until instrument defined

                # --- Parse Rest (R) ---
                elif command == "R":
                    # R:<Track>:<Duration>
                    data_parts = value.split(":", 1)  # Split into Track and Duration
                    if len(data_parts) < 2:
                        print(
                            f"Warning: Malformed R command on Line {current_line_num}: '{line}'. Requires 2 parts. Skipping."
                        )
                        continue

                    track_id = data_parts[0].strip()
                    duration_sym_raw = data_parts[1].strip()

                    if not track_id or not duration_sym_raw:
                        print(
                            f"Warning: Empty part in R command on Line {current_line_num}: '{line}'. Skipping."
                        )
                        continue

                    # Clean potential comments from duration part
                    duration_sym = duration_sym_raw.split("#", 1)[0].strip()

                    if not duration_sym:
                        print(
                            f"Warning: Empty duration for R command on Line {current_line_num}: '{line}'. Skipping."
                        )
                        continue

                    inst_track_key = (inst_name_for_event_lower, track_id)
                    event_duration_sec = duration_to_seconds(
                        duration_sym, current_tempo, current_ts_den
                    )

                    # Get current time for this specific track/instrument combination
                    track_specific_start_offset = time_within_bar_per_track.get(
                        inst_track_key, 0.0
                    )
                    event_start_time = (
                        current_bar_start_time + track_specific_start_offset
                    )
                    # Rests don't generate notes, just advance time

                # --- Post-Event Processing (Common to N, C, R) ---
                if inst_track_key:
                    # Define instrument if not seen before
                    if inst_track_key not in instrument_definitions:
                        # Use original case instrument name for the pretty_midi name
                        pm_instrument_name = (
                            f"{inst_name_for_event_orig}-{inst_track_key[1]}"
                        )
                        # Lookup program using lowercase name
                        program = INSTRUMENT_PROGRAM_MAP.get(
                            inst_track_key[0], 0
                        )  # Default to Piano if unknown
                        is_drum = inst_track_key[0] in DRUM_INSTRUMENT_NAMES

                        # Override program to 0 if it's a drum track, as per GM convention
                        if is_drum:
                            program = 0  # Program for drums doesn't matter much, channel does

                        instrument_definitions[inst_track_key] = {
                            "program": program,
                            "is_drum": is_drum,
                            "name": pm_instrument_name,
                            "orig_inst_name": inst_name_for_event_orig,  # Store original name too
                        }
                        print(
                            f"Defined instrument: {pm_instrument_name} (Program: {program}, IsDrum: {is_drum})"
                        )
                        # Initialize time and note list for new instrument/track
                        # Start time should be the global bar start + any offset already accumulated in this bar
                        initial_track_offset = time_within_bar_per_track.get(
                            inst_track_key, 0.0
                        )
                        current_track_times[inst_track_key] = (
                            current_bar_start_time + initial_track_offset
                        )
                        time_within_bar_per_track[inst_track_key] = initial_track_offset
                        notes_by_instrument_track[inst_track_key] = []

                    # Add parsed notes (if any) to the correct list
                    if (
                        command == "N"
                        and "note_event" in locals()
                        and note_event is not None
                    ):
                        notes_by_instrument_track[inst_track_key].append(note_event)
                        # Clear note_event to avoid adding duplicates if next line is invalid
                        del note_event
                    elif (
                        command == "C" and "chord_notes" in locals() and chord_notes
                    ):
                        notes_by_instrument_track[inst_track_key].extend(chord_notes)
                        # Clear chord_notes
                        del chord_notes

                    # Advance time for this specific track within the current bar context
                    new_track_time_absolute = event_start_time + event_duration_sec
                    current_track_times[inst_track_key] = new_track_time_absolute
                    time_within_bar_per_track[inst_track_key] = (
                        new_track_time_absolute - current_bar_start_time
                    )

                    # Update the global last event time marker
                    last_event_end_time = max(
                        last_event_end_time, new_track_time_absolute
                    )
                    # `current_global_time` is advanced primarily by BAR markers for synchronization.

            else:
                print(
                    f"Warning: Unknown command '{command}' on line {current_line_num}: '{line}'. Skipping."
                )

        except Exception as e:
            print(f"FATAL Error parsing line {current_line_num}: '{line}' - {e}")
            traceback.print_exc()
            # Decide whether to continue or stop on fatal errors
            # For robustness, try to continue unless it's clearly unrecoverable
            # continue

    print(
        f"Symbolic text parsing complete. Estimated total duration: {last_event_end_time:.3f} seconds."
    )
    # Sanity check: Ensure all defined instruments actually have notes
    final_instrument_defs = {}
    final_notes_data = {}
    for key, definition in instrument_definitions.items():
        # key is (inst_name_lower, track_id)
        if key in notes_by_instrument_track and notes_by_instrument_track[key]:
            final_instrument_defs[key] = definition
            final_notes_data[key] = notes_by_instrument_track[key]
        else:
            print(
                f"Info: Instrument '{definition['name']}' defined but had no notes parsed. Excluding from MIDI."
            )

    # Return final state along with parsed data
    return (
        final_notes_data,
        final_instrument_defs,
        tempo_changes,
        time_signature_changes,
        key_signature_changes,
        last_event_end_time,
        current_key,  # Return the final key state
        current_tempo,  # Return the final tempo state
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
        # Use the first tempo change as the initial tempo
        initial_tempo = (
            tempo_changes[0][1] if tempo_changes else CONFIG["default_tempo"]
        )
        midi_obj = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo)
        print(f"Initialized MIDI with tempo: {initial_tempo:.2f} BPM")

        # --- Apply Meta-Messages (Tempo, Time Sig, Key Sig) ---
        # Remove duplicate times for the same event type before applying

        # Tempo Changes (pretty_midi handles initial tempo via constructor)
        if len(tempo_changes) > 0:
            print("Applying tempo changes...")
            # Filter out redundant changes at the same time, keeping the last one
            unique_tempo_times = {}
            for t, bpm in tempo_changes:
                unique_tempo_times[round(t, 6)] = bpm  # Round time slightly for comparison

            # Sort by time
            sorted_tempo_times = sorted(unique_tempo_times.keys())

            # Get times and corresponding BPMs, skipping time 0.0 if initial tempo matches
            final_tempo_times = []
            final_tempo_bpm = []
            initial_tempo_set_by_event = False
            if 0.0 in unique_tempo_times:
                if abs(unique_tempo_times[0.0] - initial_tempo) < 1e-3:
                    initial_tempo_set_by_event = True
                else:
                    # If tempo at 0.0 differs from constructor, warn and override?
                    # pretty_midi seems to handle this; the constructor sets the first tempo.
                    # Let's trust pretty_midi's internal handling.
                    pass

            for t in sorted_tempo_times:
                # Skip adding tempo at time 0 if it was already set by constructor
                if t == 0.0 and initial_tempo_set_by_event:
                    continue
                if t >= 0:  # Ensure non-negative time
                    final_tempo_times.append(t)
                    final_tempo_bpm.append(unique_tempo_times[t])

            if final_tempo_times:
                # pretty_midi expects tempo in microseconds per quarter note (MPQN)
                tempo_change_mpq = [
                    pretty_midi.bpm_to_tempo(bpm) for bpm in final_tempo_bpm
                ]
                # Use the internal method to add tempo changes after initialization
                midi_obj._load_tempo_changes(final_tempo_times, tempo_change_mpq)
                print(
                    f"Applied {len(final_tempo_times)} tempo change events (excluding initial)."
                )
            else:
                print("No additional tempo changes applied.")

        # Time Signature Changes
        time_sig_changes.sort(key=lambda x: x[0])  # Sort by time
        unique_ts = {}
        # Keep the *last* definition at each specific time
        for time, num, den in time_sig_changes:
            # Denominator must be power of 2 (or 1) for standard MIDI
            actual_den = den
            if den <= 0 or (den & (den - 1) != 0 and den != 1):
                # Find closest power of 2
                if den <= 0:
                    actual_den = 4
                else:
                    actual_den = 2 ** math.ceil(math.log2(den))
                print(
                    f"Warning: Invalid time signature denominator {den} at time {time:.3f}s. Using closest power of 2: {actual_den}."
                )

            # Use rounded time as key to merge close events
            unique_ts[round(time, 6)] = (num, actual_den)

        midi_obj.time_signature_changes = []  # Clear default if any
        applied_ts_count = 0
        last_ts = None
        # Add sorted, unique time signatures
        for time in sorted(unique_ts.keys()):
            if time >= 0:
                num, den = unique_ts[time]
                # Avoid adding duplicate consecutive time signatures
                if last_ts != (num, den):
                    ts_change = pretty_midi.TimeSignature(num, den, time)
                    midi_obj.time_signature_changes.append(ts_change)
                    applied_ts_count += 1
                    last_ts = (num, den)

        if applied_ts_count > 0:
            print(f"Applied {applied_ts_count} unique time signature changes.")
        # Ensure at least one time signature exists (usually at time 0)
        if not midi_obj.time_signature_changes:
            default_num, default_den = CONFIG["default_timesig"]
            midi_obj.time_signature_changes.append(
                pretty_midi.TimeSignature(default_num, default_den, 0.0)
            )
            print(f"Applied default time signature: {default_num}/{default_den}")

        # Key Signature Changes
        key_sig_changes.sort(key=lambda x: x[0])  # Sort by time
        unique_ks = {}
        last_valid_key_name = CONFIG["default_key"]
        # Keep the *last* definition at each specific time
        for time, key_name in key_sig_changes:
            # Use rounded time as key
            unique_ks[round(time, 6)] = key_name

        midi_obj.key_signature_changes = []  # Clear default if any
        applied_key_count = 0
        last_key_number = None
        # Add sorted, unique key signatures
        for time in sorted(unique_ks.keys()):
            if time >= 0:
                key_name = unique_ks[time]
                try:
                    key_number = pretty_midi.key_name_to_key_number(key_name)
                    # Avoid adding duplicate consecutive key signatures
                    if key_number != last_key_number:
                        key_change = pretty_midi.KeySignature(
                            key_number=key_number, time=time
                        )
                        midi_obj.key_signature_changes.append(key_change)
                        applied_key_count += 1
                        last_key_number = key_number
                        last_valid_key_name = key_name  # Track last successfully applied name
                except ValueError as e:
                    print(
                        f"Warning: Could not parse key signature '{key_name}' at time {time:.3f}s. Skipping. Error: {e}"
                    )

        if applied_key_count > 0:
            print(f"Applied {applied_key_count} unique key signature changes.")
        # Ensure at least one key signature exists
        if not midi_obj.key_signature_changes:
            try:
                # Use the last valid key name seen during parsing, or the config default
                final_default_key = (
                    last_valid_key_name
                    if last_key_number is not None
                    else CONFIG["default_key"]
                )
                default_key_num = pretty_midi.key_name_to_key_number(
                    final_default_key
                )
                midi_obj.key_signature_changes.append(
                    pretty_midi.KeySignature(key_number=default_key_num, time=0.0)
                )
                print(f"Applied default key signature: {final_default_key}")
            except ValueError as e:
                print(
                    f"Warning: Could not parse default key signature '{final_default_key}'. No key signature applied. Error: {e}"
                )

        # --- Create instruments and add notes ---
        available_channels = list(range(16))
        drum_channel = 9  # Standard GM drum channel
        if drum_channel in available_channels:
            available_channels.remove(drum_channel)  # Reserve channel 9 for drums
        channel_index = 0  # Index into available_channels for non-drum tracks

        # Sort definitions to ensure consistent channel assignment if possible
        # Key: (inst_name_lower, track_id)
        sorted_inst_keys = sorted(instrument_defs.keys())

        assigned_channels = {}  # Track assigned channel for each instrument object

        for inst_track_key in sorted_inst_keys:
            definition = instrument_defs[inst_track_key]
            # inst_track_key is (inst_name_lower, track_id)
            # definition is {program, is_drum, name, orig_inst_name}

            if not notes_data.get(
                inst_track_key
            ):  # Should have been filtered by parser, but double check
                print(
                    f"Skipping instrument '{definition['name']}' as it has no parsed notes."
                )
                continue

            is_drum = definition["is_drum"]
            program = definition["program"]
            pm_instrument_name = definition["name"]

            # Assign channel
            channel = None
            if is_drum:
                channel = drum_channel
            else:
                if not available_channels:
                    print(
                        f"ERROR: No available non-drum MIDI channels left for instrument '{pm_instrument_name}'. Skipping."
                    )
                    continue  # Skip this instrument if no channels left

                # Cycle through available channels if we run out
                channel = available_channels[channel_index % len(available_channels)]
                if channel_index >= len(available_channels):
                    print(
                        f"Warning: Ran out of unique non-drum MIDI channels! Reusing channel {channel} for {pm_instrument_name}."
                    )
                channel_index += 1

            # Create the instrument object
            instrument_obj = pretty_midi.Instrument(
                program=program, is_drum=is_drum, name=pm_instrument_name
            )
            # Add instrument to the MIDI object *before* adding notes
            # pretty_midi assigns channel automatically when adding the instrument
            midi_obj.instruments.append(instrument_obj)
            # Store the assigned channel (pretty_midi doesn't expose it easily after adding)
            assigned_channels[id(instrument_obj)] = channel

            print(
                f"Created MIDI instrument: {pm_instrument_name} (Program: {program}, IsDrum: {is_drum}, Target Channel: {channel})"
            )

            # Add notes to the instrument object
            note_count = 0
            skipped_notes = 0
            for note_info in notes_data[inst_track_key]:
                start_time = max(0.0, note_info["start"])
                # Ensure minimum duration to avoid zero-length notes in MIDI
                min_duration = 0.001  # 1 millisecond seems safe
                end_time = max(start_time + min_duration, note_info["end"])

                # Clamp velocity to valid MIDI range (1-127 for NoteOn, 0 is NoteOff)
                velocity = max(1, min(127, int(note_info["velocity"])))
                # Clamp pitch to valid MIDI range
                pitch = max(0, min(127, int(note_info["pitch"])))

                # Additional check for duration validity
                if (
                    end_time - start_time < min_duration / 2
                ):  # Check if duration is effectively zero
                    print(
                        f"Warning: Skipping note for '{pm_instrument_name}' with near-zero duration (Start: {start_time:.4f}, End: {end_time:.4f}, Pitch: {pitch})."
                    )
                    skipped_notes += 1
                    continue

                try:
                    note = pretty_midi.Note(
                        velocity=velocity, pitch=pitch, start=start_time, end=end_time,
                    )
                    instrument_obj.notes.append(note)
                    note_count += 1
                except ValueError as e:
                    # This might catch issues if times are invalid (e.g., negative)
                    print(
                        f"Error creating pretty_midi.Note for {pm_instrument_name}: {e}. Skipping note."
                    )
                    print(
                        f"  Note data: Pitch={pitch}, Vel={velocity}, Start={start_time:.4f}, End={end_time:.4f}"
                    )
                    skipped_notes += 1

            if note_count == 0 and skipped_notes == 0:
                print(f"  Warning: Instrument '{pm_instrument_name}' had no notes added.")
            else:
                print(
                    f"  Added {note_count} notes. ({skipped_notes} notes skipped due to errors/duration)."
                )

        # Ensure output directory exists
        os.makedirs(CONFIG["output_dir"], exist_ok=True)
        full_output_path = os.path.join(CONFIG["output_dir"], filename)

        # Write the MIDI file
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

    # Step 1: Enrich the initial description and get structure hint
    enriched_description, structure_hint = enrich_music_description(
        CONFIG["initial_description"]
    )
    if not enriched_description:
        enriched_description = CONFIG["initial_description"]  # Fallback

    # Step 2: Generate the section plan
    section_plan = generate_section_plan(enriched_description, structure_hint)
    if not section_plan:
        print("Exiting due to failure in generating section plan.")
        exit(1)

    # Step 3 (Section Generation) & Concatenation
    all_symbolic_text = ""
    current_bar_count = 1
    last_section_summary_info = None
    generated_sections_count = 0
    total_planned_bars = sum(info.get("bars", 0) for info in section_plan.values())

    print(
        f"\n--- Step 3: Generating {len(section_plan)} Sections ({total_planned_bars} planned bars) ---"
    )
    section_order = list(section_plan.keys())  # Get the order
    for section_name in section_order:
        section_info = section_plan[section_name]
        # Basic validation of section_info before passing
        if not isinstance(section_info.get("bars"), int) or section_info["bars"] <= 0:
            print(
                f"ERROR: Invalid 'bars' value ({section_info.get('bars')}) for section {section_name}. Skipping."
            )
            continue  # Skip this section

        symbolic_section, current_section_summary_info = generate_symbolic_section(
            enriched_description,
            section_plan,  # Pass the whole plan for context
            section_name,
            current_bar_count,
            last_section_summary_info,
        )

        if symbolic_section is not None and current_section_summary_info is not None:
            # Add newline if missing for cleaner concatenation between sections
            symbolic_section_cleaned = symbolic_section.strip()
            if symbolic_section_cleaned:  # Avoid adding empty sections
                all_symbolic_text += symbolic_section_cleaned + "\n"
                generated_sections_count += 1
                last_section_summary_info = current_section_summary_info
                current_bar_count += section_info["bars"]  # Advance bar counter
            else:
                print(
                    f"Warning: Section {section_name} generated empty symbolic text. Skipping concatenation."
                )
                # Don't advance bar counter if section was empty/invalid

        else:
            print(
                f"Failed to generate or validate section {section_name}. Stopping generation."
            )
            # Option: decide whether to proceed with partial generation or stop
            # break # Stop the whole process if a section fails critically
            print("Attempting to proceed with previously generated sections...")
            break  # For now, stop on critical failure

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
    print("------------------------------------")

    # Step 4: Parse the combined symbolic text
    _pitch_parse_cache.clear()  # Clear cache before parsing
    (
        parsed_notes,
        instrument_definitions,
        tempo_changes,
        time_sig_changes,
        key_sig_changes,
        estimated_duration,
        final_key,
        final_tempo,
    ) = parse_symbolic_to_structured_data(all_symbolic_text)

    # Step 5: Create the MIDI file
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
        print(
            "\nError: No valid notes or instruments were successfully parsed from the symbolic text. MIDI file not created."
        )

    print("\n--- MidiMaker Generator Pipeline Finished ---")