# -*- coding: utf-8 -*-
"""
Generates MIDI music from textual descriptions using LLMs and a compact symbolic format.

This script implements a pipeline:
1. Enrich a basic music description using an LLM.
2. Generate symbolic music notation section by section using an LLM.
3. Concatenate the symbolic sections.
4. Parse the symbolic text into structured data.
5. Convert the structured data into a MIDI file using pretty_midi.
"""

import datetime
import os
import re
import time
import traceback
import math

import google.generativeai as genai
import pretty_midi

# --- Configuration ---
# IMPORTANT: Use environment variables or secrets management for API keys in real applications!
# Set the API_KEY environment variable or replace the placeholder below.
API_KEY = os.environ.get("API_KEY", "AIzaSyC5jbwgP050qfurqK9GyvgrUYvpwEy0n8s") # Placeholder - Replace or use env var

# Configure the Gemini model to use
GEMINI_MODEL = "gemini-2.5-pro-exp-03-25" #"gemini-2.0-flash-thinking-exp-01-21" 

# Initial high-level description of the music
INITIAL_DESCRIPTION = (
    "trans electronic hiphop cosmic ambient music fast 160 bpm optimistic"
)

# Define the structure and goals for each section
# TODO: Generate section descriptions by LLM as well, do not use hardcoded ones
SECTIONS = {
    "A1": {
        "bars": 8,
        "goal": "Establish the main theme",
    },
    "B": {
        "bars": 16,
        "goal": "Introduce a slight variation or counter-melody, Slightly more active rhythm.",
    },
    "A2": {
        "bars": 8,
        "goal": "Return to the main theme, similar to A1 but maybe with slight embellishment or a fuller chord in the left hand. End conclusively for looping.",
    },
    "C": {
        "bars": 8,
        "goal": "Final and conclusive section, possibly a coda or outro. Use a different instrument or sound to signal the end.",
    },
}

# Default musical parameters (can be overridden by enriched description or symbolic commands)
DEFAULT_TEMPO = 120
DEFAULT_TIMESIG = (4, 4)
DEFAULT_KEY = "Cmin"
OUTPUT_DIR = "output"  # Directory to save generated MIDI and symbolic text files

# --- Symbolic Format Definition (for prompts and parsing) ---
SYMBOLIC_FORMAT_DEFINITION = """
Use this compact symbolic format ONLY:
- `INST:<InstrumentName>` (Instrument: e.g., Pno, Gtr, Bass, Drs, Str, Flt, Tpt, SynPad, SynLead)
- `T:<BPM>` (Tempo: e.g., T:120)
- `TS:<N>/<D>` (Time Signature: e.g., TS:4/4)
- `K:<Key>` (Key Signature: e.g., K:Cmin, K:Gmaj, K:Ddor)
- `BAR:<Num>` (Bar marker, starting from 1)
- `N:<Track>:<Pitch>:<Duration>:<Velocity>` (Note: TrackID: PitchName: DurationSymbol: Velocity[0-127])
- `C:<Track>:<Pitches>:<Duration>:<Velocity>` (Chord: TrackID: [Pitch1,Pitch2,...]: DurationSymbol: Velocity)
- `R:<Track>:<Duration>` (Rest: TrackID: DurationSymbol)

TrackIDs: Use simple names like RH (Right Hand), LH (Left Hand), Melody, Bass, Drums, Tr1, Tr2, PnoLH, PnoRH, SynPad, SynLead etc.
PitchNames: Use standard notation (e.g., C4, F#5, Gb3). Middle C is C4. For Drums, use names like Kick, Snare, HHC, HHO, Crash.
DurationSymbols: W (Whole), H (Half), Q (Quarter), E (Eighth), S (Sixteenth). '.' can be appended for dotted notes (e.g., Q.).
Velocity: MIDI velocity (0-127).

Example Note: N:RH:C5:H:70
Example Chord: C:LH:[C3,Eb3,G3]:W:60
Example Rest: R:RH:Q
Example Drums: N:Drums:Kick:Q:90
Example Multi-Instrument:
INST:Pno
BAR:1
N:PnoLH:C3:Q:60
N:PnoRH:G4:Q:70
...
INST:SynPad
BAR:5
C:PadTrk:[D3,F3,A3]:W:50
...
"""

# --- Helper Functions ---


def configure_genai():
    """Configures the Google Generative AI library."""
    if not API_KEY:
        print(
            "ERROR: API_KEY environment variable is not set."
            " Please set the API_KEY environment variable."
        )
        exit(1)
    try:
        genai.configure(api_key=API_KEY)
        print(f"Google Generative AI configured using model: {GEMINI_MODEL}")
    except Exception as e:
        print(f"Error configuring Generative AI: {e}")
        print("Please ensure your API_KEY is set correctly and valid.")
        exit(1)


def call_gemini(prompt, retries=3, delay=5):
    """
    Calls the Gemini API with the specified prompt and handles retries.

    Args:
        prompt (str): The prompt to send to the LLM.
        retries (int): Maximum number of retry attempts.
        delay (int): Delay in seconds between retries.

    Returns:
        str or None: The generated text content, or None if generation failed after retries.
    """
    model = genai.GenerativeModel(GEMINI_MODEL)
    for attempt in range(retries):
        try:
            # generation_config = genai.types.GenerationConfig(
            #     # candidate_count=1, # Already default
            #     # stop_sequences=["\n"], # Example stop sequence
            #     # max_output_tokens=8192, # Set max tokens if needed
            #     temperature=0.7 # Adjust creativity vs coherence
            # )
            # response = model.generate_content(prompt, generation_config=generation_config)
            response = model.generate_content(prompt)

            # Handle potential API response variations
            if response.parts:
                return response.text.strip()

            # Handle blocked prompts or safety issues
            if (
                hasattr(response, "prompt_feedback")
                and response.prompt_feedback.block_reason
            ):
                print(
                    f"Prompt blocked (Attempt {attempt + 1}):"
                    f" {response.prompt_feedback.block_reason}"
                )
                return None  # Retrying won't help

            # Handle other finish reasons (e.g., MAX_TOKENS, SAFETY)
            if (
                hasattr(response, "candidates")
                and response.candidates
                and response.candidates[0].finish_reason != "STOP"
            ):
                print(
                    f"Generation stopped for reason: {response.candidates[0].finish_reason}"
                )
                # Try to return partial content if available
                if (
                    response.candidates[0].content
                    and response.candidates[0].content.parts
                ):
                    print("Returning partial content due to non-STOP finish reason.")
                    return response.candidates[0].content.parts[0].text.strip()
                return None  # Indicate non-standard stop without content

            # Check candidates directly if parts is empty but no explicit block/error
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    print(
                        "Accessing content via candidates[0] as response.parts was empty."
                    )
                    return candidate.content.parts[0].text.strip()

            print(
                f"Warning: Received empty or unexpected response structure from Gemini (Attempt {attempt + 1})."
            )
            # Allow retries

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
ACCIDENTAL_MAP = {"#": 1, "b": -1, "": 0}

# General MIDI Instrument Program Numbers (Selected)
# See https://en.wikipedia.org/wiki/General_MIDI#Program_change_events
INSTRUMENT_PROGRAM_MAP = {
    # Piano
    "Pno": 0, "Piano": 0, "Acoustic Grand Piano": 0,
    "Bright Acoustic Piano": 1, "Electric Grand Piano": 2, "Honky-tonk Piano": 3,
    "Electric Piano 1": 4, "Rhodes Piano": 4, "Electric Piano 2": 5,
    # Chromatic Percussion
    "Celesta": 8, "Glockenspiel": 9, "Music Box": 10, "Vibraphone": 11,
    # Organ
    "Org": 16, "Organ": 16, "Drawbar Organ": 16, "Percussive Organ": 17, "Rock Organ": 18,
    # Guitar
    "Gtr": 25, "Acoustic Guitar": 25, "Nylon Guitar": 24, "Steel Guitar": 25,
    "Electric Guitar": 27, "Jazz Guitar": 26, "Clean Electric Guitar": 27,
    "Muted Electric Guitar": 28, "Overdriven Guitar": 29, "Distortion Guitar": 30,
    # Bass
    "Bass": 33, "Acoustic Bass": 32, "Electric Bass": 33, "Finger Bass": 33,
    "Pick Bass": 34, "Fretless Bass": 35, "Slap Bass": 36, "Synth Bass": 38, "Synth Bass 2": 39,
    # Strings
    "Str": 48, "Strings": 48, "Violin": 40, "Viola": 41, "Cello": 42, "Contrabass": 43,
    "Tremolo Strings": 44, "Pizzicato Strings": 45, "Orchestral Harp": 46,
    "String Ensemble 1": 48, "String Ensemble 2": 49, "Synth Strings 1": 50, "Synth Strings 2": 51,
    # Brass
    "Tpt": 56, "Trumpet": 56, "Trombone": 57, "Tuba": 58, "Muted Trumpet": 59, "French Horn": 60,
    # Reed
    "Sax": 65, "Soprano Sax": 64, "Alto Sax": 65, "Tenor Sax": 66, "Baritone Sax": 67,
    "Oboe": 68, "English Horn": 69, "Bassoon": 70, "Clarinet": 71,
    # Pipe
    "Flt": 73, "Flute": 73, "Piccolo": 72, "Recorder": 74, "Pan Flute": 75,
    # Synth Lead
    "SynLead": 81, "Synth Lead": 81, "Lead 1 (Square)": 80, "Lead 2 (Sawtooth)": 81,
    "Lead 3 (Calliope)": 82, "Lead 8 (Bass + Lead)": 87,
    # Synth Pad
    "SynPad": 89, "Synth Pad": 89, "Pad 1 (New Age)": 88, "Pad 2 (Warm)": 89,
    "Pad 3 (Polysynth)": 90, "Pad 4 (Choir)": 91, "Pad 5 (Bowed)": 92,
    # Drums are a special case (channel 10 / index 9)
    "Drs": 0, "Drums": 0, # Program 0 is often used, but channel is key
}

# Standard drum note map (MIDI channel 10 / index 9) - General MIDI Level 1 Percussion Key Map
DRUM_PITCH_MAP = {
    # Bass Drum
    "Kick": 36, "BD": 36, "Bass Drum 1": 36, "Acoustic Bass Drum": 35,
    # Snare
    "Snare": 38, "SD": 38, "Acoustic Snare": 38, "Electric Snare": 40,
    # Hi-Hat
    "HiHatClosed": 42, "HHC": 42, "Closed Hi Hat": 42, "Closed Hi-Hat": 42,
    "HiHatOpen": 46, "HHO": 46, "Open Hi Hat": 46, "Open Hi-Hat": 46,
    "HiHatPedal": 44, "HH P": 44, "Pedal Hi Hat": 44, "Pedal Hi-Hat": 44,
    # Cymbals
    "Crash": 49, "CR": 49, "Crash Cymbal 1": 49, "Crash Cymbal 2": 57,
    "Ride": 51, "RD": 51, "Ride Cymbal 1": 51, "Ride Cymbal 2": 59, "Ride Bell": 53,
    "Splash Cymbal": 55, "Chinese Cymbal": 52,
    # Toms
    "High Tom": 50, "HT": 50, "Hi Tom": 50,
    "Mid Tom": 47, "MT": 47, "Hi-Mid Tom": 48, "Low-Mid Tom": 47,
    "Low Tom": 43, "LT": 43, "High Floor Tom": 43, "Low Floor Tom": 41,
    "Floor Tom": 41, "FT": 41,
    # Other
    "Rimshot": 37, "RS": 37, "Side Stick": 37,
    "Clap": 39, "CP": 39, "Hand Clap": 39,
    "Cowbell": 56, "CB": 56,
    "Tambourine": 54, "Tmb": 54,
    "Claves": 75, "Wood Block": 76, "High Wood Block": 76, "Low Wood Block": 77,
}


def pitch_to_midi(pitch_name):
    """Converts pitch name (e.g., C4, F#5) to MIDI number."""
    pitch_name = pitch_name.strip()
    # Regex allows for optional accidental and multi-digit/negative octaves
    match = re.match(r"([A-G])([#b]?)(\-?\d+)", pitch_name, re.IGNORECASE)
    if not match:
        print(f"Warning: Could not parse pitch name: '{pitch_name}'. Defaulting to Middle C (60).")
        return 60

    note, acc, oct_str = match.groups()
    octave = int(oct_str)

    base_midi = PITCH_MAP.get(note.upper())
    if base_midi is None:
        print(f"Warning: Invalid note name: '{note}'. Defaulting to Middle C (60).")
        return 60

    acc_val = ACCIDENTAL_MAP.get(acc, 0)

    # Middle C (C4) is MIDI note 60. Formula: base + accidental + (octave + 1) * 12
    midi_val = base_midi + acc_val + (octave + 1) * 12

    # Clamp to valid MIDI range 0-127
    return max(0, min(127, midi_val))


def duration_to_seconds(duration_symbol, tempo, time_sig_denominator=4):
    """Converts duration symbol (W, H, Q, E, S, W., H., etc.) to seconds."""
    duration_symbol = duration_symbol.strip().upper()
    if not duration_symbol:
        print("Warning: Empty duration symbol. Defaulting to Quarter note duration.")
        duration_symbol = "Q"

    try:
        beats_per_minute = float(tempo)
        if beats_per_minute <= 0:
            print(f"Warning: Invalid tempo {tempo}. Using default 120.")
            beats_per_minute = 120

        # Tempo (BPM) is almost always defined as Quarter Notes Per Minute.
        quarter_note_duration_sec = 60.0 / beats_per_minute

        # Duration map relative to a quarter note (Q=1.0)
        # Assumes time signature denominator defines the 'beat' unit if not 4,
        # but MIDI tempo is typically quarter-note based. We stick to quarter notes for BPM.
        duration_map = {
            "W": 4.0,  # Whole note = 4 quarter notes
            "H": 2.0,  # Half note = 2 quarter notes
            "Q": 1.0,  # Quarter note = 1 quarter note
            "E": 0.5,  # Eighth note = 0.5 quarter notes
            "S": 0.25, # Sixteenth note = 0.25 quarter notes
            "T": 0.125, # Thirty-second note (Added)
        }

        base_symbol = duration_symbol.replace(".", "")
        is_dotted = duration_symbol.endswith(".")

        relative_duration_quarters = duration_map.get(base_symbol)
        if relative_duration_quarters is None:
            print(
                f"Warning: Unknown duration symbol: '{duration_symbol}'. Defaulting to Quarter (1.0)."
            )
            relative_duration_quarters = 1.0

        # Adjust for dotted notes (adds half of the base duration)
        if is_dotted:
            relative_duration_quarters *= 1.5

        # Calculate actual duration in seconds
        actual_duration_sec = relative_duration_quarters * quarter_note_duration_sec
        return actual_duration_sec

    except ValueError:
        print(f"Warning: Could not parse tempo '{tempo}' as float. Using default 120.")
        # Recalculate with default tempo
        return duration_to_seconds(duration_symbol, 120, time_sig_denominator)
    except Exception as e:
        print(f"Error calculating duration for '{duration_symbol}' at tempo {tempo}: {e}. Using default 0.5s.")
        return 0.5 # Default duration (e.g., quarter note at 120 BPM)


# --- Main Pipeline Functions ---


def enrich_music_description(description):
    """Step 1: Use LLM to enrich the initial music description."""
    print("\n--- Step 1: Enriching Description ---")
    prompt = f"""
    Take the following basic music description and enrich it with more detail for a generative music task.
    Suggest a plausible key signature (e.g., K:Cmin, K:Gmaj, K:F#dor), tempo (e.g., T:90), time signature (e.g., TS:4/4),
    and primary instrumentation (e.g., INST:Pno, INST:Gtr).
    Also, briefly elaborate on the mood and potential simple structure (like AABA).
    Keep the output concise and focused on these musical parameters. Output should start directly with the parameters.

    Basic Description: "{description}"

    Enriched Description (Example Format: K:Cmin T:60 TS:4/4 INST:Pno Mood: Melancholic, sparse. Structure: A-B-A):
    """
    enriched = call_gemini(prompt)

    # Use global defaults which might be updated by the enriched description
    global DEFAULT_KEY, DEFAULT_TEMPO, DEFAULT_TIMESIG

    if enriched:
        print(f"Enriched Description:\n{enriched}\n")
        # Attempt to parse key parameters from the enriched description to update defaults
        # Use regex for more robust parsing
        key_match = re.search(r"K:([A-Ga-g][#b]?(?:maj|min|dor|phr|lyd|mix|loc|aeo|ion)?)", enriched)
        tempo_match = re.search(r"T:(\d+)", enriched)
        ts_match = re.search(r"TS:(\d+)/(\d+)", enriched)

        if key_match:
            DEFAULT_KEY = key_match.group(1)
            print(f"Updated Default Key: {DEFAULT_KEY}")
        if tempo_match:
            try:
                tempo_val = int(tempo_match.group(1))
                if tempo_val > 0:
                    DEFAULT_TEMPO = tempo_val
                    print(f"Updated Default Tempo: {DEFAULT_TEMPO}")
            except ValueError: pass
        if ts_match:
            try:
                ts_num, ts_den = int(ts_match.group(1)), int(ts_match.group(2))
                # Basic validation for common time signatures
                if ts_num > 0 and ts_den > 0 and ts_den in [1, 2, 4, 8, 16, 32]:
                     DEFAULT_TIMESIG = (ts_num, ts_den)
                     print(f"Updated Default Time Signature: {DEFAULT_TIMESIG}")
            except ValueError: pass
        return enriched
    else:
        print("Failed to enrich description. Using initial description and defaults.")
        return description


def generate_symbolic_section(
    overall_desc, section_name, section_info, current_bar, previous_section_summary=None
):
    """Step 2: Generate symbolic music for one section using LLM."""
    print(
        f"--- Step 2: Generating Symbolic Section {section_name} (Starting Bar: {current_bar}) ---"
    )
    bars = section_info["bars"]
    goal = section_info["goal"]

    context_prompt = ""
    if previous_section_summary:
        prev_name = previous_section_summary.get('name', 'Previous')
        prev_summary = previous_section_summary.get('summary', 'No summary')
        prev_key = previous_section_summary.get('key', 'Unknown')
        prev_tempo = previous_section_summary.get('tempo', 'Unknown')
        context_prompt = (
            f"Context from previous section ({prev_name}): {prev_summary}\n"
            f"It aimed for key {prev_key} and tempo {prev_tempo}.\n"
            "Ensure a smooth musical transition if possible.\n"
        )

    prompt = f"""
You are a symbolic music generator. Your task is to generate ONLY the symbolic music notation for a specific section of a piece, following the provided format strictly.

Overall Music Goal: {overall_desc}
{context_prompt}
Current Section: {section_name}
Target Bars: {bars} (Start this section exactly at BAR:{current_bar}, end before BAR:{current_bar + bars})
Section Goal: {goal}

Instructions:
1. Generate music ONLY for this section, starting *exactly* with `BAR:{current_bar}`.
2. If tempo (T), time signature (TS), key (K), or instrument (INST) need to be set or changed *at the very beginning* of this section (before the first BAR marker), include those commands *before* `BAR:{current_bar}`. Otherwise, assume they carry over or use defaults (T:{DEFAULT_TEMPO}, TS:{DEFAULT_TIMESIG[0]}/{DEFAULT_TIMESIG[1]}, K:{DEFAULT_KEY}). You can change INST multiple times within the section if needed for different instruments/tracks.
3. Strictly adhere to the compact symbolic format defined below. Do NOT include any other text, explanations, apologies, or formatting like ```mus``` or ```. Output ONLY the commands, each on a new line.
4. Ensure musical coherence within the section and try to follow the Section Goal. Make smooth transitions if context from a previous section is provided (e.g., connect harmonically or rhythmically).
5. The total duration of notes/rests/chords within each bar MUST add up correctly according to the active time signature (e.g., 4 quarter notes in 4/4). Be precise. Use rests (R:<Track>:<Duration>) to fill empty time.
6. End the generation cleanly *after* the content for bar {current_bar + bars - 1} is complete, before the next section would start (i.e., do NOT include BAR:{current_bar + bars}).

{SYMBOLIC_FORMAT_DEFINITION}

Generate Section {section_name} symbolic music now (starting BAR:{current_bar}):
"""

    symbolic_text = call_gemini(prompt)

    if symbolic_text:
        # Clean the response: remove potential markdown code fences and surrounding whitespace/text
        symbolic_text = re.sub(r"^```[a-z]*\n", "", symbolic_text, flags=re.MULTILINE)
        symbolic_text = re.sub(r"\n```$", "", symbolic_text)
        symbolic_text = symbolic_text.strip()

        # Find the first meaningful line (non-empty, not a comment)
        lines = symbolic_text.split('\n')
        first_meaningful_line = ""
        first_meaningful_line_index = -1
        for idx, line in enumerate(lines):
            line = line.strip()
            if line and not line.startswith('#'):
                first_meaningful_line = line
                first_meaningful_line_index = idx
                break

        if first_meaningful_line_index == -1:
             print(f"Warning: Generated text for {section_name} appears empty or only contains comments.")
             return "", None # Treat as failure

        bar_marker = f"BAR:{current_bar}"
        # Check if the first meaningful line is the expected BAR marker or a setting (T/TS/K/INST)
        is_valid_start = first_meaningful_line.startswith(bar_marker) or \
                         re.match(r"^(T:|TS:|K:|INST:)", first_meaningful_line)

        if not is_valid_start:
            print(f"Warning: Generated text for {section_name} doesn't seem to start correctly.")
            print(f"Expected '{bar_marker}' or INST/T/TS/K. Got: '{first_meaningful_line[:60]}...'")
            # Attempt to find the correct starting BAR marker if LLM added preamble
            bar_marker_index_in_text = symbolic_text.find(bar_marker)
            if bar_marker_index_in_text != -1:
                print(f"Found '{bar_marker}' later in the text. Attempting to use text from that point.")
                # Find the beginning of the line containing the bar marker
                start_pos = symbolic_text.rfind('\n', 0, bar_marker_index_in_text) + 1
                symbolic_text = symbolic_text[start_pos:]
            else:
                print(f"Could not find '{bar_marker}'. The generated section might be incorrect or incomplete.")
                # Return the potentially incorrect text, parsing will handle errors later
        elif first_meaningful_line_index > 0:
             # If the valid start was found but not on the first line, trim preceding lines
             print(f"Trimming {first_meaningful_line_index} lines of potential preamble from {section_name} output.")
             symbolic_text = "\n".join(lines[first_meaningful_line_index:])


        print(f"Generated symbolic text for Section {section_name} (first 300 chars):\n{symbolic_text[:300]}...\n")

        # Create a simple summary based on the *request* for the next section's context
        summary_info = {
            "name": section_name,
            "summary": f"Generated {bars} bars targeting goal: {goal}.",
            "key": DEFAULT_KEY, # Use current default/enriched key
            "tempo": DEFAULT_TEMPO # Use current default/enriched tempo
            # Note: These key/tempo values are from the *start* of generation,
            # the actual end key/tempo might differ if changed within the section.
        }
        return symbolic_text, summary_info
    else:
        print(f"Failed to generate symbolic text for Section {section_name}.")
        return "", None # Return empty string and no summary on failure


def parse_symbolic_to_structured_data(symbolic_text):
    """Step 3: Parse concatenated symbolic text into structured data for MIDI."""
    print("\n--- Step 3: Parsing Symbolic Text ---")
    # Data structures to hold parsed information
    notes_by_instrument_track = {} # { ('InstName', 'TrackID'): [note_obj1, ...], ... }
    tempo_changes = [] # Store changes with time: [(time, value), ...]
    time_signature_changes = [] # Store time sig changes with time: [(time, num, den), ...]
    key_signature_changes = [] # Store key sig changes: [(time, key_name)]
    instrument_definitions = {} # { ('InstName', 'TrackID'): {'program': X, 'is_drum': Y, 'name': Z}, ... }

    # --- Time Tracking State ---
    # Use a dictionary to track current time *per track* to handle polyphony correctly
    # Global time represents the latest time reached across all tracks for settings changes (T, TS, K)
    current_track_times = {} # { ('InstName', 'TrackID'): time, ... }
    current_global_time = 0.0 # Tracks the time for BAR markers and global settings

    # --- Musical State ---
    current_tempo = float(DEFAULT_TEMPO)
    current_ts_num, current_ts_den = DEFAULT_TIMESIG
    current_key = DEFAULT_KEY
    active_instrument_name = "Pno" # Default instrument if none specified early

    # --- Bar Tracking ---
    current_bar_number = 0
    current_bar_start_time = 0.0
    # Track accumulated duration within the current bar *per track*
    time_within_bar_per_track = {} # { ('InstName', 'TrackID'): time, ... }
    expected_bar_duration_sec = (60.0 / current_tempo) * current_ts_num * (4.0 / current_ts_den)
    last_event_end_time = 0.0 # Track the maximum end time of any note/event

    initial_commands_set = {'T': False, 'TS': False, 'K': False}
    lines = symbolic_text.strip().split('\n')
    line_num = 0

    # --- Pre-pass for initial settings before the first BAR marker ---
    print("Processing initial settings (before first BAR marker)...")
    while line_num < len(lines):
        line = lines[line_num].strip()
        if not line or line.startswith('#'):
            line_num += 1
            continue
        if line.startswith("BAR:") or not re.match(r"^(T:|TS:|K:|INST:)", line):
            break

        parts = line.split(':', 1)
        command = parts[0].upper()
        value = parts[1].strip() if len(parts) > 1 else ""
        ini_line_num = line_num + 1

        try:
            if command == "INST": # Initial INST sets the default
                if value:
                    active_instrument_name = value
                    print(f"Initial Instrument context set to {active_instrument_name}")
            elif command == "T" and not initial_commands_set['T']:
                new_tempo = float(value)
                if new_tempo > 0:
                    current_tempo = new_tempo
                    tempo_changes.append((0.0, current_tempo))
                    expected_bar_duration_sec = (60.0 / current_tempo) * current_ts_num * (4.0 / current_ts_den)
                    initial_commands_set['T'] = True
                    print(f"Initial Tempo set to {current_tempo} BPM")
            elif command == "TS" and not initial_commands_set['TS']:
                num_str, den_str = value.split('/')
                new_ts_num, new_ts_den = int(num_str), int(den_str)
                if new_ts_num > 0 and new_ts_den > 0 and (new_ts_num, new_ts_den) != (current_ts_num, current_ts_den):
                    current_ts_num, current_ts_den = new_ts_num, new_ts_den
                    time_signature_changes.append((0.0, current_ts_num, current_ts_den))
                    expected_bar_duration_sec = (60.0 / current_tempo) * current_ts_num * (4.0 / current_ts_den)
                    initial_commands_set['TS'] = True
                    print(f"Initial Time Signature set to {current_ts_num}/{current_ts_den}")
            elif command == "K" and not initial_commands_set['K']:
                if value:
                    current_key = value
                    key_signature_changes.append((0.0, current_key))
                    initial_commands_set['K'] = True
                    print(f"Initial Key set to {current_key}")

        except Exception as e:
            print(f"Error parsing initial setting line {ini_line_num}: '{line}' - {e}")
        line_num += 1

    # Add default initial settings if not overridden
    if not initial_commands_set['T']: tempo_changes.append((0.0, current_tempo))
    if not initial_commands_set['TS']: time_signature_changes.append((0.0, current_ts_num, current_ts_den))
    if not initial_commands_set['K']: key_signature_changes.append((0.0, current_key))

    # --- Main Parsing Loop ---
    print("Parsing main body...")
    for i in range(line_num, len(lines)):
        current_line_num = i + 1
        line = lines[i].strip()
        if not line or line.startswith('#'): continue

        parts = line.split(':', 1)
        command = parts[0].upper()
        value = parts[1].strip() if len(parts) > 1 else ""

        try:
            # Handle global settings (affect future calculations/state)
            if command == "INST":
                if value:
                    active_instrument_name = value
                    # print(f"Time ~{current_global_time:.2f}s: Instrument context changed to {active_instrument_name}")
            elif command == "T":
                new_tempo = float(value)
                if new_tempo > 0 and new_tempo != current_tempo:
                    # Tempo changes apply globally at the current global time
                    tempo_changes.append((current_global_time, new_tempo))
                    current_tempo = new_tempo
                    expected_bar_duration_sec = (60.0 / current_tempo) * current_ts_num * (4.0 / current_ts_den)
                    print(f"Time {current_global_time:.2f}s (Bar ~{current_bar_number}): Tempo change to {new_tempo} BPM")
            elif command == "TS":
                num_str, den_str = value.split('/')
                new_ts_num, new_ts_den = int(num_str), int(den_str)
                if new_ts_num > 0 and new_ts_den > 0 and (new_ts_num, new_ts_den) != (current_ts_num, current_ts_den):
                    time_signature_changes.append((current_global_time, new_ts_num, new_ts_den))
                    current_ts_num, current_ts_den = new_ts_num, new_ts_den
                    expected_bar_duration_sec = (60.0 / current_tempo) * current_ts_num * (4.0 / current_ts_den)
                    print(f"Time {current_global_time:.2f}s (Bar ~{current_bar_number}): Time Sig change to {new_ts_num}/{new_ts_den}")
            elif command == "K":
                if value and value != current_key:
                    key_signature_changes.append((current_global_time, value))
                    current_key = value
                    print(f"Time {current_global_time:.2f}s (Bar ~{current_bar_number}): Key change to {current_key}")

            # Handle BAR marker (synchronization point)
            elif command == "BAR":
                bar_number = int(value)
                # --- Bar Time Synchronization ---
                # Calculate the expected start time of this new bar based on the previous bar's start and duration
                expected_new_bar_start_time = current_bar_start_time + expected_bar_duration_sec if current_bar_number > 0 else 0.0

                # Check timing consistency for tracks that had events in the previous bar
                if current_bar_number > 0:
                    max_accumulated_time_in_prev_bar = 0.0
                    for track_key, accumulated_time in time_within_bar_per_track.items():
                         max_accumulated_time_in_prev_bar = max(max_accumulated_time_in_prev_bar, accumulated_time)

                    duration_error = max_accumulated_time_in_prev_bar - expected_bar_duration_sec
                    tolerance = max(0.01, expected_bar_duration_sec * 0.01) # 1% or 10ms tolerance

                    if abs(duration_error) > tolerance:
                        print(
                            f"Warning: Bar {current_bar_number} timing mismatch on Line {current_line_num}. "
                            f"Expected duration {expected_bar_duration_sec:.3f}s, max accumulated {max_accumulated_time_in_prev_bar:.3f}s "
                            f"(Error: {duration_error:.3f}s). Setting bar {bar_number} start to expected time {expected_new_bar_start_time:.3f}s."
                        )
                        current_global_time = expected_new_bar_start_time
                    else:
                         # If within tolerance, advance global time based on the *actual* end time of the longest track in the bar
                         current_global_time = current_bar_start_time + max_accumulated_time_in_prev_bar


                # Handle jumps in bar numbers (advance global time)
                bars_jumped = bar_number - (current_bar_number + 1)
                if bars_jumped > 0 and current_bar_number > 0:
                    jump_duration = bars_jumped * expected_bar_duration_sec
                    print(
                        f"Warning: Jump detected from Bar {current_bar_number} to {bar_number} (Line {current_line_num}). "
                        f"Advancing global time by ~{jump_duration:.3f}s ({bars_jumped} bars)."
                    )
                    current_global_time += jump_duration

                # Update bar state
                current_bar_number = bar_number
                current_bar_start_time = current_global_time
                time_within_bar_per_track = {} # Reset accumulated time for the new bar
                # Reset individual track times to the start of the new bar
                current_track_times = {track: current_bar_start_time for track in current_track_times}
                # print(f"Time {current_global_time:.2f}s: --- Reached BAR {bar_number} ---")


            # Handle Note, Chord, Rest (track-specific events)
            elif command in ["N", "C", "R"]:
                if current_bar_number == 0:
                    print(f"Warning: Event '{line}' on Line {current_line_num} found before first BAR marker. Processing at time 0.")
                    # Time will be 0 unless settings changed it, handled by track time logic below

                data_parts = value.split(':')
                min_parts = 3 if command == "R" else 4 # R:Track:Dur, N/C:Track:Data:Dur:Vel
                if len(data_parts) < min_parts:
                    print(f"Warning: Malformed {command} command on Line {current_line_num}: '{line}'. Requires at least {min_parts} parts. Skipping.")
                    continue

                track_id = data_parts[0].strip()
                # Use the *currently active* instrument name from the last INST command
                inst_name_for_event = active_instrument_name
                inst_track_key = (inst_name_for_event, track_id)

                # Determine if this track/instrument is drums
                is_drum_track = inst_name_for_event.lower() in ["drs", "drums"] or track_id.lower() == "drums"
                # Find appropriate program number, default to Piano(0) if name not found
                program = INSTRUMENT_PROGRAM_MAP.get(inst_name_for_event, 0)

                # Define instrument properties if not seen before
                if inst_track_key not in instrument_definitions:
                    pm_instrument_name = f"{inst_name_for_event}-{track_id}"
                    instrument_definitions[inst_track_key] = {
                        "program": program,
                        "is_drum": is_drum_track,
                        "name": pm_instrument_name
                    }
                    print(f"Defined instrument: {pm_instrument_name} (Program: {program}, IsDrum: {is_drum_track})")
                    # Initialize time for this new track
                    current_track_times[inst_track_key] = current_bar_start_time + time_within_bar_per_track.get(inst_track_key, 0.0)
                    notes_by_instrument_track[inst_track_key] = []


                # Get the current start time for this specific track
                event_start_time = current_track_times.get(inst_track_key, current_bar_start_time + time_within_bar_per_track.get(inst_track_key, 0.0))
                event_duration_sec = 0.0

                if command == "N": # Note:Track:Pitch:Duration:Velocity
                    if len(data_parts) < 4: continue # Already checked min_parts, but check specific length
                    pitch_name = data_parts[1].strip()
                    duration_sym = data_parts[2].strip()
                    velocity_str = data_parts[3].strip()
                    try: velocity = int(velocity_str)
                    except ValueError: velocity = 64; print(f"Warning: Invalid velocity '{velocity_str}' on Line {current_line_num}. Using 64.")

                    event_duration_sec = duration_to_seconds(duration_sym, current_tempo, current_ts_den)

                    if is_drum_track:
                        midi_pitch = DRUM_PITCH_MAP.get(pitch_name.capitalize()) # Try capitalized first
                        if midi_pitch is None: midi_pitch = DRUM_PITCH_MAP.get(pitch_name) # Try original case
                        if midi_pitch is None: midi_pitch = 36; print(f"Warning: Unknown drum sound '{pitch_name}' on Line {current_line_num}. Using Kick (36).")
                    else:
                        midi_pitch = pitch_to_midi(pitch_name)

                    note_event = {
                        "pitch": midi_pitch,
                        "start": event_start_time,
                        "end": event_start_time + event_duration_sec,
                        "velocity": max(0, min(127, velocity))
                    }
                    notes_by_instrument_track[inst_track_key].append(note_event)

                elif command == "C": # Chord:Track:[P1,P2,...]:Duration:Velocity
                    if len(data_parts) < 4: continue
                    pitches_str = data_parts[1].strip()
                    duration_sym = data_parts[2].strip()
                    velocity_str = data_parts[3].strip()
                    try: velocity = int(velocity_str)
                    except ValueError: velocity = 64; print(f"Warning: Invalid velocity '{velocity_str}' on Line {current_line_num}. Using 64.")

                    if pitches_str.startswith('[') and pitches_str.endswith(']'):
                        pitches_str = pitches_str[1:-1]
                    pitch_names = [p.strip() for p in pitches_str.split(',') if p.strip()]

                    if not pitch_names: continue # Skip if no valid pitches

                    event_duration_sec = duration_to_seconds(duration_sym, current_tempo, current_ts_den)

                    for pitch_name in pitch_names:
                        if is_drum_track: print(f"Warning: Chord command used on drum track (Line {current_line_num}). Treating as standard notes.")
                        midi_pitch = pitch_to_midi(pitch_name)
                        note_event = {
                            "pitch": midi_pitch,
                            "start": event_start_time,
                            "end": event_start_time + event_duration_sec,
                            "velocity": max(0, min(127, velocity))
                        }
                        notes_by_instrument_track[inst_track_key].append(note_event)

                elif command == "R": # Rest:Track:Duration
                    if len(data_parts) < 2: continue # R needs Track:Duration
                    duration_sym = data_parts[1].strip()
                    event_duration_sec = duration_to_seconds(duration_sym, current_tempo, current_ts_den)
                    # Rests only advance time for the specific track

                # --- Advance Time for the specific track ---
                new_track_time = event_start_time + event_duration_sec
                current_track_times[inst_track_key] = new_track_time
                # Update accumulated time within the bar for this track
                time_within_bar_per_track[inst_track_key] = new_track_time - current_bar_start_time
                # Update the latest event end time globally
                last_event_end_time = max(last_event_end_time, new_track_time)
                # Update global time marker if this event pushes it forward (relevant for T/TS/K changes)
                current_global_time = max(current_global_time, new_track_time)


            else:
                print(f"Warning: Unknown command '{command}' on line {current_line_num}: '{line}'. Skipping.")

        except Exception as e:
            print(f"Error parsing line {current_line_num}: '{line}' - {e}")
            traceback.print_exc()
            continue

    print(f"Symbolic text parsing complete. Estimated total duration: {last_event_end_time:.2f} seconds.")
    return (
        notes_by_instrument_track,
        instrument_definitions,
        tempo_changes,
        time_signature_changes,
        key_signature_changes,
        last_event_end_time,
        current_key,
        current_tempo
    )


def create_midi_file(
    notes_data,
    instrument_defs,
    tempo_changes,
    time_sig_changes,
    key_sig_changes,
    filename,
):
    """Step 4: Create MIDI file using pretty_midi."""
    print(f"\n--- Step 4: Creating MIDI File ({filename}) ---")
    if not notes_data or not instrument_defs:
        print("Error: No instrument or note data was successfully parsed. Cannot create MIDI file.")
        return

    try:
        # Initialize with the first tempo change or default
        initial_tempo = tempo_changes[0][1] if tempo_changes else DEFAULT_TEMPO
        midi_obj = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo)
        print(f"Initialized MIDI with tempo: {initial_tempo:.2f} BPM")

        # Add Time Signature Changes (sorted)
        time_sig_changes.sort(key=lambda x: x[0])
        applied_ts_count = 0
        last_ts_time = -1.0
        for time, num, den in time_sig_changes:
            # Denominator needs to be power of 2 for standard MIDI
            if den == 0 or (den & (den - 1) != 0): # Check if not power of 2
                 # Find closest power of 2 (e.g., 3 -> 4, 6 -> 8, 7 -> 8)
                 if den == 0: actual_den = 4 # Avoid log(0)
                 else: actual_den = 2**math.ceil(math.log2(den))
                 print(
                    f"Warning: Non-power-of-2 time signature denominator {den} at time {time:.2f}s."
                    f" Using closest power of 2: {actual_den}."
                 )
            else:
                actual_den = den

            if time >= last_ts_time: # Allow changes at the same time, replacing previous ones at that exact time
                # Remove potential duplicates at the *exact* same time before adding
                midi_obj.time_signature_changes = [ts for ts in midi_obj.time_signature_changes if ts.time != time]
                ts_change = pretty_midi.TimeSignature(num, actual_den, time)
                midi_obj.time_signature_changes.append(ts_change)
                applied_ts_count += 1
                last_ts_time = time
            # else: # Ignore changes that are chronologically before the last applied one (shouldn't happen with sorting)
            #     print(f"Ignoring out-of-order time signature change ({num}/{den}) at time {time:.2f}s.")

        # Re-sort just in case additions messed order (unlikely but safe)
        midi_obj.time_signature_changes.sort(key=lambda ts: ts.time)
        if applied_ts_count > 0: print(f"Applied {applied_ts_count} time signature changes.")
        if not midi_obj.time_signature_changes:
             default_num, default_den = DEFAULT_TIMESIG
             midi_obj.time_signature_changes.append(pretty_midi.TimeSignature(default_num, default_den, 0.0))
             print(f"Applied default time signature: {default_num}/{default_den}")


        # Add Key Signature Changes (sorted)
        key_sig_changes.sort(key=lambda x: x[0])
        applied_key_count = 0
        last_key_time = -1.0
        for time, key_name in key_sig_changes:
            try:
                key_number = pretty_midi.key_name_to_key_number(key_name)
                if time >= last_key_time:
                    # Remove potential duplicates at the *exact* same time before adding
                    midi_obj.key_signature_changes = [ks for ks in midi_obj.key_signature_changes if ks.time != time]
                    key_change = pretty_midi.KeySignature(key_number=key_number, time=time)
                    midi_obj.key_signature_changes.append(key_change)
                    applied_key_count += 1
                    last_key_time = time
                # else: # Ignore out-of-order changes
                #     print(f"Ignoring out-of-order key signature change ({key_name}) at time {time:.2f}s.")
            except ValueError as e:
                print(f"Warning: Could not parse key signature '{key_name}' at time {time:.2f}s. Skipping. Error: {e}")

        # Re-sort just in case
        midi_obj.key_signature_changes.sort(key=lambda ks: ks.time)
        if applied_key_count > 0: print(f"Applied {applied_key_count} key signature changes.")
        if not midi_obj.key_signature_changes:
            try:
                default_key_num = pretty_midi.key_name_to_key_number(DEFAULT_KEY)
                midi_obj.key_signature_changes.append(pretty_midi.KeySignature(key_number=default_key_num, time=0.0))
                print(f"Applied default key signature: {DEFAULT_KEY}")
            except ValueError as e: print(f"Warning: Could not parse default key signature '{DEFAULT_KEY}'. Error: {e}")


        # Create instruments and add notes
        # Assign channels, ensuring drums are on channel 9 (0-indexed)
        available_channels = list(range(16))
        drum_channel = 9
        if drum_channel in available_channels: available_channels.remove(drum_channel)
        channel_index = 0
        assigned_channels = {} # Track assigned channel per instrument object

        instrument_objects = {} # { inst_track_key: pretty_midi.Instrument }

        # Sort instrument definitions to process drums first (if any)
        sorted_inst_defs = sorted(instrument_defs.items(), key=lambda item: item[1]['is_drum'], reverse=True)

        for inst_track_key, definition in sorted_inst_defs:
            if inst_track_key not in notes_data or not notes_data[inst_track_key]:
                print(f"Skipping instrument '{definition['name']}' as it has no parsed notes.")
                continue

            is_drum = definition['is_drum']
            program = definition['program']
            pm_instrument_name = definition['name']

            instrument_obj = pretty_midi.Instrument(program=program, is_drum=is_drum, name=pm_instrument_name)
            midi_obj.instruments.append(instrument_obj)
            instrument_objects[inst_track_key] = instrument_obj

            # Assign MIDI channel
            if is_drum:
                channel = drum_channel
                print(f"Created MIDI instrument: {pm_instrument_name} (Program: {program}, IsDrum: {is_drum}, Assigned Channel: {channel})")
            else:
                if channel_index >= len(available_channels):
                    reuse_channel_index = channel_index % len(available_channels)
                    channel = available_channels[reuse_channel_index]
                    print(f"Warning: Ran out of non-drum MIDI channels! Reusing channel {channel} for {pm_instrument_name}.")
                else:
                    channel = available_channels[channel_index]
                print(f"Created MIDI instrument: {pm_instrument_name} (Program: {program}, IsDrum: {is_drum}, Assigned Channel: {channel})")
                channel_index += 1
            assigned_channels[instrument_obj] = channel # Store assigned channel if needed later, although pretty_midi handles it


            # Add notes to this instrument
            note_count = 0
            skipped_notes = 0
            for note_info in notes_data[inst_track_key]:
                start_time = max(0.0, note_info['start'])
                min_duration = 0.001 # 1ms minimum duration
                end_time = max(start_time + min_duration, note_info['end'])
                velocity = max(1, min(127, note_info['velocity'])) # Ensure velocity > 0 for note-on
                pitch = max(0, min(127, note_info['pitch']))

                if start_time >= end_time:
                     # print(f"Warning: Skipping note with non-positive duration for {pm_instrument_name}. Pitch: {pitch}, Start: {start_time:.3f}, End: {end_time:.3f}")
                     skipped_notes += 1
                     continue

                try:
                    note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=end_time)
                    instrument_obj.notes.append(note)
                    note_count += 1
                except ValueError as e:
                    print(f"Error creating pretty_midi.Note for {pm_instrument_name}: {e}. Skipping note.")
                    print(f"  Note data: Pitch={pitch}, Vel={velocity}, Start={start_time}, End={end_time}")
                    skipped_notes += 1

            print(f"Added {note_count} notes to instrument {pm_instrument_name}. ({skipped_notes} notes skipped).")


        # Add explicit tempo changes as MIDI meta messages (optional, pretty_midi handles timing internally)
        # This might be needed for some external sequencers to display tempo changes correctly.
        if len(tempo_changes) > 1:
             print("Applying tempo changes as MIDI meta-messages...")
             # Get tempo change times and BPM values from our list
             tempo_change_times, tempo_change_bpm = zip(*tempo_changes)
             # Convert BPM to microseconds per beat (MIDI standard tempo)
             tempo_change_mpb = [pretty_midi.bpm_to_tempo(bpm) for bpm in tempo_change_bpm]
             # Add the tempo changes to the MIDI object
             midi_obj._load_tempo_changes(list(tempo_change_times), list(tempo_change_mpb))


        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        full_output_path = os.path.join(OUTPUT_DIR, filename)

        # Write the MIDI file
        midi_obj.write(full_output_path)
        print(f"\nSuccessfully created MIDI file: {full_output_path}")

    except Exception as e:
        print(f"Error writing MIDI file '{filename}': {e}")
        traceback.print_exc()


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting AutoMusic Generator Pipeline...")
    configure_genai()

    # Step 1: Enrich the initial description
    enriched_description = enrich_music_description(INITIAL_DESCRIPTION)
    if not enriched_description:
        enriched_description = INITIAL_DESCRIPTION # Fallback

    # Step 2 (Section Generation) & Concatenation
    all_symbolic_text = ""
    current_bar_count = 1
    last_section_summary_info = None
    generated_sections_count = 0

    for section_name, section_info in SECTIONS.items():
        symbolic_section, current_section_summary_info = generate_symbolic_section(
            enriched_description,
            section_name,
            section_info,
            current_bar_count,
            last_section_summary_info,
        )

        if symbolic_section and current_section_summary_info:
            if not symbolic_section.endswith('\n'): symbolic_section += '\n'
            all_symbolic_text += symbolic_section
            generated_sections_count += 1
            last_section_summary_info = current_section_summary_info
            current_bar_count += section_info['bars']
        else:
            print(f"Skipping section {section_name} due to generation failure.")
            # break # Option: Stop if any section fails

    if not all_symbolic_text.strip():
        print("\nERROR: No symbolic text was generated successfully. Cannot proceed.")
        exit(1)
    if generated_sections_count < len(SECTIONS):
        print(f"\nWarning: Only {generated_sections_count}/{len(SECTIONS)} sections were generated successfully.")

    print("\n--- Combined Symbolic Text ---")
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    symbolic_filename = os.path.join(OUTPUT_DIR, f"symbolic_music_{timestamp_str}.txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        with open(symbolic_filename, "w", encoding="utf-8") as f:
            f.write(all_symbolic_text)
        print(f"Saved combined symbolic text to: {symbolic_filename}")
    except IOError as e: print(f"Error saving symbolic text file: {e}")
    print("----------------------------")

    # Step 3: Parse the combined symbolic text
    (
        parsed_notes,
        instrument_definitions,
        tempo_changes,
        time_sig_changes,
        key_sig_changes,
        estimated_duration,
        final_key,
        final_tempo
    ) = parse_symbolic_to_structured_data(all_symbolic_text)


    # Step 4: Create the MIDI file
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
        print("\nError: No notes or instruments were successfully parsed. MIDI file not created.")

    print("\n--- AutoMusic Generator Pipeline Finished ---")