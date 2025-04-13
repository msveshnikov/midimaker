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

import google.generativeai as genai
import pretty_midi

# --- Configuration ---
# IMPORTANT: Use environment variables or secrets management for API keys in real applications!
# Set the API_KEY environment variable or replace the placeholder below.
API_KEY = os.environ.get("API_KEY", "AIzaSyC5jbwgP050qfurqK9GyvgrUYvpwEy0n8s") # Placeholder - Replace or use env var

# Configure the Gemini model to use
GEMINI_MODEL = "gemini-2.0-flash-thinking-exp-01-21" # Or other suitable models like 'gemini-1.0-pro'

# Initial high-level description of the music
INITIAL_DESCRIPTION = (
    "A simple, slightly melancholic loopable piano piece suitable for a game background."
)

# Define the structure and goals for each section
# TODO: Generate section descriptions by LLM as well, do not use hardcoded ones
SECTIONS = {
    "A1": {
        "bars": 8,
        "goal": "Establish the main melancholic theme in C minor. Simple texture, sparse piano.",
    },
    "B": {
        "bars": 8,
        "goal": "Introduce a slight variation or counter-melody, perhaps brightening slightly towards Eb major. Slightly more active rhythm.",
    },
    "A2": {
        "bars": 8,
        "goal": "Return to the main theme, similar to A1 but maybe with slight embellishment or a fuller chord in the left hand. End conclusively for looping.",
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
- `INST:<ID>` (Instrument: Pno, Gtr, Bass, Drs, Str, Flt, Tpt ...)
- `T:<BPM>` (Tempo: e.g., T:120)
- `TS:<N>/<D>` (Time Signature: e.g., TS:4/4)
- `K:<Key>` (Key Signature: e.g., K:Cmin, K:Gmaj)
- `BAR:<Num>` (Bar marker)
- `N:<Track>:<Pitch>:<Duration>:<Velocity>` (Note: TrackID: PitchName: DurationSymbol: Velocity[0-127])
- `C:<Track>:<Pitches>:<Duration>:<Velocity>` (Chord: TrackID: [Pitch1,Pitch2,...]: DurationSymbol: Velocity)
- `R:<Track>:<Duration>` (Rest: TrackID: DurationSymbol)

TrackIDs: Use simple names like RH (Right Hand), LH (Left Hand), Melody, Bass, Drums.
PitchNames: Use standard notation (e.g., C4, F#5, Gb3). Middle C is C4.
DurationSymbols: W (Whole), H (Half), Q (Quarter), E (Eighth), S (Sixteenth). '.' can be appended for dotted notes (e.g., Q.).
Velocity: MIDI velocity (0-127).

Example Note: N:RH:C5:H:70
Example Chord: C:LH:[C3,Eb3,G3]:W:60
Example Rest: R:RH:Q
"""

# --- Helper Functions ---


def configure_genai():
    """Configures the Google Generative AI library."""
    if not API_KEY or API_KEY == "YOUR_API_KEY":
        print(
            "ERROR: API_KEY is not set or is using a placeholder value."
            " Please set the API_KEY environment variable or update the script."
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
            # Example of using generation_config if needed:
            # generation_config = genai.types.GenerationConfig(
            #     temperature=0.7,
            #     max_output_tokens=4096
            # )
            # response = model.generate_content(prompt, generation_config=generation_config)
            response = model.generate_content(prompt)

            # Debugging: Print raw response structure if needed
            # print(f"Gemini Response (Attempt {attempt+1}): {response}")

            # Access text, handling potential blocking or empty responses
            if response.parts:
                return response.text.strip() # Return stripped text

            # Handle blocked prompts
            if (
                hasattr(response, "prompt_feedback")
                and response.prompt_feedback.block_reason
            ):
                print(
                    f"Prompt blocked (Attempt {attempt + 1}):"
                    f" {response.prompt_feedback.block_reason}"
                )
                # If blocked, retrying usually won't help with the same prompt
                return None

            # Handle non-standard finish reasons
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
                return None # Indicate non-standard stop without content

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
            # Don't return None immediately, allow retries

        except Exception as e:
            print(
                f"Error calling Gemini API (Attempt {attempt + 1}/{retries}): {e}"
            )
            traceback.print_exc() # Print full traceback for debugging

        if attempt < retries - 1:
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
        else:
            print("Max retries reached. Failing.")
            return None # Indicate failure after all retries
    return None # Fallback


# --- Music Data Structures and Mappings ---

PITCH_MAP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
ACCIDENTAL_MAP = {"#": 1, "b": -1, "": 0}

# General MIDI Instrument Program Numbers
# Expand this map as needed for more instruments
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
    "Gtr": 27, "Acoustic Guitar": 25, "Nylon Guitar": 24, "Steel Guitar": 25,
    "Electric Guitar": 27, "Jazz Guitar": 26, "Clean Electric Guitar": 27,
    "Muted Electric Guitar": 28, "Overdriven Guitar": 29, "Distortion Guitar": 30,
    # Bass
    "Bass": 33, "Acoustic Bass": 32, "Electric Bass": 33, "Finger Bass": 33,
    "Pick Bass": 34, "Fretless Bass": 35, "Slap Bass": 36, "Synth Bass": 38,
    # Strings
    "Str": 48, "Strings": 48, "Violin": 40, "Viola": 41, "Cello": 42, "Contrabass": 43,
    "Tremolo Strings": 44, "Pizzicato Strings": 45, "Orchestral Harp": 46,
    "String Ensemble 1": 48, "Synth Strings 1": 50,
    # Brass
    "Tpt": 56, "Trumpet": 56, "Trombone": 57, "Tuba": 58, "Muted Trumpet": 59, "French Horn": 60,
    # Reed
    "Sax": 65, "Soprano Sax": 64, "Alto Sax": 65, "Tenor Sax": 66, "Baritone Sax": 67,
    "Oboe": 68, "English Horn": 69, "Bassoon": 70, "Clarinet": 71,
    # Pipe
    "Flt": 73, "Flute": 73, "Piccolo": 72, "Recorder": 74, "Pan Flute": 75,
    # Synth Lead
    "Synth Lead": 80, "Lead 1 (Square)": 80, "Lead 2 (Sawtooth)": 81,
    # Drums are a special case (channel 10 / index 9)
    "Drs": 0, "Drums": 0, # Program 0 is often used, but channel is key
    # Add more instruments...
}

# Standard drum note map (MIDI channel 10 / index 9) - General MIDI Level 1 Percussion Key Map
# Expand this map as needed for more drum sounds
DRUM_PITCH_MAP = {
    # Bass Drum
    "Kick": 36, "BD": 36, "Bass Drum 1": 36, "Acoustic Bass Drum": 35,
    # Snare
    "Snare": 38, "SD": 38, "Acoustic Snare": 38, "Electric Snare": 40,
    # Hi-Hat
    "HiHatClosed": 42, "HHC": 42, "Closed Hi Hat": 42,
    "HiHatOpen": 46, "HHO": 46, "Open Hi Hat": 46,
    "HiHatPedal": 44, "HH P": 44, "Pedal Hi Hat": 44,
    # Cymbals
    "Crash": 49, "CR": 49, "Crash Cymbal 1": 49, "Crash Cymbal 2": 57,
    "Ride": 51, "RD": 51, "Ride Cymbal 1": 51, "Ride Cymbal 2": 59, "Ride Bell": 53,
    "Splash Cymbal": 55, "Chinese Cymbal": 52,
    # Toms
    "High Tom": 50, "HT": 50, "Hi Tom": 50,
    "Mid Tom": 47, "MT": 47, "Hi-Mid Tom": 48, "Low-Mid Tom": 47,
    "Low Tom": 43, "LT": 43, "High Floor Tom": 43,
    "Floor Tom": 41, "FT": 41, "Low Floor Tom": 41,
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
    match = re.match(r"([A-G])([#b]?)(\-?\d+)", pitch_name)
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
    duration_symbol = duration_symbol.strip()
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
        duration_map = {
            "W": 4.0,  # Whole note = 4 quarter notes
            "H": 2.0,  # Half note = 2 quarter notes
            "Q": 1.0,  # Quarter note = 1 quarter note
            "E": 0.5,  # Eighth note = 0.5 quarter notes
            "S": 0.25, # Sixteenth note = 0.25 quarter notes
            # TODO: Add support for triplets, complex durations if needed
        }

        base_symbol = duration_symbol.replace(".", "")
        is_dotted = duration_symbol.endswith(".")

        relative_duration_quarters = duration_map.get(base_symbol.upper())
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
        enriched_parts = enriched.strip().split()
        key_found, tempo_found, ts_found = False, False, False
        for part in enriched_parts:
            if part.startswith("K:") and not key_found:
                try:
                    key_val = part.split(":", 1)[1]
                    if key_val:
                        DEFAULT_KEY = key_val
                        print(f"Updated Default Key: {DEFAULT_KEY}")
                        key_found = True
                except IndexError: pass
            elif part.startswith("T:") and not tempo_found:
                try:
                    tempo_val = int(part.split(":", 1)[1])
                    if tempo_val > 0:
                        DEFAULT_TEMPO = tempo_val
                        print(f"Updated Default Tempo: {DEFAULT_TEMPO}")
                        tempo_found = True
                except (IndexError, ValueError): pass
            elif part.startswith("TS:") and not ts_found:
                try:
                    num_str, den_str = part.split(":", 1)[1].split('/')
                    ts_num, ts_den = int(num_str), int(den_str)
                    if ts_num > 0 and ts_den > 0:
                         DEFAULT_TIMESIG = (ts_num, ts_den)
                         print(f"Updated Default Time Signature: {DEFAULT_TIMESIG}")
                         ts_found = True
                except (IndexError, ValueError): pass
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

    # Basic context from the previous section's *intended* parameters/goal
    # Note: More advanced context would involve parsing the *actual* end state of the previous section.
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
2. If tempo (T), time signature (TS), key (K), or instrument (INST) need to be set or changed *at the very beginning* of this section (before the first note/rest), include those commands *before* `BAR:{current_bar}`. Otherwise, assume they carry over or use defaults (T:{DEFAULT_TEMPO}, TS:{DEFAULT_TIMESIG[0]}/{DEFAULT_TIMESIG[1]}, K:{DEFAULT_KEY}).
3. Strictly adhere to the compact symbolic format defined below. Do NOT include any other text, explanations, apologies, or formatting like ```mus``` or ```. Output ONLY the commands, each on a new line.
4. Ensure musical coherence within the section and try to follow the Section Goal. Make smooth transitions if context from a previous section is provided (e.g., connect harmonically or rhythmically).
5. The total duration of notes/rests/chords within each bar MUST add up correctly according to the active time signature (e.g., 4 quarter notes in 4/4). Be precise.
6. End the generation cleanly *after* the content for bar {current_bar + bars - 1} is complete, before the next section would start (i.e., do NOT include BAR:{current_bar + bars}).

{SYMBOLIC_FORMAT_DEFINITION}

Generate Section {section_name} symbolic music now (starting BAR:{current_bar}):
"""

    symbolic_text = call_gemini(prompt)

    if symbolic_text:
        # Clean the response: remove potential markdown code fences and surrounding whitespace/text
        symbolic_text = re.sub(r"^```[a-z]*\n", "", symbolic_text, flags=re.MULTILINE)
        symbolic_text = re.sub(r"\n```$", "", symbolic_text)
        symbolic_text = symbolic_text.strip() # Remove leading/trailing whitespace

        # Basic validation: Check if it starts roughly correctly or find the start
        lines = symbolic_text.split('\n')
        first_meaningful_line = ""
        start_index = 0
        for idx, line in enumerate(lines):
            line = line.strip()
            if line and not line.startswith('#'): # Skip comments/empty
                first_meaningful_line = line
                start_index = idx
                break

        bar_marker = f"BAR:{current_bar}"
        # Check if the first non-empty line is the expected BAR marker or a setting (T/TS/K/INST)
        is_valid_start = first_meaningful_line.startswith(bar_marker) or \
                         re.match(r"^(T:|TS:|K:|INST:)", first_meaningful_line)

        if not is_valid_start:
            print(f"Warning: Generated text for {section_name} doesn't seem to start correctly.")
            print(f"Expected '{bar_marker}' or INST/T/TS/K. Got: '{first_meaningful_line[:60]}...'")
            # Attempt to find the correct starting BAR marker if LLM added preamble
            bar_marker_index = symbolic_text.find(bar_marker)
            if bar_marker_index != -1:
                print(f"Found '{bar_marker}' later in the text. Attempting to use text from that point.")
                # Find the beginning of the line containing the bar marker
                start_pos = symbolic_text.rfind('\n', 0, bar_marker_index) + 1
                symbolic_text = symbolic_text[start_pos:]
            else:
                print(f"Could not find '{bar_marker}'. The generated section might be incorrect or incomplete.")
                # Decide whether to return None or the potentially incorrect text
                # return "", None # Option: reject the section if start is critical

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
    # { ('InstName', 'TrackID'): [note_obj1, ...], ... }
    notes_by_instrument_track = {}
    # Store changes with time: [(time, value), ...]
    tempo_changes = []
    # Store time sig changes with time: [(time, num, den), ...]
    time_signature_changes = []
    # Store key sig changes: [(time, key_name)]
    key_signature_changes = []
    # Store instrument definitions: { ('InstName', 'TrackID'): {'program': X, 'is_drum': Y, 'name': Z}, ... }
    instrument_definitions = {}

    # --- Time Tracking State ---
    current_global_time = 0.0
    time_within_bar = 0.0
    current_bar_start_time = 0.0
    # --- Musical State ---
    current_tempo = float(DEFAULT_TEMPO)
    current_ts_num, current_ts_den = DEFAULT_TIMESIG
    current_key = DEFAULT_KEY
    # Track active instrument *per track ID* - allows multi-instrument definition
    # If INST is used without a specific track, it might apply as a default?
    # Current implementation: INST sets a global default instrument context.
    active_instrument_name = "Pno" # Default instrument if none specified early
    # --- Bar Tracking ---
    current_bar_number = 0 # Start before bar 1
    expected_bar_duration_sec = (60.0 / current_tempo) * current_ts_num * (4.0 / current_ts_den)
    last_event_time = 0.0 # Track the end time of the last event

    # Ensure initial tempo/TS/Key are set at time 0 if not already set by early commands
    initial_commands_set = {'T': False, 'TS': False, 'K': False, 'INST': False}

    lines = symbolic_text.strip().split('\n')
    line_num = 0

    # --- Pre-pass for initial settings before the first BAR marker ---
    initial_settings_lines = []
    while line_num < len(lines):
        line = lines[line_num].strip()
        if not line or line.startswith('#'):
            line_num += 1
            continue
        # Stop pre-pass when first BAR is encountered or if non-setting command found
        if line.startswith("BAR:") or not re.match(r"^(T:|TS:|K:|INST:)", line):
            break
        initial_settings_lines.append((line_num + 1, line))
        line_num += 1

    # Process initial settings at time 0.0
    print("Processing initial settings (before first BAR marker)...")
    for ini_line_num, ini_line in initial_settings_lines:
        parts = ini_line.split(':', 1)
        command = parts[0].upper()
        value = parts[1].strip() if len(parts) > 1 else ""

        try:
            if command == "INST" and not initial_commands_set['INST']:
                if value:
                    active_instrument_name = value
                    initial_commands_set['INST'] = True
                    print(f"Initial Instrument set to {active_instrument_name}")
            elif command == "T" and not initial_commands_set['T']:
                new_tempo = float(value)
                if new_tempo > 0 and new_tempo != current_tempo:
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
            # else: # Ignore commands that are not settings or already set
            #     print(f"Warning: Command '{command}' before first BAR on line {ini_line_num} ignored or already set.")

        except Exception as e:
            print(f"Error parsing initial setting line {ini_line_num}: '{ini_line}' - {e}")

    # Add default initial settings if not overridden by commands before the first BAR
    if not initial_commands_set['T']: tempo_changes.append((0.0, current_tempo))
    if not initial_commands_set['TS']: time_signature_changes.append((0.0, current_ts_num, current_ts_den))
    if not initial_commands_set['K']: key_signature_changes.append((0.0, current_key))
    # Initial instrument is handled via active_instrument_name state


    # --- Main Parsing Loop (starting from where pre-pass left off) ---
    print("Parsing main body...")
    for i in range(line_num, len(lines)):
        current_line_num = i + 1 # Use 1-based indexing for messages
        line = lines[i].strip()
        if not line or line.startswith('#'): # Skip empty lines and comments
            continue

        parts = line.split(':', 1)
        command = parts[0].upper()
        value = parts[1] if len(parts) > 1 else ""

        try:
            if command == "INST":
                new_instrument = value.strip()
                if new_instrument and new_instrument != active_instrument_name:
                    active_instrument_name = new_instrument
                    print(f"Time {current_global_time:.2f}s (Bar ~{current_bar_number}): Instrument context changed to {active_instrument_name}")
                    # Note: This only sets the *default* for subsequent N/C/R commands.
                    # The actual instrument object is created when a note for that inst/track appears.

            elif command == "T":
                new_tempo = float(value.strip())
                if new_tempo > 0 and new_tempo != current_tempo:
                    print(f"Time {current_global_time:.2f}s (Bar ~{current_bar_number}): Tempo change to {new_tempo} BPM")
                    tempo_changes.append((current_global_time, new_tempo))
                    current_tempo = new_tempo
                    # Recalculate expected bar duration based on new tempo
                    expected_bar_duration_sec = (60.0 / current_tempo) * current_ts_num * (4.0 / current_ts_den)

            elif command == "TS":
                num_str, den_str = value.strip().split('/')
                new_ts_num, new_ts_den = int(num_str), int(den_str)
                if new_ts_num > 0 and new_ts_den > 0 and (new_ts_num, new_ts_den) != (current_ts_num, current_ts_den):
                    print(f"Time {current_global_time:.2f}s (Bar ~{current_bar_number}): Time Signature change to {new_ts_num}/{new_ts_den}")
                    time_signature_changes.append((current_global_time, new_ts_num, new_ts_den))
                    current_ts_num, current_ts_den = new_ts_num, new_ts_den
                    # Recalculate expected bar duration based on new time signature
                    expected_bar_duration_sec = (60.0 / current_tempo) * current_ts_num * (4.0 / current_ts_den)

            elif command == "K":
                new_key = value.strip()
                if new_key and new_key != current_key:
                    current_key = new_key
                    print(f"Time {current_global_time:.2f}s (Bar ~{current_bar_number}): Key set to {current_key}")
                    key_signature_changes.append((current_global_time, current_key))

            elif command == "BAR":
                bar_number = int(value.strip())
                # --- Bar Time Synchronization Logic ---
                if current_bar_number > 0: # If this isn't the first bar marker encountered
                    # Expected end time of the previous bar
                    expected_prev_bar_end_time = current_bar_start_time + expected_bar_duration_sec
                    # Calculate the time difference (error)
                    duration_error = time_within_bar - expected_bar_duration_sec
                    # Allow a small tolerance (e.g., 1% of bar duration or a fixed small amount like 10ms)
                    tolerance = max(0.01, expected_bar_duration_sec * 0.01)

                    if abs(duration_error) > tolerance:
                        print(
                            f"Warning: Bar {current_bar_number} timing mismatch on Line {current_line_num}. "
                            f"Expected duration {expected_bar_duration_sec:.3f}s, accumulated {time_within_bar:.3f}s "
                            f"(Error: {duration_error:.3f}s). Adjusting global time to expected bar start."
                        )
                        # Force global time to the expected start time of the new bar to prevent drift
                        current_global_time = expected_prev_bar_end_time
                    else:
                        # If within tolerance, use the accumulated time for smoother transitions
                        current_global_time = current_bar_start_time + time_within_bar

                elif bar_number != 1:
                    # If the first BAR command is not BAR:1, assume time starts at 0 but log warning
                    print(f"Warning: First BAR marker is BAR:{bar_number} (Line {current_line_num}), not BAR:1. Starting time from 0.")
                    current_global_time = 0.0 # Ensure time starts at 0 if first bar isn't 1


                # Handle jumps in bar numbers (assuming silent bars passed)
                # This happens *after* synchronizing the end of the previous bar
                bars_jumped = bar_number - (current_bar_number + 1)
                if bars_jumped > 0 and current_bar_number > 0: # Only jump if not the first bar
                    jump_duration = bars_jumped * expected_bar_duration_sec
                    print(
                        f"Warning: Jump detected from Bar {current_bar_number} to {bar_number} (Line {current_line_num}). "
                        f"Advancing time by ~{jump_duration:.3f}s ({bars_jumped} bars)."
                    )
                    current_global_time += jump_duration

                # Update current bar state for the new bar
                current_bar_number = bar_number
                current_bar_start_time = current_global_time # This bar starts now
                time_within_bar = 0.0 # Reset time accumulation for the new bar
                # Optional: print(f"Time {current_global_time:.2f}s: --- Reached BAR {bar_number} ---")

            elif command in ["N", "C", "R"]: # Note, Chord, or Rest
                if current_bar_number == 0:
                    print(f"Warning: Event '{line}' on Line {current_line_num} found before first BAR marker. Processing at time {current_global_time:.2f}s.")
                    # Process anyway, assuming time 0.0 or current time if settings changed it.

                data_parts = value.split(':')
                min_parts = 3 if command == "R" else 4 # R:Track:Dur, N/C:Track:Data:Dur:Vel
                if len(data_parts) < min_parts:
                    print(f"Warning: Malformed {command} command on Line {current_line_num}: '{line}'. Requires at least {min_parts} parts separated by ':'. Skipping.")
                    continue

                track_id = data_parts[0].strip()
                # Use the globally set instrument context for this event
                inst_name_for_event = active_instrument_name

                # Determine if this track/instrument is drums
                # Treat track ID "Drums" or instrument name "Drs"/"Drums" as drum track
                is_drum_track = inst_name_for_event.lower() in ["drs", "drums"] or track_id.lower() == "drums"
                program = INSTRUMENT_PROGRAM_MAP.get(inst_name_for_event, 0) # Default to Piano if name not found

                # Define instrument properties if not seen before for this specific InstName/TrackID combo
                inst_track_key = (inst_name_for_event, track_id)
                if inst_track_key not in instrument_definitions:
                    pm_instrument_name = f"{inst_name_for_event}-{track_id}" # Unique name for pretty_midi instrument
                    instrument_definitions[inst_track_key] = {
                        "program": program,
                        "is_drum": is_drum_track,
                        "name": pm_instrument_name
                    }
                    print(f"Defined instrument: {pm_instrument_name} (Program: {program}, IsDrum: {is_drum_track})")

                # Ensure note list exists for this instrument/track combination
                if inst_track_key not in notes_by_instrument_track:
                    notes_by_instrument_track[inst_track_key] = []

                event_start_time = current_global_time # Event starts at the current time
                event_duration_sec = 0.0

                if command == "N": # Note:Track:Pitch:Duration:Velocity
                    if len(data_parts) < 4:
                        print(f"Warning: Malformed Note command on Line {current_line_num}: '{line}'. Requires Track:Pitch:Duration:Velocity. Skipping.")
                        continue
                    pitch_name = data_parts[1].strip()
                    duration_sym = data_parts[2].strip()
                    velocity_str = data_parts[3].strip()
                    try:
                        velocity = int(velocity_str)
                    except ValueError:
                        print(f"Warning: Invalid velocity '{velocity_str}' on Line {current_line_num}. Using 64.")
                        velocity = 64

                    event_duration_sec = duration_to_seconds(duration_sym, current_tempo, current_ts_den)

                    # Drum mapping or regular pitch mapping
                    if is_drum_track:
                        # Use original pitch name (e.g., "Kick", "Snare") for drum map lookup
                        midi_pitch = DRUM_PITCH_MAP.get(pitch_name)
                        if midi_pitch is None:
                             # Try case-insensitive match as fallback
                             midi_pitch = DRUM_PITCH_MAP.get(pitch_name.capitalize())
                             if midi_pitch is None:
                                 print(f"Warning: Unknown drum sound '{pitch_name}' on Line {current_line_num}. Using Kick (36).")
                                 midi_pitch = 36 # Default to Kick
                    else:
                        midi_pitch = pitch_to_midi(pitch_name)

                    note_event = {
                        "pitch": midi_pitch,
                        "start": event_start_time,
                        "end": event_start_time + event_duration_sec,
                        "velocity": max(0, min(127, velocity)) # Clamp velocity
                    }
                    notes_by_instrument_track[inst_track_key].append(note_event)

                elif command == "C": # Chord:Track:[P1,P2,...]:Duration:Velocity
                    if len(data_parts) < 4:
                        print(f"Warning: Malformed Chord command on Line {current_line_num}: '{line}'. Requires Track:[Pitches]:Duration:Velocity. Skipping.")
                        continue
                    pitches_str = data_parts[1].strip()
                    duration_sym = data_parts[2].strip()
                    velocity_str = data_parts[3].strip()
                    try:
                        velocity = int(velocity_str)
                    except ValueError:
                        print(f"Warning: Invalid velocity '{velocity_str}' on Line {current_line_num}. Using 64.")
                        velocity = 64


                    # Handle brackets and split pitches, filter empty strings
                    if pitches_str.startswith('[') and pitches_str.endswith(']'):
                        pitches_str = pitches_str[1:-1]
                    pitch_names = [p.strip() for p in pitches_str.split(',') if p.strip()]

                    if not pitch_names:
                        print(f"Warning: No valid pitches found in Chord command on Line {current_line_num}: '{line}'. Skipping.")
                        continue

                    event_duration_sec = duration_to_seconds(duration_sym, current_tempo, current_ts_den)

                    for pitch_name in pitch_names:
                        # Assume chords aren't typically on drum tracks, use standard pitch conversion
                        if is_drum_track:
                            print(f"Warning: Chord command used on drum track (Line {current_line_num}). Treating pitches as standard notes, not drum sounds.")
                        midi_pitch = pitch_to_midi(pitch_name)
                        note_event = {
                            "pitch": midi_pitch,
                            "start": event_start_time,
                            "end": event_start_time + event_duration_sec,
                            "velocity": max(0, min(127, velocity)) # Clamp velocity
                        }
                        # Add chord notes to the same instrument/track key
                        notes_by_instrument_track[inst_track_key].append(note_event)

                elif command == "R": # Rest:Track:Duration
                    if len(data_parts) < 2:
                        print(f"Warning: Malformed Rest command on Line {current_line_num}: '{line}'. Requires Track:Duration. Skipping.")
                        continue
                    # Track ID is data_parts[0], Duration is data_parts[1]
                    duration_sym = data_parts[1].strip()
                    event_duration_sec = duration_to_seconds(duration_sym, current_tempo, current_ts_den)
                    # Rests just advance time, no note event added. They belong to a track conceptually,
                    # but don't generate sound. The time advance handles the silence for all tracks implicitly.

                # --- Advance Time ---
                # Advance global time based on the duration of the note/chord/rest
                current_global_time += event_duration_sec
                # Accumulate time within the current bar
                time_within_bar += event_duration_sec
                # Keep track of the latest time point reached by any event
                last_event_time = max(last_event_time, current_global_time)

            else:
                print(f"Warning: Unknown command '{command}' on line {current_line_num}: '{line}'. Skipping.")

        except Exception as e:
            print(f"Error parsing line {current_line_num}: '{line}' - {e}")
            traceback.print_exc() # Print full traceback for debugging
            continue # Skip problematic line

    print(f"Symbolic text parsing complete. Estimated total duration: {last_event_time:.2f} seconds.")
    # Return all parsed information
    return (
        notes_by_instrument_track,
        instrument_definitions,
        tempo_changes,
        time_signature_changes,
        key_signature_changes,
        last_event_time, # Return the calculated end time
        current_key, # Return the final key found during parsing
        current_tempo # Return the final tempo found during parsing
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
    if not notes_data:
        print("Error: No note data was successfully parsed. Cannot create MIDI file.")
        return

    try:
        # Initialize with the first tempo change or default
        initial_tempo = tempo_changes[0][1] if tempo_changes else DEFAULT_TEMPO
        midi_obj = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo)
        print(f"Initialized MIDI with tempo: {initial_tempo:.2f} BPM")

        # Add Time Signature Changes
        # Sort by time just in case they weren't added chronologically during parsing
        time_sig_changes.sort(key=lambda x: x[0])
        applied_ts_count = 0
        last_ts_time = -1.0 # Track time of last added TS to avoid duplicates
        for time, num, den in time_sig_changes:
            # Denominator needs to be power of 2 for standard MIDI
            den_power = den.bit_length() - 1
            actual_den = 2**den_power
            if actual_den != den:
                print(
                    f"Warning: Non-power-of-2 time signature denominator {den} at time {time:.2f}s."
                    f" Using closest power of 2: {actual_den}."
                )
            # Prevent adding duplicate TS at the exact same time
            if time > last_ts_time:
                ts_change = pretty_midi.TimeSignature(num, actual_den, time)
                midi_obj.time_signature_changes.append(ts_change)
                applied_ts_count += 1
                last_ts_time = time
                # print(f"Added Time Signature: {num}/{actual_den} at {time:.2f}s") # Verbose
            elif time == last_ts_time:
                print(f"Ignoring duplicate time signature change ({num}/{den}) at time {time:.2f}s.")

        if applied_ts_count > 0:
            print(f"Applied {applied_ts_count} time signature changes.")
        # Add default if none were added and the default wasn't added at time 0
        if not midi_obj.time_signature_changes:
            default_num, default_den = DEFAULT_TIMESIG
            midi_obj.time_signature_changes.append(
                pretty_midi.TimeSignature(default_num, default_den, 0.0)
            )
            print(f"Applied default time signature: {default_num}/{default_den}")


        # Add Key Signature Changes
        # Sort by time
        key_sig_changes.sort(key=lambda x: x[0])
        applied_key_count = 0
        last_key_time = -1.0 # Track time of last added key sig
        for time, key_name in key_sig_changes:
            try:
                key_number = pretty_midi.key_name_to_key_number(key_name)
                # Prevent adding duplicate key sig at the exact same time
                if time > last_key_time:
                    key_change = pretty_midi.KeySignature(key_number=key_number, time=time)
                    midi_obj.key_signature_changes.append(key_change)
                    applied_key_count += 1
                    last_key_time = time
                    # print(f"Added Key Signature: {key_name} ({key_number}) at {time:.2f}s") # Verbose
                elif time == last_key_time:
                    print(f"Ignoring duplicate key signature change ({key_name}) at time {time:.2f}s.")

            except ValueError as e:
                print(
                    f"Warning: Could not parse key signature '{key_name}' at time {time:.2f}s. Skipping. Error: {e}"
                )
        if applied_key_count > 0:
            print(f"Applied {applied_key_count} key signature changes.")
        # Add default if none were added and the default wasn't added at time 0
        if not midi_obj.key_signature_changes:
            try:
                default_key_num = pretty_midi.key_name_to_key_number(DEFAULT_KEY)
                midi_obj.key_signature_changes.append(
                    pretty_midi.KeySignature(key_number=default_key_num, time=0.0)
                )
                print(f"Applied default key signature: {DEFAULT_KEY}")
            except ValueError as e:
                print(
                    f"Warning: Could not parse default key signature '{DEFAULT_KEY}'. Error: {e}"
                )


        # Create instruments and add notes
        midi_instruments = {} # { ('InstName', 'TrackID'): <Instrument object>, ... }
        # Assign channels, ensuring drums are on channel 9 (0-indexed)
        # MIDI channels 0-15, channel 9 is conventionally drums.
        available_channels = list(range(16))
        available_channels.pop(9) # Remove drum channel 9
        channel_index = 0

        for inst_track_key, definition in instrument_defs.items():
            # Check if this instrument actually has notes associated with it
            if inst_track_key not in notes_data or not notes_data[inst_track_key]:
                print(f"Skipping instrument '{definition['name']}' as it has no parsed notes.")
                continue # Skip instruments without notes

            is_drum = definition['is_drum']
            program = definition['program']
            pm_instrument_name = definition['name']

            # Create Instrument object
            instrument_obj = pretty_midi.Instrument(
                program=program, is_drum=is_drum, name=pm_instrument_name
            )

            # Assign MIDI channel
            if is_drum:
                channel = 9 # Standard MIDI drum channel (index 9)
                print(
                    f"Created MIDI instrument: {pm_instrument_name} (Program: {program}, IsDrum: {is_drum}, Assigned Channel: 9)"
                )
            else:
                if channel_index >= len(available_channels):
                    print(f"Warning: Ran out of non-drum MIDI channels! Reusing channel {available_channels[channel_index % len(available_channels)]} for {pm_instrument_name}.")
                    channel = available_channels[channel_index % len(available_channels)]
                else:
                    channel = available_channels[channel_index]
                print(
                    f"Created MIDI instrument: {pm_instrument_name} (Program: {program}, IsDrum: {is_drum}, Assigned Channel: {channel})"
                )
                channel_index += 1


            midi_obj.instruments.append(instrument_obj)
            midi_instruments[inst_track_key] = instrument_obj # Store for adding notes

            # Add notes to this instrument
            note_count = 0
            skipped_notes = 0
            for note_info in notes_data[inst_track_key]:
                # Ensure start time is not negative and end time is strictly after start time
                start_time = max(0.0, note_info['start'])
                # Ensure minimum duration to avoid zero-length notes which can cause issues
                min_duration = 0.001 # 1ms minimum duration
                end_time = max(start_time + min_duration, note_info['end'])

                # Additional validation: Ensure velocity and pitch are within MIDI range
                velocity = max(0, min(127, note_info['velocity']))
                pitch = max(0, min(127, note_info['pitch']))

                if start_time >= end_time:
                     print(f"Warning: Skipping note with non-positive duration for {pm_instrument_name}. Pitch: {pitch}, Start: {start_time:.3f}, End: {end_time:.3f}")
                     skipped_notes += 1
                     continue

                try:
                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=start_time,
                        end=end_time,
                    )
                    instrument_obj.notes.append(note)
                    note_count += 1
                except ValueError as e:
                    print(f"Error creating pretty_midi.Note for {pm_instrument_name}: {e}. Skipping note.")
                    print(f"  Note data: Pitch={pitch}, Vel={velocity}, Start={start_time}, End={end_time}")
                    skipped_notes += 1

            print(f"Added {note_count} notes to instrument {pm_instrument_name}. ({skipped_notes} notes skipped due to errors/duration issues).")


        # Handle Tempo Changes (beyond the initial one)
        # pretty_midi applies tempo changes based on note timings relative to the initial tempo.
        # For explicit tempo change meta-messages in the MIDI file (needed by some sequencers):
        if len(tempo_changes) > 1:
            print(
                f"Note: {len(tempo_changes)} tempo changes found. pretty_midi primarily uses initial tempo. "
                "Explicit tempo change meta-messages are not directly added by this script, "
                "but timing should reflect tempo changes."
            )
            # If precise MIDI tempo events are needed, it requires manually creating
            # tempo change events (`_tick_scale`, `_tempo_change_ticks`, `_tempo_change_times`)
            # or manipulating the underlying MIDI track events, which is complex.
            # For most players, the note timings derived from the parser's `current_global_time`
            # (which accounts for tempo changes) should suffice.


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
    last_section_summary_info = None # Store dict with name, summary, key, tempo for context
    generated_sections_count = 0

    for section_name, section_info in SECTIONS.items():
        symbolic_section, current_section_summary_info = generate_symbolic_section(
            enriched_description,
            section_name,
            section_info,
            current_bar_count,
            last_section_summary_info, # Pass summary dict of previous section's goal
        )

        if symbolic_section and current_section_summary_info: # Check both generated successfully
            # Ensure section ends with a newline for clean concatenation
            if not symbolic_section.endswith('\n'):
                symbolic_section += '\n'
            all_symbolic_text += symbolic_section
            generated_sections_count += 1

            # Update the summary info to pass to the *next* iteration's context prompt
            last_section_summary_info = current_section_summary_info

            # Advance bar count for the next section's *starting* point
            current_bar_count += section_info['bars']
        else:
            print(f"Skipping section {section_name} due to generation failure.")
            # Decide whether to stop or continue without the failed section
            # break # Uncomment to stop the whole process if one section fails

    if not all_symbolic_text.strip():
        print("\nERROR: No symbolic text was generated successfully. Cannot proceed.")
        exit(1)
    if generated_sections_count < len(SECTIONS):
        print(f"\nWarning: Only {generated_sections_count}/{len(SECTIONS)} sections were generated successfully.")

    print("\n--- Combined Symbolic Text ---")
    # Save combined text to a file for debugging/inspection
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    symbolic_filename = os.path.join(OUTPUT_DIR, f"symbolic_music_{timestamp_str}.txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        with open(symbolic_filename, "w", encoding="utf-8") as f:
            f.write(all_symbolic_text)
        print(f"Saved combined symbolic text to: {symbolic_filename}")
    except IOError as e:
        print(f"Error saving symbolic text file: {e}")

    # print(all_symbolic_text[:1000] + "...") # Optional: Print beginning of combined text
    print("----------------------------")

    # Step 3: Parse the combined symbolic text
    (
        parsed_notes,
        instrument_definitions,
        tempo_changes,
        time_sig_changes,
        key_sig_changes,
        # Ignore end time, key, tempo from parser return here, as create_midi uses the change lists
        _, _, _
    ) = parse_symbolic_to_structured_data(all_symbolic_text)


    # Step 4: Create the MIDI file
    if parsed_notes: # Only create MIDI if parsing yielded notes for at least one instrument
        # Generate filename with timestamp
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
        print("\nError: No notes were successfully parsed from the symbolic text. MIDI file not created.")

    print("\n--- AutoMusic Generator Pipeline Finished ---")