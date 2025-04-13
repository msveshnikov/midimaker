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
GEMINI_KEY = os.environ.get("GEMINI_KEY", "") # Placeholder - Replace or use env var

# Configure the Gemini model to use
GEMINI_MODEL = "gemini-2.5-pro-exp-03-25" #"gemini-2.0-flash-thinking-exp-01-21" 

# Configuration dictionary
CONFIG = {
    "api_key": GEMINI_KEY,
    "gemini_model": GEMINI_MODEL,
    "initial_description": "trans hiphop cosmic ambient music fast 160 bpm",
    "output_dir": "output",
    "default_tempo": 120,
    "default_timesig": (4, 4),
    "default_key": "Amin",
    "generation_retries": 3,
    "generation_delay": 5,
    "max_total_bars": 128,  # Limit total length for safety/cost
    "min_section_bars": 16,
    "max_section_bars": 32,
    "temperature": 0.7,
}


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
    if not CONFIG["api_key"]:
        print(
            "ERROR: GEMINI_KEY environment variable is not set or config is None."
            " Please set the GEMINI_KEY environment variable."
        )
        exit(1)
    try:
        genai.configure(api_key=CONFIG["api_key"])
        print(
            f"Google Generative AI configured using model: {CONFIG['gemini_model']}"
        )
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
    retries = retries if retries is not None else CONFIG["generation_retries"]
    delay = delay if delay is not None else CONFIG["generation_delay"]
    model = genai.GenerativeModel(CONFIG["gemini_model"])
    gen_config_args = {}
    if output_format == "json":
        gen_config_args["response_mime_type"] = "application/json"
        gen_config_args["temperature"] = CONFIG["temperature"]

    generation_config = genai.types.GenerationConfig(**gen_config_args)

    for attempt in range(retries):
        try:
            response = model.generate_content(
                prompt, generation_config=generation_config
            )

            # Handle potential API response variations
            if response.parts:
                content = response.text.strip()
                if output_format == "json":
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError as json_e:
                        print(
                            f"Error decoding JSON response (Attempt {attempt + 1}): {json_e}"
                        )
                        print(f"Received text: {content[:500]}...")
                        # Fall through to retry logic
                else:
                    return content  # Return text directly

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
                # Try to return partial content if available and valid
                if (
                    response.candidates[0].content
                    and response.candidates[0].content.parts
                ):
                    partial_content = response.candidates[0].content.parts[
                        0
                    ].text.strip()
                    print(
                        "Returning partial content due to non-STOP finish reason."
                    )
                    if output_format == "json":
                        try:
                            # Attempt to parse potentially incomplete JSON
                            return json.loads(partial_content)
                        except json.JSONDecodeError:
                            print(
                                "Partial content is not valid JSON. Failing."
                            )
                            return None
                    else:
                        return partial_content
                return None  # Indicate non-standard stop without content

            # Check candidates directly if parts is empty but no explicit block/error
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    print(
                        "Accessing content via candidates[0] as response.parts was empty."
                    )
                    content = candidate.content.parts[0].text.strip()
                    if output_format == "json":
                        try:
                            return json.loads(content)
                        except json.JSONDecodeError as json_e:
                            print(
                                f"Error decoding JSON from candidate (Attempt {attempt + 1}): {json_e}"
                            )
                            print(f"Received text: {content[:500]}...")
                            # Fall through to retry logic
                    else:
                        return content  # Return text directly

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
INSTRUMENT_PROGRAM_MAP = {
    # Piano
    "Pno": 0,
    "Piano": 0,
    "Acoustic Grand Piano": 0,
    "Bright Acoustic Piano": 1,
    "Electric Grand Piano": 2,
    "Honky-tonk Piano": 3,
    "Electric Piano 1": 4,
    "Rhodes Piano": 4,
    "Electric Piano 2": 5,
    # Chromatic Percussion
    "Celesta": 8,
    "Glockenspiel": 9,
    "Music Box": 10,
    "Vibraphone": 11,
    # Organ
    "Org": 16,
    "Organ": 16,
    "Drawbar Organ": 16,
    "Percussive Organ": 17,
    "Rock Organ": 18,
    # Guitar
    "Gtr": 25,
    "Acoustic Guitar": 25,
    "Nylon Guitar": 24,
    "Steel Guitar": 25,
    "Electric Guitar": 27,
    "Jazz Guitar": 26,
    "Clean Electric Guitar": 27,
    "Muted Electric Guitar": 28,
    "Overdriven Guitar": 29,
    "Distortion Guitar": 30,
    # Bass
    "Bass": 33,
    "Acoustic Bass": 32,
    "Electric Bass": 33,
    "Finger Bass": 33,
    "Pick Bass": 34,
    "Fretless Bass": 35,
    "Slap Bass": 36,
    "Synth Bass": 38,
    "Synth Bass 2": 39,
    # Strings
    "Str": 48,
    "Strings": 48,
    "Violin": 40,
    "Viola": 41,
    "Cello": 42,
    "Contrabass": 43,
    "Tremolo Strings": 44,
    "Pizzicato Strings": 45,
    "Orchestral Harp": 46,
    "String Ensemble 1": 48,
    "String Ensemble 2": 49,
    "Synth Strings 1": 50,
    "Synth Strings 2": 51,
    # Brass
    "Tpt": 56,
    "Trumpet": 56,
    "Trombone": 57,
    "Tuba": 58,
    "Muted Trumpet": 59,
    "French Horn": 60,
    # Reed
    "Sax": 65,
    "Soprano Sax": 64,
    "Alto Sax": 65,
    "Tenor Sax": 66,
    "Baritone Sax": 67,
    "Oboe": 68,
    "English Horn": 69,
    "Bassoon": 70,
    "Clarinet": 71,
    # Pipe
    "Flt": 73,
    "Flute": 73,
    "Piccolo": 72,
    "Recorder": 74,
    "Pan Flute": 75,
    # Synth Lead
    "SynLead": 81,
    "Synth Lead": 81,
    "Lead 1 (Square)": 80,
    "Lead 2 (Sawtooth)": 81,
    "Lead 3 (Calliope)": 82,
    "Lead 8 (Bass + Lead)": 87,
    # Synth Pad
    "SynPad": 89,
    "Synth Pad": 89,
    "Pad 1 (New Age)": 88,
    "Pad 2 (Warm)": 89,
    "Pad 3 (Polysynth)": 90,
    "Pad 4 (Choir)": 91,
    "Pad 5 (Bowed)": 92,
    # Drums are a special case (channel 10 / index 9)
    "Drs": 0,
    "Drums": 0,  # Program 0 is often used, but channel is key
}

# Standard drum note map (MIDI channel 10 / index 9)
DRUM_PITCH_MAP = {
    # Bass Drum
    "Kick": 36,
    "BD": 36,
    "Bass Drum 1": 36,
    "Acoustic Bass Drum": 35,
    # Snare
    "Snare": 38,
    "SD": 38,
    "Acoustic Snare": 38,
    "Electric Snare": 40,
    # Hi-Hat
    "HiHatClosed": 42,
    "HHC": 42,
    "Closed Hi Hat": 42,
    "Closed Hi-Hat": 42,
    "HiHatOpen": 46,
    "HHO": 46,
    "Open Hi Hat": 46,
    "Open Hi-Hat": 46,
    "HiHatPedal": 44,
    "HH P": 44,
    "Pedal Hi Hat": 44,
    "Pedal Hi-Hat": 44,
    # Cymbals
    "Crash": 49,
    "CR": 49,
    "Crash Cymbal 1": 49,
    "Crash Cymbal 2": 57,
    "Ride": 51,
    "RD": 51,
    "Ride Cymbal 1": 51,
    "Ride Cymbal 2": 59,
    "Ride Bell": 53,
    "Splash Cymbal": 55,
    "Chinese Cymbal": 52,
    # Toms
    "High Tom": 50,
    "HT": 50,
    "Hi Tom": 50,
    "Mid Tom": 47,
    "MT": 47,
    "Hi-Mid Tom": 48,
    "Low-Mid Tom": 47,
    "Low Tom": 43,
    "LT": 43,
    "High Floor Tom": 43,
    "Low Floor Tom": 41,
    "Floor Tom": 41,
    "FT": 41,
    # Other
    "Rimshot": 37,
    "RS": 37,
    "Side Stick": 37,
    "Clap": 39,
    "CP": 39,
    "Hand Clap": 39,
    "Cowbell": 56,
    "CB": 56,
    "Tambourine": 54,
    "Tmb": 54,
    "Claves": 75,
    "Wood Block": 76,
    "High Wood Block": 76,
    "Low Wood Block": 77,
}


def pitch_to_midi(pitch_name):
    """Converts pitch name (e.g., C4, F#5) to MIDI number."""
    pitch_name = pitch_name.strip()
    match = re.match(r"([A-G])([#b]?)(\-?\d+)", pitch_name, re.IGNORECASE)
    if not match:
        print(
            f"Warning: Could not parse pitch name: '{pitch_name}'. Defaulting to Middle C (60)."
        )
        return 60

    note, acc, oct_str = match.groups()
    octave = int(oct_str)

    base_midi = PITCH_MAP.get(note.upper())
    if base_midi is None:
        print(f"Warning: Invalid note name: '{note}'. Defaulting to Middle C (60).")
        return 60

    acc_val = ACCIDENTAL_MAP.get(acc, 0)
    midi_val = base_midi + acc_val + (octave + 1) * 12
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
    Take the following basic music description and enrich it with more detail for a generative music task.
    Suggest a plausible key signature (e.g., K:Cmin, K:Gmaj, K:F#dor), tempo (e.g., T:90), time signature (e.g., TS:4/4),
    and primary instrumentation (e.g., INST:Pno, INST:Gtr).
    Also, briefly elaborate on the mood and suggest a simple musical structure (like AABA, ABAC, Verse-Chorus).
    Keep the output concise and focused on these musical parameters. Output should start directly with the parameters.

    Basic Description: "{description}"

    Enriched Description (Example Format: K:Amin T:70 TS:4/4 INST:Pno Mood: Melancholic, sparse, simple melody over block chords. Structure: A-B-A-Coda):
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
        key_match = re.search(
            r"K:([A-Ga-g][#b]?(?:maj|min|dor|phr|lyd|mix|loc|aeo|ion)?)",
            enriched,
        )
        tempo_match = re.search(r"T:(\d+)", enriched)
        ts_match = re.search(r"TS:(\d+)/(\d+)", enriched)
        inst_match = re.search(r"INST:(\w+)", enriched, re.IGNORECASE)
        struct_match = re.search(
            r"Structure:\s*([\w\-]+)", enriched, re.IGNORECASE
        )

        if key_match:
            current_key = key_match.group(1)
            print(f"Updated Default Key: {current_key}")
        if tempo_match:
            try:
                tempo_val = int(tempo_match.group(1))
                if tempo_val > 0:
                    current_tempo = tempo_val
                    print(f"Updated Default Tempo: {current_tempo}")
            except ValueError:
                pass
        if ts_match:
            try:
                ts_num, ts_den = int(ts_match.group(1)), int(ts_match.group(2))
                if ts_num > 0 and ts_den > 0 and ts_den in [1, 2, 4, 8, 16, 32]:
                    current_timesig = (ts_num, ts_den)
                    print(f"Updated Default Time Signature: {current_timesig}")
            except ValueError:
                pass
        if inst_match:
            primary_instrument = inst_match.group(1)
            print(f"Identified Primary Instrument: {primary_instrument}")
        if struct_match:
            structure_hint = struct_match.group(1)
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
The plan should be a JSON object where keys are section names (e.g., "A1", "B", "A2", "Coda") and values are objects containing:
1. "bars": An integer number of bars for the section (between {CONFIG['min_section_bars']} and {CONFIG['max_section_bars']}).
2. "goal": A concise string describing the musical purpose or content of the section (e.g., "Introduce main theme", "Develop theme with variation", "Contrasting bridge section", "Return to main theme, fuller texture", "Concluding Coda").

The total number of bars across all sections should ideally be around {CONFIG['max_total_bars'] // 2} to {CONFIG['max_total_bars']}, but strictly not exceed {CONFIG['max_total_bars']}.
Use the suggested structure "{structure_hint}" as a guide, potentially adapting it (e.g., AABA -> A1 B A2 Coda). Ensure section names are unique.

Enriched Description:
{enriched_desc}

Generate the JSON plan now:
"""

    plan_json = call_gemini(prompt, output_format="json")

    if plan_json and isinstance(plan_json, dict):
        # Validate the plan
        validated_plan = {}
        total_bars = 0
        is_valid = True
        for name, info in plan_json.items():
            if (
                isinstance(info, dict)
                and "bars" in info
                and isinstance(info["bars"], int)
                and "goal" in info
                and isinstance(info["goal"], str)
                and CONFIG["min_section_bars"]
                <= info["bars"]
                <= CONFIG["max_section_bars"]
            ):
                if total_bars + info["bars"] <= CONFIG["max_total_bars"]:
                    validated_plan[name] = info
                    total_bars += info["bars"]
                else:
                    print(
                        f"Warning: Section '{name}' exceeds max total bars ({CONFIG['max_total_bars']}). Truncating plan."
                    )
                    is_valid = False
                    break  # Stop adding sections
            else:
                print(
                    f"Warning: Invalid format for section '{name}' in generated plan. Skipping."
                )
                is_valid = False
                # Continue validating other sections if possible

        if not validated_plan:
            print("ERROR: Failed to generate a valid section plan. Cannot proceed.")
            return None

        print("Generated Section Plan:")
        for name, info in validated_plan.items():
            print(f"  - {name} ({info['bars']} bars): {info['goal']}")
        print(f"Total Bars in Plan: {total_bars}")
        return validated_plan
    else:
        print("ERROR: Failed to generate section plan JSON from LLM. Cannot proceed.")
        return None


def generate_symbolic_section(
    overall_desc, section_name, section_info, current_bar, previous_section_summary=None
):
    """Step 3: Generate symbolic music for one section using LLM."""
    print(
        f"--- Step 3: Generating Symbolic Section {section_name} (Starting Bar: {current_bar}) ---"
    )
    bars = section_info["bars"]
    goal = section_info["goal"]

    context_prompt = ""
    if previous_section_summary:
        prev_name = previous_section_summary.get("name", "Previous")
        prev_summary = previous_section_summary.get("summary", "No summary")
        prev_key = previous_section_summary.get("key", "Unknown")
        prev_tempo = previous_section_summary.get("tempo", "Unknown")
        context_prompt = (
            f"Context from previous section ({prev_name}): {prev_summary}\n"
            f"It aimed for key {prev_key} and tempo {prev_tempo}.\n"
            "Ensure a smooth musical transition if possible.\n"
        )

    # Use the potentially updated defaults from CONFIG
    default_tempo = CONFIG["default_tempo"]
    default_timesig = CONFIG["default_timesig"]
    default_key = CONFIG["default_key"]

    prompt = f"""
You are a symbolic music generator. Your task is to generate ONLY the symbolic music notation for a specific section of a piece, following the provided format strictly.

Overall Music Goal: {overall_desc}
{context_prompt}
Current Section: {section_name}
Target Bars: {bars} (Start this section exactly at BAR:{current_bar}, end before BAR:{current_bar + bars})
Section Goal: {goal}

Instructions:
1. Generate music ONLY for this section, starting *exactly* with `BAR:{current_bar}`.
2. If tempo (T), time signature (TS), key (K), or instrument (INST) need to be set or changed *at the very beginning* of this section (before the first BAR marker), include those commands *before* `BAR:{current_bar}`. Otherwise, assume they carry over or use defaults (T:{default_tempo}, TS:{default_timesig[0]}/{default_timesig[1]}, K:{default_key}). You can change INST multiple times within the section if needed for different instruments/tracks.
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
        symbolic_text = re.sub(
            r"^```[a-z]*\n", "", symbolic_text, flags=re.MULTILINE
        )
        symbolic_text = re.sub(r"\n```$", "", symbolic_text)
        symbolic_text = symbolic_text.strip()

        lines = symbolic_text.split("\n")
        first_meaningful_line = ""
        first_meaningful_line_index = -1
        for idx, line in enumerate(lines):
            line = line.strip()
            if line and not line.startswith("#"):
                first_meaningful_line = line
                first_meaningful_line_index = idx
                break

        if first_meaningful_line_index == -1:
            print(
                f"Warning: Generated text for {section_name} appears empty or only contains comments."
            )
            return "", None  # Treat as failure

        bar_marker = f"BAR:{current_bar}"
        is_valid_start = first_meaningful_line.startswith(
            bar_marker
        ) or re.match(r"^(T:|TS:|K:|INST:)", first_meaningful_line)

        if not is_valid_start:
            print(
                f"Warning: Generated text for {section_name} doesn't seem to start correctly."
            )
            print(
                f"Expected '{bar_marker}' or INST/T/TS/K. Got: '{first_meaningful_line[:60]}...'"
            )
            bar_marker_index_in_text = symbolic_text.find(bar_marker)
            if bar_marker_index_in_text != -1:
                print(
                    f"Found '{bar_marker}' later in the text. Attempting to use text from that point."
                )
                start_pos = (
                    symbolic_text.rfind("\n", 0, bar_marker_index_in_text) + 1
                )
                symbolic_text = symbolic_text[start_pos:]
            else:
                print(
                    f"Could not find '{bar_marker}'. The generated section might be incorrect or incomplete."
                )
        elif first_meaningful_line_index > 0:
            print(
                f"Trimming {first_meaningful_line_index} lines of potential preamble from {section_name} output."
            )
            symbolic_text = "\n".join(lines[first_meaningful_line_index:])

        print(
            f"Generated symbolic text for Section {section_name} (first 300 chars):\n{symbolic_text[:300]}...\n"
        )

        summary_info = {
            "name": section_name,
            "summary": f"Generated {bars} bars targeting goal: {goal}.",
            "key": default_key,
            "tempo": default_tempo,
        }
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
    instrument_definitions = {}

    current_track_times = {}
    current_global_time = 0.0

    current_tempo = float(CONFIG["default_tempo"])
    current_ts_num, current_ts_den = CONFIG["default_timesig"]
    current_key = CONFIG["default_key"]
    active_instrument_name = "Pno"

    current_bar_number = 0
    current_bar_start_time = 0.0
    time_within_bar_per_track = {}
    expected_bar_duration_sec = (
        (60.0 / current_tempo) * current_ts_num * (4.0 / current_ts_den)
    )
    last_event_end_time = 0.0

    initial_commands_set = {"T": False, "TS": False, "K": False}
    lines = symbolic_text.strip().split("\n")
    line_num = 0

    # --- Pre-pass for initial settings ---
    print("Processing initial settings (before first BAR marker)...")
    while line_num < len(lines):
        line = lines[line_num].strip()
        if not line or line.startswith("#"):
            line_num += 1
            continue
        if line.startswith("BAR:") or not re.match(r"^(T:|TS:|K:|INST:)", line):
            break

        parts = line.split(":", 1)
        command = parts[0].upper()
        value = parts[1].strip() if len(parts) > 1 else ""
        ini_line_num = line_num + 1

        try:
            if command == "INST":
                if value:
                    active_instrument_name = value
                    print(
                        f"Initial Instrument context set to {active_instrument_name}"
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
                    and new_ts_den > 0
                    and (new_ts_num, new_ts_den)
                    != (current_ts_num, current_ts_den)
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
                    current_key = value
                    key_signature_changes.append((0.0, current_key))
                    initial_commands_set["K"] = True
                    print(f"Initial Key set to {current_key}")

        except Exception as e:
            print(
                f"Error parsing initial setting line {ini_line_num}: '{line}' - {e}"
            )
        line_num += 1

    if not initial_commands_set["T"]:
        tempo_changes.append((0.0, current_tempo))
    if not initial_commands_set["TS"]:
        time_signature_changes.append((0.0, current_ts_num, current_ts_den))
    if not initial_commands_set["K"]:
        key_signature_changes.append((0.0, current_key))

    # --- Main Parsing Loop ---
    print("Parsing main body...")
    for i in range(line_num, len(lines)):
        current_line_num = i + 1
        line = lines[i].strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split(":", 1)
        command = parts[0].upper()
        value = parts[1].strip() if len(parts) > 1 else ""

        try:
            if command == "INST":
                if value:
                    active_instrument_name = value
            elif command == "T":
                new_tempo = float(value)
                if new_tempo > 0 and new_tempo != current_tempo:
                    tempo_changes.append((current_global_time, new_tempo))
                    current_tempo = new_tempo
                    expected_bar_duration_sec = (
                        (60.0 / current_tempo)
                        * current_ts_num
                        * (4.0 / current_ts_den)
                    )
                    print(
                        f"Time {current_global_time:.2f}s (Bar ~{current_bar_number}): Tempo change to {new_tempo} BPM"
                    )
            elif command == "TS":
                num_str, den_str = value.split("/")
                new_ts_num, new_ts_den = int(num_str), int(den_str)
                if (
                    new_ts_num > 0
                    and new_ts_den > 0
                    and (new_ts_num, new_ts_den)
                    != (current_ts_num, current_ts_den)
                ):
                    time_signature_changes.append(
                        (current_global_time, new_ts_num, new_ts_den)
                    )
                    current_ts_num, current_ts_den = new_ts_num, new_ts_den
                    expected_bar_duration_sec = (
                        (60.0 / current_tempo)
                        * current_ts_num
                        * (4.0 / current_ts_den)
                    )
                    print(
                        f"Time {current_global_time:.2f}s (Bar ~{current_bar_number}): Time Sig change to {new_ts_num}/{new_ts_den}"
                    )
            elif command == "K":
                if value and value != current_key:
                    key_signature_changes.append((current_global_time, value))
                    current_key = value
                    print(
                        f"Time {current_global_time:.2f}s (Bar ~{current_bar_number}): Key change to {current_key}"
                    )

            elif command == "BAR":
                bar_number = int(value)
                expected_new_bar_start_time = (
                    current_bar_start_time + expected_bar_duration_sec
                    if current_bar_number > 0
                    else 0.0
                )

                if current_bar_number > 0:
                    max_accumulated_time_in_prev_bar = 0.0
                    for (
                        _track_key,
                        accumulated_time,
                    ) in time_within_bar_per_track.items():
                        max_accumulated_time_in_prev_bar = max(
                            max_accumulated_time_in_prev_bar, accumulated_time
                        )

                    duration_error = (
                        max_accumulated_time_in_prev_bar
                        - expected_bar_duration_sec
                    )
                    tolerance = max(0.01, expected_bar_duration_sec * 0.01)

                    if abs(duration_error) > tolerance:
                        print(
                            f"Warning: Bar {current_bar_number} timing mismatch on Line {current_line_num}. "
                            f"Expected duration {expected_bar_duration_sec:.3f}s, max accumulated {max_accumulated_time_in_prev_bar:.3f}s "
                            f"(Error: {duration_error:.3f}s). Setting bar {bar_number} start to expected time {expected_new_bar_start_time:.3f}s."
                        )
                        current_global_time = expected_new_bar_start_time
                    else:
                        current_global_time = (
                            current_bar_start_time
                            + max_accumulated_time_in_prev_bar
                        )

                bars_jumped = bar_number - (current_bar_number + 1)
                if bars_jumped > 0 and current_bar_number > 0:
                    jump_duration = bars_jumped * expected_bar_duration_sec
                    print(
                        f"Warning: Jump detected from Bar {current_bar_number} to {bar_number} (Line {current_line_num}). "
                        f"Advancing global time by ~{jump_duration:.3f}s ({bars_jumped} bars)."
                    )
                    current_global_time += jump_duration

                current_bar_number = bar_number
                current_bar_start_time = current_global_time
                time_within_bar_per_track = {}
                current_track_times = {
                    track: current_bar_start_time
                    for track in current_track_times
                }

            elif command in ["N", "C", "R"]:
                if current_bar_number == 0:
                    print(
                        f"Warning: Event '{line}' on Line {current_line_num} found before first BAR marker. Processing at time 0."
                    )

                data_parts = value.split(":")
                min_parts = 3 if command == "R" else 4
                if len(data_parts) < min_parts:
                    print(
                        f"Warning: Malformed {command} command on Line {current_line_num}: '{line}'. Requires at least {min_parts} parts. Skipping."
                    )
                    continue

                track_id = data_parts[0].strip()
                inst_name_for_event = active_instrument_name
                inst_track_key = (inst_name_for_event, track_id)

                is_drum_track = (
                    inst_name_for_event.lower() in ["drs", "drums"]
                    or track_id.lower() == "drums"
                )
                program = INSTRUMENT_PROGRAM_MAP.get(inst_name_for_event, 0)

                if inst_track_key not in instrument_definitions:
                    pm_instrument_name = f"{inst_name_for_event}-{track_id}"
                    instrument_definitions[inst_track_key] = {
                        "program": program,
                        "is_drum": is_drum_track,
                        "name": pm_instrument_name,
                    }
                    print(
                        f"Defined instrument: {pm_instrument_name} (Program: {program}, IsDrum: {is_drum_track})"
                    )
                    current_track_times[inst_track_key] = (
                        current_bar_start_time
                        + time_within_bar_per_track.get(inst_track_key, 0.0)
                    )
                    notes_by_instrument_track[inst_track_key] = []

                event_start_time = current_track_times.get(
                    inst_track_key,
                    current_bar_start_time
                    + time_within_bar_per_track.get(inst_track_key, 0.0),
                )
                event_duration_sec = 0.0

                if command == "N":
                    if len(data_parts) < 4:
                        continue
                    pitch_name = data_parts[1].strip()
                    duration_sym = data_parts[2].strip()
                    velocity_str = data_parts[3].strip()
                    try:
                        velocity = int(velocity_str)
                    except ValueError:
                        velocity = 64
                        print(
                            f"Warning: Invalid velocity '{velocity_str}' on Line {current_line_num}. Using 64."
                        )

                    event_duration_sec = duration_to_seconds(
                        duration_sym, current_tempo, current_ts_den
                    )

                    if is_drum_track:
                        midi_pitch = DRUM_PITCH_MAP.get(pitch_name.capitalize())
                        if midi_pitch is None:
                            midi_pitch = DRUM_PITCH_MAP.get(pitch_name)
                        if midi_pitch is None:
                            midi_pitch = 36
                            print(
                                f"Warning: Unknown drum sound '{pitch_name}' on Line {current_line_num}. Using Kick (36)."
                            )
                    else:
                        midi_pitch = pitch_to_midi(pitch_name)

                    note_event = {
                        "pitch": midi_pitch,
                        "start": event_start_time,
                        "end": event_start_time + event_duration_sec,
                        "velocity": max(0, min(127, velocity)),
                    }
                    notes_by_instrument_track[inst_track_key].append(note_event)

                elif command == "C":
                    if len(data_parts) < 4:
                        continue
                    pitches_str = data_parts[1].strip()
                    duration_sym = data_parts[2].strip()
                    velocity_str = data_parts[3].strip()
                    try:
                        velocity = int(velocity_str)
                    except ValueError:
                        velocity = 64
                        print(
                            f"Warning: Invalid velocity '{velocity_str}' on Line {current_line_num}. Using 64."
                        )

                    if pitches_str.startswith("[") and pitches_str.endswith("]"):
                        pitches_str = pitches_str[1:-1]
                    pitch_names = [
                        p.strip() for p in pitches_str.split(",") if p.strip()
                    ]

                    if not pitch_names:
                        continue

                    event_duration_sec = duration_to_seconds(
                        duration_sym, current_tempo, current_ts_den
                    )

                    for pitch_name in pitch_names:
                        if is_drum_track:
                            print(
                                f"Warning: Chord command used on drum track (Line {current_line_num}). Treating as standard notes."
                            )
                        midi_pitch = pitch_to_midi(pitch_name)
                        note_event = {
                            "pitch": midi_pitch,
                            "start": event_start_time,
                            "end": event_start_time + event_duration_sec,
                            "velocity": max(0, min(127, velocity)),
                        }
                        notes_by_instrument_track[inst_track_key].append(
                            note_event
                        )

                elif command == "R":
                    if len(data_parts) < 2:
                        continue
                    duration_sym = data_parts[1].strip()
                    event_duration_sec = duration_to_seconds(
                        duration_sym, current_tempo, current_ts_den
                    )

                new_track_time = event_start_time + event_duration_sec
                current_track_times[inst_track_key] = new_track_time
                time_within_bar_per_track[inst_track_key] = (
                    new_track_time - current_bar_start_time
                )
                last_event_end_time = max(last_event_end_time, new_track_time)
                current_global_time = max(current_global_time, new_track_time)

            else:
                print(
                    f"Warning: Unknown command '{command}' on line {current_line_num}: '{line}'. Skipping."
                )

        except Exception as e:
            print(f"Error parsing line {current_line_num}: '{line}' - {e}")
            traceback.print_exc()
            continue

    print(
        f"Symbolic text parsing complete. Estimated total duration: {last_event_end_time:.2f} seconds."
    )
    return (
        notes_by_instrument_track,
        instrument_definitions,
        tempo_changes,
        time_signature_changes,
        key_signature_changes,
        last_event_end_time,
        current_key,
        current_tempo,
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
        initial_tempo = (
            tempo_changes[0][1] if tempo_changes else CONFIG["default_tempo"]
        )
        midi_obj = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo)
        print(f"Initialized MIDI with tempo: {initial_tempo:.2f} BPM")

        # Add Time Signature Changes
        time_sig_changes.sort(key=lambda x: x[0])
        applied_ts_count = 0
        last_ts_time = -1.0
        for time, num, den in time_sig_changes:
            if den == 0 or (den & (den - 1) != 0):
                if den == 0:
                    actual_den = 4
                else:
                    actual_den = 2 ** math.ceil(math.log2(den))
                print(
                    f"Warning: Non-power-of-2 time signature denominator {den} at time {time:.2f}s."
                    f" Using closest power of 2: {actual_den}."
                )
            else:
                actual_den = den

            if time >= last_ts_time:
                midi_obj.time_signature_changes = [
                    ts
                    for ts in midi_obj.time_signature_changes
                    if ts.time != time
                ]
                ts_change = pretty_midi.TimeSignature(num, actual_den, time)
                midi_obj.time_signature_changes.append(ts_change)
                applied_ts_count += 1
                last_ts_time = time

        midi_obj.time_signature_changes.sort(key=lambda ts: ts.time)
        if applied_ts_count > 0:
            print(f"Applied {applied_ts_count} time signature changes.")
        if not midi_obj.time_signature_changes:
            default_num, default_den = CONFIG["default_timesig"]
            midi_obj.time_signature_changes.append(
                pretty_midi.TimeSignature(default_num, default_den, 0.0)
            )
            print(f"Applied default time signature: {default_num}/{default_den}")

        # Add Key Signature Changes
        key_sig_changes.sort(key=lambda x: x[0])
        applied_key_count = 0
        last_key_time = -1.0
        for time, key_name in key_sig_changes:
            try:
                key_number = pretty_midi.key_name_to_key_number(key_name)
                if time >= last_key_time:
                    midi_obj.key_signature_changes = [
                        ks
                        for ks in midi_obj.key_signature_changes
                        if ks.time != time
                    ]
                    key_change = pretty_midi.KeySignature(
                        key_number=key_number, time=time
                    )
                    midi_obj.key_signature_changes.append(key_change)
                    applied_key_count += 1
                    last_key_time = time
            except ValueError as e:
                print(
                    f"Warning: Could not parse key signature '{key_name}' at time {time:.2f}s. Skipping. Error: {e}"
                )

        midi_obj.key_signature_changes.sort(key=lambda ks: ks.time)
        if applied_key_count > 0:
            print(f"Applied {applied_key_count} key signature changes.")
        if not midi_obj.key_signature_changes:
            try:
                default_key_num = pretty_midi.key_name_to_key_number(
                    CONFIG["default_key"]
                )
                midi_obj.key_signature_changes.append(
                    pretty_midi.KeySignature(key_number=default_key_num, time=0.0)
                )
                print(f"Applied default key signature: {CONFIG['default_key']}")
            except ValueError as e:
                print(
                    f"Warning: Could not parse default key signature '{CONFIG['default_key']}'. Error: {e}"
                )

        # Create instruments and add notes
        available_channels = list(range(16))
        drum_channel = 9
        if drum_channel in available_channels:
            available_channels.remove(drum_channel)
        channel_index = 0
        instrument_objects = {}

        sorted_inst_defs = sorted(
            instrument_defs.items(), key=lambda item: item[1]["is_drum"], reverse=True
        )

        for inst_track_key, definition in sorted_inst_defs:
            if (
                inst_track_key not in notes_data
                or not notes_data[inst_track_key]
            ):
                print(
                    f"Skipping instrument '{definition['name']}' as it has no parsed notes."
                )
                continue

            is_drum = definition["is_drum"]
            program = definition["program"]
            pm_instrument_name = definition["name"]

            instrument_obj = pretty_midi.Instrument(
                program=program, is_drum=is_drum, name=pm_instrument_name
            )
            midi_obj.instruments.append(instrument_obj)
            instrument_objects[inst_track_key] = instrument_obj

            if is_drum:
                channel = drum_channel
                print(
                    f"Created MIDI instrument: {pm_instrument_name} (Program: {program}, IsDrum: {is_drum}, Assigned Channel: {channel})"
                )
            else:
                if not available_channels:
                    print(f"ERROR: No available non-drum channels left for {pm_instrument_name}")
                    continue # Skip adding notes if no channel

                if channel_index >= len(available_channels):
                    reuse_channel_index = channel_index % len(available_channels)
                    channel = available_channels[reuse_channel_index]
                    print(
                        f"Warning: Ran out of non-drum MIDI channels! Reusing channel {channel} for {pm_instrument_name}."
                    )
                else:
                    channel = available_channels[channel_index]
                print(
                    f"Created MIDI instrument: {pm_instrument_name} (Program: {program}, IsDrum: {is_drum}, Assigned Channel: {channel})"
                )
                channel_index += 1

            note_count = 0
            skipped_notes = 0
            for note_info in notes_data[inst_track_key]:
                start_time = max(0.0, note_info["start"])
                min_duration = 0.001
                end_time = max(start_time + min_duration, note_info["end"])
                velocity = max(1, min(127, note_info["velocity"]))
                pitch = max(0, min(127, note_info["pitch"]))

                if start_time >= end_time:
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
                    print(
                        f"Error creating pretty_midi.Note for {pm_instrument_name}: {e}. Skipping note."
                    )
                    print(
                        f"  Note data: Pitch={pitch}, Vel={velocity}, Start={start_time}, End={end_time}"
                    )
                    skipped_notes += 1

            print(
                f"Added {note_count} notes to instrument {pm_instrument_name}. ({skipped_notes} notes skipped)."
            )

        # Apply tempo changes as MIDI meta messages
        if len(tempo_changes) > 1:
            print("Applying tempo changes as MIDI meta-messages...")
            tempo_change_times, tempo_change_bpm = zip(*tempo_changes)
            tempo_change_mpb = [
                pretty_midi.bpm_to_tempo(bpm) for bpm in tempo_change_bpm
            ]
            midi_obj._load_tempo_changes(
                list(tempo_change_times), list(tempo_change_mpb)
            )

        # Ensure output directory exists
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

    for section_name, section_info in section_plan.items():
        symbolic_section, current_section_summary_info = (
            generate_symbolic_section(
                enriched_description,
                section_name,
                section_info,
                current_bar_count,
                last_section_summary_info,
            )
        )

        if symbolic_section and current_section_summary_info:
            if not symbolic_section.endswith("\n"):
                symbolic_section += "\n"
            all_symbolic_text += symbolic_section
            generated_sections_count += 1
            last_section_summary_info = current_section_summary_info
            # Ensure section_info['bars'] is valid before adding
            if isinstance(section_info.get("bars"), int) and section_info["bars"] > 0:
                 current_bar_count += section_info["bars"]
            else:
                 print(f"Warning: Invalid or missing 'bars' for section {section_name}. Bar count may be inaccurate.")
                 # Decide how to handle this: stop, or estimate, or just warn. Warning for now.
        else:
            print(f"Skipping section {section_name} due to generation failure.")
            # break # Option: Stop if any section fails

    if not all_symbolic_text.strip():
        print("\nERROR: No symbolic text was generated successfully. Cannot proceed.")
        exit(1)
    if generated_sections_count < len(section_plan):
        print(
            f"\nWarning: Only {generated_sections_count}/{len(section_plan)} sections were generated successfully."
        )

    print("\n--- Combined Symbolic Text ---")
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
    print("----------------------------")

    # Step 4: Parse the combined symbolic text
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
            "\nError: No notes or instruments were successfully parsed. MIDI file not created."
        )

    print("\n--- MidiMaker Generator Pipeline Finished ---")