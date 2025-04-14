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
GEMINI_MODEL = "gemini-2.5-pro-preview-03-25" #"gemini-2.0-flash-thinking-exp-01-21" 

# Configuration dictionary
CONFIG = {
    "api_key": GEMINI_KEY,
    "gemini_model": GEMINI_MODEL,
    "initial_description": "A fast (160 bpm), optimistic, intellectual, and complex piece of music with a simple melody over block chords. The piece should be in C minor, 4/4 time signature, and feature piano, guitar, and synth lead.",
    "output_dir": "output",
    "default_tempo": 120,
    "default_timesig": (4, 4),
    "default_key": "Cmin", # Updated default based on description
    "generation_retries": 3,
    "generation_delay": 5,
    "max_total_bars": 128,  # Limit total length for safety/cost
    "min_section_bars": 8,   # Reduced min for more flexibility
    "max_section_bars": 32,
    "temperature": 0.75, # Slightly higher temperature for more creativity
}


# --- Symbolic Format Definition (for prompts and parsing) ---
SYMBOLIC_FORMAT_DEFINITION = """
Use this compact symbolic format ONLY. Each command must be on a new line. Do NOT include comments after the command parameters on the same line.
- `INST:<InstrumentName>` (Instrument: e.g., Pno, Gtr, Bass, Drs, Str, Flt, Tpt, SynPad, SynLead, Arp). Use 'Drs' or 'Drums' for drum kits.
- `T:<BPM>` (Tempo: e.g., T:120)
- `TS:<N>/<D>` (Time Signature: e.g., TS:4/4)
- `K:<Key>` (Key Signature: e.g., K:Cmin, K:Gmaj, K:Ddor). Use standard `pretty_midi` key names.
- `BAR:<Num>` (Bar marker, starting from 1, strictly sequential)
- `N:<Track>:<Pitch>:<Duration>:<Velocity>` (Note: TrackID: PitchName: DurationSymbol: Velocity[0-127])
- `C:<Track>:<Pitches>:<Duration>:<Velocity>` (Chord: TrackID: [Pitch1,Pitch2,...]: DurationSymbol: Velocity)
- `R:<Track>:<Duration>` (Rest: TrackID: DurationSymbol)

TrackIDs: Use simple names like RH, LH, Melody, Bass, Drums, Arp1, Pad, Lead etc. A TrackID combined with the current INST defines a unique part.
PitchNames: Standard notation (e.g., C4, F#5, Gb3). Middle C is C4. For Drums (when INST is Drs/Drums), use names like Kick, Snare, HHC, HHO, Crash, Ride, HT, MT, LT.
DurationSymbols: W (Whole), H (Half), Q (Quarter), E (Eighth), S (Sixteenth), T (Thirty-second). Append '.' for dotted notes (e.g., Q., E.).
Velocity: MIDI velocity (0-127). Must be a number.

Example Note: N:Melody:G5:E:95
Example Chord: C:PnoLH:[C3,Eb3,G3]:H:60
Example Rest: R:Bass:W
Example Drums:
INST:Drs
N:Drums:Kick:Q:100
N:Drums:HHC:E:80
N:Drums:Snare:Q:110
Example Multi-Instrument:
INST:SynLead
BAR:1
N:LeadMelody:C5:Q:90
...
INST:SynPad
BAR:1
C:PadChord:[C3,Eb3,G3,Bb4]:W:50
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
    gen_config_args = {"temperature": CONFIG["temperature"]}
    if output_format == "json":
        gen_config_args["response_mime_type"] = "application/json"

    generation_config = genai.types.GenerationConfig(**gen_config_args)

    for attempt in range(retries):
        try:
            response = model.generate_content(
                prompt, generation_config=generation_config
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

            if not response.parts:
                 # Check candidates if parts is empty but no explicit block/error
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if candidate.finish_reason != 'STOP':
                         print(f"Generation stopped for reason: {candidate.finish_reason} (Candidate Level)")
                         # Try to get content even if stopped early
                         if candidate.content and candidate.content.parts:
                             print("Attempting to use partial content from candidate.")
                             content = candidate.content.parts[0].text.strip()
                         else:
                             print("No content available from early-stopped candidate.")
                             content = None # No content to return
                    elif candidate.content and candidate.content.parts:
                        print("Accessing content via candidates[0] as response.parts was empty.")
                        content = candidate.content.parts[0].text.strip()
                    else:
                        print(f"Warning: Received response with no parts and no usable candidate content (Attempt {attempt + 1}).")
                        content = None # No content found
                else:
                    print(f"Warning: Received response with no parts (Attempt {attempt + 1}).")
                    content = None # No content found
            else:
                 # Standard case: content is in response.parts
                 content = response.text.strip()


            # Process the content based on expected format
            if content is not None:
                if output_format == "json":
                    try:
                        # Remove potential markdown fences before parsing JSON
                        content = re.sub(r"^```json\n", "", content)
                        content = re.sub(r"\n```$", "", content)
                        return json.loads(content)
                    except json.JSONDecodeError as json_e:
                        print(
                            f"Error decoding JSON response (Attempt {attempt + 1}): {json_e}"
                        )
                        print(f"Received text: {content[:500]}...")
                        # Fall through to retry logic
                else:
                     # Remove potential markdown fences from text output
                    content = re.sub(r"^```[a-z]*\n", "", content, flags=re.MULTILINE)
                    content = re.sub(r"\n```$", "", content)
                    return content.strip() # Return cleaned text

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
ACCIDENTAL_MAP = {"#": 1, "S": 1, "B": -1, "": 0} # Allow S for sharp

# General MIDI Instrument Program Numbers (Selected)
INSTRUMENT_PROGRAM_MAP = {
    # Piano
    "Pno": 0, "Piano": 0, "Acoustic Grand Piano": 0, "Bright Acoustic Piano": 1,
    "Electric Grand Piano": 2, "Honky-tonk Piano": 3, "Electric Piano 1": 4,
    "Rhodes Piano": 4, "Electric Piano 2": 5,
    # Chromatic Percussion
    "Celesta": 8, "Glockenspiel": 9, "Music Box": 10, "Vibraphone": 11, "Marimba": 12, "Xylophone": 13,
    # Organ
    "Org": 16, "Organ": 16, "Drawbar Organ": 16, "Percussive Organ": 17, "Rock Organ": 18, "Church Organ": 19,
    # Guitar
    "Gtr": 25, "Acoustic Guitar": 25, "Nylon Guitar": 24, "Steel Guitar": 25,
    "Electric Guitar": 27, "Jazz Guitar": 26, "Clean Electric Guitar": 27,
    "Muted Electric Guitar": 28, "Overdriven Guitar": 29, "Distortion Guitar": 30,
    # Bass
    "Bass": 33, "Acoustic Bass": 32, "Electric Bass": 33, "Finger Bass": 33,
    "Pick Bass": 34, "Fretless Bass": 35, "Slap Bass": 36, "Synth Bass": 38,
    "SynthBass": 38, "Synth Bass 2": 39,
    # Strings
    "Str": 48, "Strings": 48, "Violin": 40, "Viola": 41, "Cello": 42,
    "Contrabass": 43, "Tremolo Strings": 44, "Pizzicato Strings": 45,
    "Orchestral Harp": 46, "String Ensemble 1": 48, "String Ensemble 2": 49,
    "Synth Strings 1": 50, "Synth Strings 2": 51,
    # Brass
    "Tpt": 56, "Trumpet": 56, "Trombone": 57, "Tuba": 58, "Muted Trumpet": 59,
    "French Horn": 60, "Brass Section": 61,
    # Reed
    "Sax": 65, "Soprano Sax": 64, "Alto Sax": 65, "Tenor Sax": 66,
    "Baritone Sax": 67, "Oboe": 68, "English Horn": 69, "Bassoon": 70, "Clarinet": 71,
    # Pipe
    "Flt": 73, "Flute": 73, "Piccolo": 72, "Recorder": 74, "Pan Flute": 75,
    # Synth Lead
    "SynLead": 81, "Synth Lead": 81, "Lead 1 (Square)": 80, "Lead 2 (Sawtooth)": 81,
    "Lead 3 (Calliope)": 82, "Lead 8 (Bass + Lead)": 87,
    # Synth Pad
    "SynPad": 89, "Synth Pad": 89, "Pad 1 (New Age)": 88, "Pad 2 (Warm)": 89,
    "Pad 3 (Polysynth)": 90, "Pad 4 (Choir)": 91, "Pad 5 (Bowed)": 92, "Pad 6 (Metallic)": 93, "Pad 7 (Halo)": 94,
    # Arp (Mapped to a synth sound)
    "Arp": 81, # Map Arp to Sawtooth Lead by default
    # Drums are a special case (channel 10 / index 9) - Program 0 is conventional
    "Drs": 0, "Drums": 0, "808Drums": 0,
}
DRUM_INSTRUMENT_NAMES = {"drs", "drums", "808drums"} # Lowercase set for checking INST

# Standard drum note map (MIDI channel 10 / index 9) - Keys should be lowercase for lookup
DRUM_PITCH_MAP = {
    # Bass Drum
    "kick": 36, "bd": 36, "bass drum 1": 36, "acoustic bass drum": 35,
    # Snare
    "snare": 38, "sd": 38, "acoustic snare": 38, "electric snare": 40,
    # Hi-Hat
    "hihatclosed": 42, "hhc": 42, "closed hi hat": 42, "closed hi-hat": 42,
    "hihatopen": 46, "hho": 46, "open hi hat": 46, "open hi-hat": 46,
    "hihatpedal": 44, "hh p": 44, "pedal hi hat": 44, "pedal hi-hat": 44,
    # Cymbals
    "crash": 49, "cr": 49, "crash cymbal 1": 49, "crash cymbal 2": 57,
    "ride": 51, "rd": 51, "ride cymbal 1": 51, "ride cymbal 2": 59,
    "ride bell": 53, "splash cymbal": 55, "chinese cymbal": 52,
    # Toms
    "high tom": 50, "ht": 50, "hi tom": 50,
    "mid tom": 47, "mt": 47, "hi-mid tom": 48, "low-mid tom": 47,
    "low tom": 43, "lt": 43, "high floor tom": 43, "low floor tom": 41,
    "floor tom": 41, "ft": 41,
    # Other
    "rimshot": 37, "rs": 37, "side stick": 37,
    "clap": 39, "cp": 39, "hand clap": 39,
    "cowbell": 56, "cb": 56,
    "tambourine": 54, "tmb": 54,
    "claves": 75, "wood block": 76, "high wood block": 76, "low wood block": 77,
}


def pitch_to_midi(pitch_name):
    """Converts pitch name (e.g., C4, F#5, Gb3) to MIDI number."""
    pitch_name = pitch_name.strip()
    match = re.match(r"([A-G])([#sb]?)(\-?\d+)", pitch_name, re.IGNORECASE) # Allow 's' for sharp
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

    # Normalize accidentals: 's' becomes '#'
    acc_norm = acc.upper() if acc else ''
    if acc_norm == 'S':
        acc_norm = '#'

    acc_val = ACCIDENTAL_MAP.get(acc_norm, 0)
    midi_val = base_midi + acc_val + (octave + 1) * 12
    return max(0, min(127, midi_val))


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
            "W": 4.0, "H": 2.0, "Q": 1.0, "E": 0.5, "S": 0.25, "T": 0.125,
        }

        base_symbol = duration_symbol.replace(".", "")
        is_dotted = duration_symbol.endswith(".")

        relative_duration_quarters = duration_map.get(base_symbol)
        if relative_duration_quarters is None:
            print(
                f"Warning: Unknown duration symbol: '{duration_symbol}'. Defaulting to Quarter (1.0)."
            )
            relative_duration_quarters = 1.0

        # Adjust for time signature denominator (relative to quarter note)
        # Example: In 4/4, Q=1 beat. In 6/8, Q=2/3 of a beat (relative to dotted quarter).
        # We calculate duration based on quarter notes per minute (tempo), so relative duration is correct.
        # The time_sig_denominator is mostly relevant for calculating expected bar duration.

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
    Infer or suggest a plausible key signature (e.g., K:Cmin, K:Gmaj, K:F#dor), tempo (e.g., T:90), time signature (e.g., TS:4/4),
    and primary instrumentation (e.g., INST:Pno, INST:Gtr, INST:SynLead, INST:Drs).
    Also, briefly elaborate on the mood and suggest a simple musical structure (like AABA, ABAC, Verse-Chorus-Bridge).
    Keep the output concise and focused on these musical parameters. Output should start directly with the parameters if possible, or clearly list them.

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
        # Use more robust regex to find parameters anywhere in the text
        key_match = re.search(
            r"[Kk](?:ey)?\s*:\s*([A-Ga-g][#sb]?(?:maj|min|dor|phr|lyd|mix|loc|aeo|ion)?)",
            enriched, re.IGNORECASE
        )
        tempo_match = re.search(r"[Tt](?:empo)?\s*:\s*(\d+)", enriched)
        ts_match = re.search(r"[Tt](?:ime)?\s*[Ss](?:ig)?\s*:\s*(\d+)\s*/\s*(\d+)", enriched)
        inst_match = re.search(r"INST\s*:\s*(\w+)", enriched, re.IGNORECASE) # Look for INST specifically if possible
        if not inst_match:
             inst_match = re.search(r"(?:instrument(?:s|ation)?|primary inst)\s*:\s*(\w+)", enriched, re.IGNORECASE)
        struct_match = re.search(
            r"[Ss]tructure\s*:\s*([\w\-]+)", enriched, re.IGNORECASE
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
                # Basic validation for common time signatures
                if ts_num > 0 and ts_den > 0 and ts_den in [1, 2, 4, 8, 16, 32]:
                    current_timesig = (ts_num, ts_den)
                    print(f"Updated Default Time Signature: {current_timesig}")
            except ValueError:
                pass
        if inst_match:
            primary_instrument = inst_match.group(1)
            print(f"Identified Primary Instrument: {primary_instrument}")
        if struct_match:
            structure_hint = struct_match.group(1).upper() # Standardize structure hint
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

Generate ONLY the JSON plan now, starting with {{ and ending with }}:
"""

    plan_json = call_gemini(prompt, output_format="json")

    if plan_json and isinstance(plan_json, dict):
        # Validate the plan
        validated_plan = {}
        total_bars = 0
        is_valid = True
        section_order = list(plan_json.keys()) # Keep original order if possible

        for name in section_order:
            info = plan_json[name]
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
                        f"Warning: Section '{name}' ({info['bars']} bars) exceeds max total bars ({CONFIG['max_total_bars']}) when added to current total {total_bars}. Truncating plan."
                    )
                    is_valid = False
                    break  # Stop adding sections
            else:
                print(
                    f"Warning: Invalid format or bar count for section '{name}' in generated plan: {info}. Skipping."
                )
                is_valid = False
                # Continue validating other sections if possible

        if not validated_plan:
            print("ERROR: Failed to generate a valid section plan from LLM response. Cannot proceed.")
            return None

        print("Generated Section Plan:")
        for name, info in validated_plan.items():
            print(f"  - {name} ({info['bars']} bars): {info['goal']}")
        print(f"Total Bars in Plan: {total_bars}")
        return validated_plan
    else:
        print("ERROR: Failed to generate or parse section plan JSON from LLM. Cannot proceed.")
        if isinstance(plan_json, str): # Log if we got string instead of JSON
            print(f"LLM Output (expected JSON):\n{plan_json[:500]}...")
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
{context_prompt}
Current Section: {section_name}
Target Bars: {bars} (Start this section exactly at BAR:{current_bar}, end *before* BAR:{current_bar + bars})
Section Goal: {goal}

Instructions:
1. Generate music ONLY for this section, starting *exactly* with `BAR:{current_bar}` unless initial T, TS, K, or INST commands are needed for this specific section start.
2. If tempo (T), time signature (TS), key (K), or instrument (INST) need to be set or changed *at the very beginning* of this section (time = start of BAR:{current_bar}), include those commands *before* the `BAR:{current_bar}` marker. Otherwise, assume they carry over from the previous section or use defaults (T:{default_tempo}, TS:{default_timesig[0]}/{default_timesig[1]}, K:{default_key}). You can change INST multiple times within the section if needed.
3. Strictly adhere to the compact symbolic format defined below. Output ONLY the commands, each on a new line.
4. DO NOT include any other text, explanations, apologies, section titles, or formatting like ```mus``` or ```.
5. DO NOT include comments (#) on the same line as commands.
6. Ensure musical coherence within the section and try to achieve the Section Goal.
7. The total duration of notes/rests/chords within each bar MUST add up precisely according to the active time signature (e.g., 4 quarter notes in 4/4, 6 eighth notes in 6/8). Be precise. Use rests (R:<Track>:<Duration>) to fill empty time accurately for each active track within a bar.
8. End the generation cleanly *after* the content for bar {current_bar + bars - 1} is complete. Do NOT include `BAR:{current_bar + bars}`.

{SYMBOLIC_FORMAT_DEFINITION}

Generate Section {section_name} symbolic music now (starting BAR:{current_bar}):
"""

    symbolic_text = call_gemini(prompt)

    if symbolic_text:
        # Basic cleaning (already done in call_gemini, but belts and suspenders)
        symbolic_text = re.sub(r"^```[a-z]*\n", "", symbolic_text, flags=re.MULTILINE)
        symbolic_text = re.sub(r"\n```$", "", symbolic_text)
        symbolic_text = symbolic_text.strip()

        # Validate start and content
        lines = symbolic_text.split("\n")
        first_meaningful_line = ""
        first_meaningful_line_index = -1
        for idx, line in enumerate(lines):
            line = line.strip()
            if line and not line.startswith("#"): # Allow comment lines
                first_meaningful_line = line
                first_meaningful_line_index = idx
                break

        if first_meaningful_line_index == -1:
            print(
                f"Warning: Generated text for {section_name} appears empty or only contains comments."
            )
            return "", None  # Treat as failure

        bar_marker = f"BAR:{current_bar}"
        # Check if the first meaningful line is a valid start command or the expected BAR marker
        is_valid_start_command = re.match(r"^(T:|TS:|K:|INST:)", first_meaningful_line)
        is_correct_bar_marker = first_meaningful_line.startswith(bar_marker)

        # Find the first occurrence of the expected BAR marker
        bar_marker_pos = -1
        for idx, line in enumerate(lines):
            if line.strip().startswith(bar_marker):
                bar_marker_pos = idx
                break

        # Logic to handle incorrect starts:
        if is_correct_bar_marker:
            # Starts with the correct bar marker, potentially after initial commands. Good.
            # If there were lines before it, they should be INST/T/TS/K.
             pass # No adjustment needed if it starts correctly
        elif is_valid_start_command and bar_marker_pos != -1 and bar_marker_pos > first_meaningful_line_index:
            # Starts with valid commands, and the correct BAR marker appears later. Good.
            pass # No adjustment needed
        elif bar_marker_pos != -1:
             # Found the correct BAR marker, but it wasn't the first line or preceded by only valid commands.
             print(f"Warning: Generated text for {section_name} had unexpected content before {bar_marker}. Trimming preamble.")
             # Trim lines before the first valid command (T/TS/K/INST) or the BAR marker itself
             start_index = bar_marker_pos
             for idx in range(bar_marker_pos - 1, -1, -1):
                 line_content = lines[idx].strip()
                 if re.match(r"^(T:|TS:|K:|INST:)", line_content):
                     start_index = idx # Keep valid initial commands
                 elif not line_content or line_content.startswith("#"):
                     continue # Ignore empty/comment lines
                 else:
                     break # Stop if we hit other content

             symbolic_text = "\n".join(lines[start_index:])
             print(f"Adjusted start for {section_name}.")
        else:
            # Cannot find the expected BAR marker at all. This section is likely unusable.
            print(f"ERROR: Generated text for {section_name} does not contain the expected start marker '{bar_marker}'. Discarding section.")
            print(f"Received text (first 500 chars):\n{symbolic_text[:500]}...")
            return "", None


        print(
            f"Generated symbolic text for Section {section_name} (first 300 chars):\n{symbolic_text[:300]}...\n"
        )

        # Extract summary info (simple version for now)
        summary_info = {
            "name": section_name,
            "summary": f"Generated {bars} bars targeting goal: {goal}.",
            "key": default_key, # TODO: Update if K: changes within section
            "tempo": default_tempo, # TODO: Update if T: changes within section
            "time_sig": f"{default_timesig[0]}/{default_timesig[1]}" # TODO: Update if TS: changes
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
    instrument_definitions = {} # Key: (inst_name, track_id), Value: {program, is_drum, name}

    # State variables
    current_track_times = {} # Key: (inst_name, track_id), Value: current time cursor for this track
    current_global_time = 0.0 # Tracks the latest event time across all tracks, adjusted by BAR markers

    current_tempo = float(CONFIG["default_tempo"])
    current_ts_num, current_ts_den = CONFIG["default_timesig"]
    current_key = CONFIG["default_key"]
    active_instrument_name = "Pno" # Default starting instrument
    active_instrument_is_drum = False

    current_bar_number = 0
    current_bar_start_time = 0.0 # Global time when the current bar started
    time_within_bar_per_track = {} # Key: (inst_name, track_id), Value: time elapsed within the current bar for this track
    expected_bar_duration_sec = (
        (60.0 / current_tempo) * current_ts_num * (4.0 / current_ts_den)
    )
    last_event_end_time = 0.0 # Tracks the absolute end time of the last note/rest

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
            break # Stop pre-pass when BAR is encountered

        parts = line.split(":", 1)
        command = parts[0].upper()
        value = parts[1].strip() if len(parts) > 1 else ""
        ini_line_num = i + 1

        try:
            if command == "INST":
                if value:
                    active_instrument_name = value
                    active_instrument_is_drum = active_instrument_name.lower() in DRUM_INSTRUMENT_NAMES
                    print(
                        f"Initial Instrument context set to {active_instrument_name} (Is Drum: {active_instrument_is_drum})"
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
                    and new_ts_den > 0 and new_ts_den in [1, 2, 4, 8, 16, 32] # Power of 2 denominator
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
                    # Validate key? pretty_midi will handle it later.
                    current_key = value
                    key_signature_changes.append((0.0, current_key))
                    initial_commands_set["K"] = True
                    print(f"Initial Key set to {current_key}")

        except Exception as e:
            print(
                f"Error parsing initial setting line {ini_line_num}: '{line}' - {e}"
            )
        parse_start_line_index = i + 1 # Ensure we start parsing after the pre-pass section

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
                    active_instrument_name = value
                    active_instrument_is_drum = active_instrument_name.lower() in DRUM_INSTRUMENT_NAMES
                    # Don't print every INST change, can be verbose
            elif command == "T":
                new_tempo = float(value)
                if new_tempo > 0 and new_tempo != current_tempo:
                    # Use current_global_time, which reflects bar starts
                    event_time = current_global_time
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
            elif command == "TS":
                num_str, den_str = value.split("/")
                new_ts_num, new_ts_den = int(num_str), int(den_str)
                if (
                    new_ts_num > 0
                    and new_ts_den > 0 and new_ts_den in [1, 2, 4, 8, 16, 32]
                    and (new_ts_num, new_ts_den) != (current_ts_num, current_ts_den)
                ):
                    event_time = current_global_time
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
            elif command == "K":
                if value and value != current_key:
                    event_time = current_global_time
                    key_signature_changes.append((event_time, value))
                    current_key = value
                    print(
                        f"Time {event_time:.3f}s (Bar ~{current_bar_number}): Key change to {current_key}"
                    )

            elif command == "BAR":
                bar_number = int(value)
                # Calculate expected start time of this new bar
                expected_new_bar_start_time = current_bar_start_time + expected_bar_duration_sec if current_bar_number > 0 else 0.0

                # Check timing of the previous bar before moving to the new one
                if current_bar_number > 0:
                    max_accumulated_time_in_prev_bar = 0.0
                    for track_key, accumulated_time in time_within_bar_per_track.items():
                        max_accumulated_time_in_prev_bar = max(max_accumulated_time_in_prev_bar, accumulated_time)

                    # Allow a small tolerance for floating point comparisons
                    tolerance = max(0.005, expected_bar_duration_sec * 0.01) # 5ms or 1%
                    duration_error = max_accumulated_time_in_prev_bar - expected_bar_duration_sec

                    if abs(duration_error) > tolerance:
                        print(
                            f"Warning: Bar {current_bar_number} timing mismatch on Line {current_line_num}. "
                            f"Expected duration {expected_bar_duration_sec:.3f}s, max accumulated {max_accumulated_time_in_prev_bar:.3f}s "
                            f"(Error: {duration_error:.3f}s). Setting bar {bar_number} start to expected time {expected_new_bar_start_time:.3f}s."
                        )
                        # Force global time to the expected start time
                        current_global_time = expected_new_bar_start_time
                    else:
                        # If within tolerance, use the actual max accumulated time to avoid drift
                        current_global_time = current_bar_start_time + max_accumulated_time_in_prev_bar

                # Handle jumps in bar numbers
                bars_jumped = bar_number - (current_bar_number + 1)
                if bars_jumped > 0 and current_bar_number > 0:
                    # Estimate jump duration based on current tempo/TS
                    jump_duration = bars_jumped * expected_bar_duration_sec
                    print(
                        f"Warning: Jump detected from Bar {current_bar_number} to {bar_number} (Line {current_line_num}). "
                        f"Advancing global time by ~{jump_duration:.3f}s ({bars_jumped} bars)."
                    )
                    current_global_time += jump_duration

                # Update bar state
                current_bar_number = bar_number
                current_bar_start_time = current_global_time
                # Reset time within the new bar for all tracks
                time_within_bar_per_track = {key: 0.0 for key in time_within_bar_per_track}
                # Sync individual track cursors to the new bar start time
                current_track_times = {key: current_bar_start_time for key in current_track_times}


            elif command in ["N", "C", "R"]:
                if current_bar_number == 0:
                    print(
                        f"Warning: Event '{line}' on Line {current_line_num} found before first BAR marker. Processing at time 0."
                    )
                    # Ensure bar state is initialized if events occur before BAR:1
                    current_bar_start_time = 0.0
                    current_global_time = 0.0


                inst_name_for_event = active_instrument_name
                is_drum_track_for_event = active_instrument_is_drum
                inst_track_key = None # Will be set below based on command

                event_duration_sec = 0.0
                event_start_time = 0.0

                # --- Parse Note (N) ---
                if command == "N":
                    data_parts = value.split(":")
                    if len(data_parts) < 4:
                        print(f"Warning: Malformed N command on Line {current_line_num}: '{line}'. Requires 4+ parts (Track:Pitch:Dur:Vel). Skipping.")
                        continue

                    track_id = data_parts[0].strip()
                    pitch_name_raw = data_parts[1].strip()
                    duration_sym_raw = data_parts[2].strip()
                    velocity_str_raw = data_parts[3].strip()

                    # Clean potential comments from last part (velocity)
                    velocity_str = velocity_str_raw.split('#', 1)[0].strip()
                    duration_sym = duration_sym_raw # Assume no comments in duration

                    inst_track_key = (inst_name_for_event, track_id)

                    try:
                        velocity = int(velocity_str)
                    except ValueError:
                        velocity = 64 # Default velocity
                        print(f"Warning: Invalid velocity '{velocity_str_raw}' on Line {current_line_num}. Using {velocity}.")

                    event_duration_sec = duration_to_seconds(duration_sym, current_tempo, current_ts_den)

                    midi_pitch = 60 # Default pitch

                    if is_drum_track_for_event:
                        pitch_name_lookup = pitch_name_raw.lower()
                        midi_pitch = DRUM_PITCH_MAP.get(pitch_name_lookup)
                        if midi_pitch is None:
                            # Try without lowercasing as fallback (though map keys are lower)
                            midi_pitch = DRUM_PITCH_MAP.get(pitch_name_raw)
                        if midi_pitch is None:
                            print(f"Warning: Unknown drum sound '{pitch_name_raw}' on Line {current_line_num}. Using Kick (36).")
                            midi_pitch = 36
                    else:
                        midi_pitch = pitch_to_midi(pitch_name_raw)

                    # Get current time for this specific track
                    event_start_time = current_track_times.get(inst_track_key, current_bar_start_time + time_within_bar_per_track.get(inst_track_key, 0.0))

                    note_event = {
                        "pitch": midi_pitch,
                        "start": event_start_time,
                        "end": event_start_time + event_duration_sec,
                        "velocity": max(0, min(127, velocity)),
                    }
                    # Defer adding note until instrument is defined

                # --- Parse Chord (C) ---
                elif command == "C":
                    data_parts = value.split(":")
                    if len(data_parts) < 4:
                        print(f"Warning: Malformed C command on Line {current_line_num}: '{line}'. Requires 4+ parts (Track:[Pitches]:Dur:Vel). Skipping.")
                        continue

                    track_id = data_parts[0].strip()
                    pitches_str_raw = data_parts[1].strip()
                    duration_sym_raw = data_parts[2].strip()
                    velocity_str_raw = data_parts[3].strip()

                    # Clean potential comments from last part (velocity)
                    velocity_str = velocity_str_raw.split('#', 1)[0].strip()
                    duration_sym = duration_sym_raw

                    inst_track_key = (inst_name_for_event, track_id)

                    try:
                        velocity = int(velocity_str)
                    except ValueError:
                        velocity = 64 # Default velocity
                        print(f"Warning: Invalid velocity '{velocity_str_raw}' on Line {current_line_num}. Using {velocity}.")

                    if pitches_str_raw.startswith("[") and pitches_str_raw.endswith("]"):
                        pitches_str = pitches_str_raw[1:-1]
                    else:
                        print(f"Warning: Chord pitches format might be incorrect on Line {current_line_num}: '{pitches_str_raw}'. Expected '[P1,P2,...]'. Attempting parse.")
                        pitches_str = pitches_str_raw

                    pitch_names = [p.strip() for p in pitches_str.split(",") if p.strip()]

                    if not pitch_names:
                        print(f"Warning: No valid pitches found in Chord command on Line {current_line_num}: '{line}'. Skipping.")
                        continue

                    event_duration_sec = duration_to_seconds(duration_sym, current_tempo, current_ts_den)

                    # Get current time for this specific track
                    event_start_time = current_track_times.get(inst_track_key, current_bar_start_time + time_within_bar_per_track.get(inst_track_key, 0.0))

                    if is_drum_track_for_event:
                        print(f"Warning: Chord command 'C:' used for drum instrument '{inst_name_for_event}' on Line {current_line_num}. Treating pitches as individual drum sounds.")

                    # Create multiple note events for the chord
                    chord_notes = []
                    for pitch_name_raw in pitch_names:
                        midi_pitch = 60 # Default
                        if is_drum_track_for_event:
                            pitch_name_lookup = pitch_name_raw.lower()
                            midi_pitch = DRUM_PITCH_MAP.get(pitch_name_lookup)
                            if midi_pitch is None: midi_pitch = DRUM_PITCH_MAP.get(pitch_name_raw) # Fallback
                            if midi_pitch is None:
                                print(f"Warning: Unknown drum sound '{pitch_name_raw}' in chord on Line {current_line_num}. Using Kick (36).")
                                midi_pitch = 36
                        else:
                            midi_pitch = pitch_to_midi(pitch_name_raw)

                        note_event = {
                            "pitch": midi_pitch,
                            "start": event_start_time,
                            "end": event_start_time + event_duration_sec,
                            "velocity": max(0, min(127, velocity)),
                        }
                        chord_notes.append(note_event)
                    # Defer adding notes until instrument defined

                # --- Parse Rest (R) ---
                elif command == "R":
                    # R format: R:Track:Duration
                    data_parts = value.split(":", 1) # Split into Track and Duration
                    if len(data_parts) < 2:
                        print(f"Warning: Malformed R command on Line {current_line_num}: '{line}'. Requires 2+ parts (Track:Duration). Skipping.")
                        continue

                    track_id = data_parts[0].strip()
                    duration_sym_raw = data_parts[1].strip()

                    # Clean potential comments from duration part
                    duration_sym = duration_sym_raw.split('#', 1)[0].strip()

                    if not duration_sym:
                         print(f"Warning: Empty duration for R command on Line {current_line_num}: '{line}'. Skipping.")
                         continue

                    inst_track_key = (inst_name_for_event, track_id)
                    event_duration_sec = duration_to_seconds(duration_sym, current_tempo, current_ts_den)

                    # Get current time for this specific track
                    event_start_time = current_track_times.get(inst_track_key, current_bar_start_time + time_within_bar_per_track.get(inst_track_key, 0.0))
                    # Rests don't generate notes, just advance time

                # --- Post-Event Processing (Common to N, C, R) ---
                if inst_track_key:
                    # Define instrument if not seen before
                    if inst_track_key not in instrument_definitions:
                        pm_instrument_name = f"{inst_track_key[0]}-{inst_track_key[1]}" # e.g., SynLead-Melody
                        program = INSTRUMENT_PROGRAM_MAP.get(inst_track_key[0], 0) # Lookup program from INST name
                        is_drum = inst_track_key[0].lower() in DRUM_INSTRUMENT_NAMES # Determine drum status from INST name

                        # Override program to 0 if it's a drum track, as per GM convention
                        if is_drum:
                            program = 0

                        instrument_definitions[inst_track_key] = {
                            "program": program,
                            "is_drum": is_drum,
                            "name": pm_instrument_name,
                        }
                        print(f"Defined instrument: {pm_instrument_name} (Program: {program}, IsDrum: {is_drum})")
                        # Initialize time and note list for new instrument/track
                        current_track_times[inst_track_key] = current_bar_start_time + time_within_bar_per_track.get(inst_track_key, 0.0)
                        time_within_bar_per_track[inst_track_key] = time_within_bar_per_track.get(inst_track_key, 0.0)
                        notes_by_instrument_track[inst_track_key] = []

                    # Add parsed notes (if any) to the correct list
                    if command == 'N' and 'note_event' in locals():
                        notes_by_instrument_track[inst_track_key].append(note_event)
                    elif command == 'C' and 'chord_notes' in locals():
                        notes_by_instrument_track[inst_track_key].extend(chord_notes)

                    # Advance time for this track
                    new_track_time = event_start_time + event_duration_sec
                    current_track_times[inst_track_key] = new_track_time
                    time_within_bar_per_track[inst_track_key] = new_track_time - current_bar_start_time

                    # Update the global last event time
                    last_event_end_time = max(last_event_end_time, new_track_time)
                    # current_global_time is primarily advanced by BAR markers, but track times advance independently within bars.


            else:
                print(
                    f"Warning: Unknown command '{command}' on line {current_line_num}: '{line}'. Skipping."
                )

        except Exception as e:
            print(f"FATAL Error parsing line {current_line_num}: '{line}' - {e}")
            traceback.print_exc()
            # Decide whether to continue or stop on fatal errors
            # continue # Attempt to continue parsing other lines

    print(
        f"Symbolic text parsing complete. Estimated total duration: {last_event_end_time:.3f} seconds."
    )
    # Sanity check: Ensure all defined instruments actually have notes
    final_instrument_defs = {}
    final_notes_data = {}
    for key, definition in instrument_definitions.items():
        if key in notes_by_instrument_track and notes_by_instrument_track[key]:
            final_instrument_defs[key] = definition
            final_notes_data[key] = notes_by_instrument_track[key]
        else:
             print(f"Info: Instrument '{definition['name']}' defined but had no notes parsed. Excluding from MIDI.")


    return (
        final_notes_data,
        final_instrument_defs,
        tempo_changes,
        time_signature_changes,
        key_signature_changes,
        last_event_end_time,
        current_key, # Return the final key state
        current_tempo, # Return the final tempo state
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
        initial_tempo = tempo_changes[0][1] if tempo_changes else CONFIG["default_tempo"]
        midi_obj = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo)
        print(f"Initialized MIDI with tempo: {initial_tempo:.2f} BPM")

        # --- Apply Meta-Messages (Tempo, Time Sig, Key Sig) ---
        # Remove duplicate times for the same event type before applying

        # Tempo Changes (already handled by PrettyMIDI constructor and _load_tempo_changes)
        if len(tempo_changes) > 1:
             print("Applying tempo changes...")
             tempo_change_times, tempo_change_bpm = zip(*tempo_changes)
             # Filter out redundant changes at the same time, keeping the last one
             unique_tempo_times = {}
             for t, bpm in zip(tempo_change_times, tempo_change_bpm):
                 unique_tempo_times[round(t, 6)] = bpm # Round time slightly for comparison
             sorted_tempo_times = sorted(unique_tempo_times.keys())
             final_tempo_times = [t for t in sorted_tempo_times if t >= 0] # Ensure non-negative time
             final_tempo_bpm = [unique_tempo_times[t] for t in final_tempo_times]

             if final_tempo_times and final_tempo_times[0] == 0.0:
                 # Initial tempo already set, remove the 0.0 entry if others exist
                 if len(final_tempo_times) > 1:
                     final_tempo_times = final_tempo_times[1:]
                     final_tempo_bpm = final_tempo_bpm[1:]
                 else: # Only initial tempo specified
                      final_tempo_times, final_tempo_bpm = [], []


             if final_tempo_times:
                 tempo_change_mpq = [pretty_midi.bpm_to_tempo(bpm) for bpm in final_tempo_bpm]
                 midi_obj._load_tempo_changes(list(final_tempo_times), list(tempo_change_mpq))
                 print(f"Applied {len(final_tempo_times)} tempo change events.")


        # Time Signature Changes
        time_sig_changes.sort(key=lambda x: x[0])
        unique_ts = {}
        for time, num, den in time_sig_changes:
            if den <= 0 or (den & (den - 1) != 0 and den != 1): # Denominator must be power of 2 (or 1)
                 actual_den = 2**math.ceil(math.log2(den)) if den > 0 else 4
                 print(f"Warning: Invalid time signature denominator {den} at time {time:.3f}s. Using closest power of 2: {actual_den}.")
            else:
                 actual_den = den
            unique_ts[round(time, 6)] = (num, actual_den)

        midi_obj.time_signature_changes = [] # Clear default
        applied_ts_count = 0
        for time in sorted(unique_ts.keys()):
             if time >= 0:
                 num, den = unique_ts[time]
                 ts_change = pretty_midi.TimeSignature(num, den, time)
                 midi_obj.time_signature_changes.append(ts_change)
                 applied_ts_count += 1
        if applied_ts_count > 0:
            print(f"Applied {applied_ts_count} time signature changes.")
        elif not midi_obj.time_signature_changes: # Add default if none applied
            default_num, default_den = CONFIG["default_timesig"]
            midi_obj.time_signature_changes.append(pretty_midi.TimeSignature(default_num, default_den, 0.0))
            print(f"Applied default time signature: {default_num}/{default_den}")


        # Key Signature Changes
        key_sig_changes.sort(key=lambda x: x[0])
        unique_ks = {}
        last_valid_key_name = CONFIG["default_key"]
        for time, key_name in key_sig_changes:
            try:
                key_number = pretty_midi.key_name_to_key_number(key_name)
                unique_ks[round(time, 6)] = key_number
                last_valid_key_name = key_name
            except ValueError as e:
                print(f"Warning: Could not parse key signature '{key_name}' at time {time:.3f}s. Skipping. Error: {e}")

        midi_obj.key_signature_changes = [] # Clear default
        applied_key_count = 0
        for time in sorted(unique_ks.keys()):
             if time >= 0:
                 key_number = unique_ks[time]
                 key_change = pretty_midi.KeySignature(key_number=key_number, time=time)
                 midi_obj.key_signature_changes.append(key_change)
                 applied_key_count += 1
        if applied_key_count > 0:
            print(f"Applied {applied_key_count} key signature changes.")
        elif not midi_obj.key_signature_changes: # Add default if none applied
            try:
                default_key_num = pretty_midi.key_name_to_key_number(CONFIG["default_key"])
                midi_obj.key_signature_changes.append(pretty_midi.KeySignature(key_number=default_key_num, time=0.0))
                print(f"Applied default key signature: {CONFIG['default_key']}")
            except ValueError as e:
                print(f"Warning: Could not parse default key signature '{CONFIG['default_key']}'. No key signature applied. Error: {e}")


        # --- Create instruments and add notes ---
        available_channels = list(range(16))
        drum_channel = 9 # Standard GM drum channel
        if drum_channel in available_channels:
            available_channels.remove(drum_channel) # Reserve channel 9 for drums
        channel_index = 0 # Index into available_channels for non-drum tracks

        # Sort definitions to potentially process drums first (though channel assignment handles it)
        sorted_inst_defs = sorted(
            instrument_defs.items(), key=lambda item: item[1]["is_drum"], reverse=True
        )

        for inst_track_key, definition in sorted_inst_defs:
            # inst_track_key is (inst_name, track_id)
            # definition is {program, is_drum, name}

            if not notes_data.get(inst_track_key): # Check if notes exist for this key
                print(f"Skipping instrument '{definition['name']}' as it has no parsed notes.")
                continue

            is_drum = definition["is_drum"]
            program = definition["program"]
            pm_instrument_name = definition["name"]

            # Assign channel
            if is_drum:
                channel = drum_channel
            else:
                if not available_channels:
                    print(f"ERROR: No available non-drum MIDI channels left for instrument '{pm_instrument_name}'. Skipping.")
                    continue # Skip this instrument if no channels left

                # Cycle through available channels if we run out
                channel = available_channels[channel_index % len(available_channels)]
                if channel_index >= len(available_channels):
                     print(f"Warning: Ran out of unique non-drum MIDI channels! Reusing channel {channel} for {pm_instrument_name}.")
                channel_index += 1

            # Create the instrument object
            instrument_obj = pretty_midi.Instrument(
                program=program, is_drum=is_drum, name=pm_instrument_name
            )
            # Add instrument to the MIDI object *before* adding notes
            midi_obj.instruments.append(instrument_obj)

            print(f"Created MIDI instrument: {pm_instrument_name} (Program: {program}, IsDrum: {is_drum}, Assigned Channel: {channel})")


            # Add notes to the instrument object
            note_count = 0
            skipped_notes = 0
            for note_info in notes_data[inst_track_key]:
                start_time = max(0.0, note_info["start"])
                # Ensure minimum duration to avoid zero-length notes
                min_duration = 0.001 # 1 millisecond
                end_time = max(start_time + min_duration, note_info["end"])
                velocity = max(1, min(127, note_info["velocity"])) # Ensure velocity is 1-127
                pitch = max(0, min(127, note_info["pitch"]))

                if start_time >= end_time:
                    print(f"Warning: Skipping note for '{pm_instrument_name}' with non-positive duration (Start: {start_time:.4f}, End: {end_time:.4f}).")
                    skipped_notes += 1
                    continue

                try:
                    note = pretty_midi.Note(
                        velocity=velocity, pitch=pitch, start=start_time, end=end_time,
                    )
                    instrument_obj.notes.append(note)
                    note_count += 1
                except ValueError as e:
                    print(f"Error creating pretty_midi.Note for {pm_instrument_name}: {e}. Skipping note.")
                    print(f"  Note data: Pitch={pitch}, Vel={velocity}, Start={start_time:.4f}, End={end_time:.4f}")
                    skipped_notes += 1

            print(f"  Added {note_count} notes. ({skipped_notes} notes skipped due to errors/duration).")

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
    total_planned_bars = sum(info.get('bars', 0) for info in section_plan.values())

    print(f"\n--- Step 3: Generating {len(section_plan)} Sections ({total_planned_bars} planned bars) ---")
    for section_name, section_info in section_plan.items():
        # Basic validation of section_info before passing
        if not isinstance(section_info.get("bars"), int) or section_info["bars"] <= 0:
             print(f"ERROR: Invalid 'bars' value ({section_info.get('bars')}) for section {section_name}. Skipping.")
             continue # Skip this section

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
            # Add newline if missing for cleaner concatenation
            if not symbolic_section.endswith("\n"):
                symbolic_section += "\n"
            all_symbolic_text += symbolic_section
            generated_sections_count += 1
            last_section_summary_info = current_section_summary_info
            current_bar_count += section_info["bars"] # Advance bar counter

        else:
            print(f"Failed to generate or validate section {section_name}. Stopping generation.")
            break # Stop the whole process if a section fails critically

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