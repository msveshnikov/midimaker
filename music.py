import google.generativeai as genai
import pretty_midi
import re
import os
import time  # For potential rate limiting
import datetime  # For timestamp in filename
import traceback

# --- Configuration ---
API_KEY = "AIzaSyC5jbwgP050qfurqK9GyvgrUYvpwEy0n8s" # IMPORTANT: Use environment variables or secrets management in real apps!
GEMINI_MODEL = 'gemini-2.5-pro-exp-03-25'  #'gemini-2.0-pro-exp-02-05'
INITIAL_DESCRIPTION = "A simple, slightly melancholic loopable piano piece suitable for a game background."
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
DEFAULT_TEMPO = 120
DEFAULT_TIMESIG = (4, 4)
DEFAULT_KEY = "Cmin"
OUTPUT_DIR = "output"  # Directory to save generated MIDI files

# --- Symbolic Format Definition (for prompts and parsing) ---
SYMBOLIC_FORMAT_DEFINITION = """
Use this compact symbolic format ONLY:
- `INST:<ID>` (Instrument: Pno, Gtr, Bass, Drs ...)
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
            " Please set the API_KEY variable or use environment variables."
        )
        exit()
    try:
        genai.configure(api_key=API_KEY)
        print("Google Generative AI configured.")
    except Exception as e:
        print(f"Error configuring Generative AI: {e}")
        print("Please ensure your API_KEY is set correctly.")
        exit()


def call_gemini(prompt, retries=3, delay=5):
    """Calls the Gemini API with retries."""
    model = genai.GenerativeModel(GEMINI_MODEL)
    for attempt in range(retries):
        try:
            # More robust generation settings if needed:
            # generation_config = genai.types.GenerationConfig(
            #     # candidate_count=1, # Usually default
            #     # stop_sequences=['\n\n'], # Example stop sequence
            #     # max_output_tokens=2048, # Limit output size
            #     temperature=0.7 # Control creativity vs coherence
            # )
            # response = model.generate_content(prompt, generation_config=generation_config)
            response = model.generate_content(prompt)

            # Debugging: Print raw response structure if needed
            # print(f"Gemini Response (Attempt {attempt+1}): {response}")

            # Access text, handling potential blocking or empty responses
            if response.parts:
                return response.text
            elif (
                hasattr(response, "prompt_feedback")
                and response.prompt_feedback.block_reason
            ):
                print(
                    f"Prompt blocked (Attempt {attempt + 1}):"
                    f" {response.prompt_feedback.block_reason}"
                )
                return None  # Indicate blockage
            elif (
                hasattr(response, "candidates")
                and response.candidates
                and response.candidates[0].finish_reason != "STOP"
            ):
                # If generation stopped for reasons other than normal completion
                print(
                    f"Generation stopped for reason: {response.candidates[0].finish_reason}"
                )
                # Try to get content even if stopped, might be partial
                if (
                    response.candidates[0].content
                    and response.candidates[0].content.parts
                ):
                    print("Returning partial content due to non-STOP finish reason.")
                    return response.candidates[0].content.parts[0].text
                return None  # Indicate non-standard stop without content

            # If parts is empty but no explicit block/error, check candidates directly
            elif hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    print(
                        "Accessing content via candidates[0] as response.parts was empty."
                    )
                    return candidate.content.parts[0].text

            print(
                f"Warning: Received empty or unexpected response structure from Gemini (Attempt {attempt + 1})."
            )
            # Consider returning None only after all retries fail for this case
            # return None # Indicate empty/unexpected response

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
            return None  # Indicate failure after retries
    return None  # Fallback


# Mapping helpers (can be expanded)
PITCH_MAP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
ACCIDENTAL_MAP = {"#": 1, "b": -1, "": 0}
INSTRUMENT_PROGRAM_MAP = {
    # Piano
    "Pno": 0,
    "Piano": 0,
    "Acoustic Grand Piano": 0,
    # Bass
    "Bass": 33,
    "Electric Bass": 33,
    # Guitar
    "Gtr": 27,
    "Electric Guitar": 27,
    "Acoustic Guitar": 25,
    # Strings
    "Violin": 40,
    "Cello": 42,
    "String Ensemble": 48,
    # Drums are special case (channel 10)
    "Drs": 0,
    "Drums": 0,
    # Add more instruments as needed
}
# Standard drum note map (MIDI channel 10) - General MIDI Level 1 Percussion Key Map
DRUM_PITCH_MAP = {
    # Bass Drum
    "Kick": 36,
    "BD": 36,
    "Bass Drum 1": 36,
    # Snare
    "Snare": 38,
    "SD": 38,
    "Acoustic Snare": 38,
    "Electric Snare": 40,
    # Hi-Hat
    "HiHatClosed": 42,
    "HHC": 42,
    "Closed Hi Hat": 42,
    "HiHatOpen": 46,
    "HHO": 46,
    "Open Hi Hat": 46,
    "HiHatPedal": 44,
    "HH P": 44,
    "Pedal Hi Hat": 44,
    # Cymbals
    "Crash": 49,
    "CR": 49,
    "Crash Cymbal 1": 49,
    "Ride": 51,
    "RD": 51,
    "Ride Cymbal 1": 51,
    # Toms
    "High Tom": 50, "HT": 50,
    "Mid Tom": 47, "MT": 47, # Often Low-Mid Tom
    "Low Tom": 43, "LT": 43, # Often High Floor Tom
    "Floor Tom": 41, "FT": 41, # Often Low Floor Tom
    # Other
    "Rimshot": 37, "RS": 37,
    "Clap": 39, "CP": 39, "Hand Clap": 39,
    "Cowbell": 56, "CB": 56,
}


def pitch_to_midi(pitch_name):
    """Converts pitch name (e.g., C4, F#5) to MIDI number."""
    pitch_name = pitch_name.strip()
    match = re.match(r"([A-G])([#b]?)(\-?\d+)", pitch_name) # Allow negative octaves
    if not match:
        print(f"Warning: Could not parse pitch name: {pitch_name}. Defaulting to Middle C (60).")
        return 60

    note, acc, oct_str = match.groups()
    octave = int(oct_str)

    base_midi = PITCH_MAP.get(note.upper())
    if base_midi is None:
        print(f"Warning: Invalid note name: {note}. Defaulting to Middle C (60).")
        return 60

    acc_val = ACCIDENTAL_MAP.get(acc, 0)

    # Middle C (C4) is MIDI note 60
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

        # Calculate duration of one beat in seconds.
        # The 'beat' is defined by the time signature denominator (e.g., 4 means quarter note is the beat)
        # However, tempo (BPM) is almost always defined as Quarter Notes Per Minute.
        # So, quarter_note_duration_sec is the fundamental unit.
        quarter_note_duration_sec = 60.0 / beats_per_minute

        # Duration relative to a whole note (W=1.0)
        duration_map = {
            "W": 4.0,  # Whole note = 4 quarter notes
            "H": 2.0,  # Half note = 2 quarter notes
            "Q": 1.0,  # Quarter note = 1 quarter note
            "E": 0.5,  # Eighth note = 0.5 quarter notes
            "S": 0.25,  # Sixteenth note = 0.25 quarter notes
            # Add triplets? Requires more complex timing adjustments.
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
        return 0.5  # Default duration (e.g., quarter note at 120 BPM)


# --- Main Pipeline Functions ---


def enrich_music_description(description):
    """Step 1: Use LLM to enrich the initial music description."""
    print("--- Step 1: Enriching Description ---")
    prompt = f"""
    Take the following basic music description and enrich it with more detail for a generative music task.
    Suggest a plausible key signature (e.g., Cmin, Gmaj, F#dor), tempo (e.g., T:90), time signature (e.g., TS:4/4),
    and primary instrumentation (e.g., INST:Pno, INST:Gtr).
    Also, briefly elaborate on the mood and potential simple structure (like AABA).
    Keep the output concise and focused on these musical parameters. Output should start directly with the parameters.

    Basic Description: "{description}"

    Enriched Description (Example Format: K:Cmin T:60 TS:4/4 INST:Pno Mood: Melancholic, sparse. Structure: A-B-A):
    """
    enriched = call_gemini(prompt)
    if enriched:
        print(f"Enriched Description:\n{enriched}\n")
        # Attempt to parse key parameters from the enriched description for defaults
        global DEFAULT_KEY, DEFAULT_TEMPO, DEFAULT_TIMESIG # Allow modification of globals
        enriched_lines = enriched.strip().split()
        key_found, tempo_found, ts_found = False, False, False
        for part in enriched_lines:
             if part.startswith("K:") and not key_found:
                 try:
                     DEFAULT_KEY = part.split(":", 1)[1]
                     print(f"Updated Default Key: {DEFAULT_KEY}")
                     key_found = True
                 except IndexError: pass
             elif part.startswith("T:") and not tempo_found:
                  try:
                      DEFAULT_TEMPO = int(part.split(":", 1)[1])
                      print(f"Updated Default Tempo: {DEFAULT_TEMPO}")
                      tempo_found = True
                  except (IndexError, ValueError): pass
             elif part.startswith("TS:") and not ts_found:
                  try:
                      num_str, den_str = part.split(":", 1)[1].split('/')
                      DEFAULT_TIMESIG = (int(num_str), int(den_str))
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

    context_prompt = ""
    if previous_section_summary:
        # Keep context concise
        context_prompt = f"Context from previous section ({previous_section_summary['name']}): {previous_section_summary['summary']}\nIt ended around time {previous_section_summary['end_time']:.2f}s with key {previous_section_summary['key']} and tempo {previous_section_summary['tempo']}.\n"

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
        # Clean the response: remove potential markdown code fences and surrounding whitespace
        symbolic_text = re.sub(r"^```[a-z]*\n", "", symbolic_text.strip(), flags=re.MULTILINE)
        symbolic_text = re.sub(r"\n```$", "", symbolic_text.strip())
        symbolic_text = symbolic_text.strip()

        # Basic validation: Check if it starts roughly correctly
        lines = symbolic_text.split('\n')
        first_meaningful_line = ""
        for line in lines:
             line = line.strip()
             if line and not line.startswith('#'): # Skip comments/empty
                 first_meaningful_line = line
                 break

        if not first_meaningful_line.startswith(f"BAR:{current_bar}") and not re.match(r"^(T:|TS:|K:|INST:)", first_meaningful_line):
             print(f"Warning: Generated text for {section_name} doesn't seem to start correctly.")
             print(f"Expected BAR:{current_bar} or INST/T/TS/K. Got: '{first_meaningful_line[:50]}...'")
             # Attempt to find the correct starting BAR marker if LLM added preamble
             bar_marker = f"BAR:{current_bar}"
             if bar_marker in symbolic_text:
                 print(f"Found '{bar_marker}' later in the text. Attempting to use text from that point.")
                 symbolic_text = symbolic_text[symbolic_text.find(bar_marker):]
             else:
                 print(f"Could not find '{bar_marker}'. The generated section might be incorrect.")
                 # Decide whether to return None or the potentially incorrect text
                 # return "", None # Option: reject the section

        print(f"Generated symbolic text for Section {section_name}:\n{symbolic_text[:300]}...\n")
        # Simple summary for next section (can be improved)
        summary_info = {
            "name": section_name,
            "summary": f"Generated {bars} bars targeting goal: {goal}.",
            # These will be updated by the parser later for accuracy
            "end_time": 0.0,
            "key": DEFAULT_KEY,
            "tempo": DEFAULT_TEMPO
        }
        return symbolic_text, summary_info
    else:
        print(f"Failed to generate symbolic text for Section {section_name}.")
        return "", None  # Return empty string and no summary on failure


def parse_symbolic_to_structured_data(symbolic_text):
    """Step 3: Parse concatenated symbolic text into structured data for MIDI."""
    print("--- Step 3: Parsing Symbolic Text ---")
    # { ('InstName', 'TrackID'): [note_obj1, ...], ... }
    notes_by_instrument_track = {}
    # Store changes with time: [(time, value), ...]
    tempo_changes = []
    # Store time sig changes with time: [(time, num, den), ...]
    time_signature_changes = []
    # Store key sig changes: [(time, key_name)]
    key_signature_changes = []
    # Store instrument definitions: { ('InstName', 'TrackID'): {'program': X, 'is_drum': Y}, ... }
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
    # If INST is used without a specific track, it might apply as a default? Let's assume INST applies globally for now.
    active_instrument_name = "Pno" # Default instrument if none specified early
    # --- Bar Tracking ---
    current_bar_number = 0 # Start before bar 1
    expected_bar_duration_sec = (60.0 / current_tempo) * current_ts_num * (4.0 / current_ts_den)
    last_event_time = 0.0 # Track the end time of the last event

    # Ensure initial tempo/TS/Key are set at time 0 if not already set by early commands
    initial_commands_set = {'T': False, 'TS': False, 'K': False, 'INST': False}

    lines = symbolic_text.strip().split('\n')
    line_num = 0

    # --- Pre-pass for initial settings before BAR:1 ---
    initial_settings_lines = []
    while line_num < len(lines):
        line = lines[line_num].strip()
        if not line or line.startswith('#'):
            line_num += 1
            continue
        if line.startswith("BAR:"):
            break # Stop pre-pass when first BAR is encountered
        initial_settings_lines.append((line_num + 1, line))
        line_num += 1

    # Process initial settings at time 0.0
    print("Processing initial settings (before BAR:1)...")
    for ini_line_num, ini_line in initial_settings_lines:
        parts = ini_line.split(':', 1)
        command = parts[0].upper()
        value = parts[1].strip() if len(parts) > 1 else ""

        try:
            if command == "INST" and not initial_commands_set['INST']:
                active_instrument_name = value
                initial_commands_set['INST'] = True
                print(f"Initial Instrument set to {active_instrument_name}")
            elif command == "T" and not initial_commands_set['T']:
                new_tempo = float(value)
                if new_tempo != current_tempo:
                    current_tempo = new_tempo
                    tempo_changes.append((0.0, current_tempo))
                    expected_bar_duration_sec = (60.0 / current_tempo) * current_ts_num * (4.0 / current_ts_den)
                    initial_commands_set['T'] = True
                    print(f"Initial Tempo set to {current_tempo} BPM")
            elif command == "TS" and not initial_commands_set['TS']:
                num_str, den_str = value.split('/')
                new_ts_num, new_ts_den = int(num_str), int(den_str)
                if (new_ts_num, new_ts_den) != (current_ts_num, current_ts_den):
                    current_ts_num, current_ts_den = new_ts_num, new_ts_den
                    time_signature_changes.append((0.0, current_ts_num, current_ts_den))
                    expected_bar_duration_sec = (60.0 / current_tempo) * current_ts_num * (4.0 / current_ts_den)
                    initial_commands_set['TS'] = True
                    print(f"Initial Time Signature set to {current_ts_num}/{current_ts_den}")
            elif command == "K" and not initial_commands_set['K']:
                 current_key = value
                 key_signature_changes.append((0.0, current_key))
                 initial_commands_set['K'] = True
                 print(f"Initial Key set to {current_key}")
            else:
                 print(f"Warning: Command '{command}' before BAR:1 on line {ini_line_num} ignored or already set.")

        except Exception as e:
            print(f"Error parsing initial setting line {ini_line_num}: '{ini_line}' - {e}")

    # Add default initial settings if not overridden
    if not initial_commands_set['T']: tempo_changes.append((0.0, current_tempo))
    if not initial_commands_set['TS']: time_signature_changes.append((0.0, current_ts_num, current_ts_den))
    if not initial_commands_set['K']: key_signature_changes.append((0.0, current_key))
    # Initial instrument is handled via active_instrument_name


    # --- Main Parsing Loop (starting from where pre-pass left off) ---
    print("Parsing main body...")
    for i in range(line_num, len(lines)):
        line_num = i + 1 # Use 1-based indexing for messages
        line = lines[i].strip()
        if not line or line.startswith('#'): # Skip empty lines and comments
            continue

        parts = line.split(':', 1)
        command = parts[0].upper()
        value = parts[1] if len(parts) > 1 else ""

        try:
            if command == "INST":
                 new_instrument = value.strip()
                 if new_instrument != active_instrument_name:
                    active_instrument_name = new_instrument
                    print(f"Time {current_global_time:.2f}s (Bar ~{current_bar_number}): Instrument context changed to {active_instrument_name}")
                    # Note: This only sets the *default* for subsequent N/C/R commands.
                    # The actual instrument object is created when a note for that inst/track appears.

            elif command == "T":
                new_tempo = float(value.strip())
                if new_tempo != current_tempo:
                    print(f"Time {current_global_time:.2f}s (Bar ~{current_bar_number}): Tempo change to {new_tempo} BPM")
                    tempo_changes.append((current_global_time, new_tempo))
                    current_tempo = new_tempo
                    expected_bar_duration_sec = (60.0 / current_tempo) * current_ts_num * (4.0 / current_ts_den)

            elif command == "TS":
                 num_str, den_str = value.strip().split('/')
                 new_ts_num, new_ts_den = int(num_str), int(den_str)
                 if (new_ts_num, new_ts_den) != (current_ts_num, current_ts_den):
                      print(f"Time {current_global_time:.2f}s (Bar ~{current_bar_number}): Time Signature change to {new_ts_num}/{new_ts_den}")
                      time_signature_changes.append((current_global_time, new_ts_num, new_ts_den))
                      current_ts_num, current_ts_den = new_ts_num, new_ts_den
                      expected_bar_duration_sec = (60.0 / current_tempo) * current_ts_num * (4.0 / current_ts_den)

            elif command == "K":
                 new_key = value.strip()
                 if new_key != current_key:
                     current_key = new_key
                     print(f"Time {current_global_time:.2f}s (Bar ~{current_bar_number}): Key set to {current_key}")
                     key_signature_changes.append((current_global_time, current_key))

            elif command == "BAR":
                bar_number = int(value.strip())
                # --- Bar Time Synchronization ---
                if current_bar_number > 0: # If this isn't the first bar marker
                    # Expected end time of the previous bar based on its start time and duration
                    expected_prev_bar_end_time = current_bar_start_time + expected_bar_duration_sec
                    # Check accumulated time within the bar vs expected duration
                    duration_error = time_within_bar - expected_bar_duration_sec
                    # Allow a small tolerance (e.g., 1% of bar duration or a fixed small amount)
                    tolerance = max(0.01, expected_bar_duration_sec * 0.01)
                    if abs(duration_error) > tolerance:
                        print(
                            f"Warning: Bar {current_bar_number} duration mismatch. "
                            f"Expected {expected_bar_duration_sec:.3f}s, got {time_within_bar:.3f}s "
                            f"(Error: {duration_error:.3f}s). Adjusting time to expected end."
                        )
                    # Advance global time to the *expected* start of the new bar to prevent drift
                    current_global_time = expected_prev_bar_end_time
                elif bar_number != 1:
                     # If the first BAR command is not BAR:1, assume time starts at 0 but log warning
                     print(f"Warning: First BAR marker is BAR:{bar_number}, not BAR:1. Starting time from 0.")


                # Handle jumps in bar numbers (assume full bars passed)
                bars_jumped = bar_number - (current_bar_number + 1)
                if bars_jumped > 0:
                    jump_duration = bars_jumped * expected_bar_duration_sec
                    print(
                        f"Warning: Jump detected from Bar {current_bar_number} to {bar_number}. "
                        f"Advancing time by {jump_duration:.3f}s ({bars_jumped} bars)."
                    )
                    current_global_time += jump_duration

                # Update current bar state
                current_bar_number = bar_number
                current_bar_start_time = current_global_time # This bar starts now
                time_within_bar = 0.0 # Reset time within the new bar
                # print(f"Time {current_global_time:.2f}s: --- Reached BAR {bar_number} ---") # Can be noisy

            elif command in ["N", "C", "R"]:
                 if current_bar_number == 0:
                     print(f"Warning: Event '{line}' on Line {line_num} found before first BAR marker. Processing at time {current_global_time:.2f}s.")
                     # Process anyway, assuming time 0.0 or current time if settings changed it.

                 data_parts = value.split(':')
                 min_parts = 3 if command == "R" else 4
                 if len(data_parts) < min_parts:
                     print(f"Warning: Malformed {command} command on Line {line_num}: '{line}'. Skipping.")
                     continue

                 track_id = data_parts[0].strip()
                 # Use the globally set instrument context for this event
                 inst_name_for_event = active_instrument_name

                 # Determine if this track/instrument is drums
                 is_drum_track = inst_name_for_event.lower() in ["drs", "drums"] or track_id.lower() in ["drs", "drums"]
                 program = INSTRUMENT_PROGRAM_MAP.get(inst_name_for_event, 0) # Default to Piano

                 # Define instrument properties if not seen before
                 inst_track_key = (inst_name_for_event, track_id)
                 if inst_track_key not in instrument_definitions:
                     instrument_definitions[inst_track_key] = {
                         "program": program,
                         "is_drum": is_drum_track,
                         "name": f"{inst_name_for_event}_{track_id}" # Unique name for pretty_midi
                     }
                     print(f"Defined instrument: {instrument_definitions[inst_track_key]['name']} (Program: {program}, IsDrum: {is_drum_track})")

                 # Ensure note list exists for this instrument/track
                 if inst_track_key not in notes_by_instrument_track:
                      notes_by_instrument_track[inst_track_key] = []

                 event_start_time = current_global_time
                 event_duration_sec = 0.0

                 if command == "N": # Note:Track:Pitch:Duration:Velocity
                      if len(data_parts) < 4:
                           print(f"Warning: Malformed Note command on Line {line_num}: '{line}'. Skipping.")
                           continue
                      pitch_name = data_parts[1].strip()
                      duration_sym = data_parts[2].strip()
                      velocity = int(data_parts[3].strip())

                      event_duration_sec = duration_to_seconds(duration_sym, current_tempo, current_ts_den)

                      # Drum mapping or regular pitch mapping
                      if is_drum_track:
                           # Use original pitch name for drum map lookup
                           midi_pitch = DRUM_PITCH_MAP.get(pitch_name, None)
                           if midi_pitch is None:
                               print(f"Warning: Unknown drum sound '{pitch_name}' on Line {line_num}. Using Kick (36).")
                               midi_pitch = 36
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
                           print(f"Warning: Malformed Chord command on Line {line_num}: '{line}'. Skipping.")
                           continue
                      pitches_str = data_parts[1].strip()
                      duration_sym = data_parts[2].strip()
                      velocity = int(data_parts[3].strip())

                      # Handle brackets and split pitches
                      if pitches_str.startswith('[') and pitches_str.endswith(']'):
                           pitches_str = pitches_str[1:-1]
                      pitch_names = [p.strip() for p in pitches_str.split(',') if p.strip()]

                      if not pitch_names:
                           print(f"Warning: No valid pitches found in Chord command on Line {line_num}: '{line}'. Skipping.")
                           continue

                      event_duration_sec = duration_to_seconds(duration_sym, current_tempo, current_ts_den)

                      for pitch_name in pitch_names:
                            # Assume chords aren't typically on drum tracks, use standard pitch conversion
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
                           print(f"Warning: Malformed Rest command on Line {line_num}: '{line}'. Skipping.")
                           continue
                      # Track ID is data_parts[0], Duration is data_parts[1]
                      duration_sym = data_parts[1].strip()
                      event_duration_sec = duration_to_seconds(duration_sym, current_tempo, current_ts_den)
                      # Rests just advance time, no note event added. They belong to a track conceptually,
                      # but don't generate sound. The time advance handles the silence.

                 # --- Advance Time ---
                 current_global_time += event_duration_sec
                 time_within_bar += event_duration_sec
                 last_event_time = max(last_event_time, current_global_time) # Keep track of the latest time point

            else:
                 print(f"Warning: Unknown command '{command}' on line {line_num}: '{line}'. Skipping.")

        except Exception as e:
            print(f"Error parsing line {line_num}: '{line}' - {e}")
            traceback.print_exc() # Print full traceback for debugging
            continue # Skip problematic line

    print(f"Symbolic text parsing complete. Total duration: ~{last_event_time:.2f} seconds.")
    # Return all parsed information
    return (
        notes_by_instrument_track,
        instrument_definitions,
        tempo_changes,
        time_signature_changes,
        key_signature_changes,
        last_event_time, # Return the calculated end time
        current_key, # Return the final key
        current_tempo # Return the final tempo
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
    print(f"--- Step 4: Creating MIDI File ({filename}) ---")
    if not notes_data:
        print("No note data found. Cannot create MIDI file.")
        return

    try:
        # Initialize with the first tempo change
        initial_tempo = tempo_changes[0][1] if tempo_changes else DEFAULT_TEMPO
        midi_obj = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo)

        # Add Time Signature Changes
        # Sort by time just in case they weren't added chronologically
        time_sig_changes.sort(key=lambda x: x[0])
        applied_ts_count = 0
        for time, num, den in time_sig_changes:
             # Denominator needs to be power of 2 for standard MIDI
             den_power = den.bit_length() - 1
             actual_den = 2**den_power
             if actual_den != den:
                 print(
                     f"Warning: Non-power-of-2 time signature denominator {den} at time {time:.2f}s."
                     f" Using closest power of 2: {actual_den}."
                 )
             # Prevent adding duplicate TS at the same time
             if not midi_obj.time_signature_changes or midi_obj.time_signature_changes[-1].time < time:
                 ts_change = pretty_midi.TimeSignature(num, actual_den, time)
                 midi_obj.time_signature_changes.append(ts_change)
                 applied_ts_count += 1
             elif midi_obj.time_signature_changes[-1].time == time:
                  print(f"Ignoring duplicate time signature change at time {time:.2f}s.")

        if applied_ts_count > 0:
            print(f"Applied {applied_ts_count} time signature changes.")
        elif not midi_obj.time_signature_changes: # Add default if none were added
            midi_obj.time_signature_changes.append(
                pretty_midi.TimeSignature(DEFAULT_TIMESIG[0], DEFAULT_TIMESIG[1], 0.0)
            )
            print(f"Applied default time signature: {DEFAULT_TIMESIG[0]}/{DEFAULT_TIMESIG[1]}")


        # Add Key Signature Changes
        # Sort by time
        key_sig_changes.sort(key=lambda x: x[0])
        applied_key_count = 0
        for time, key_name in key_sig_changes:
             try:
                 key_number = pretty_midi.key_name_to_key_number(key_name)
                 # Prevent adding duplicate key sig at the same time
                 if not midi_obj.key_signature_changes or midi_obj.key_signature_changes[-1].time < time:
                     key_change = pretty_midi.KeySignature(key_number=key_number, time=time)
                     midi_obj.key_signature_changes.append(key_change)
                     applied_key_count += 1
                 elif midi_obj.key_signature_changes[-1].time == time:
                      print(f"Ignoring duplicate key signature change at time {time:.2f}s.")

             except ValueError as e:
                 print(
                     f"Warning: Could not parse key signature '{key_name}' at time {time:.2f}s. Skipping. Error: {e}"
                 )
        if applied_key_count > 0:
            print(f"Applied {applied_key_count} key signature changes.")
        elif not midi_obj.key_signature_changes: # Add default if none were added
            try:
                midi_obj.key_signature_changes.append(
                    pretty_midi.KeySignature(
                        pretty_midi.key_name_to_key_number(DEFAULT_KEY), 0.0
                    )
                )
                print(f"Applied default key signature: {DEFAULT_KEY}")
            except ValueError as e:
                print(
                    f"Warning: Could not parse default key signature '{DEFAULT_KEY}'. Error: {e}"
                )


        # Create instruments and add notes
        midi_instruments = {} # { ('InstName', 'TrackID'): <Instrument object>, ... }
        # Assign channels, ensuring drums are on channel 9 (0-indexed)
        next_channel = 0
        for inst_track_key, definition in instrument_defs.items():
            if inst_track_key not in notes_data or not notes_data[inst_track_key]:
                print(f"Skipping instrument '{definition['name']}' as it has no notes.")
                continue # Skip instruments without notes

            is_drum = definition['is_drum']
            program = definition['program']
            pm_instrument_name = definition['name']

            # Create Instrument object
            if is_drum:
                channel = 9 # Standard MIDI drum channel
            else:
                # Find next available non-drum channel
                while next_channel == 9:
                    next_channel = (next_channel + 1) % 16 # Cycle through 0-15, skipping 9
                channel = next_channel
                next_channel = (next_channel + 1) % 16 # Move to next for subsequent instruments


            instrument_obj = pretty_midi.Instrument(
                program=program, is_drum=is_drum, name=pm_instrument_name
            )
            midi_obj.instruments.append(instrument_obj)
            midi_instruments[inst_track_key] = instrument_obj # Store for adding notes
            print(
                f"Created MIDI instrument: {pm_instrument_name} (Program: {program}, IsDrum: {is_drum}, Assigned Channel: {channel})"
            )


            # Add notes to this instrument
            note_count = 0
            for note_info in notes_data[inst_track_key]:
                 # Ensure start time is not negative and end time is strictly after start time
                 start_time = max(0.0, note_info['start'])
                 # Ensure minimum duration to avoid zero-length notes which can cause issues
                 min_duration = 0.001 # 1ms minimum duration
                 end_time = max(start_time + min_duration, note_info['end'])

                 note = pretty_midi.Note(
                      velocity=note_info['velocity'],
                      pitch=note_info['pitch'],
                      start=start_time,
                      end=end_time,
                 )
                 instrument_obj.notes.append(note)
                 note_count += 1
            print(f"Added {note_count} notes to instrument {pm_instrument_name}.")


        # Add tempo changes (beyond the initial one) using MIDI Tempo Change events
        # pretty_midi handles this internally when writing the file based on initial_tempo
        # and the timing of notes. For explicit tempo change events within the MIDI file itself:
        if len(tempo_changes) > 1:
             # Get the MIDI tick scale (ticks per second)
             # Note: This is complex as tick scale can change if time signature changes.
             # pretty_midi's internal handling based on note times is generally sufficient.
             # Adding explicit tempo change meta-messages requires deeper manipulation.
             print(
                 f"Note: Multiple tempo changes detected ({len(tempo_changes)}). "
                 "pretty_midi sets initial tempo. Playback software behavior with "
                 "internal tempo changes varies. Relying on note timings for now."
             )
             # If precise MIDI tempo events are needed:
             # midi_obj.tick_to_time = # Needs careful calculation
             # for time, bpm in tempo_changes[1:]: # Skip the initial one
             #     tick = midi_obj.time_to_tick(time)
             #     tempo_event = pretty_midi.containers.TempoChange(bpm, tick)
             #     # This attribute doesn't exist directly, need to add to MIDI track events manually


        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        full_output_path = os.path.join(OUTPUT_DIR, filename)

        # Write the MIDI file
        midi_obj.write(full_output_path)
        print(f"Successfully created MIDI file: {full_output_path}")

    except Exception as e:
        print(f"Error writing MIDI file '{filename}': {e}")
        traceback.print_exc()


# --- Main Execution ---
if __name__ == "__main__":
    configure_genai()

    # Step 1
    enriched_description = enrich_music_description(INITIAL_DESCRIPTION)
    if not enriched_description:
        enriched_description = INITIAL_DESCRIPTION # Fallback

    # Step 2 (Section Generation) & Concatenation
    all_symbolic_text = ""
    current_bar_count = 1
    last_section_summary_info = None # Store dict with name, summary, end_time, key, tempo
    generated_sections_count = 0

    for section_name, section_info in SECTIONS.items():
        symbolic_section, current_section_summary_info = generate_symbolic_section(
            enriched_description,
            section_name,
            section_info,
            current_bar_count,
            last_section_summary_info, # Pass summary dict of previous section
        )

        if symbolic_section and current_section_summary_info: # Check both
            # Ensure section ends with a newline for clean concatenation
            if not symbolic_section.endswith('\n'):
                symbolic_section += '\n'
            all_symbolic_text += symbolic_section
            generated_sections_count += 1

            # Update the summary info with parsed data *after* parsing this section
            # (We'll parse the whole concatenated text later)
            last_section_summary_info = current_section_summary_info # Pass this to the *next* iteration

            # Advance bar count for the next section's *starting* point
            current_bar_count += section_info['bars']
        else:
            print(f"Skipping section {section_name} due to generation failure.")
            # Decide whether to stop or continue without the failed section
            # break # Uncomment to stop the whole process if one section fails

    if not all_symbolic_text.strip():
        print("ERROR: No symbolic text was generated successfully. Cannot proceed.")
        exit()
    if generated_sections_count < len(SECTIONS):
         print(f"Warning: Only {generated_sections_count}/{len(SECTIONS)} sections were generated.")

    print("\n--- Combined Symbolic Text ---")
    # Save combined text to a file for debugging
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    symbolic_filename = os.path.join(OUTPUT_DIR, f"symbolic_music_{timestamp_str}.txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        with open(symbolic_filename, "w", encoding="utf-8") as f:
            f.write(all_symbolic_text)
        print(f"Saved combined symbolic text to: {symbolic_filename}")
    except IOError as e:
        print(f"Error saving symbolic text file: {e}")

    # print(all_symbolic_text[:1000] + "...") # Print beginning of combined text
    print("----------------------------\n")

    # Step 3
    (
        parsed_notes,
        instrument_definitions,
        tempo_changes,
        time_sig_changes,
        key_sig_changes,
        _, # Ignore end time, key, tempo from parser here as create_midi uses the change lists
        _,
        _
    ) = parse_symbolic_to_structured_data(all_symbolic_text)


    # Step 4
    if parsed_notes: # Only create MIDI if parsing yielded notes
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
        print("No notes were successfully parsed. MIDI file not created.")

    print("\n--- Pipeline Finished ---")