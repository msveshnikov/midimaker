import google.generativeai as genai
import pretty_midi
import re
import os
import time # For potential rate limiting
import datetime # For timestamp in filename

# --- Configuration ---
API_KEY = "AIzaSyC5jbwgP050qfurqK9GyvgrUYvpwEy0n8s" # IMPORTANT: Use environment variables or secrets management in real apps!
GEMINI_MODEL = 'gemini-2.5-pro-exp-03-25'  #'gemini-2.0-pro-exp-02-05'
INITIAL_DESCRIPTION = "A simple, slightly melancholic loopable piano piece suitable for a game background."
SECTIONS = {
    "A1": {"bars": 8, "goal": "Establish the main melancholic theme in C minor. Simple texture."},
    "B": {"bars": 8, "goal": "Introduce a slight variation or counter-melody, perhaps brightening slightly towards Eb major."},
    "A2": {"bars": 8, "goal": "Return to the main theme, similar to A1 but maybe with slight embellishment."},
}
DEFAULT_TEMPO = 120
DEFAULT_TIMESIG = (4, 4)
DEFAULT_KEY = "Cmin"

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
            response = model.generate_content(prompt)
            # Add basic safety check - sometimes response might be empty or blocked
            if response.parts:
                 return response.text
            elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                 print(f"Prompt blocked: {response.prompt_feedback.block_reason}")
                 return None # Indicate blockage
            else:
                 # Check candidates for potential content if parts is empty
                 if hasattr(response, 'candidates') and response.candidates:
                     candidate = response.candidates[0]
                     if candidate.content and candidate.content.parts:
                         return candidate.content.parts[0].text
                     elif candidate.finish_reason != 'STOP':
                         print(f"Generation stopped for reason: {candidate.finish_reason}")
                         return None # Indicate non-standard stop

                 print("Warning: Received empty or blocked response from Gemini.")
                 return None # Indicate empty/blocked response

        except Exception as e:
            print(f"Error calling Gemini API (Attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Failing.")
                return None # Indicate failure after retries
    return None # Fallback

# Mapping helpers (can be expanded)
PITCH_MAP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
ACCIDENTAL_MAP = {"#": 1, "b": -1, "": 0}
INSTRUMENT_PROGRAM_MAP = {
    "Pno": 0, "Piano": 0, # Acoustic Grand Piano
    "Bass": 33, # Electric Bass (finger)
    "Gtr": 27, # Electric Guitar (clean)
    "Drs": 0 # Drums are usually on channel 10, handled differently
    # Add more instruments as needed
}
# Standard drum note map (MIDI channel 10) - very basic example
DRUM_PITCH_MAP = {
    "Kick": 36, "BD": 36,
    "Snare": 38, "SD": 38,
    "HiHatClosed": 42, "HHC": 42,
    "HiHatOpen": 46, "HHO": 46,
    "Crash": 49, "CR": 49,
    "Ride": 51, "RD": 51,
    "Tom1": 48, "HT": 48, # High Tom
    "Tom2": 45, "MT": 45, # Mid Tom
    "Tom3": 41, "LT": 41, # Low Tom
}

def pitch_to_midi(pitch_name):
    """Converts pitch name (e.g., C4, F#5) to MIDI number."""
    match = re.match(r"([A-G])([#b]?)(\d+)", pitch_name.strip())
    if not match:
        print(f"Warning: Could not parse pitch name: {pitch_name}")
        return 60 # Default to Middle C

    note, acc, oct_str = match.groups()
    octave = int(oct_str)

    base_midi = PITCH_MAP.get(note.upper())
    if base_midi is None:
         print(f"Warning: Invalid note name: {note}")
         return 60

    acc_val = ACCIDENTAL_MAP.get(acc, 0)

    midi_val = base_midi + acc_val + (octave + 1) * 12
    # Clamp to valid MIDI range 0-127
    return max(0, min(127, midi_val))


def duration_to_seconds(duration_symbol, tempo, time_sig_denominator):
    """Converts duration symbol (W, H, Q, E, S, W., H., etc.) to seconds."""
    try:
        beats_per_minute = float(tempo)
        # Calculate duration of a quarter note in seconds
        quarter_note_duration_sec = 60.0 / beats_per_minute

        # Duration relative to a quarter note (Q=1.0)
        # This assumes 4/4 style beat understanding, adjust if TS denominator implies different beat unit
        duration_map = {'W': 4.0, 'H': 2.0, 'Q': 1.0, 'E': 0.5, 'S': 0.25}

        base_symbol = duration_symbol.replace('.', '')
        is_dotted = duration_symbol.endswith('.')

        relative_duration = duration_map.get(base_symbol.upper())
        if relative_duration is None:
             print(f"Warning: Unknown duration symbol: {duration_symbol}. Defaulting to Quarter.")
             relative_duration = 1.0 # Default to Quarter Note

        # Adjust for dotted notes
        if is_dotted:
            relative_duration *= 1.5

        # Calculate actual duration in seconds
        # Assumes the denominator of time signature represents the beat unit for tempo definition (e.g., x/4 means tempo is quarter notes per minute)
        # A more robust calculation might consider the time signature denominator directly
        # For common usage (x/4), this works:
        actual_duration_sec = relative_duration * quarter_note_duration_sec

        return actual_duration_sec

    except Exception as e:
        print(f"Error calculating duration for '{duration_symbol}': {e}. Using default.")
        return 0.5 # Default duration if calculation fails


# --- Main Pipeline Functions ---

def enrich_music_description(description):
    """Step 1: Use LLM to enrich the initial music description."""
    print("--- Step 1: Enriching Description ---")
    prompt = f"""
    Take the following basic music description and enrich it with more detail for a generative music task. Suggest specific musical elements like key signature, tempo range, instrumentation (if not specified), mood nuances, and potential structure (e.g., AABA, verse-chorus). Keep the output concise and focused on musical parameters.

    Basic Description: "{description}"

    Enriched Description:
    """
    enriched = call_gemini(prompt)
    if enriched:
        print(f"Enriched Description:\n{enriched}\n")
        return enriched
    else:
        print("Failed to enrich description. Using initial description.")
        return description

def generate_symbolic_section(overall_desc, section_name, section_info, current_bar, previous_section_summary=None):
    """Step 2: Generate symbolic music for one section using LLM."""
    print(f"--- Step 2: Generating Symbolic Section {section_name} (Starting Bar: {current_bar}) ---")
    bars = section_info['bars']
    goal = section_info['goal']

    context_prompt = ""
    if previous_section_summary:
        context_prompt = f"Context from previous section: {previous_section_summary}\n"

    prompt = f"""
    You are a symbolic music generator. Your task is to generate ONLY the symbolic music notation for a specific section of a piece.

    Overall Music Goal: {overall_desc}
    {context_prompt}
    Current Section: {section_name}
    Target Bars: {bars} (Start this section exactly at BAR:{current_bar}, end before BAR:{current_bar + bars})
    Section Goal: {goal}

    Instructions:
    1. Generate music ONLY for this section, starting exactly with `BAR:{current_bar}` and continuing for {bars} bars total.
    2. Strictly adhere to the compact symbolic format defined below. Do NOT include any other text, explanations, or apologies outside the format. Output ONLY the commands.
    3. If tempo (T), time signature (TS), key (K), or instrument (INST) need to be set or changed AT THE BEGINNING of this section, include those commands BEFORE `BAR:{current_bar}`. Assume defaults T:{DEFAULT_TEMPO}, TS:{DEFAULT_TIMESIG[0]}/{DEFAULT_TIMESIG[1]}, K:{DEFAULT_KEY} if not specified otherwise or in previous context.
    4. Ensure musical coherence within the section and try to follow the Section Goal. Make smooth transitions if context from a previous section is provided.
    5. Ensure the total duration of notes/rests within each bar adds up correctly according to the time signature.
    6. End the generation cleanly before the next section would start (i.e., before BAR:{current_bar + bars}).

    {SYMBOLIC_FORMAT_DEFINITION}

    Generate Section {section_name} symbolic music now (starting BAR:{current_bar}):
    """

    symbolic_text = call_gemini(prompt)
    if symbolic_text:
         # Basic validation: Check if it starts roughly correctly
         if not symbolic_text.strip().startswith(f"BAR:{current_bar}") and not re.match(r"^(T:|TS:|K:|INST:).*\nBAR:", symbolic_text.strip(), re.MULTILINE):
              print(f"Warning: Generated text for {section_name} doesn't seem to start correctly with BAR:{current_bar} or initial commands.")
         print(f"Generated symbolic text for Section {section_name}:\n{symbolic_text[:300]}...\n") # Print beginning
         # Simple summary for next section (can be improved)
         summary = f"Ended section {section_name} targeting goal: {goal}."
         return symbolic_text.strip(), summary
    else:
         print(f"Failed to generate symbolic text for Section {section_name}.")
         return "", None # Return empty string and no summary on failure

def parse_symbolic_to_structured_data(symbolic_text):
    """Step 3: Parse concatenated symbolic text into structured data for MIDI."""
    print("--- Step 3: Parsing Symbolic Text ---")
    notes_by_instrument_track = {} # { ('Pno', 'RH'): [note_obj1, ...], ('Bass', 'Main'): [...] }
    tempo_changes = [] # Store tempo changes with time: [(time, tempo), ...]
    time_signature_changes = [] # Store time sig changes with time: [(time, num, den), ...]
    key_signature_changes = [] # Store key sig changes: [(time, key_name)] - for info/MIDI meta

    # --- Time Tracking State ---
    # Global time advances with every note/rest/chord based on its duration
    current_global_time = 0.0
    # Time within the current bar, reset by BAR commands
    time_within_bar = 0.0
    # Start time of the current bar, used for calculating global time advancement at BAR markers
    current_bar_start_time = 0.0
    # --- Musical State ---
    current_tempo = float(DEFAULT_TEMPO)
    current_ts_num, current_ts_den = DEFAULT_TIMESIG
    current_key = DEFAULT_KEY
    active_instrument = "Pno" # Default instrument
    # --- Bar Tracking ---
    current_bar_number = 0 # Start before bar 1
    expected_bar_duration_sec = (60.0 / current_tempo) * current_ts_num * (4.0 / current_ts_den)

    # Ensure initial tempo/TS/Key are set at time 0
    tempo_changes.append((0.0, current_tempo))
    time_signature_changes.append((0.0, current_ts_num, current_ts_den))
    key_signature_changes.append((0.0, current_key))

    lines = symbolic_text.strip().split('\n')

    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('#'): # Skip empty lines and comments
            continue

        parts = line.split(':', 1)
        command = parts[0].upper()
        value = parts[1] if len(parts) > 1 else ""

        try:
            if command == "INST":
                 new_instrument = value.strip()
                 if new_instrument != active_instrument:
                    active_instrument = new_instrument
                    print(f"Time {current_global_time:.2f} (Bar ~{current_bar_number}): Instrument set to {active_instrument}")

            elif command == "T":
                new_tempo = float(value)
                if new_tempo != current_tempo:
                    print(f"Time {current_global_time:.2f} (Bar ~{current_bar_number}): Tempo change to {new_tempo} BPM")
                    # Add tempo change at the *current* time position
                    tempo_changes.append((current_global_time, new_tempo))
                    current_tempo = new_tempo
                    # Recalculate expected bar duration
                    expected_bar_duration_sec = (60.0 / current_tempo) * current_ts_num * (4.0 / current_ts_den)

            elif command == "TS":
                 num_str, den_str = value.split('/')
                 new_ts_num, new_ts_den = int(num_str), int(den_str)
                 if (new_ts_num, new_ts_den) != (current_ts_num, current_ts_den):
                      print(f"Time {current_global_time:.2f} (Bar ~{current_bar_number}): Time Signature change to {new_ts_num}/{new_ts_den}")
                      # Add TS change at the *current* time position
                      time_signature_changes.append((current_global_time, new_ts_num, new_ts_den))
                      current_ts_num, current_ts_den = new_ts_num, new_ts_den
                      # Recalculate expected bar duration
                      expected_bar_duration_sec = (60.0 / current_tempo) * current_ts_num * (4.0 / current_ts_den)

            elif command == "K":
                 new_key = value.strip()
                 if new_key != current_key:
                     current_key = new_key
                     print(f"Time {current_global_time:.2f} (Bar ~{current_bar_number}): Key set to {current_key}")
                     key_signature_changes.append((current_global_time, current_key))


            elif command == "BAR":
                bar_number = int(value)
                # --- Bar Time Synchronization ---
                # If this isn't the first bar marker encountered
                if current_bar_number > 0:
                    # Calculate the expected end time of the previous bar
                    expected_prev_bar_end_time = current_bar_start_time + expected_bar_duration_sec
                    # Check if the accumulated time within the bar roughly matches expectations
                    if abs(time_within_bar - expected_bar_duration_sec) > 0.01: # Tolerance for float errors
                        print(f"Warning: Bar {current_bar_number} duration mismatch. Expected {expected_bar_duration_sec:.3f}s, got {time_within_bar:.3f}s. Adjusting time.")
                    # Advance global time to the start of the new bar, potentially correcting drift
                    current_global_time = expected_prev_bar_end_time

                # If there's a jump in bar numbers, advance time accordingly (assuming full bars)
                bars_jumped = bar_number - current_bar_number - 1
                if bars_jumped > 0:
                    print(f"Warning: Jump detected from Bar {current_bar_number} to {bar_number}. Assuming {bars_jumped} full bars passed.")
                    current_global_time += bars_jumped * expected_bar_duration_sec

                # Update current bar state
                current_bar_number = bar_number
                current_bar_start_time = current_global_time # Set the start time for this new bar
                time_within_bar = 0.0 # Reset time within the new bar
                print(f"Time {current_global_time:.2f}: --- Reached BAR {bar_number} ---")


            elif command in ["N", "C", "R"]:
                 if current_bar_number == 0:
                     print(f"Warning: Note/Chord/Rest event found before first BAR marker (Line {line_num+1}). Events should follow BAR:1.")
                     # Optionally skip or try to process assuming Bar 1 starts at time 0

                 data_parts = value.split(':')
                 if len(data_parts) < (3 if command == "R" else 4):
                     print(f"Warning: Malformed {command} command on Line {line_num+1}: '{line}'. Skipping.")
                     continue

                 track_id = data_parts[0].strip()

                 # Determine instrument - drums handled differently
                 is_drum_track = active_instrument.lower() in ["drs", "drums"] or track_id.lower() in ["drs", "drums"]
                 instrument_key = "Drs" if is_drum_track else active_instrument

                 # Instrument + Track combination key
                 inst_track_key = (instrument_key, track_id)
                 if inst_track_key not in notes_by_instrument_track:
                      notes_by_instrument_track[inst_track_key] = []

                 event_duration_sec = 0.0

                 if command == "N": # Note:Track:Pitch:Duration:Velocity
                      pitch_name = data_parts[1].strip()
                      duration_sym = data_parts[2].strip()
                      velocity = int(data_parts[3].strip())

                      event_duration_sec = duration_to_seconds(duration_sym, current_tempo, current_ts_den)

                      # Drum mapping or regular pitch mapping
                      if is_drum_track:
                           midi_pitch = DRUM_PITCH_MAP.get(pitch_name, DRUM_PITCH_MAP.get("Kick")) # Default to Kick if unknown
                           if midi_pitch is None: midi_pitch = 36 # Absolute fallback
                      else:
                           midi_pitch = pitch_to_midi(pitch_name)

                      note_event = {
                           "pitch": midi_pitch,
                           "start": current_global_time,
                           "end": current_global_time + event_duration_sec,
                           "velocity": max(0, min(127, velocity)) # Clamp velocity
                      }
                      notes_by_instrument_track[inst_track_key].append(note_event)


                 elif command == "C": # Chord:Track:[P1,P2,...]:Duration:Velocity
                      pitches_str = data_parts[1].strip()
                      # Handle potential spaces within brackets: remove brackets, then split
                      if pitches_str.startswith('[') and pitches_str.endswith(']'):
                           pitches_str = pitches_str[1:-1]
                      pitch_names = [p.strip() for p in pitches_str.split(',') if p.strip()]

                      duration_sym = data_parts[2].strip()
                      velocity = int(data_parts[3].strip())

                      event_duration_sec = duration_to_seconds(duration_sym, current_tempo, current_ts_den)

                      for pitch_name in pitch_names:
                            # Assume chords aren't typically on drum tracks
                            midi_pitch = pitch_to_midi(pitch_name)
                            note_event = {
                                "pitch": midi_pitch,
                                "start": current_global_time,
                                "end": current_global_time + event_duration_sec,
                                "velocity": max(0, min(127, velocity)) # Clamp velocity
                            }
                            notes_by_instrument_track[inst_track_key].append(note_event)


                 elif command == "R": # Rest:Track:Duration
                      duration_sym = data_parts[1].strip()
                      event_duration_sec = duration_to_seconds(duration_sym, current_tempo, current_ts_den)
                      # Rests just advance time, no note event added

                 # --- Advance Time ---
                 current_global_time += event_duration_sec
                 time_within_bar += event_duration_sec

        except Exception as e:
            print(f"Error parsing line {line_num+1}: '{line}' - {e}")
            # Decide whether to skip line, stop parsing, etc.
            continue # Skip problematic line

    print("Symbolic text parsing complete.")
    # Return key signatures as well, though pretty_midi might not use them directly for playback
    return notes_by_instrument_track, tempo_changes, time_signature_changes, key_signature_changes


def create_midi_file(notes_data, tempo_changes, time_sig_changes, key_sig_changes, filename):
    """Step 4: Create MIDI file using pretty_midi."""
    print(f"--- Step 4: Creating MIDI File ({filename}) ---")
    try:
        midi_obj = pretty_midi.PrettyMIDI(initial_tempo=tempo_changes[0][1] if tempo_changes else DEFAULT_TEMPO)

        # Add Time Signature Changes (pretty_midi supports multiple)
        for time, num, den in time_sig_changes:
             # Denominator needs to be power of 2 for standard MIDI
             den_power = den.bit_length() - 1
             if 2**den_power != den:
                 print(f"Warning: Non-power-of-2 time signature denominator {den} at time {time:.2f}. MIDI requires powers of 2 (e.g., 2, 4, 8). Using closest power of 2: {2**den_power}.")
                 den = 2**den_power
             ts_change = pretty_midi.TimeSignature(num, den, time)
             midi_obj.time_signature_changes.append(ts_change)
        if time_sig_changes:
             print(f"Applied {len(time_sig_changes)} time signature changes.")
        else:
             # Add default if none specified
             midi_obj.time_signature_changes.append(pretty_midi.TimeSignature(DEFAULT_TIMESIG[0], DEFAULT_TIMESIG[1], 0.0))
             print(f"Applied default time signature: {DEFAULT_TIMESIG[0]}/{DEFAULT_TIMESIG[1]}")


        # Add Key Signature Changes (pretty_midi supports multiple)
        for time, key_name in key_sig_changes:
             try:
                 key_change = pretty_midi.KeySignature(
                     key_number=pretty_midi.key_name_to_key_number(key_name),
                     time=time
                 )
                 midi_obj.key_signature_changes.append(key_change)
             except ValueError as e:
                 print(f"Warning: Could not parse key signature '{key_name}' at time {time:.2f}. Skipping. Error: {e}")
        if key_sig_changes:
             print(f"Applied {len(key_sig_changes)} key signature changes.")
        else:
            # Add default if none specified
             try:
                midi_obj.key_signature_changes.append(pretty_midi.KeySignature(pretty_midi.key_name_to_key_number(DEFAULT_KEY), 0.0))
                print(f"Applied default key signature: {DEFAULT_KEY}")
             except ValueError as e:
                 print(f"Warning: Could not parse default key signature '{DEFAULT_KEY}'. Error: {e}")


        # Create instruments and add notes
        instruments = {} # { ('Pno', 'RH'): <Instrument object>, ... }

        for (inst_name, track_id), notes in notes_data.items():
            is_drum = inst_name.lower() in ["drs", "drums"]

            # Get MIDI program number
            program = INSTRUMENT_PROGRAM_MAP.get(inst_name, 0) # Default to Piano if unknown

            # Combine Inst+Track into a unique instrument name for pretty_midi
            pm_instrument_name = f"{inst_name}_{track_id}"

            # Create Instrument object if it doesn't exist
            if pm_instrument_name not in instruments:
                 # Drums go to channel 9 (0-indexed) in pretty_midi/standard MIDI
                 channel = 9 if is_drum else len(midi_obj.instruments) % 15 # Avoid channel 9 for non-drums
                 if channel >= 9: channel += 1 # Skip channel 9 explicitly if we wrap around

                 instruments[pm_instrument_name] = pretty_midi.Instrument(
                     program=program,
                     is_drum=is_drum,
                     name=pm_instrument_name
                 )
                 midi_obj.instruments.append(instruments[pm_instrument_name])
                 print(f"Created instrument: {pm_instrument_name} (Program: {program}, IsDrum: {is_drum}, Channel: {channel})") # Channel is implicit via order in pretty_midi

            # Add notes to this instrument
            instrument_obj = instruments[pm_instrument_name]
            note_count = 0
            for note_info in notes:
                 # Ensure start time is not negative and end time is after start time
                 start_time = max(0.0, note_info['start'])
                 end_time = max(start_time + 0.001, note_info['end']) # Ensure minimum duration

                 note = pretty_midi.Note(
                      velocity=note_info['velocity'],
                      pitch=note_info['pitch'],
                      start=start_time,
                      end=end_time
                 )
                 instrument_obj.notes.append(note)
                 note_count += 1
            print(f"Added {note_count} notes to instrument {pm_instrument_name}.")


        # Add tempo changes (beyond the initial one) using MIDI Tempo Change events
        # Need to convert pretty_midi times (seconds) to MIDI ticks first.
        # This part is more complex and often handled sufficiently by DAWs interpreting the initial tempo.
        # For simplicity here, we rely on the initial tempo setting and assume DAWs handle it.
        # If more precise intra-file tempo changes are needed, would require `_tick_scales` manipulation
        # or direct MIDI event insertion, which is beyond the scope of typical pretty_midi usage.
        if len(tempo_changes) > 1:
            print(f"Note: Multiple tempo changes detected ({len(tempo_changes)}). pretty_midi sets initial tempo. Precise intra-file changes might require manual MIDI event handling depending on playback software.")


        # Write the MIDI file
        midi_obj.write(filename)
        print(f"Successfully created MIDI file: {filename}")

    except Exception as e:
        print(f"Error writing MIDI file '{filename}': {e}")
        import traceback
        traceback.print_exc()


# --- Main Execution ---
if __name__ == "__main__":

    if not API_KEY or API_KEY == "YOUR_API_KEY" or "AIzaSyC5jbwgP050qfurqK9GyvgrUYvpwEy0n8s": # Basic check
         print("WARNING: API_KEY is not securely set or is using the placeholder.")
         # Consider adding a stronger check or exiting if needed
         # exit() # Uncomment to force setting a real key

    configure_genai()

    # Step 1
    enriched_description = enrich_music_description(INITIAL_DESCRIPTION)
    if not enriched_description:
         enriched_description = INITIAL_DESCRIPTION # Fallback

    # Step 2 (Section Generation) & Concatenation
    all_symbolic_text = ""
    current_bar_count = 1
    last_section_summary = None
    for section_name, section_info in SECTIONS.items():
        symbolic_section, last_section_summary = generate_symbolic_section(
            enriched_description,
            section_name,
            section_info,
            current_bar_count,
            last_section_summary # Pass summary of previous section
        )
        if symbolic_section: # Only append if generation succeeded
             # Ensure section ends with a newline for clean concatenation
             if not symbolic_section.endswith('\n'):
                 symbolic_section += '\n'
             all_symbolic_text += symbolic_section
             current_bar_count += section_info['bars'] # Advance bar count for next section
        else:
             print(f"Skipping section {section_name} due to generation failure.")
             # Decide if you want to stop or continue without the failed section
             # break # Uncomment to stop the whole process if one section fails

    if not all_symbolic_text.strip():
         print("ERROR: No symbolic text was generated. Cannot proceed.")
         exit()

    print("\n--- Combined Symbolic Text ---")
    print(all_symbolic_text[:1000] + "...") # Print beginning of combined text
    print("----------------------------\n")

    # Step 3
    parsed_notes, tempo_changes, time_sig_changes, key_sig_changes = parse_symbolic_to_structured_data(all_symbolic_text)

    # Step 4
    if parsed_notes: # Only create MIDI if parsing yielded notes
        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"generated_music_{timestamp}.mid"
        create_midi_file(parsed_notes, tempo_changes, time_sig_changes, key_sig_changes, output_filename)
    else:
        print("No notes were successfully parsed. MIDI file not created.")

    print("\n--- Pipeline Finished ---")