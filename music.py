import google.generativeai as genai
import pretty_midi
import re
import os
import time # For potential rate limiting

# --- Configuration ---
API_KEY = "AIzaSyC5jbwgP050qfurqK9GyvgrUYvpwEy0n8s" # IMPORTANT: Use environment variables or secrets management in real apps!
GEMINI_MODEL = 'gemini-2.0-pro-exp-02-05'  #'gemini-2.0-flash-001'
INITIAL_DESCRIPTION = "A simple, slightly melancholic loopable piano piece suitable for a game background."
SECTIONS = {
    "A1": {"bars": 8, "goal": "Establish the main melancholic theme in C minor. Simple texture."},
    "B": {"bars": 8, "goal": "Introduce a slight variation or counter-melody, perhaps brightening slightly towards Eb major."},
    "A2": {"bars": 8, "goal": "Return to the main theme, similar to A1 but maybe with slight embellishment."},
}
OUTPUT_FILENAME = "generated_music.mid"
DEFAULT_TEMPO = 90
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
DurationSymbols: W (Whole), H (Half), Q (Quarter), E (Eighth), S (Sixteenth).
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
            elif response.prompt_feedback.block_reason:
                 print(f"Prompt blocked: {response.prompt_feedback.block_reason}")
                 return None # Indicate blockage
            else:
                 print("Warning: Received empty response from Gemini.")
                 return None # Indicate empty response

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
}
# Standard drum note map (MIDI channel 10) - very basic example
DRUM_PITCH_MAP = {
    "Kick": 36, "BD": 36,
    "Snare": 38, "SD": 38,
    "HiHatClosed": 42, "HHC": 42,
    "HiHatOpen": 46, "HHO": 46,
    "Crash": 49, "CR": 49
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
    
    return base_midi + acc_val + (octave + 1) * 12

def duration_to_seconds(duration_symbol, tempo, time_sig_denominator):
    """Converts duration symbol (W, H, Q, E, S) to seconds."""
    try:
        beats_per_minute = float(tempo)
        beats_per_second = beats_per_minute / 60.0
        
        # Duration relative to a whole note
        duration_map = {'W': 4.0, 'H': 2.0, 'Q': 1.0, 'E': 0.5, 'S': 0.25}
        
        # Calculate duration of a quarter note in seconds
        # Assumes the denominator of time signature represents the beat unit (e.g., 4/4 means quarter note is the beat)
        # This simplification works for common time signatures but might need adjustment for complex ones.
        quarter_note_duration_sec = 60.0 / beats_per_minute
        
        relative_duration = duration_map.get(duration_symbol.upper())
        if relative_duration is None:
             print(f"Warning: Unknown duration symbol: {duration_symbol}. Defaulting to Quarter.")
             relative_duration = 1.0 # Default to Quarter Note

        # Calculate actual duration in seconds based on quarter note = 1.0 relative duration
        return relative_duration * quarter_note_duration_sec

    except Exception as e:
        print(f"Error calculating duration: {e}. Using default.")
        return 0.5 # Default duration if calculation fails


# --- Main Pipeline Functions ---

def enrich_music_description(description):
    """Step 1: Use LLM to enrich the initial music description."""
    print("--- Step 1: Enriching Description ---")
    prompt = f"""
    Take the following basic music description and enrich it with more detail for a generative music task. Suggest specific musical elements like key signature, tempo range, instrumentation (if not specified), mood nuances, and potential structure (e.g., AABA, verse-chorus).

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

def generate_symbolic_section(overall_desc, section_name, section_info, current_bar):
    """Step 2: Generate symbolic music for one section using LLM."""
    print(f"--- Step 2: Generating Symbolic Section {section_name} ---")
    bars = section_info['bars']
    goal = section_info['goal']
    
    prompt = f"""
    You are a symbolic music generator. Your task is to generate ONLY the symbolic music notation for a specific section of a piece.

    Overall Music Goal: {overall_desc}

    Current Section: {section_name}
    Target Bars: {bars} (Start this section at BAR:{current_bar})
    Section Goal: {goal}

    Instructions:
    1. Generate music ONLY for this section, starting exactly at BAR:{current_bar} and continuing for {bars} bars.
    2. Strictly adhere to the compact symbolic format defined below. Do NOT include any other text, explanations, or apologies.
    3. If tempo (T), time signature (TS), key (K), or instrument (INST) need to be set or changed at the start of this section, include those commands before BAR:{current_bar}. Assume defaults T:{DEFAULT_TEMPO}, TS:{DEFAULT_TIMESIG[0]}/{DEFAULT_TIMESIG[1]}, K:{DEFAULT_KEY} if not specified.
    4. Ensure musical coherence within the section and try to follow the Section Goal.

    {SYMBOLIC_FORMAT_DEFINITION}

    Generate Section {section_name} symbolic music now:
    """
    
    symbolic_text = call_gemini(prompt)
    if symbolic_text:
         print(f"Generated symbolic text for Section {section_name}:\n{symbolic_text[:300]}...\n") # Print beginning
    else:
         print(f"Failed to generate symbolic text for Section {section_name}.")
    return symbolic_text or "" # Return empty string on failure

def parse_symbolic_to_structured_data(symbolic_text):
    """Step 3: Parse concatenated symbolic text into structured data for MIDI."""
    print("--- Step 3: Parsing Symbolic Text ---")
    notes_by_instrument_track = {} # { ('Pno', 'RH'): [note_obj1, ...], ('Bass', 'Main'): [...] }
    tempo_changes = [] # Store tempo changes with time: [(time, tempo), ...]
    time_signature_changes = [] # Store time sig changes with time: [(time, num, den), ...]

    current_time = 0.0
    current_tempo = float(DEFAULT_TEMPO)
    current_ts_num, current_ts_den = DEFAULT_TIMESIG
    current_key = DEFAULT_KEY
    current_bar_start_time = 0.0
    last_bar_number = 0
    beat_duration = 60.0 / current_tempo * (4.0 / current_ts_den) # Duration of one beat (defined by denominator)

    # Ensure initial tempo/TS are set at time 0
    tempo_changes.append((0.0, current_tempo))
    time_signature_changes.append((0.0, current_ts_num, current_ts_den))

    lines = symbolic_text.strip().split('\n')
    
    active_instrument = "Pno" # Default instrument

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'): # Skip empty lines and comments
            continue

        parts = line.split(':', 1)
        command = parts[0].upper()
        value = parts[1] if len(parts) > 1 else ""

        try:
            if command == "INST":
                 active_instrument = value.strip()
                 print(f"Time {current_time:.2f}: Instrument set to {active_instrument}")

            elif command == "T":
                new_tempo = float(value)
                if new_tempo != current_tempo:
                    print(f"Time {current_time:.2f}: Tempo change to {new_tempo} BPM")
                    current_tempo = new_tempo
                    tempo_changes.append((current_time, current_tempo))
                    beat_duration = 60.0 / current_tempo * (4.0 / current_ts_den)
            
            elif command == "TS":
                 num_str, den_str = value.split('/')
                 new_ts_num, new_ts_den = int(num_str), int(den_str)
                 if (new_ts_num, new_ts_den) != (current_ts_num, current_ts_den):
                      print(f"Time {current_time:.2f}: Time Signature change to {new_ts_num}/{new_ts_den}")
                      current_ts_num, current_ts_den = new_ts_num, new_ts_den
                      time_signature_changes.append((current_time, current_ts_num, current_ts_den))
                      beat_duration = 60.0 / current_tempo * (4.0 / current_ts_den) # Recalculate beat duration

            elif command == "K":
                 current_key = value.strip()
                 print(f"Time {current_time:.2f}: Key set to {current_key}") # Info only for now

            elif command == "BAR":
                bar_number = int(value)
                # Estimate bar start time based on previous bar number and time sig
                # This is approximate, actual time advances with notes/rests
                if bar_number > last_bar_number:
                     # Assume regular bars passed if there's a jump
                     bars_passed = bar_number - last_bar_number -1
                     current_time += bars_passed * current_ts_num * beat_duration

                     current_bar_start_time = current_time # Rough start time
                     print(f"Time {current_time:.2f}: Reached BAR {bar_number}")
                     last_bar_number = bar_number


            elif command in ["N", "C", "R"]:
                 data_parts = value.split(':')
                 track_id = data_parts[0].strip()
                 
                 # Determine instrument - drums handled differently
                 is_drum_track = active_instrument.lower() in ["drs", "drums"] or track_id.lower() in ["drs", "drums"]
                 instrument_key = "Drs" if is_drum_track else active_instrument

                 # Instrument + Track combination key
                 inst_track_key = (instrument_key, track_id)
                 if inst_track_key not in notes_by_instrument_track:
                      notes_by_instrument_track[inst_track_key] = []

                 if command == "N": # Note:Track:Pitch:Duration:Velocity
                      pitch_name = data_parts[1].strip()
                      duration_sym = data_parts[2].strip()
                      velocity = int(data_parts[3].strip())
                      
                      duration_sec = duration_to_seconds(duration_sym, current_tempo, current_ts_den)
                      
                      # Drum mapping or regular pitch mapping
                      if is_drum_track:
                           midi_pitch = DRUM_PITCH_MAP.get(pitch_name, 36) # Default to Kick if unknown
                      else:
                           midi_pitch = pitch_to_midi(pitch_name)

                      note_event = {
                           "pitch": midi_pitch,
                           "start": current_time,
                           "end": current_time + duration_sec,
                           "velocity": velocity
                      }
                      notes_by_instrument_track[inst_track_key].append(note_event)
                      current_time += duration_sec # Advance time based on this note

                 elif command == "C": # Chord:Track:[P1,P2,...]:Duration:Velocity
                      pitches_str = data_parts[1].strip()[1:-1] # Remove brackets
                      pitch_names = [p.strip() for p in pitches_str.split(',')]
                      duration_sym = data_parts[2].strip()
                      velocity = int(data_parts[3].strip())

                      duration_sec = duration_to_seconds(duration_sym, current_tempo, current_ts_den)

                      for pitch_name in pitch_names:
                            # Assume chords aren't typically on drum tracks
                            midi_pitch = pitch_to_midi(pitch_name)
                            note_event = {
                                "pitch": midi_pitch,
                                "start": current_time,
                                "end": current_time + duration_sec,
                                "velocity": velocity
                            }
                            notes_by_instrument_track[inst_track_key].append(note_event)
                      current_time += duration_sec # Advance time only ONCE per chord

                 elif command == "R": # Rest:Track:Duration
                      duration_sym = data_parts[1].strip()
                      duration_sec = duration_to_seconds(duration_sym, current_tempo, current_ts_den)
                      current_time += duration_sec # Advance time for the rest

        except Exception as e:
            print(f"Error parsing line: '{line}' - {e}")
            # Decide whether to skip line, stop parsing, etc.
            continue # Skip problematic line

    print("Symbolic text parsing complete.")
    return notes_by_instrument_track, tempo_changes, time_signature_changes


def create_midi_file(notes_data, tempo_changes, time_sig_changes, filename):
    """Step 4: Create MIDI file using pretty_midi."""
    print(f"--- Step 4: Creating MIDI File ({filename}) ---")
    midi_obj = pretty_midi.PrettyMIDI()

    # Apply tempo changes
    # pretty_midi handles tempo changes internally via its time_to_tick method based on the initial tempo
    # For accurate tempo changes *within* the file, we need to set the initial tempo correctly
    # and potentially manage tempo change events if the library fully supports writing them (it's complex).
    # Setting initial tempo:
    if tempo_changes:
         midi_obj.initial_tempo = tempo_changes[0][1] # Set initial tempo from first event
         print(f"Set initial tempo to: {midi_obj.initial_tempo}")
         # Note: pretty_midi primarily uses tempo for tick conversion, writing multiple tempo changes
         # might require digging into lower-level MIDI event creation if needed.

    # Apply time signature changes
    # Similar to tempo, pretty_midi uses this info but writing multiple changes needs care.
    if time_sig_changes:
        ts = time_sig_changes[0] # Use first time signature
        midi_obj.time_signature_changes.append(pretty_midi.TimeSignature(ts[1], ts[2], ts[0]))
        print(f"Set initial time signature to: {ts[1]}/{ts[2]}")


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
             instruments[pm_instrument_name] = pretty_midi.Instrument(program=program, is_drum=is_drum, name=pm_instrument_name)
             midi_obj.instruments.append(instruments[pm_instrument_name])
             print(f"Created instrument: {pm_instrument_name} (Program: {program}, IsDrum: {is_drum})")
        
        # Add notes to this instrument
        instrument_obj = instruments[pm_instrument_name]
        for note_info in notes:
             note = pretty_midi.Note(
                  velocity=note_info['velocity'],
                  pitch=note_info['pitch'],
                  start=note_info['start'],
                  end=note_info['end']
             )
             instrument_obj.notes.append(note)

    # Write the MIDI file
    try:
        midi_obj.write(filename)
        print(f"Successfully created MIDI file: {filename}")
    except Exception as e:
        print(f"Error writing MIDI file: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    
    # *** IMPORTANT: Set your API Key securely! ***
    # Example using environment variable:
    # API_KEY = os.getenv("GOOGLE_API_KEY") 
    # Or use Colab secrets: from google.colab import userdata; API_KEY = userdata.get('GOOGLE_API_KEY')
    
    if API_KEY == "YOUR_API_KEY" or not API_KEY:
         print("ERROR: Please set your GOOGLE_API_KEY in the script.")
         exit()

    configure_genai()

    # Step 1
    enriched_description = enrich_music_description(INITIAL_DESCRIPTION)
    if not enriched_description:
         enriched_description = INITIAL_DESCRIPTION # Fallback

    # Step 2 (Section Generation) & Concatenation
    all_symbolic_text = ""
    current_bar_count = 1
    for section_name, section_info in SECTIONS.items():
        symbolic_section = generate_symbolic_section(enriched_description, section_name, section_info, current_bar_count)
        if symbolic_section: # Only append if generation succeeded
             all_symbolic_text += symbolic_section + "\n" # Add newline separator
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
    parsed_notes, tempo_changes, time_sig_changes = parse_symbolic_to_structured_data(all_symbolic_text)

    # Step 4
    if parsed_notes: # Only create MIDI if parsing yielded notes
        create_midi_file(parsed_notes, tempo_changes, time_sig_changes, OUTPUT_FILENAME)
    else:
        print("No notes were successfully parsed. MIDI file not created.")

    print("\n--- Pipeline Finished ---")
