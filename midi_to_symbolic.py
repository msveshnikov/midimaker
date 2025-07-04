# -*- coding: utf-8 -*-
"""
Converts a MIDI file into the custom Compact Symbolic Format text.
This is the reverse process of symbolic_parser.py and midi_generator.py.
"""

import pretty_midi
import os
from collections import defaultdict
import traceback

# Import from local modules
import config
import music_defs # Assuming this contains PITCH_MAP, DRUM_PITCH_MAP, DURATION_RELATIVE_MAP, etc.

# Reverse maps for conversion (initialized below)
_MIDI_TO_PITCH_CLASS_NAME = {v: k for k, v in music_defs.PITCH_MAP.items()}
_DRUM_PITCH_TO_NAME = {v: k for k, v in music_defs.DRUM_PITCH_MAP.items()}
_PROGRAM_TO_INSTRUMENT_NAME = {v: k for k, v in music_defs.INSTRUMENT_PROGRAM_MAP.items()}

# Helper function to get the effective relative quarter duration from the map
def _get_relative_quarter_duration(symbol):
    """Helper to get the relative quarter duration for a symbolic note/rest symbol."""
    symbol = symbol.strip().upper()
    if not symbol: return None

    is_dotted = symbol.endswith(".")
    base_symbol = symbol.replace(".", "")

    rel_dur = music_defs.DURATION_RELATIVE_MAP.get(base_symbol)
    if rel_dur is None:
        # print(f"Warning: Unknown symbolic duration base '{base_symbol}'.") # Avoid spamming warnings during map creation
        return None

    if is_dotted:
        rel_dur *= 1.5

    return rel_dur

# Create and sort symbolic durations by their relative quarter duration, descending
_all_symbol_relative_quarters = []
# Add standard notes/rests
for symbol in set(music_defs.DURATION_RELATIVE_MAP.keys()):
    rel_q = _get_relative_quarter_duration(symbol)
    if rel_q is not None and rel_q > 0: # Only include positive durations
        _all_symbol_relative_quarters.append((rel_q, symbol))
# Add dotted versions
for symbol in set(music_defs.DURATION_RELATIVE_MAP.keys()):
    if symbol + "." not in music_defs.DURATION_RELATIVE_MAP: # Avoid double adding if map already has dotted
        rel_q_dotted = _get_relative_quarter_duration(symbol + ".")
        if rel_q_dotted is not None and rel_q_dotted > 0:
             _all_symbol_relative_quarters.append((rel_q_dotted, symbol + "."))


# Sort and remove duplicates based on relative quarter duration, keeping the first encounter.
_SORTED_SYMBOLIC_DURATIONS = sorted(
    _all_symbol_relative_quarters,
    key=lambda item: item[0],
    reverse=True
)
_SORTED_SYMBOLIC_DURATIONS_UNIQUE = []
seen_rel_quarters = set()
for rel_q, symbol in _SORTED_SYMBOLIC_DURATIONS:
    # Use a small tolerance for floating point comparison of relative quarters
    is_seen = False
    for seen_q in seen_rel_quarters:
        if abs(rel_q - seen_q) < 1e-6:
            is_seen = True
            break
    if not is_seen:
        _SORTED_SYMBOLIC_DURATIONS_UNIQUE.append((rel_q, symbol))
        seen_rel_quarters.add(rel_q)

_SORTED_SYMBOLIC_DURATIONS = _SORTED_SYMBOLIC_DURATIONS_UNIQUE


def midi_pitch_to_symbolic(midi_pitch, is_drum):
    """Converts MIDI pitch number to symbolic pitch name (e.g., C4) or drum name."""
    if is_drum:
        return _DRUM_PITCH_TO_NAME.get(midi_pitch, f"Drum{midi_pitch}")

    if not (0 <= midi_pitch <= 127):
        return None

    pitch_class = midi_pitch % 12
    octave = (midi_pitch // 12) - 1  # C4 is MIDI 60, octave 4 in symbolic, so C0 is MIDI 12, octave 0
    pitch_name = _MIDI_TO_PITCH_CLASS_NAME.get(pitch_class)

    if pitch_name is None:
        # Should not happen with standard MIDI pitches 0-127 and PITCH_MAP
        return None

    return f"{pitch_name}{octave}"


def seconds_to_symbolic_duration_sequence(duration_sec, start_time, tempo_map, ts_map, tolerance=0.005):
    """
    Converts a duration in seconds to a sequence of symbolic duration strings (e.g., ['Q', 'E.']).
    Uses the tempo/TS at the start_time for conversion.
    """
    if duration_sec <= tolerance:
        return []

    # Get tempo at the start of the duration
    tempo = get_active_state(start_time, tempo_map, ts_map, {})[0]
    if tempo <= 0:
        print(f"Warning: Tempo is zero or negative ({tempo}) at time {start_time:.3f}s. Cannot convert duration {duration_sec:.3f}s to symbolic.")
        return []

    # Calculate seconds per quarter note at this tempo
    sec_per_quarter = 60.0 / tempo

    # Convert total duration in seconds to total duration in relative quarter notes
    # Avoid division by zero sec_per_quarter
    total_relative_quarters = duration_sec / sec_per_quarter if sec_per_quarter > 1e-9 else 0

    remaining_relative_quarters = total_relative_quarters
    symbolic_sequence = []

    # Use sorted durations (longest first) to greedily decompose
    for rel_quarters_value, symbol in _SORTED_SYMBOLIC_DURATIONS:
        if rel_quarters_value > 1e-9: # Avoid division by zero/near zero
            # How many times does this duration symbol fit into the remaining duration?
            # Use tolerance when checking if it fits
            while remaining_relative_quarters >= rel_quarters_value * (1.0 - tolerance / rel_quarters_value if rel_quarters_value > 1e-9 else 1.0): # avoid div by zero for tolerance
                 # Check if remaining is very close to this duration
                 if abs(remaining_relative_quarters - rel_quarters_value) < tolerance * rel_quarters_value:
                      symbolic_sequence.append(symbol)
                      remaining_relative_quarters = 0 # Consumed the remaining duration
                      break # Found the best fit for the remainder

                 # If not a close fit, but larger, take the symbol
                 if remaining_relative_quarters >= rel_quarters_value * (1.0 + tolerance / rel_quarters_value if rel_quarters_value > 1e-9 else 1.0): # avoid div by zero for tolerance
                      symbolic_sequence.append(symbol)
                      remaining_relative_quarters -= rel_quarters_value
                 else:
                      # Remaining is smaller than current symbol, try the next one
                      break
        if remaining_relative_quarters < tolerance * total_relative_quarters and remaining_relative_quarters < tolerance: # Break if remainder is negligible
            break
        # If loop finishes and remaining_relative_quarters is still > tolerance, it couldn't be perfectly decomposed

    # If there's still a significant duration left
    if remaining_relative_quarters > tolerance * total_relative_quarters and remaining_relative_quarters > tolerance:
         # Remaining duration is significant relative to total, or significant in absolute terms
         # print(f"Warning: Could not perfectly convert duration {duration_sec:.3f}s (={total_relative_quarters:.3f} relative quarters) at time {start_time:.3f}s into standard symbolic notes. Remaining: {remaining_relative_quarters:.3f} relative quarters. Approximation used.")
         # The greedy approach provides the best standard approximation.
         pass # The sequence generated is the greedy best fit.

    # Handle cases where duration is very short but > tolerance, might not fit any standard symbol
    # If sequence is empty but duration was > tolerance, maybe add the smallest unit?
    # Smallest unit is 'T' (Thirty-second)
    smallest_rel_quarters = _get_relative_quarter_duration('T')
    if not symbolic_sequence and duration_sec > tolerance and smallest_rel_quarters is not None:
         # If duration is at least half of the smallest unit, represent it with the smallest unit
         if total_relative_quarters >= smallest_rel_quarters / 2.0 - 1e-9:
              symbolic_sequence.append('T')
              # print(f"Info: Representing small duration {duration_sec:.3f}s (~{total_relative_quarters:.3f} rel qtr) as 'T' at time {start_time:.3f}s.")


    return symbolic_sequence


def get_active_state(time, tempo_changes, ts_changes, key_changes):
    """Helper to find the active tempo, TS, and key signature at a given time."""
    # Assumes change lists are sorted by time

    def find_active(time, changes, default_value):
        active_value = default_value
        for change_time, *value in changes:
            if change_time <= time + 1e-6: # Use tolerance for time comparison
                active_value = value[0] if len(value) == 1 else tuple(value)
            else:
                break
        return active_value

    active_tempo = find_active(time, tempo_changes, config.CONFIG["default_tempo"])
    active_ts = find_active(time, ts_changes, config.CONFIG["default_timesig"])
    active_key = find_active(time, key_changes, config.CONFIG["default_key"])

    return active_tempo, active_ts, active_key


def midi_to_symbolic(midi_file_path):
    """
    Converts a MIDI file into the custom Compact Symbolic Format text.
    """
    print(f"\n--- Converting MIDI File to Symbolic Format ({midi_file_path}) ---")

    if not os.path.exists(midi_file_path):
        print(f"Error: MIDI file not found at {midi_file_path}")
        return None

    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file_path)
        print("MIDI file loaded successfully.")
    except Exception as e:
        print(f"Error loading MIDI file: {e}")
        traceback.print_exc()
        return None

    symbolic_lines = []
    time_tolerance = 1e-6 # Use a consistent time tolerance

    # 1. Extract and process global meta-messages
    # --- FIXED LINE BELOW ---
    tempo_change_times, tempo_qpm_values = midi_data.get_tempo_changes()
    tempo_changes = list(zip(tempo_change_times, tempo_qpm_values))
    # --- END OF FIX ---
    
    ts_changes = [(c.time, c.numerator, c.denominator) for c in midi_data.time_signature_changes]
    key_changes = [(c.time, pretty_midi.key_number_to_key_name(c.key_number)) for c in midi_data.key_signature_changes]

    # Ensure time 0 has state if not present
    if not tempo_changes or tempo_changes[0][0] > time_tolerance:
        tempo_changes.insert(0, (0.0, config.CONFIG["default_tempo"]))
    if not ts_changes or ts_changes[0][0] > time_tolerance:
        ts_changes.insert(0, (0.0, *config.CONFIG["default_timesig"]))
    if not key_changes or key_changes[0][0] > time_tolerance:
        key_changes.insert(0, (0.0, config.CONFIG["default_key"]))

    # Sort changes just in case (pretty_midi usually sorts them, but good practice)
    tempo_changes.sort(key=lambda x: x[0])
    ts_changes.sort(key=lambda x: x[0])
    key_changes.sort(key=lambda x: x[0])

    # 2. Collect all unique significant time points
    time_points = set([0.0]) # Always start at time 0
    # Add times from meta-messages
    for time, _ in tempo_changes: time_points.add(time)
    for time, _, _ in ts_changes: time_points.add(time)
    for time, _ in key_changes: time_points.add(time)

    note_events = []
    inst_map = {} # Map pretty_midi.Instrument index to (symbolic_inst_name, symbolic_track_id, is_drum_track)
    inst_initial_symbolic_lines = [] # Collect initial INST lines

    for inst_idx, instrument in enumerate(midi_data.instruments):
        # Determine symbolic instrument name
        symbolic_inst_name = instrument.name.strip() if instrument.name and instrument.name.lower() != 'untitled' else None
        if not symbolic_inst_name:
             symbolic_inst_name = _PROGRAM_TO_INSTRUMENT_NAME.get(instrument.program, f"Program{instrument.program}")

        is_drum_track = instrument.is_drum # pretty_midi.Instrument.is_drum is a boolean
        # GM Percussion Kit is MIDI channel 9 (0-indexed), not program 128. 
        # However, some MIDI files might assign program 128 to a drum track.
        # pretty_midi's is_drum usually handles this by checking the channel.
        # Keeping the check for instrument.program == 128 can be a fallback,
        # but is_drum should be the primary indicator.

        # Determine symbolic track ID - make it unique and somewhat descriptive
        # Combine cleaned instrument name and index
        cleaned_inst_name = "".join(c for c in symbolic_inst_name if c.isalnum() or c in ('_', '-')).strip()
        if not cleaned_inst_name: cleaned_inst_name = "Track"
        symbolic_track_id = f"{cleaned_inst_name}_{inst_idx}"

        inst_map[inst_idx] = (symbolic_inst_name, symbolic_track_id, is_drum_track)
        print(f"Mapped MIDI Instrument {inst_idx} ('{instrument.name}', Program {instrument.program}, IsDrum {instrument.is_drum}) to Symbolic: INST='{symbolic_inst_name}', TrackID='{symbolic_track_id}', IsDrumTrack={is_drum_track}")

        # Add INST command for non-drum tracks before the first BAR marker, only add if not already added
        # Check against existing INST lines by parsing them
        existing_inst_names = []
        for line in inst_initial_symbolic_lines:
            if line.startswith('INST:'):
                try:
                    existing_inst_names.append(line.split(':')[1])
                except IndexError:
                    pass # Malformed line, ignore

        if not is_drum_track and symbolic_inst_name not in existing_inst_names:
            inst_initial_symbolic_lines.append(f"INST:{symbolic_inst_name}")


        for note in instrument.notes:
            time_points.add(note.start)
            time_points.add(note.end) # Note ends are important for calculating rests
            note_events.append((note.start, note.end, note.pitch, note.velocity, inst_idx))

    sorted_time_points = sorted(list(time_points))
    # Remove time points that are extremely close to each other
    unique_time_points = []
    if sorted_time_points:
         unique_time_points.append(sorted_time_points[0])
         for t in sorted_time_points[1:]:
              if t - unique_time_points[-1] > time_tolerance: # Use tolerance
                   unique_time_points.append(t)
    sorted_time_points = unique_time_points


    # 3. Initialize state and output initial lines
    global_time = 0.0
    current_bar = 1
    bar_start_time = 0.0
    current_tempo, (current_ts_num, current_ts_den), current_key = get_active_state(0.0, tempo_changes, ts_changes, key_changes)

    # Add initial global parameters
    symbolic_lines.append(f"T:{current_tempo:.2f}")
    symbolic_lines.append(f"TS:{current_ts_num}/{current_ts_den}")
    # Validate key name before adding initial K
    if music_defs.KEY_SIGNATURE_PATTERN.match(current_key):
        symbolic_lines.append(f"K:{current_key}")
    else:
        print(f"Warning: Invalid default key name '{current_key}'. Skipping initial K command.")
        current_key = config.CONFIG["default_key"] # Fallback state

    # Add initial INST lines *after* global parameters but *before* the first BAR
    symbolic_lines.extend(inst_initial_symbolic_lines)

    symbolic_lines.append(f"BAR:{current_bar}")

    # Track the current time position for each symbolic track
    track_current_time = {inst_idx: 0.0 for inst_idx in inst_map.keys()}
    # Store notes grouped by instrument index and sorted by start time
    notes_by_inst = defaultdict(list)
    for note_event in note_events:
         notes_by_inst[note_event[4]].append(note_event)
    for inst_idx in notes_by_inst:
         notes_by_inst[inst_idx].sort(key=lambda x: x[0]) # Sort by start time

    # Keep track of notes already processed at a time point for each track
    track_notes_processed_indices = {inst_idx: 0 for inst_idx in inst_map.keys()}


    # 4. Iterate through time points and generate symbolic events
    print(f"Processing {len(sorted_time_points)} unique time points...")
    for time_point in sorted_time_points:
        if time_point < global_time - time_tolerance:
            # Should not happen with sorted_time_points
            print(f"Warning: Time point {time_point:.3f} is earlier than current global time {global_time:.3f}. Skipping.")
            continue

        # Handle time elapsed since last event (potential rests and bar crossings)
        time_elapsed_segment = time_point - global_time

        if time_elapsed_segment > time_tolerance:
            # Process bar crossings within this time segment
            time_to_process = time_elapsed_segment
            current_segment_start_time = global_time

            while time_to_process > time_tolerance:
                 temp_tempo, (temp_ts_num, temp_ts_den), _ = get_active_state(current_segment_start_time, tempo_changes, ts_changes, key_changes)
                 beats_per_bar_at_current_time = temp_ts_num * (4.0 / temp_ts_den) if temp_ts_den > 0 else 4.0
                 sec_per_beat_at_current_time = 60.0 / temp_tempo if temp_tempo > 0 else 0.5 # Fallback to 120 BPM quarter
                 sec_per_bar_at_current_time = beats_per_bar_at_current_time * sec_per_beat_at_current_time

                 if sec_per_bar_at_current_time <= time_tolerance: # Avoid infinite loop with zero bar duration
                     print(f"Warning: Zero or near-zero bar duration calculated at time {current_segment_start_time:.3f}s. Cannot advance time reliably by bars.")
                     break 

                 time_into_current_bar = current_segment_start_time - bar_start_time
                 time_to_next_bar_line = sec_per_bar_at_current_time - (time_into_current_bar % sec_per_bar_at_current_time if sec_per_bar_at_current_time > 1e-9 else 0)
                 if abs(time_to_next_bar_line - sec_per_bar_at_current_time) < time_tolerance or time_to_next_bar_line < time_tolerance : # If on the bar line or very close
                     time_to_next_bar_line = sec_per_bar_at_current_time


                 advance_time = min(time_to_process, time_to_next_bar_line)

                 if advance_time >= time_to_next_bar_line - time_tolerance and time_to_next_bar_line > time_tolerance :
                      current_segment_start_time += time_to_next_bar_line
                      time_to_process -= time_to_next_bar_line
                      current_bar += 1
                      bar_start_time = current_segment_start_time
                      symbolic_lines.append(f"BAR:{current_bar}")
                 else:
                      current_segment_start_time += time_to_process
                      time_to_process = 0 

            global_time = time_point

        # Update global state if meta-messages occur exactly at this time point
        temp_tempo, (temp_ts_num, temp_ts_den), temp_key = get_active_state(global_time, tempo_changes, ts_changes, key_changes)

        if abs(temp_tempo - current_tempo) > 1e-3: 
            symbolic_lines.append(f"T:{temp_tempo:.2f}")
            current_tempo = temp_tempo

        if (temp_ts_num, temp_ts_den) != (current_ts_num, current_ts_den):
             symbolic_lines.append(f"TS:{temp_ts_num}/{temp_ts_den}")
             current_ts_num, current_ts_den = temp_ts_num, temp_ts_den

        if temp_key != current_key:
            if music_defs.KEY_SIGNATURE_PATTERN.match(temp_key):
                symbolic_lines.append(f"K:{temp_key}")
                current_key = temp_key
            else:
                 print(f"Warning: Invalid key name '{temp_key}' found at time {global_time:.3f}s. Ignoring key change.")


        for inst_idx, (sym_inst_name, sym_track_id, is_drum_track) in inst_map.items():
            track_notes = notes_by_inst.get(inst_idx, [])
            notes_starting_now = []
            start_idx = track_notes_processed_indices[inst_idx]
            notes_to_check = track_notes[start_idx:]

            for j, note_event in enumerate(notes_to_check):
                 note_start, note_end, pitch, vel, _ = note_event
                 if abs(note_start - global_time) < time_tolerance:
                      notes_starting_now.append(note_event)
                 elif note_start > global_time + time_tolerance:
                      break

            if notes_starting_now:
                track_notes_processed_indices[inst_idx] += len(notes_starting_now)

                rest_duration_sec = global_time - track_current_time[inst_idx]
                if rest_duration_sec > time_tolerance: 
                     rest_symbols = seconds_to_symbolic_duration_sequence(rest_duration_sec, track_current_time[inst_idx], tempo_changes, ts_changes)
                     if rest_symbols:
                         symbolic_lines.append(f"R:{sym_track_id}:{' '.join(rest_symbols)}")
                     elif rest_duration_sec > 0.05: 
                          print(f"Warning: Could not symbolize rest of {rest_duration_sec:.3f}s for track '{sym_track_id}' starting at time {track_current_time[inst_idx]:.3f}s and ending at {global_time:.3f}s.")
                
                note_groups = defaultdict(list)
                for note_start, note_end, pitch, vel, _ in notes_starting_now:
                     group_key = (round(note_start, 6), round(note_end, 6), int(round(vel)))
                     note_groups[group_key].append(pitch) 

                event_end_time_for_this_track = track_current_time[inst_idx] 

                for (event_start_time, event_end_time, event_velocity), pitches in sorted(note_groups.items()):
                    valid_pitches = []
                    pitches.sort()
                    for pitch in pitches:
                        symbolic_pitch = midi_pitch_to_symbolic(pitch, is_drum_track)
                        if symbolic_pitch:
                            valid_pitches.append(symbolic_pitch)
                        else:
                            print(f"Warning: Could not convert pitch {pitch} to symbolic for track '{sym_track_id}' in group starting at time {event_start_time:.3f}s. Skipping pitch.")

                    if valid_pitches:
                        event_duration_sec = event_end_time - event_start_time

                        if event_duration_sec > time_tolerance:
                            duration_symbols = seconds_to_symbolic_duration_sequence(event_duration_sec, event_start_time, tempo_changes, ts_changes)

                            if not duration_symbols and event_duration_sec > 0.01: # Only warn if duration symbols are empty for a somewhat significant duration
                                 print(f"Warning: Could not symbolize duration {event_duration_sec:.3f}s for note/chord at time {event_start_time:.3f}s. Using 'T' as fallback if very short, or it might be missed.")
                                 # Attempt to use smallest if very short and not covered by main logic
                                 if event_duration_sec < _get_relative_quarter_duration('T') * (60.0/current_tempo) * 2: # if duration is less than 2x a thirty-second note
                                     duration_symbols = ["T"]
                                 else: # Otherwise, if still no symbols for a longer duration, it's problematic
                                     duration_symbols = ["Q"] # A more noticeable fallback for longer un-symbolized durations

                            if not duration_symbols and event_duration_sec > time_tolerance: # If still no symbols after fallback, might be an issue or very short
                                 # For very short notes that couldn't be symbolized, and if no fallback worked,
                                 # we might skip them or use a minimal representation if essential.
                                 # Current logic in seconds_to_symbolic_duration_sequence tries to handle this with 'T'
                                 # If it still comes back empty here, it means it was too short even for 'T' or some other issue.
                                 # print(f"Info: Note/chord at {event_start_time:.3f}s with duration {event_duration_sec:.3f}s was too short to symbolize or resulted in no symbols.")
                                 pass # Skip adding if no duration symbols and it's truly negligible or unrepresentable

                            if duration_symbols: # Only proceed if we have duration symbols
                                safe_velocity = max(1, min(127, int(event_velocity)))
                                if len(valid_pitches) > 1:
                                    symbolic_lines.append(f"C:{sym_track_id}:[{','.join(valid_pitches)}]:{' '.join(duration_symbols)}:{safe_velocity}")
                                else:
                                    symbolic_lines.append(f"N:{sym_track_id}:{valid_pitches[0]}:{' '.join(duration_symbols)}:{safe_velocity}")
                                event_end_time_for_this_track = max(event_end_time_for_this_track, event_end_time)
                
                track_current_time[inst_idx] = event_end_time_for_this_track


    last_event_time = 0.0
    if note_events:
        last_event_time = max(note[1] for note in note_events) # Max end time of notes

    all_meta_times = []
    if tempo_changes: all_meta_times.extend([t[0] for t in tempo_changes])
    if ts_changes: all_meta_times.extend([t[0] for t in ts_changes])
    if key_changes: all_meta_times.extend([t[0] for t in key_changes])
    if all_meta_times:
        last_event_time = max(last_event_time, max(all_meta_times))
    
    if sorted_time_points: # The last processed time_point in the loop
        last_event_time = max(last_event_time, sorted_time_points[-1])


    time_to_process_after_loop = last_event_time - global_time
    current_segment_start_time_final = global_time # Use a different var name to avoid confusion with loop var

    while time_to_process_after_loop > time_tolerance:
        temp_tempo, (temp_ts_num, temp_ts_den), _ = get_active_state(current_segment_start_time_final, tempo_changes, ts_changes, key_changes)
        beats_per_bar_at_current_time = temp_ts_num * (4.0 / temp_ts_den) if temp_ts_den > 0 else 4.0
        sec_per_beat_at_current_time = 60.0 / temp_tempo if temp_tempo > 0 else 0.5
        sec_per_bar_at_current_time = beats_per_bar_at_current_time * sec_per_beat_at_current_time

        if sec_per_bar_at_current_time <= time_tolerance:
            print(f"Warning: Zero or near-zero bar duration calculated at time {current_segment_start_time_final:.3f}s during final bar processing.")
            break

        time_into_current_bar = current_segment_start_time_final - bar_start_time
        time_to_next_bar_line = sec_per_bar_at_current_time - (time_into_current_bar % sec_per_bar_at_current_time if sec_per_bar_at_current_time > 1e-9 else 0)
        if abs(time_to_next_bar_line - sec_per_bar_at_current_time) < time_tolerance or time_to_next_bar_line < time_tolerance :
             time_to_next_bar_line = sec_per_bar_at_current_time
        
        advance_time = min(time_to_process_after_loop, time_to_next_bar_line)

        if advance_time >= time_to_next_bar_line - time_tolerance and time_to_next_bar_line > time_tolerance:
             current_segment_start_time_final += time_to_next_bar_line
             time_to_process_after_loop -= time_to_next_bar_line
             if current_segment_start_time_final <= last_event_time + time_tolerance : # Only add bar if it's within or at the end of the piece
                current_bar += 1
                bar_start_time = current_segment_start_time_final # bar_start_time should update to the time of the new bar
                symbolic_lines.append(f"BAR:{current_bar}")
        else:
             current_segment_start_time_final += time_to_process_after_loop
             time_to_process_after_loop = 0

    for inst_idx, (sym_inst_name, sym_track_id, is_drum_track) in inst_map.items():
        rest_duration_sec = last_event_time - track_current_time[inst_idx]
        if rest_duration_sec > time_tolerance:
            rest_symbols = seconds_to_symbolic_duration_sequence(rest_duration_sec, track_current_time[inst_idx], tempo_changes, ts_changes)
            if rest_symbols:
                symbolic_lines.append(f"R:{sym_track_id}:{' '.join(rest_symbols)}")
            elif rest_duration_sec > 0.05: 
                 print(f"Warning: Could not symbolize final rest of {rest_duration_sec:.3f}s for track '{sym_track_id}' ending at time {track_current_time[inst_idx]:.3f}s.")


    print("MIDI to Symbolic conversion complete.")
    return "\n".join(symbolic_lines)

if __name__ == "__main__":
    print("converting midi to symbolic format")
    midi_file_path = "output/generated_music_20250506_180304.mid"  # Replace with your MIDI file path
    symbolic_text = midi_to_symbolic(midi_file_path)
    if symbolic_text:
        print("Symbolic text generated successfully.")
        print(symbolic_text)
        output_file_path = "output/output_symbolic.txt"
        try:
            with open(output_file_path, "w") as f:
                 f.write(symbolic_text)
            print(f"Symbolic text saved to {output_file_path}")
        except IOError as e:
            print(f"Error saving symbolic text to file: {e}")
    else:
        print("Failed to generate symbolic text.")