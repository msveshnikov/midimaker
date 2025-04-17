# -*- coding: utf-8 -*-
"""
Parses the symbolic music text format into structured data suitable for MIDI conversion.
Includes helper functions for pitch and duration conversion.
"""

import re
import math
import traceback

# Import from local modules
import config
import music_defs

# Caches for parsing results
_pitch_parse_cache = {}
_duration_cache = {}


def pitch_to_midi(pitch_name):
    """Converts pitch name (e.g., C4, F#5, Gb3) to MIDI number. Returns None if invalid."""
    pitch_name = pitch_name.strip()
    if pitch_name in _pitch_parse_cache:
        return _pitch_parse_cache[pitch_name]

    match = re.match(r"([A-G])([#sb]?)(\-?\d+)", pitch_name, re.IGNORECASE)
    if not match:
        _pitch_parse_cache[pitch_name] = None
        return None

    note, acc, oct_str = match.groups()
    try:
        octave = int(oct_str)
    except ValueError:
        _pitch_parse_cache[pitch_name] = None
        return None

    base_midi = music_defs.PITCH_MAP.get(note.upper())
    if base_midi is None:
        _pitch_parse_cache[pitch_name] = None
        return None

    acc_norm = acc.upper() if acc else ""
    if acc_norm == "S":
        acc_norm = "#"
    acc_val = music_defs.ACCIDENTAL_MAP.get(acc_norm, 0)
    # MIDI standard: Middle C (C4) is MIDI note 60. C0 is MIDI 12.
    # Formula: base + accidental + (octave + 1) * 12
    midi_val = base_midi + acc_val + (octave + 1) * 12

    if 0 <= midi_val <= 127:
        _pitch_parse_cache[pitch_name] = midi_val
        return midi_val
    else:
        _pitch_parse_cache[pitch_name] = None
        return None


def duration_to_seconds(duration_symbol, tempo, time_sig_denominator=4):
    """Converts duration symbol (W, H, Q, E, S, T, W., H., etc.) to seconds."""
    cache_key = (duration_symbol, tempo, time_sig_denominator)
    if cache_key in _duration_cache:
        return _duration_cache[cache_key]

    duration_symbol = duration_symbol.strip().upper()
    if not duration_symbol:
        print("Warning: Empty duration symbol. Defaulting to Quarter note duration.")
        duration_symbol = "Q"

    try:
        beats_per_minute = float(tempo)
        if beats_per_minute <= 0:
            print(f"Warning: Invalid tempo {tempo}. Using default {config.CONFIG['default_tempo']}.")
            beats_per_minute = float(config.CONFIG['default_tempo'])

        # Duration relative to a quarter note
        base_symbol = duration_symbol.replace(".", "")
        is_dotted = duration_symbol.endswith(".")

        relative_duration_quarters = music_defs.DURATION_RELATIVE_MAP.get(base_symbol)
        if relative_duration_quarters is None:
            print(
                f"Warning: Unknown duration symbol: '{duration_symbol}'. Defaulting to Quarter (1.0)."
            )
            relative_duration_quarters = 1.0

        if is_dotted:
            relative_duration_quarters *= 1.5

        # Calculate seconds per quarter note
        quarter_note_duration_sec = 60.0 / beats_per_minute

        # Calculate actual duration in seconds
        actual_duration_sec = relative_duration_quarters * quarter_note_duration_sec
        _duration_cache[cache_key] = actual_duration_sec
        return actual_duration_sec

    except ValueError:
        print(f"Warning: Could not parse tempo '{tempo}' as float. Using default {config.CONFIG['default_tempo']}.")
        # Recurse with default tempo, avoiding infinite loop by ensuring tempo is valid
        return duration_to_seconds(duration_symbol, config.CONFIG['default_tempo'], time_sig_denominator)
    except Exception as e:
        print(
            f"Error calculating duration for '{duration_symbol}' at tempo {tempo}: {e}. Using default 0.5s."
        )
        return 0.5


def parse_symbolic_to_structured_data(symbolic_text):
    """Step 4: Parse concatenated symbolic text into structured data for MIDI."""
    print("\n--- Step 4: Parsing Symbolic Text ---")
    notes_by_instrument_track = {}
    tempo_changes = []
    time_signature_changes = []
    key_signature_changes = []
    # Key: unique inst_track_key tuple (inst_name_lower, track_id)
    # Value: {program, is_drum, name (for MIDI), orig_inst_name}
    instrument_definitions = {}

    # State variables for parsing context
    current_track_times = {}
    current_global_time = 0.0
    current_tempo = float(config.CONFIG["default_tempo"])
    current_ts_num, current_ts_den = config.CONFIG["default_timesig"]
    current_key = config.CONFIG["default_key"]
    active_melodic_program = config.CONFIG["default_program"]
    active_melodic_instrument_orig_name = config.CONFIG["default_instrument_name"]
    active_melodic_instrument_lookup_name = active_melodic_instrument_orig_name.lower()

    current_bar_number = 0
    current_bar_start_time = 0.0
    time_within_bar_per_track = {}
    expected_bar_duration_sec = duration_to_seconds("W", current_tempo) * (
        current_ts_num / 4.0
    ) * (4.0 / current_ts_den)

    last_event_end_time = 0.0

    lines = symbolic_text.strip().split("\n")
    parse_start_line_index = 0

    # --- Pre-pass for initial settings (before first BAR marker) ---
    print("Processing initial settings (before first BAR marker)...")
    initial_settings = {"T": None, "TS": None, "K": None, "INST": None}
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("BAR:"):
            parse_start_line_index = i
            break

        parts = line.split(":", 1)
        command = parts[0].upper()
        value = parts[1].strip() if len(parts) > 1 else ""
        ini_line_num = i + 1

        try:
            if command == "INST" and initial_settings["INST"] is None:
                if value:
                    inst_name_lower = value.lower()
                    program = music_defs.INSTRUMENT_PROGRAM_MAP.get(
                        inst_name_lower, config.CONFIG["default_program"]
                    )
                    active_melodic_program = program
                    active_melodic_instrument_orig_name = value
                    active_melodic_instrument_lookup_name = inst_name_lower
                    initial_settings["INST"] = (value, program)
                    print(f"Initial Melodic Instrument context set to '{value}' (Program: {program})")
                else: print(f"Warning line {ini_line_num}: INST command has empty value.")
            elif command == "T" and initial_settings["T"] is None:
                new_tempo = float(value)
                if new_tempo > 0:
                    current_tempo = new_tempo
                    initial_settings["T"] = current_tempo
                    print(f"Initial Tempo set to {current_tempo} BPM")
                else: print(f"Warning line {ini_line_num}: Invalid tempo value {value}.")
            elif command == "TS" and initial_settings["TS"] is None:
                ts_match = music_defs.TIME_SIGNATURE_PATTERN.match(value)
                if ts_match:
                    new_ts_num, new_ts_den = int(ts_match.group(1)), int(ts_match.group(2))
                    if new_ts_num > 0 and new_ts_den > 0:
                        current_ts_num, current_ts_den = new_ts_num, new_ts_den
                        initial_settings["TS"] = (current_ts_num, current_ts_den)
                        print(f"Initial Time Signature set to {current_ts_num}/{current_ts_den}")
                    else: print(f"Warning line {ini_line_num}: Invalid time signature value {value}.")
                else: print(f"Warning line {ini_line_num}: Invalid TS format '{value}'.")
            elif command == "K" and initial_settings["K"] is None:
                if value:
                    if music_defs.KEY_SIGNATURE_PATTERN.match(value):
                        current_key = value
                        initial_settings["K"] = current_key
                        print(f"Initial Key set to {current_key}")
                    else: print(f"Warning line {ini_line_num}: Invalid key signature format '{value}'.")
                else: print(f"Warning line {ini_line_num}: K command has empty value.")
        except Exception as e:
            print(f"Error parsing initial setting line {ini_line_num}: '{line}' - {e}")
        parse_start_line_index = i + 1

    # Apply initial settings as events at time 0.0
    tempo_changes.append((0.0, current_tempo))
    time_signature_changes.append((0.0, current_ts_num, current_ts_den))
    key_signature_changes.append((0.0, current_key))
    if initial_settings["INST"] is None:
        print(f"Using default initial Melodic Instrument: '{active_melodic_instrument_orig_name}' (Program: {active_melodic_program})")
    expected_bar_duration_sec = duration_to_seconds("W", current_tempo) * (current_ts_num / 4.0) * (4.0 / current_ts_den)

    # --- Main Parsing Loop ---
    print(f"Parsing main body starting from line {parse_start_line_index + 1}...")
    for i in range(parse_start_line_index, len(lines)):
        current_line_num = i + 1
        line = lines[i].strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split(":", 1)
        command = parts[0].upper()
        value = parts[1].strip() if len(parts) > 1 else ""

        try:
            # --- Handle Global Parameter Changes ---
            if command == "INST":
                if value:
                    inst_name_lower = value.lower()
                    program = music_defs.INSTRUMENT_PROGRAM_MAP.get(inst_name_lower)
                    if program is not None:
                        if program != active_melodic_program or value != active_melodic_instrument_orig_name:
                            active_melodic_program = program
                            active_melodic_instrument_orig_name = value
                            active_melodic_instrument_lookup_name = inst_name_lower
                    else: print(f"Warning line {current_line_num}: Unknown instrument name '{value}' in INST command. Ignoring.")
                else: print(f"Warning line {current_line_num}: INST command has empty value. Ignoring.")
            elif command == "T":
                new_tempo = float(value)
                if new_tempo > 0 and abs(new_tempo - current_tempo) > 1e-3:
                    event_time = current_global_time
                    if not tempo_changes or abs(tempo_changes[-1][0] - event_time) > 1e-6 or abs(tempo_changes[-1][1] - new_tempo) > 1e-3:
                        tempo_changes.append((event_time, new_tempo))
                        print(f"Time {event_time:.3f}s (Bar ~{current_bar_number}): Tempo change to {new_tempo} BPM")
                    current_tempo = new_tempo
                    expected_bar_duration_sec = duration_to_seconds("W", current_tempo) * (current_ts_num / 4.0) * (4.0 / current_ts_den)
            elif command == "TS":
                ts_match = music_defs.TIME_SIGNATURE_PATTERN.match(value)
                if ts_match:
                    new_ts_num, new_ts_den = int(ts_match.group(1)), int(ts_match.group(2))
                    if new_ts_num > 0 and new_ts_den > 0 and (new_ts_num, new_ts_den) != (current_ts_num, current_ts_den):
                        event_time = current_global_time
                        if not time_signature_changes or abs(time_signature_changes[-1][0] - event_time) > 1e-6 \
                           or (time_signature_changes[-1][1], time_signature_changes[-1][2]) != (new_ts_num, new_ts_den):
                            time_signature_changes.append((event_time, new_ts_num, new_ts_den))
                            print(f"Time {event_time:.3f}s (Bar ~{current_bar_number}): Time Sig change to {new_ts_num}/{new_ts_den}")
                        current_ts_num, current_ts_den = new_ts_num, new_ts_den
                        expected_bar_duration_sec = duration_to_seconds("W", current_tempo) * (current_ts_num / 4.0) * (4.0 / current_ts_den)
                else: print(f"Warning line {current_line_num}: Invalid TS format '{value}'. Ignoring TS command.")
            elif command == "K":
                if value and value != current_key:
                    if music_defs.KEY_SIGNATURE_PATTERN.match(value):
                        event_time = current_global_time
                        if not key_signature_changes or abs(key_signature_changes[-1][0] - event_time) > 1e-6 or key_signature_changes[-1][1] != value:
                            key_signature_changes.append((event_time, value))
                            print(f"Time {event_time:.3f}s (Bar ~{current_bar_number}): Key change to {value}")
                        current_key = value
                    else: print(f"Warning line {current_line_num}: Invalid key signature format '{value}'. Ignoring K command.")
                elif not value: print(f"Warning line {current_line_num}: K command has empty value. Ignoring.")

            # --- Handle Bar Marker ---
            elif command == "BAR":
                bar_number = int(value)
                expected_new_bar_start_time = current_bar_start_time + expected_bar_duration_sec if current_bar_number > 0 else 0.0

                if current_bar_number > 0:
                    max_accumulated_time_in_prev_bar = max(time_within_bar_per_track.values()) if time_within_bar_per_track else 0.0
                    tolerance = max(0.005, expected_bar_duration_sec * 0.01)
                    duration_error = max_accumulated_time_in_prev_bar - expected_bar_duration_sec

                    if abs(duration_error) > tolerance:
                        print(f"Warning: Bar {current_bar_number} timing mismatch. Expected {expected_bar_duration_sec:.3f}s, got {max_accumulated_time_in_prev_bar:.3f}s. Forcing bar {bar_number} start to expected time {expected_new_bar_start_time:.3f}s.")
                    current_global_time = expected_new_bar_start_time

                bars_jumped = bar_number - (current_bar_number + 1)
                if bars_jumped > 0 and current_bar_number > 0:
                    jump_duration = bars_jumped * expected_bar_duration_sec
                    print(f"Warning: Jump detected from Bar {current_bar_number} to {bar_number}. Advancing global time by ~{jump_duration:.3f}s.")
                    current_global_time += jump_duration
                elif bar_number <= current_bar_number and current_bar_number > 0:
                    print(f"Warning line {current_line_num}: Bar number {bar_number} is not sequential (previous was {current_bar_number}). Timing might be incorrect.")

                current_bar_number = bar_number
                current_bar_start_time = current_global_time
                time_within_bar_per_track = {key: 0.0 for key in time_within_bar_per_track}
                current_track_times = {key: current_bar_start_time for key in current_track_times}

            # --- Handle Note, Chord, Rest Events ---
            elif command in ["N", "C", "R"]:
                if current_bar_number == 0:
                    print(f"Warning: Event '{line}' on Line {current_line_num} occurred before the first BAR marker. Processing at time 0.")
                    current_bar_number = 1
                    current_bar_start_time = 0.0
                    current_global_time = 0.0

                data_parts = value.split(":")
                min_parts = 2 if command == "R" else 4
                if len(data_parts) < min_parts:
                    print(f"Warning: Malformed {command} command on Line {current_line_num}: '{line}'. Skipping.")
                    continue

                track_id = data_parts[0].strip()
                if not track_id:
                    print(f"Warning: Empty TrackID in {command} command on Line {current_line_num}. Skipping.")
                    continue

                event_is_drum = track_id.lower() in music_defs.DRUM_TRACK_IDS
                event_program = 0
                event_instrument_base_name = track_id
                inst_track_key = None

                if event_is_drum:
                    inst_track_key = ("drums", track_id.lower())
                    midi_instrument_name = f"{track_id} (Drums)"
                else:
                    event_program = active_melodic_program
                    event_instrument_base_name = active_melodic_instrument_orig_name
                    inst_track_key = (active_melodic_instrument_lookup_name, track_id)
                    midi_instrument_name = f"{track_id} ({active_melodic_instrument_orig_name})"

                if inst_track_key not in instrument_definitions:
                    instrument_definitions[inst_track_key] = {
                        "program": event_program, "is_drum": event_is_drum,
                        "name": midi_instrument_name, "orig_inst_name": event_instrument_base_name,
                    }
                    print(f"Defined instrument/track: {midi_instrument_name} (Key: {inst_track_key}, Program: {event_program}, IsDrum: {event_is_drum})")
                    initial_track_offset = time_within_bar_per_track.get(inst_track_key, 0.0)
                    current_track_times[inst_track_key] = current_bar_start_time + initial_track_offset
                    time_within_bar_per_track[inst_track_key] = initial_track_offset
                    notes_by_instrument_track[inst_track_key] = []

                track_specific_start_offset = time_within_bar_per_track.get(inst_track_key, 0.0)
                event_start_time = current_bar_start_time + track_specific_start_offset
                event_duration_sec = 0.0

                if command == "N":
                    pitch_name_raw = data_parts[1].strip()
                    duration_sym_raw = data_parts[2].strip()
                    velocity_str_raw = data_parts[3].strip()
                    if not pitch_name_raw or not duration_sym_raw or not velocity_str_raw:
                        print(f"Warning: Empty part in N command on Line {current_line_num}. Skipping.")
                        continue

                    velocity_str = velocity_str_raw.split("#", 1)[0].strip()
                    duration_sym = duration_sym_raw.split("#", 1)[0].strip()
                    if not duration_sym: print(f"Warning: Empty duration for N command '{line}'. Skipping."); continue

                    try: velocity = max(0, min(127, int(velocity_str)))
                    except ValueError: velocity = 90; print(f"Warning line {current_line_num}: Invalid velocity '{velocity_str_raw}'. Using {velocity}.")

                    event_duration_sec = duration_to_seconds(duration_sym, current_tempo, current_ts_den)
                    midi_pitch = None

                    if event_is_drum:
                        pitch_name_lookup = pitch_name_raw.lower()
                        midi_pitch = music_defs.DRUM_PITCH_MAP.get(pitch_name_lookup)
                        if midi_pitch is None: print(f"Warning line {current_line_num}: Unknown drum sound '{pitch_name_raw}' for track '{track_id}'. Skipping note."); continue
                    else:
                        midi_pitch = pitch_to_midi(pitch_name_raw)
                        if midi_pitch is None: print(f"Warning line {current_line_num}: Cannot parse pitch '{pitch_name_raw}' for track '{track_id}'. Skipping note."); continue

                    note_event = {"pitch": midi_pitch, "start": event_start_time, "end": event_start_time + event_duration_sec, "velocity": velocity}
                    notes_by_instrument_track[inst_track_key].append(note_event)

                elif command == "C":
                    pitches_str_raw = data_parts[1].strip()
                    duration_sym_raw = data_parts[2].strip()
                    velocity_str_raw = data_parts[3].strip()
                    if not pitches_str_raw or not duration_sym_raw or not velocity_str_raw:
                        print(f"Warning: Empty part in C command on Line {current_line_num}. Skipping.")
                        continue

                    velocity_str = velocity_str_raw.split("#", 1)[0].strip()
                    duration_sym = duration_sym_raw.split("#", 1)[0].strip()
                    if not duration_sym: print(f"Warning: Empty duration for C command '{line}'. Skipping."); continue

                    try: velocity = max(0, min(127, int(velocity_str)))
                    except ValueError: velocity = 70; print(f"Warning line {current_line_num}: Invalid velocity '{velocity_str_raw}'. Using {velocity}.")

                    if not (pitches_str_raw.startswith("[") and pitches_str_raw.endswith("]")):
                        print(f"Warning line {current_line_num}: Chord pitches format incorrect: '{pitches_str_raw}'. Skipping chord."); continue
                    pitches_str = pitches_str_raw[1:-1]
                    pitch_names = [p.strip() for p in pitches_str.split(",") if p.strip()]
                    if not pitch_names: print(f"Warning line {current_line_num}: No pitches found in Chord command '{line}'. Skipping."); continue

                    event_duration_sec = duration_to_seconds(duration_sym, current_tempo, current_ts_den)
                    chord_notes = []
                    valid_pitches_in_chord = 0
                    for pitch_name_raw in pitch_names:
                        midi_pitch = None
                        if event_is_drum:
                            pitch_name_lookup = pitch_name_raw.lower()
                            midi_pitch = music_defs.DRUM_PITCH_MAP.get(pitch_name_lookup)
                            if midi_pitch is None: print(f"Warning line {current_line_num}: Unknown drum sound '{pitch_name_raw}' in chord for track '{track_id}'. Skipping this pitch."); continue
                        else:
                            midi_pitch = pitch_to_midi(pitch_name_raw)
                            if midi_pitch is None: print(f"Warning line {current_line_num}: Cannot parse pitch '{pitch_name_raw}' in chord for track '{track_id}'. Skipping this pitch."); continue

                        note_event = {"pitch": midi_pitch, "start": event_start_time, "end": event_start_time + event_duration_sec, "velocity": velocity}
                        chord_notes.append(note_event)
                        valid_pitches_in_chord += 1

                    if valid_pitches_in_chord > 0: notes_by_instrument_track[inst_track_key].extend(chord_notes)
                    else: print(f"Warning line {current_line_num}: Chord command had no valid notes after parsing. Skipping chord."); continue

                elif command == "R":
                    duration_sym_raw = data_parts[1].strip()
                    if not duration_sym_raw: print(f"Warning line {current_line_num}: Empty duration in R command '{line}'. Skipping."); continue
                    duration_sym = duration_sym_raw.split("#", 1)[0].strip()
                    if not duration_sym: print(f"Warning line {current_line_num}: Empty duration for R command '{line}'. Skipping."); continue
                    event_duration_sec = duration_to_seconds(duration_sym, current_tempo, current_ts_den)

                # --- Post-Event Time Advancement ---
                if event_duration_sec > 0:
                    new_track_time_absolute = event_start_time + event_duration_sec
                    current_track_times[inst_track_key] = new_track_time_absolute
                    time_within_bar_per_track[inst_track_key] = new_track_time_absolute - current_bar_start_time
                    last_event_end_time = max(last_event_end_time, new_track_time_absolute)
                else: print(f"Warning line {current_line_num}: Event '{line}' resulted in zero duration. Not advancing time.")

            else: # Unknown command
                print(f"Warning: Unknown command '{command}' on line {current_line_num}: '{line}'. Skipping.")

        except Exception as e:
            print(f"FATAL Error parsing line {current_line_num}: '{line}' - {e}")
            traceback.print_exc()

    # --- Final Cleanup and Summary ---
    print(f"Symbolic text parsing complete. Estimated total duration: {last_event_end_time:.3f} seconds.")
    final_instrument_defs = {}
    final_notes_data = {}
    for key, definition in instrument_definitions.items():
        if key in notes_by_instrument_track and notes_by_instrument_track[key]:
            final_instrument_defs[key] = definition
            final_notes_data[key] = notes_by_instrument_track[key]
        else: print(f"Info: Instrument/Track '{definition['name']}' (Key: {key}) defined but had no notes parsed. Excluding from MIDI.")

    _pitch_parse_cache.clear()
    _duration_cache.clear()

    return (
        final_notes_data,
        final_instrument_defs,
        tempo_changes,
        time_signature_changes,
        key_signature_changes,
        last_event_end_time,
        current_key,
        current_tempo
    )