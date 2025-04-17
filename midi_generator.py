# -*- coding: utf-8 -*-
"""
Handles the creation of the final MIDI file from structured music data
using the pretty_midi library.
"""

import os
import math
import traceback
import pretty_midi

# Import from local modules
import config

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
        initial_tempo = tempo_changes[0][1] if tempo_changes else config.CONFIG["default_tempo"]
        midi_obj = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo)
        print(f"Initialized MIDI with tempo: {initial_tempo:.2f} BPM")

        # --- Apply Meta-Messages ---
        if len(tempo_changes) > 1:
             print(f"Tempo changes detected ({len(tempo_changes)} total). Note timings reflect these changes.")

        # Time Signature Changes
        time_sig_changes.sort(key=lambda x: x[0])
        unique_ts = {}
        for time, num, den in time_sig_changes:
            actual_den = den
            if den <= 0 or (den & (den - 1) != 0):
                 if den > 0: actual_den = 2**math.ceil(math.log2(den))
                 else: actual_den = 4
                 print(f"Warning: Invalid TS denominator {den} at {time:.3f}s. Using nearest power of 2: {actual_den}.")
            unique_ts[round(time, 6)] = (num, actual_den)

        midi_obj.time_signature_changes = []
        applied_ts_count = 0
        last_ts_tuple = None
        for time in sorted(unique_ts.keys()):
            num, den = unique_ts[time]
            current_ts_tuple = (num, den)
            if current_ts_tuple != last_ts_tuple:
                try:
                    ts_event = pretty_midi.TimeSignature(num, den, time)
                    midi_obj.time_signature_changes.append(ts_event)
                    applied_ts_count += 1
                    last_ts_tuple = current_ts_tuple
                except ValueError as e:
                     print(f"Error creating TimeSignature({num}, {den}, {time:.3f}): {e}. Skipping.")

        if applied_ts_count > 0: print(f"Applied {applied_ts_count} unique time signature changes.")
        if not midi_obj.time_signature_changes:
            default_num, default_den = config.CONFIG["default_timesig"]
            midi_obj.time_signature_changes.append(
                pretty_midi.TimeSignature(default_num, default_den, 0.0)
            )
            print(f"Applied default time signature: {default_num}/{default_den}")

        # Key Signature Changes
        key_sig_changes.sort(key=lambda x: x[0])
        unique_ks = {}
        last_valid_key_name = config.CONFIG["default_key"]
        for time, key_name in key_sig_changes:
            try:
                key_number = pretty_midi.key_name_to_key_number(key_name)
                unique_ks[round(time, 6)] = key_name
                last_valid_key_name = key_name
            except ValueError:
                print(f"Warning: Invalid key name '{key_name}' found at time {time:.3f}s during processing. Ignoring.")

        midi_obj.key_signature_changes = []
        applied_key_count = 0
        last_key_number_added = None
        for time in sorted(unique_ks.keys()):
            key_name = unique_ks[time]
            try:
                key_number = pretty_midi.key_name_to_key_number(key_name)
                if key_number != last_key_number_added:
                    ks_event = pretty_midi.KeySignature(key_number=key_number, time=time)
                    midi_obj.key_signature_changes.append(ks_event)
                    applied_key_count += 1
                    last_key_number_added = key_number
            except ValueError as e:
                print(f"Error creating KeySignature for '{key_name}' at {time:.3f}s: {e}. Skipping.")

        if applied_key_count > 0: print(f"Applied {applied_key_count} unique key signature changes.")
        if not midi_obj.key_signature_changes:
            try:
                final_default_key = last_valid_key_name
                default_key_num = pretty_midi.key_name_to_key_number(final_default_key)
                midi_obj.key_signature_changes.append(
                    pretty_midi.KeySignature(key_number=default_key_num, time=0.0)
                )
                print(f"Applied default key signature: {final_default_key}")
            except ValueError as e:
                print(f"Warning: Invalid default key '{final_default_key}'. No key signature applied. Error: {e}")

        # --- Create instruments and add notes ---
        available_channels = list(range(16))
        drum_channel = 9
        if drum_channel in available_channels: available_channels.remove(drum_channel)
        channel_index = 0

        sorted_inst_keys = sorted(instrument_defs.keys())

        for inst_track_key in sorted_inst_keys:
            definition = instrument_defs[inst_track_key]
            if not notes_data.get(inst_track_key): continue

            is_drum = definition["is_drum"]
            program = definition["program"]
            pm_instrument_name = definition["name"]

            channel = -1
            if is_drum:
                channel = drum_channel
            else:
                if not available_channels:
                    print(f"Warning: Ran out of unique MIDI channels! Reusing channel for {pm_instrument_name}.")
                    channel = (channel_index % 15)
                    if channel >= drum_channel: channel += 1
                else:
                    channel = available_channels[channel_index % len(available_channels)]
                channel_index += 1

            instrument_obj = pretty_midi.Instrument(program=program, is_drum=is_drum, name=pm_instrument_name)
            midi_obj.instruments.append(instrument_obj)
            print(f"Created MIDI instrument: {pm_instrument_name} (Program: {program}, IsDrum: {is_drum}, Target Channel: {channel})")

            note_count, skipped_notes = 0, 0
            for note_info in notes_data[inst_track_key]:
                start_time = max(0.0, note_info["start"])
                end_time = note_info["end"]
                min_duration = 0.001
                if end_time <= start_time: end_time = start_time + min_duration
                elif end_time - start_time < min_duration: end_time = start_time + min_duration

                velocity = max(1, min(127, int(note_info["velocity"])))
                pitch = max(0, min(127, int(note_info["pitch"])))

                try:
                    note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=end_time)
                    instrument_obj.notes.append(note)
                    note_count += 1
                except ValueError as e:
                    print(f"Error creating pretty_midi.Note for {pm_instrument_name}: {e}. Skipping note. Data: P={pitch}, V={velocity}, S={start_time:.4f}, E={end_time:.4f}")
                    skipped_notes += 1

            print(f"  Added {note_count} notes. ({skipped_notes} skipped due to errors/duration).")

        # --- Write MIDI File ---
        os.makedirs(config.CONFIG["output_dir"], exist_ok=True)
        full_output_path = os.path.join(config.CONFIG["output_dir"], filename)
        midi_obj.write(full_output_path)
        print(f"\nSuccessfully created MIDI file: {full_output_path}")

    except Exception as e:
        print(f"Error writing MIDI file '{filename}': {e}")
        traceback.print_exc()