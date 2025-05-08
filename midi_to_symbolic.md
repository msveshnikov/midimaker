# Documentation for midi_to_symbolic.py

## Overview

`midi_to_symbolic.py` is a core component of the project responsible for converting standard MIDI files (`.mid`) into a custom text-based format called the Compact Symbolic Format. This format represents musical information (notes, rests, tempo, time signatures, key signatures, instrument assignments) in a structured, human-readable text format.

This file effectively performs the reverse operation of `symbolic_parser.py` (which reads the symbolic format) and `midi_generator.py` (which creates MIDI from the symbolic format). It analyzes the timing and properties of MIDI events to reconstruct the musical structure in terms of symbolic durations, pitches, and metadata changes.

Its primary role in the project pipeline is to act as an initial processing step for MIDI input, making the musical content accessible to subsequent analysis, transformation, or generation modules that operate on the symbolic format.

## Dependencies

*   **External:**
    *   `pretty_midi`: For loading and parsing MIDI files.
    *   `math`: For mathematical operations (not explicitly used in the provided code snippet, but often useful with time/tempo calculations).
    *   `os`: For file path validation.
    *   `collections.defaultdict`: For grouping notes by instrument.
    *   `traceback`: For printing detailed error information.
    *   `re`: For validating key signature strings.
*   **Internal:**
    *   `config`: Assumed to contain default values like tempo, time signature, and key.
    *   `music_defs`: Assumed to contain mappings and definitions like `PITCH_MAP`, `DRUM_PITCH_MAP`, `DURATION_RELATIVE_MAP`, `INSTRUMENT_PROGRAM_MAP`, and `KEY_SIGNATURE_PATTERN`.

## Global Variables and Constants

*   `_MIDI_TO_PITCH_CLASS_NAME`: `dict`
    *   A reverse mapping from MIDI pitch class numbers (0-11) to their symbolic names (e.g., `0` -> `'C'`, `1` -> `'C#'`). Derived from `music_defs.PITCH_MAP`.
*   `_DRUM_PITCH_TO_NAME`: `dict`
    *   A reverse mapping from MIDI drum pitch numbers to symbolic drum names. Derived from `music_defs.DRUM_PITCH_MAP`.
*   `_PROGRAM_TO_INSTRUMENT_NAME`: `dict`
    *   A reverse mapping from MIDI program numbers to symbolic instrument names. Derived from `music_defs.INSTRUMENT_PROGRAM_MAP`.
*   `_SORTED_SYMBOLIC_DURATIONS`: `list of tuple`
    *   A list of tuples `(relative_quarter_duration, symbol_string)`, sorted in descending order by `relative_quarter_duration`. This list is used for greedily decomposing MIDI durations into symbolic duration sequences (e.g., converting a 1.5-quarter duration into `['Q', 'E.']` or `['Q', 'E']`). It includes standard and dotted durations derived from `music_defs.DURATION_RELATIVE_MAP`. Unique relative quarter durations are kept, prioritizing the first symbol encountered (based on the initial map order).

## Helper Functions

### `_get_relative_quarter_duration(symbol)`

*   **Purpose:** Calculates the duration of a single symbolic note or rest symbol in terms of relative quarter notes, based on the `music_defs.DURATION_RELATIVE_MAP`. Handles dotted durations.
*   **Parameters:**
    *   `symbol` (`str`): The symbolic duration string (e.g., 'Q', 'W.', 'E'). Case-insensitive and leading/trailing whitespace is ignored.
*   **Return Value:**
    *   `float`: The duration in relative quarter notes (e.g., 1.0 for 'Q', 4.0 for 'W', 6.0 for 'W.').
    *   `None`: If the base symbol is not found in `music_defs.DURATION_RELATIVE_MAP`.

### `midi_pitch_to_symbolic(midi_pitch, is_drum)`

*   **Purpose:** Converts a MIDI pitch number into its symbolic representation (e.g., `60` -> `'C4'`, `36` with `is_drum=True` -> `'BassDrum1'`).
*   **Parameters:**
    *   `midi_pitch` (`int`): The MIDI pitch number (0-127).
    *   `is_drum` (`bool`): `True` if the pitch should be interpreted as a drum sound, `False` otherwise.
*   **Return Value:**
    *   `str`: The symbolic pitch or drum name (e.g., 'C4', 'BassDrum1'). If a drum pitch is not mapped, returns a default format like 'Drum{pitch}'.
    *   `None`: If the MIDI pitch is outside the valid range (0-127) and `is_drum` is `False`.

### `seconds_to_symbolic_duration_sequence(duration_sec, start_time, tempo_map, ts_map, tolerance=0.005)`

*   **Purpose:** Converts a duration given in seconds into a sequence of symbolic duration strings (e.g., `['Q', 'E.']`). This is done by determining the active tempo and time signature at the `start_time` to calculate the duration in relative quarter notes, and then greedily decomposing this value using the `_SORTED_SYMBOLIC_DURATIONS` list.
*   **Parameters:**
    *   `duration_sec` (`float`): The duration in seconds to convert.
    *   `start_time` (`float`): The time in seconds at which the duration begins. Used to determine the active tempo and time signature.
    *   `tempo_map` (`list of tuple`): A sorted list of `(time, qpm)` tuples representing tempo changes.
    *   `ts_map` (`list of tuple`): A sorted list of `(time, numerator, denominator)` tuples representing time signature changes.
    *   `tolerance` (`float`, optional): A small value used for floating-point comparisons when matching durations during decomposition. Defaults to 0.005.
*   **Return Value:**
    *   `list of str`: A list of symbolic duration strings representing the decomposed duration (e.g., `['Q', 'E.']`). Returns an empty list if the duration is zero or negative, or if conversion is not possible.

### `get_active_state(time, tempo_changes, ts_changes, key_changes)`

*   **Purpose:** Helper function to find the active tempo, time signature, and key signature at a given time point, based on lists of changes sorted by time.
*   **Parameters:**
    *   `time` (`float`): The time in seconds to query the state.
    *   `tempo_changes` (`list of tuple`): Sorted list of `(time, qpm)` tempo changes.
    *   `ts_changes` (`list of tuple`): Sorted list of `(time, numerator, denominator)` time signature changes.
    *   `key_changes` (`list of tuple`): Sorted list of `(time, key_name)` key signature changes.
*   **Return Value:**
    *   `tuple`: A tuple `(active_tempo, active_ts, active_key)`. `active_ts` is a tuple `(numerator, denominator)`. The values reflect the last change that occurred at or before the given `time`. Default values from `config` are used if no changes occur before `time`.

## Main Function

### `midi_to_symbolic(midi_file_path)`

*   **Purpose:** Loads a MIDI file, analyzes its structure (notes, timing, meta-messages), and converts it into a sequence of lines in the custom Compact Symbolic Format.
*   **Parameters:**
    *   `midi_file_path` (`str`): The path to the input MIDI file.
*   **Return Value:**
    *   `str`: A single string containing the entire symbolic representation of the MIDI file, with lines separated by newline characters.
    *   `None`: If the MIDI file cannot be found or loaded, or if an error occurs during processing.

*   **Process Details:**
    1.  **Load MIDI:** Uses `pretty_midi.PrettyMIDI` to load the file. Handles file not found and loading errors.
    2.  **Extract Meta-messages:** Retrieves tempo, time signature, and key signature changes from the MIDI data. Stores them as lists of `(time, value)` tuples.
    3.  **Ensure Initial State:** Adds default tempo, time signature, and key signature entries at time 0.0 if the MIDI file doesn't explicitly define them there. Sorts the change lists.
    4.  **Collect Time Points:** Gathers all significant time points from meta-message changes and the start/end times of all notes across all instruments. Sorts and removes duplicate time points (within a small tolerance).
    5.  **Initialize State:** Sets the global time, current bar number, bar start time, and current tempo/TS/key based on the state at time 0.0.
    6.  **Add Initial Lines:** Appends initial `T:`, `TS:`, and `K:` commands to the symbolic output based on the state at time 0.0. Validates the key signature string. Adds initial `INST:` commands for non-drum tracks found in the MIDI file. Appends the first `BAR:1` command.
    7.  **Map Instruments:** Creates a mapping from `pretty_midi.Instrument` index to a tuple containing the symbolic instrument name, a unique symbolic track ID, and whether it's a drum track. Derives symbolic names from MIDI instrument names or program numbers.
    8.  **Group Notes:** Organizes all notes from the MIDI file by their instrument index and sorts them by start time within each instrument group.
    9.  **Iterate Time Points:** Loops through the sorted, unique significant time points.
        *   **Handle Time Elapsed & Bars:** Calculates the duration between the current global time and the next time point. Processes this duration, adding `BAR:` commands whenever a bar line is crossed based on the active tempo and time signature. Updates the global time and bar tracking.
        *   **Update Global State:** Checks if tempo, time signature, or key signature changes occur exactly at the current time point and appends the corresponding `T:`, `TS:`, or `K:` commands if the state changes. Validates key signature strings.
        *   **Process Notes/Chords:** For each instrument/track:
            *   Identifies all notes starting exactly at the current time point.
            *   Calculates the rest duration from the track's last event end time up to the current time point and adds `R:` commands if the rest is significant and can be symbolized.
            *   Groups notes starting at the same time on the same track that also have the same end time and velocity. Each group represents a potential symbolic event (single note or chord).
            *   For each group, converts the pitches to symbolic names and calculates the duration in seconds.
            *   Converts the event duration to a sequence of symbolic duration strings using `seconds_to_symbolic_duration_sequence`.
            *   Appends `N:` (for single notes) or `C:` (for chords) commands to the symbolic output, including the track ID, pitches, duration symbols, and velocity.
            *   Updates the track's current time to the latest end time among the events processed for that track at the current global time point.
    10. **Final Rests and Bars:** After processing all time points, calculates the overall end time of the piece (latest note end or meta-message time). Continues adding `BAR:` commands if the end time extends beyond the last bar processed. Calculates and adds final `R:` commands for each track from its last event end time up to the overall end time.
    11. **Return Output:** Joins all generated symbolic lines into a single string and returns it.

## Usage Example

See __main__ section for an example of how to use the `midi_to_symbolic` function. The example demonstrates loading a MIDI file, converting it to the symbolic format, and printing the output.

```python
if __name__ == "__main__":
    midi_file_path = "path/to/your/midi/file.mid"
    symbolic_output = midi_to_symbolic(midi_file_path)
    print(symbolic_output)
```