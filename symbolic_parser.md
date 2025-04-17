# File: symbolic_parser.py

## Overview

The `symbolic_parser.py` file is a core component of the music generation pipeline. Its primary function is to take a string containing music notation in a custom symbolic text format and convert it into a structured Python data representation. This structured data is designed to be easily consumable by the `midi_generator.py` module for creating MIDI files.

The parser handles various musical elements including notes, chords, rests, tempo changes, time signature changes, key signature changes, and instrument definitions, processing them sequentially to determine their timing and properties. It relies on definitions and configurations provided by `music_defs.py` and `config.py`.

## Role in the Project

In the context of the project structure:

-   It takes the raw symbolic text, potentially generated or retrieved by other parts of the system (like `llm_interface.py` if it were generating this format, or assumed input for `pipeline.py`).
-   It uses constants and mappings from `config.py` and `music_defs.py`.
-   It produces structured data (notes with precise start/end times, instrument mappings, tempo/time signature/key signature events) which is then passed to `midi_generator.py` for the final MIDI file creation.
-   `pipeline.py` would likely coordinate the flow from text input through this parser to the MIDI generator.

## Dependencies

-   `re`: Used for regular expression matching (e.g., parsing pitch names, time signatures).
-   `math`: Imported but not explicitly used in the provided code snippet. (Potentially remnants or for future use).
-   `traceback`: Used for printing detailed error information during parsing.
-   `config`: Provides default settings like tempo, time signature, program numbers.
-   `music_defs`: Provides mappings for pitch names, accidentals, duration symbols, drum sounds, instrument programs, and regex patterns.

## Global Variables

-   `_pitch_parse_cache`: A dictionary used to cache results of the `pitch_to_midi` function for performance. Maps pitch name strings to MIDI numbers or `None`.
-   `_duration_cache`: A dictionary used to cache results of the `duration_to_seconds` function. Maps `(duration_symbol, tempo, time_sig_denominator)` tuples to duration in seconds.

These caches improve performance by avoiding repeated calculations for common pitches or durations at the same tempo.

## Functions

### `pitch_to_midi(pitch_name)`

Converts a standard musical pitch name string (e.g., "C4", "F#5", "Gb3", "Dbb2", "E##6") into its corresponding MIDI note number (0-127).

-   **Parameters:**
    -   `pitch_name` (`str`): The string representation of the pitch. Expected format is Note (A-G), optional accidental (#, b, s, sb), and octave number. Case-insensitive for the note and accidental.
-   **Returns:**
    -   `int` or `None`: The MIDI note number (an integer between 0 and 127) if the pitch name is valid and within the MIDI range. Returns `None` if the pitch name is malformed or the resulting MIDI number is outside the 0-127 range.
-   **Details:**
    -   Uses a regular expression to extract the note letter, accidental, and octave.
    -   Looks up the base MIDI value from `music_defs.PITCH_MAP`.
    -   Looks up the accidental value from `music_defs.ACCIDENTAL_MAP`.
    -   Calculates the final MIDI value using the formula: `base_midi + accidental_value + (octave + 1) * 12` (based on C0 being MIDI note 12, C4 being 60).
    -   Caches valid and invalid results to speed up subsequent calls with the same pitch name.
    -   Prints no warnings or errors itself, simply returns `None` on failure.

### `duration_to_seconds(duration_symbol, tempo, time_sig_denominator=4)`

Converts a symbolic duration string (e.g., "W", "H.", "Q", "E", "S") into its equivalent duration in seconds, based on the provided tempo.

-   **Parameters:**
    -   `duration_symbol` (`str`): The symbol representing the duration (e.g., "W" for whole, "H" for half, "Q" for quarter, "E" for eighth, "S" for sixteenth, "T" for thirty-second). Can include a dot (`.`) for dotted notes. Case-insensitive.
    -   `tempo` (`float` or `int`): The tempo in beats per minute (BPM).
    -   `time_sig_denominator` (`int`, optional): The denominator of the time signature. Used _only_ as part of the cache key; the calculation itself is relative to the quarter note duration based on BPM, which is standard in MIDI timing. Defaults to 4.
-   **Returns:**
    -   `float`: The duration in seconds. Returns 0.5 seconds as a fallback in case of errors during calculation.
-   **Details:**
    -   Expects standard duration symbols, optionally followed by a dot for dotted values.
    -   Looks up the duration's value relative to a quarter note from `music_defs.DURATION_RELATIVE_MAP`.
    -   Applies the 1.5 multiplier for dotted notes.
    -   Calculates the duration of a quarter note in seconds (`60.0 / tempo`).
    -   Multiplies the relative duration by the quarter note duration to get the final duration in seconds.
    -   Caches results based on the symbol, tempo, and time signature denominator.
    -   Includes error handling for invalid tempo values (uses default) and unknown duration symbols (uses quarter note relative value, prints warnings).
    -   Returns a default duration (0.5s) if a general exception occurs during calculation.

### `parse_symbolic_to_structured_data(symbolic_text)`

The main parsing function. Takes the entire symbolic music text content as a string and processes it line by line to extract musical events and parameters, organizing them into structured data suitable for MIDI generation.

-   **Parameters:**
    -   `symbolic_text` (`str`): The complete content of the symbolic music text file or string.
-   **Returns:**
    -   `tuple`: A tuple containing the parsed data:
        1.  `notes_data` (`dict`): A dictionary where keys are `(instrument_lookup_name_lower, track_id)` tuples, and values are lists of note event dictionaries. Each note event dictionary has keys: `{"pitch": int, "start": float, "end": float, "velocity": int}`. Only includes instrument/track keys for which at least one note was parsed.
        2.  `instrument_definitions` (`dict`): A dictionary where keys are `(instrument_lookup_name_lower, track_id)` tuples, and values are dictionaries defining the instrument/track: `{"program": int, "is_drum": bool, "name": str, "orig_inst_name": str}`. Only includes definitions for instrument/track keys present in `notes_data`.
        3.  `tempo_changes` (`list`): A list of tuples `(time, tempo)`, indicating tempo changes at specific times (in seconds from the start). Includes the initial tempo at time 0.0.
        4.  `time_signature_changes` (`list`): A list of tuples `(time, numerator, denominator)`, indicating time signature changes at specific times. Includes the initial time signature at time 0.0.
        5.  `key_signature_changes` (`list`): A list of tuples `(time, key_string)`, indicating key signature changes at specific times. Includes the initial key signature at time 0.0.
        6.  `estimated_total_duration` (`float`): The time in seconds when the last parsed event ends.
        7.  `final_key` (`str`): The key signature string active at the end of the parsed text.
        8.  `final_tempo` (`float`): The tempo (BPM) active at the end of the parsed text.
-   **Details:**
    -   Parses line by line, ignoring comments (`#`) and empty lines.
    -   Identifies commands based on a `COMMAND:` format.
    -   **Commands Handled:**
        -   `INST:<name>`: Sets the _default_ melodic instrument for subsequent non-drum tracks. Affects the program number assigned to new melodic tracks.
        -   `T:<tempo>`: Sets the current tempo (BPM). Records a tempo change event at the current global time.
        -   `TS:<num>/<den>`: Sets the current time signature. Records a time signature change event at the current global time.
        -   `K:<key>`: Sets the current key signature. Records a key signature change event at the current global time.
        -   `BAR:<number>`: Marks the beginning of a new bar. Advances the global time and resets the time within bar for all tracks. Includes logic to estimate the expected start time of the new bar based on the previous bar's duration and warns if the timing is significantly off, adjusting the global time to the expected bar start. Handles bar number jumps.
        -   `N:<track_id>:<pitch>:<duration>:<velocity>[#comment]`: Parses a single note event. Uses `pitch_to_midi` and `duration_to_seconds`. Assigns the note to the specified track ID. Determines if it's a drum or melodic track based on `music_defs.DRUM_TRACK_IDS`. Appends the note to the `notes_data` for the relevant instrument/track key. Advances the time for that specific track within the current bar.
        -   `C:<track_id>:[<pitch1>,<pitch2>,...]:<duration>:<velocity>[#comment]`: Parses a chord event (multiple notes starting simultaneously). Similar to `N`, but parses multiple pitches. Each valid pitch becomes a separate note event with the same start time, duration, and velocity. Advances the time for the track based on the _single_ chord duration.
        -   `R:<track_id>:<duration>[#comment]`: Parses a rest event. Uses `duration_to_seconds`. Does _not_ add a note event, but _does_ advance the time for the specified track within the current bar.
    -   Maintains state variables for the current tempo, time signature, key, bar number, global time, and accumulated time _within the current bar_ for _each active track_.
    -   Processes an initial pass for `INST`, `T`, `TS`, `K` commands that appear _before_ the first `BAR` marker to establish initial settings at time 0.0.
    -   Calculates note/rest start times based on the `current_bar_start_time` and the track's accumulated `time_within_bar`.
    -   Calculates note/rest end times by adding the parsed duration to the start time.
    -   Updates the `time_within_bar` and the track's absolute time after processing an event.
    -   Includes extensive warning messages for malformed lines, unknown commands, invalid values (tempo, pitch, duration, velocity, time sig, key), and timing discrepancies at bar markers.
    -   Clears the pitch and duration caches before returning.
    -   Filters out instrument/track definitions and data for tracks that ultimately had no valid notes parsed.

## Usage Example
