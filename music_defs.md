# `music_defs.py` Documentation

## Overview

The `music_defs.py` file serves as a central repository for constants, mappings, and definitions crucial to interpreting and generating music data within the MidiMaker project. It standardizes the symbolic music format used for input/output and provides the necessary mappings to translate between this symbolic representation and the technical details of the MIDI standard (like pitch numbers, instrument program numbers, and timing values).

This file contains no executable logic (functions or classes) but provides foundational data structures used throughout the project, particularly by the symbolic parser (`symbolic_parser.py`) and MIDI generator (`midi_generator.py`).

## Contents

This file defines the following key constants and mappings:

### `SYMBOLIC_FORMAT_DEFINITION`

A multi-line string constant that provides a comprehensive definition of the custom symbolic music format used by the MidiMaker project. This string is intended to serve as documentation for the format itself and can potentially be used as part of a prompt for language models generating this format.

-   **Type:** `str`
-   **Description:** Details each command (`INST`, `T`, `TS`, `K`, `BAR`, `N`, `C`, `R`), their parameters, expected data types, allowed values (like track IDs, pitch names, duration symbols, velocities), and provides examples. It explicitly states formatting rules (new lines per command, no inline comments).
-   **Usage:** Primarily used as a reference or instruction source for understanding or generating the symbolic format. Modules like `llm_interface.py` might use this in prompts, while `symbolic_parser.py` adheres to the structure and rules defined here.

### `PITCH_MAP`

A dictionary mapping standard musical note letters to their base MIDI pitch value within an octave (where C=0, C#=1, D=2, etc., up to B=11).

-   **Type:** `dict`
-   **Description:** `{ "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11 }`
-   **Usage:** Used in conjunction with `ACCIDENTAL_MAP` and octave numbers to calculate the absolute MIDI note number (0-127) for melodic pitches. Used by `symbolic_parser.py`.

### `ACCIDENTAL_MAP`

A dictionary mapping symbols used for accidentals to the integer value they add to a pitch's base value.

-   **Type:** `dict`
-   **Description:** `{ "#": 1, "S": 1, "B": -1, "": 0 }`. Includes both `#` and `S` for sharp and an empty string for natural notes.
-   **Usage:** Used in conjunction with `PITCH_MAP` to adjust the base pitch value based on accidentals. Used by `symbolic_parser.py`.

### `INSTRUMENT_PROGRAM_MAP`

A dictionary mapping lowercase instrument names (both common abbreviations and more descriptive names) to their corresponding General MIDI program number (0-127).

-   **Type:** `dict`
-   **Description:** A comprehensive mapping covering pianos, chromatic percussion, organs, guitars, basses, strings, brass, reeds, pipes, synth leads/pads/FX, ethnic instruments, percussive instruments, and sound effects. Includes common synonyms (e.g., "pno", "piano", "acoustic grand piano" all map to 0).
-   **Usage:** Used to determine the MIDI program number for melodic tracks based on the active instrument set by the `INST:` command. Used by `symbolic_parser.py` and potentially `midi_generator.py`.

### `DRUM_TRACK_IDS`

A set of lowercase strings representing track IDs that are recognized as drum tracks. These tracks are treated specially (assigned to MIDI channel 10) and their pitches are interpreted using `DRUM_PITCH_MAP` instead of `PITCH_MAP`.

-   **Type:** `set`
-   **Description:** Contains common drum-related names like `"drums"`, `"drumkit"`, `"percussion"`, `"elecdrums"`, `"808drums"`, and includes keys from `INSTRUMENT_PROGRAM_MAP` that map to program 0 and contain drum-related terms.
-   **Usage:** Used by `symbolic_parser.py` to determine if a given track ID corresponds to a drum track, influencing how pitch and MIDI channel are handled.

### `DRUM_PITCH_MAP`

A dictionary mapping lowercase drum sound names (used as pitches on drum tracks) to their corresponding MIDI note number on channel 10.

-   **Type:** `dict`
-   **Description:** Includes mappings for standard MIDI percussion sounds like `"kick"` (36), `"snare"` (38), `"hhc"` (42), `"hho"` (46), `"crash"` (49), `"ride"` (51), various toms, and other hand percussion and cymbals.
-   **Usage:** Used by `symbolic_parser.py` to convert drum sound names specified in `N:` or `C:` commands on drum tracks into the correct MIDI note number for channel 10.

### `KEY_SIGNATURE_PATTERN`

A compiled regular expression pattern used for validating key signature strings.

-   **Type:** `re.Pattern`
-   **Description:** Matches strings like `Cmaj`, `Amin`, `G#dor`, `Fbmin`, `Dloc`. It allows a note letter (A-G), optional accidental (#, s, b), and an optional mode suffix (maj, min, dor, phr, lyd, mix, loc, aeo, ion) case-insensitively.
-   **Usage:** Used by `symbolic_parser.py` to validate the format of the value provided with the `K:` command.

### `TIME_SIGNATURE_PATTERN`

A compiled regular expression pattern used for validating time signature strings.

-   **Type:** `re.Pattern`
-   **Description:** Matches strings like `4/4`, `3 / 8`, `12/16`, capturing the numerator and denominator as groups.
-   **Usage:** Used by `symbolic_parser.py` to validate the format of the value provided with the `TS:` command and extract the numerator and denominator.

### `DURATION_RELATIVE_MAP`

A dictionary mapping the symbolic duration characters to their duration relative to a quarter note (which has a value of 1.0).

-   **Type:** `dict`
-   **Description:** `{ "W": 4.0, "H": 2.0, "Q": 1.0, "E": 0.5, "S": 0.25, "T": 0.125 }`. Dotted notes are handled by the parser logic using this base value.
-   **Usage:** Used by `symbolic_parser.py` to calculate the length of notes and rests in beats (or quarter notes) based on the duration symbol.

### `KNOWN_MELODIC_INSTRUMENTS`

A sorted list of lowercase instrument names derived from `INSTRUMENT_PROGRAM_MAP` that are _not_ considered drum track IDs.

-   **Type:** `list`
-   **Description:** Provides a list of the recognized names that can be used with the `INST:` command for melodic tracks.
-   **Usage:** Can be used for validation or listing available instrument options.

### `KNOWN_DRUM_SOUNDS`

A sorted list of lowercase drum sound names derived from `DRUM_PITCH_MAP`.

-   **Type:** `list`
-   **Description:** Provides a list of the recognized names that can be used as pitches on drum tracks.
-   **Usage:** Can be used for validation or listing available drum sound options.

## Role in the Project

`music_defs.py` is a foundational module in the MidiMaker project.

-   **`symbolic_parser.py`:** Heavily relies on `music_defs.py`. It uses `SYMBOLIC_FORMAT_DEFINITION` as the specification, `PITCH_MAP`, `ACCIDENTAL_MAP`, and `DRUM_PITCH_MAP` for pitch conversion, `INSTRUMENT_PROGRAM_MAP` and `DRUM_TRACK_IDS` for instrument/channel mapping, `DURATION_RELATIVE_MAP` for timing calculations, and the regex patterns for validation (`KEY_SIGNATURE_PATTERN`, `TIME_SIGNATURE_PATTERN`).
-   **`midi_generator.py`:** While the parser does most of the conversion to MIDI numbers and durations, the generator implicitly uses the standards defined here (e.g., understanding that channel 10 is for drums, using the program numbers from the map).
-   **`pipeline.py`:** Orchestrates the process and uses the parser and generator, thus depending indirectly on the definitions provided here.
-   **`llm_interface.py`:** May use `SYMBOLIC_FORMAT_DEFINITION` as part of its interaction with language models designed to generate the symbolic format.

In essence, `music_defs.py` provides the shared vocabulary and translation tables that allow the different components of MidiMaker to understand and process musical information consistently.

## Usage Examples

Other modules access the definitions using standard Python imports:
