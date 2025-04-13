Okay, here is the updated Product Backlog based on the provided `README.md`, current project state, and today's date (Sun Apr 13 2025).

**Product Owner Agent:** Backlog Update - AutoMusic Generator
**Date:** 2025-04-13

**Project Goal:** To create a system that generates coherent MIDI music from textual descriptions using LLMs via a sectional generation approach and a compact symbolic format parsed by a Python script.

**Current State:** Minimal project structure (`landing.html`, `music.py`). Core parsing logic and LLM interaction workflow need implementation.

---

## Product Backlog - AutoMusic Generator

*(Items are ordered by priority within each section)*

### P1: High Priority (Core Functionality - MVP Focus)

1.  **Feature: Implement Core Symbolic Format Parser (`music.py`)**
    *   **User Story:** As a developer, I need `music.py` to parse the basic elements (`N:`, `C:`, `R:`, `BAR:`) of the compact symbolic format from a text file so that musical events can be translated into a structured representation.
    *   **Acceptance Criteria:** Script can read a `.txt` file, identify note, chord, rest, and bar lines, and store them internally (e.g., in lists or objects). Handles basic pitch notation (e.g., C4, F#5) and duration abbreviations (W, H, Q, E, S). Handles track identifiers.
    *   **Notes:** This is the absolute core of the conversion process. Focus on correct parsing logic first.

2.  **Feature: Implement Basic MIDI File Generation (`music.py`)**
    *   **User Story:** As a developer, I need `music.py` to convert the parsed symbolic representation into a valid `.mid` file using a MIDI library (e.g., `mido`, `pretty_midi`).
    *   **Acceptance Criteria:** Script takes the internal representation from P1.1, calculates absolute timings based on a *default* tempo/time signature, assigns MIDI note numbers, velocities, and durations, and saves a playable MIDI file.
    *   **Notes:** Requires selecting and integrating a Python MIDI library. Initial version can use hardcoded defaults for tempo/TS.

3.  **Feature: Implement Metadata Parsing (`music.py`)**
    *   **User Story:** As a developer, I need `music.py` to parse metadata (`T:`, `TS:`, `K:`, `INST:`) from the symbolic format so that the generated MIDI reflects the intended tempo, time signature, key, and instrumentation.
    *   **Acceptance Criteria:** Script correctly extracts BPM, time signature (numerator/denominator), key signature, and instrument names. Tempo and time signature are used for timing calculations in MIDI generation. Key and Instrument data are included in the MIDI file if the library supports it.
    *   **Notes:** Builds upon P1.1 and P1.2. Essential for making the output musically meaningful according to the LLM's intent.

4.  **Feature: Define & Document Initial Symbolic Format v1.0**
    *   **User Story:** As a project team member, I need a clearly defined and documented specification for the initial Compact Symbolic Format (including all elements in the README) so that both LLM prompts and the `music.py` parser have a consistent target.
    *   **Acceptance Criteria:** A document (e.g., separate `FORMAT.md` or detailed section in README) exists, precisely defining syntax, abbreviations, expected values, and providing clear examples for `INST:`, `T:`, `TS:`, `K:`, `BAR:`, `N:`, `C:`, `R:`.
    *   **Notes:** Crucial for alignment. Should be finalized before/during P1.1-P1.3 development.

5.  **Feature: Basic Command-Line Interface for `music.py`**
    *   **User Story:** As a user/developer, I want to run `music.py` from the command line, specifying an input symbolic text file and an output MIDI file path, so I can easily test the conversion process.
    *   **Acceptance Criteria:** Script accepts arguments like `python music.py input.txt output.mid`. Provides basic feedback (e.g., "Parsing complete", "MIDI file saved").
    *   **Notes:** Enables testing and usage without a complex UI.

### P2: Medium Priority (Improvements & Usability)

1.  **Feature: Implement Dotted Duration Parsing (`music.py`)**
    *   **User Story:** As a developer, I need the `music.py` parser to handle dotted duration notations (e.g., `Q.`, `H.`) so that more complex rhythms can be accurately represented and generated.
    *   **Acceptance Criteria:** Parser correctly interprets `.` notation and calculates the corresponding duration (1.5x base duration). MIDI generation reflects these durations accurately.

2.  **Feature: Robust Parser Error Handling (`music.py`)**
    *   **User Story:** As a developer, I need `music.py` to gracefully handle common errors or variations in the symbolic input (e.g., malformed lines, unknown abbreviations) so that the script doesn't crash and provides informative error messages.
    *   **Acceptance Criteria:** Script identifies invalid lines/syntax. Reports errors with line numbers or context. May attempt to skip invalid lines and continue processing where possible.
    *   **Notes:** Improves reliability, especially given potential LLM output inconsistencies.

3.  **Documentation: Develop Example LLM Prompts for Sectional Generation**
    *   **User Story:** As a user/developer, I need documented examples of effective LLM prompts for generating individual music sections using the symbolic format, including how to pass context between sections, so I can successfully use the intended generation pipeline.
    *   **Acceptance Criteria:** Examples covering initial section generation, subsequent section generation (referencing previous), specifying goals (mood, key, length), and explicitly defining the symbolic format within the prompt. Stored in README or separate documentation.
    *   **Notes:** Critical for guiding users/developers on *how* to generate the symbolic input for `music.py`.

4.  **Testing: Create Test Suite for `music.py`**
    *   **User Story:** As a developer, I need a set of test cases (example symbolic files and expected MIDI outputs/properties) for `music.py` so that I can verify parser correctness and prevent regressions.
    *   **Acceptance Criteria:** A collection of `.txt` files covering various format features (notes, chords, rests, metadata, dotted notes). Corresponding checks (manual or automated) for the generated MIDI files.
    *   **Notes:** Essential for maintaining code quality as features are added.

### P3: Low Priority (Future Enhancements & Nice-to-Haves)

1.  **Feature: Enhance Symbolic Format (v1.1+)**
    *   **User Story:** As a musician, I want the symbolic format to support additional musical nuances like gradual tempo changes (accelerando/ritardando), dynamics (crescendo/diminuendo), articulations (staccato, legato), and pedal markings so that the generated music can be more expressive.
    *   **Acceptance Criteria:** Define syntax for new elements. Implement parsing and MIDI generation logic in `music.py` for these features. Update format documentation.
    *   **Notes:** Based on "Future Enhancements" in README. Requires careful design for LLM compatibility.

2.  **Feature: Configuration File**
    *   **User Story:** As a user/developer, I want to manage default settings (e.g., fallback tempo/TS, MIDI resolution, symbolic format definitions) via a configuration file (e.g., `config.yaml`, `config.ini`) so that I can customize behavior without modifying the script.
    *   **Acceptance Criteria:** `music.py` reads settings from a config file. If the file is missing or settings are absent, sensible defaults are used.
    *   **Notes:** Improves flexibility and maintainability.

3.  **Feature: Basic Web UI (`landing.html` + Backend)**
    *   **User Story:** As a user, I want a simple web interface where I can paste or upload my concatenated symbolic music text and click a button to generate and download the corresponding MIDI file.
    *   **Acceptance Criteria:** `landing.html` provides a text area and a button. A simple backend (e.g., Flask/Django) receives the text, runs `music.py`, and returns the MIDI file for download.
    *   **Notes:** Makes the tool more accessible but adds web development complexity. The current `landing.html` is likely just a placeholder.

4.  **Exploration: Music Theory Constraints**
    *   **User Story:** As a developer, I want to explore methods (e.g., prompt constraints, post-processing rules) to enforce basic music theory principles (e.g., staying within key, avoiding excessive dissonance) during or after generation so that the output quality is improved.
    *   **Acceptance Criteria:** Research documented. Potential implementation ideas proposed. Maybe a proof-of-concept post-processing script.
    *   **Notes:** Advanced topic, significant research and development effort.

### Completed / Removed

*   *(None - This is the initial backlog based on the project definition)*

---

**Next Steps:**

*   Focus development efforts on the P1 items, starting with the core parser (`music.py`) and MIDI generation.
*   Finalize and document the v1.0 symbolic format specification.
*   Select and integrate a suitable Python MIDI library.