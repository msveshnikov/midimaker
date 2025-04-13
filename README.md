# AutoMusic Generator

This project explores the generation of MIDI music from textual descriptions using Large Language Models (LLMs). It employs a pipeline that translates high-level musical ideas into a compact symbolic representation, generates music section by section for better coherence, and finally converts this representation into a standard MIDI file using a Python script.

![alt text](image.jpg)

## Core Concept

The primary challenge in generating long musical pieces (e.g., 5 minutes) with LLMs is maintaining coherence and structure. This project addresses this by:

1.  **Sectional Generation:** Breaking down the generation process into smaller, manageable sections (A, B, C...).
2.  **Compact Symbolic Format:** Using a concise text-based format for music representation, which is easier for LLMs to handle than verbose formats and simplifies parsing.
3.  **Algorithmic Conversion:** Utilizing a Python script (`music.py`) to parse the symbolic format and generate the final MIDI file.

## Generation Pipeline

1.  **Small Text Description -> LLM Enrichment -> Detailed Text Description (Overall Plan)**
    -   User provides a high-level idea (e.g., "a sad piano piece in C minor").
    -   An LLM elaborates this into a more detailed plan, potentially outlining structure, mood, instrumentation, key, tempo, etc.
2.  **Detailed Plan + Section Goal -> LLM -> Compact Symbolic Representation (Section A)**
    -   Using the overall plan, provide a specific goal for the first section (e.g., "Section A: 60 bars, establish the C minor theme, sparse texture").
    -   Instruct the LLM to generate the music for this section using the defined **Compact Symbolic Format**.
3.  **Detailed Plan + Section Goal -> LLM -> Compact Symbolic Representation (Section B, C...)**
    -   Repeat the process for subsequent sections, providing goals that guide musical development and transitions (e.g., "Section B: 80 bars, contrast Section A, move to Eb major, increase rhythmic activity"). Ensure prompts reference previous sections as needed for coherence.
4.  **Concatenate Symbolic Sections**
    -   Combine the symbolic text outputs for all sections (A, B, C...) into a single text file (e.g., `symbolic_music.txt`).
5.  **Symbolic Representation -> `music.py` Script -> MIDI File**
    -   The Python script (`music.py`) parses the concatenated symbolic text file.
    -   It interprets the compact format, calculating note timings, durations, velocities, etc., based on tempo, time signature, and bar markers.
    -   It uses a MIDI library (e.g., `mido`, `pretty_midi`) to construct and save the final `.mid` file.

## Compact Symbolic Format

A concise format is crucial for efficient LLM generation and parsing. The recommended format uses delimited fields:

-   `INST:<InstrumentName>` (e.g., `INST:Pno`, `INST:Gtr`)
-   `T:<BPM>` (e.g., `T:60`)
-   `TS:<Numerator>/<Denominator>` (e.g., `TS:4/4`)
-   `K:<KeySignature>` (e.g., `K:Cmin`, `K:Gmaj`)
-   `BAR:<Number>` (e.g., `BAR:1`, `BAR:61`) - Marks the beginning of a bar.
-   `N:<Track>:<Pitch>:<Duration>:<Velocity>` (Note event)
    -   Example: `N:RH:C5:H:70` (Note: Right Hand, C5, Half note, Velocity 70)
-   `C:<Track>:<[Pitches]>:<Duration>:<Velocity>` (Chord event)
    -   Example: `C:LH:[C3,Eb3,G3]:W:60` (Chord: Left Hand, Cmin triad, Whole note, Velocity 60)
-   `R:<Track>:<Duration>` (Rest event)
    -   Example: `R:RH:Q` (Rest: Right Hand, Quarter note)

**Key Abbreviations:**

-   **Track:** `RH` (Right Hand), `LH` (Left Hand), `Tr1`, `Tr2`, etc. (User-defined)
-   **Pitch:** Standard notation (C4, F#5, Gb3). C4 is Middle C.
-   **Duration:** `W` (Whole), `H` (Half), `Q` (Quarter), `E` (Eighth), `S` (Sixteenth). `.` can be appended for dotted notes (e.g., `Q.`).
-   **Velocity:** MIDI velocity (0-127).

**Note:** The LLM must be explicitly instructed and provided with examples of this format during the generation steps.

## Sectional Generation Strategy

-   **Process:** Generate each musical section (A, B, C...) via separate LLM prompts.
-   **Prompting:** Each prompt should include:
    -   The overall musical plan/context.
    -   Specific goals for the current section (length, musical function, relation to previous sections).
    -   The definition and examples of the compact symbolic format.
    -   The starting bar number for the section.
-   **Benefits:** Improves feasibility for longer pieces, allows finer control over musical development, encourages clear structural form.
-   **Challenges:** Requires careful prompt engineering to ensure smooth transitions and overall coherence between independently generated sections. Passing summaries or key features of the previous section in the prompt for the next section can help.

## Design Considerations & Challenges

-   **LLM Prompt Engineering:** Crafting effective prompts is critical for generating musically coherent and structurally sound sections that adhere to the symbolic format.
-   **Symbolic Format Robustness:** The format needs to balance expressiveness (capturing necessary musical detail) with simplicity (for LLM generation and parsing). It might need extensions for more complex features (e.g., dynamics changes, articulations).
-   **Parsing Reliability:** The `music.py` script needs robust parsing logic to handle potential variations or minor errors in the LLM's symbolic output.
-   **Musical Cohesion:** Ensuring smooth transitions and consistent musical language across sections generated in separate steps remains a key challenge.

## Future Enhancements

-   **Enhanced Symbolic Format:** Add support for more musical nuances like gradual tempo/dynamic changes, articulations, pedal markings, etc.
-   **Improved State Management:** Implement more sophisticated ways to pass context between sectional generation prompts.
-   **Configuration File:** Manage settings like default tempo, key, time signature, and symbolic format definitions via a config file.
-   **Music Theory Constraints:** Explore adding rules or constraints to the LLM prompts or post-processing steps to enforce basic music theory principles.

# TODO
