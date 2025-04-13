# AutoMusic Generator

This project explores the generation of MIDI music from textual descriptions using Large Language Models (LLMs). It employs a pipeline that translates high-level musical ideas into a compact symbolic representation, generates music section by section for better coherence, and finally converts this representation into a standard MIDI file using a Python script.

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
-   **Evaluation:** Assessing the musical quality of the generated output is subjective and requires careful listening or potentially computational analysis.

## Future Enhancements

-   **Automated LLM Interaction:** Integrate LLM API calls directly into the Python script to streamline the generation process.
-   **Enhanced Symbolic Format:** Add support for more musical nuances like gradual tempo/dynamic changes, articulations, pedal markings, etc.
-   **Improved State Management:** Implement more sophisticated ways to pass context between sectional generation prompts.
-   **Configuration File:** Manage settings like default tempo, key, time signature, and symbolic format definitions via a config file.
-   **User Interface:** Develop a simple GUI or web interface for easier interaction.
-   **Error Handling:** Implement more comprehensive error checking and reporting in the parser.
-   **Music Theory Constraints:** Explore adding rules or constraints to the LLM prompts or post-processing steps to enforce basic music theory principles.


# TODO

- generate section descriptions by LLM as well, do not use hardcoded ones
- add more instruments, examples:

K:Ddor
T:100
TS:4/4
INST:Pno
BAR:1
C:LH:[D3,A3]:Q:60
R:LH:Q
C:LH:[D3,A3,C4]:Q:60
R:LH:Q
N:RH:A4:Q:70
N:RH:C5:E:70
N:RH:D5:E:75
N:RH:C5:Q:70
R:RH:Q
BAR:2
C:LH:[D3,A3]:Q:60
R:LH:Q
C:LH:[D3,A3,C4]:Q:60
R:LH:Q
N:RH:A4:Q:70
N:RH:G4:E:65
N:RH:A4:E:70
N:RH:F4:H:65
BAR:3
C:LH:[G3,D4]:Q:62
R:LH:Q
C:LH:[G3,D4,F4]:Q:62
R:LH:Q
N:RH:C5:Q:72
N:RH:D5:E:72
N:RH:F5:E:77
N:RH:D5:Q:72
R:RH:Q
BAR:4
C:LH:[G3,D4]:Q:62
R:LH:Q
C:LH:[G3,D4,F4]:Q:62
R:LH:Q
N:RH:C5:Q:72
N:RH:A4:E:68
N:RH:C5:E:72
N:RH:Bb4:H:68
BAR:5
C:LH:[C3,G3]:Q:60
R:LH:Q
C:LH:[C3,G3,E4]:Q:60
R:LH:Q
N:RH:G4:Q:70
N:RH:E5:Q:75
N:RH:D5:H:70
BAR:6
C:LH:[C3,G3]:Q:60
R:LH:Q
C:LH:[C3,G3,E4]:Q:60
R:LH:Q
N:RH:C5:Q:70
N:RH:D5:E:75
N:RH:E5:E:75
N:RH:G4:Q:70
R:RH:Q
BAR:7
C:LH:[F3,C4]:Q:62
R:LH:Q
C:LH:[F3,C4,A4]:Q:62
R:LH:Q
N:RH:A4:Q:72
N:RH:C5:Q:77
N:RH:A4:H:72
BAR:8
C:LH:[F3,C4]:Q:62
R:LH:Q
C:LH:[F3,C4,A4]:Q:62
R:LH:Q
N:RH:G4:Q:72
N:RH:A4:E:77
N:RH:C5:E:77
N:RH:F4:H:72
BAR:9
C:LH:[D3,A3]:Q:60
R:LH:Q
C:LH:[D3,A3,C4]:Q:60
R:LH:Q
N:RH:A4:Q:70
N:RH:C5:E:70
N:RH:D5:E:75
N:RH:C5:Q:70
R:RH:Q
BAR:10
C:LH:[D3,A3]:Q:60
R:LH:Q
C:LH:[D3,A3,C4]:Q:60
R:LH:Q
N:RH:A4:H:70
N:RH:F4:H:65
BAR:11
C:LH:[G3,D4]:Q:62
R:LH:Q
C:LH:[G3,D4,F4]:Q:62
R:LH:Q
N:RH:C5:Q:72
N:RH:D5:E:72
N:RH:F5:E:77
N:RH:D5:H:72
BAR:12
C:LH:[G3,D4]:Q:62
R:LH:Q
C:LH:[G3,D4,F4]:Q:62
R:LH:Q
N:RH:C5:H:72
N:RH:A4:H:68
BAR:13
C:LH:[A3,E4]:Q:64
R:LH:Q
C:LH:[A3,E4,G4]:Q:64
R:LH:Q
N:RH:E5:Q:74
N:RH:G5:Q:78
N:RH:E5:H:74
BAR:14
C:LH:[A3,E4]:Q:64
R:LH:Q
C:LH:[A3,E4,G4]:Q:64
R:LH:Q
N:RH:D5:Q:74
N:RH:C5:E:70
N:RH:A4:E:70
N:RH:C5:H:74
BAR:15
C:LH:[D3,A3]:Q:60
R:LH:Q
C:LH:[D3,A3,F4]:Q:60
R:LH:Q
N:RH:D5:H:75
N:RH:A4:H:65
BAR:16
C:LH:[D3,A3,F4]:H.:60
R:LH:Q
N:RH:D4:W:60
INST:PnoLH
INST:PnoRH
INST:SynPad
INST:SynLead
BAR:17
C:SynPad:[D3,F3,A3,C4]:W:60
N:PnoLH:D3:H:65
N:PnoLH:A3:H:65
R:PnoRH:Q
C:PnoRH:[F4,A4,C5]:Q:65
C:PnoRH:[F4,A4,C5]:H:65
R:SynLead:W
BAR:18
C:SynPad:[G3,Bb3,D4,F4]:W:60
N:PnoLH:G3:H:65
N:PnoLH:D4:H:65
R:PnoRH:Q
C:PnoRH:[G4,Bb4,D5]:Q:65
C:PnoRH:[G4,Bb4,D5]:H:65
R:SynLead:W
BAR:19
C:SynPad:[C3,E3,G3,Bb3]:W:62
N:PnoLH:C3:H:65
N:PnoLH:G3:H:65
R:PnoRH:Q
C:PnoRH:[E4,G4,C5]:Q:65
C:PnoRH:[E4,G4,C5]:H:65
R:SynLead:W
BAR:20
C:SynPad:[F3,A3,C4,E4]:W:62
N:PnoLH:F3:H:65
N:PnoLH:C4:H:65
R:PnoRH:Q
C:PnoRH:[F4,A4,C5]:Q:65
C:PnoRH:[F4,A4,C5]:H:65
R:SynLead:W
BAR:21
C:SynPad:[D3,F3,A3,C4]:W:65
N:PnoLH:D3:Q:68
N:PnoLH:A3:Q:68
N:PnoLH:D3:Q:68
N:PnoLH:A3:Q:68
R:PnoRH:Q
N:PnoRH:A4:E:65
N:PnoRH:C5:E:65
N:PnoRH:D5:Q:70
R:PnoRH:Q
R:SynLead:H
N:SynLead:F5:Q:75
N:SynLead:E5:Q:75
BAR:22
C:SynPad:[G3,Bb3,D4,F4]:W:65
N:PnoLH:G3:Q:68
N:PnoLH:D4:Q:68
N:PnoLH:G3:Q:68
N:PnoLH:D4:Q:68
R:PnoRH:Q
N:PnoRH:Bb4:E:65
N:PnoRH:D5:E:65
N:PnoRH:F5:Q:70
R:PnoRH:Q
N:SynLead:G5:Q:75
N:SynLead:F5:Q:75
N:SynLead:D5:H:75
BAR:23
C:SynPad:[Bb2,D3,F3,A3]:W:68
N:PnoLH:Bb2:H:70
N:PnoLH:F3:H:70
R:PnoRH:Q
C:PnoRH:[D4,F4,Bb4]:Q:68
C:PnoRH:[D4,F4,Bb4]:H:68
N:SynLead:C5:Q:78
N:SynLead:D5:Q:78
N:SynLead:Bb4:H:78
BAR:24
C:SynPad:[A2,C#3,E3,G3]:W:68
N:PnoLH:A2:H:70
N:PnoLH:E3:H:70
R:PnoRH:Q
C:PnoRH:[C#4,E4,G4]:Q:70
C:PnoRH:[C#4,E4,G4]:H:70
N:SynLead:A4:Q:80
N:SynLead:G4:Q:80
N:SynLead:E4:H:80
BAR:25
C:SynPad:[D3,F3,A3,C4]:W:70
N:PnoLH:D3:Q:72
N:PnoLH:D3:E:72
N:PnoLH:D3:E:72
N:PnoLH:A3:Q:72
N:PnoLH:D3:Q:72
C:PnoRH:[F4,A4,D5]:Q:70
R:PnoRH:Q
C:PnoRH:[F4,A4,D5]:Q:70
R:PnoRH:Q
N:SynLead:F5:E:80
N:SynLead:G5:E:80
N:SynLead:A5:Q:82
N:SynLead:G5:Q:82
N:SynLead:F5:Q:80
BAR:26
C:SynPad:[G3,Bb3,D4,F4]:W:70
N:PnoLH:G3:Q:72
N:PnoLH:G3:E:72
N:PnoLH:G3:E:72
N:PnoLH:D4:Q:72
N:PnoLH:G3:Q:72
C:PnoRH:[G4,Bb4,D5]:Q:70
R:PnoRH:Q
C:PnoRH:[G4,Bb4,D5]:Q:70
R:PnoRH:Q
N:SynLead:E5:E:80
N:SynLead:F5:E:80
N:SynLead:G5:Q:82
N:SynLead:F5:Q:82
N:SynLead:D5:Q:80
BAR:27
C:SynPad:[C3,Eb3,G3,Bb3]:W:72
N:PnoLH:C3:H:72
N:PnoLH:G3:H:72
R:PnoRH:Q
C:PnoRH:[Eb4,G4,C5]:Q:72
C:PnoRH:[Eb4,G4,C5]:H:72
N:SynLead:Eb5:Q:85
N:SynLead:D5:Q:85
N:SynLead:C5:H:85
BAR:28
C:SynPad:[F3,A3,C4,Eb4]:W:72
N:PnoLH:F3:H:72
N:PnoLH:C4:H:72
R:PnoRH:Q
C:PnoRH:[F4,A4,C5]:Q:72
C:PnoRH:[F4,A4,C5]:H:72
N:SynLead:F5:Q:85
N:SynLead:Eb5:Q:85
N:SynLead:C5:H:85
BAR:29
C:SynPad:[Bb2,D3,F3,Ab3]:W:75
N:PnoLH:Bb2:Q:75
N:PnoLH:Bb2:E:75
N:PnoLH:Bb2:E:75
N:PnoLH:F3:Q:75
N:PnoLH:Bb2:Q:75
C:PnoRH:[D4,F4,Bb4]:Q:75
R:PnoRH:Q
C:PnoRH:[D4,F4,Bb4]:Q:75
R:PnoRH:Q
N:SynLead:G5:E:88
N:SynLead:F5:E:88
N:SynLead:Eb5:Q:88
N:SynLead:D5:Q:88
N:SynLead:F5:Q:86
BAR:30
C:SynPad:[Eb3,G3,Bb3,D4]:W:78
N:PnoLH:Eb3:Q:78
N:PnoLH:Eb3:E:78
N:PnoLH:Eb3:E:78
N:PnoLH:Bb3:Q:78
N:PnoLH:Eb3:Q:78
C:PnoRH:[G4,Bb4,Eb5]:Q:78
R:PnoRH:Q
C:PnoRH:[G4,Bb4,Eb5]:Q:78
R:PnoRH:Q
N:SynLead:G5:Q:90
N:SynLead:F5:Q:90
N:SynLead:Eb5:H:90
BAR:31
C:SynPad:[G3,Bb3,D4,F4]:W:75
N:PnoLH:G3:H:75
N:PnoLH:D4:H:75
R:PnoRH:Q
C:PnoRH:[G4,Bb4,D5]:Q:75
C:PnoRH:[G4,Bb4,D5]:H:75
N:SynLead:D5:Q:88
N:SynLead:F5:Q:88
N:SynLead:G5:H:88
BAR:32
C:SynPad:[A2,C#3,E3,G3]:H.:78
R:SynPad:Q
N:PnoLH:A2:H:78
N:PnoLH:E3:H:78
R:PnoRH:Q
C:PnoRH:[C#4,E4,G4]:Q.:78
N:PnoRH:A4:E:78
N:PnoRH:G4:Q:78
N:SynLead:A4:Q:85
N:SynLead:C5:Q:85
N:SynLead:B4:E:85
N:SynLead:A4:E:85
N:SynLead:G4:Q:85
INST:Pno
BAR:33
C:LH:[D3,A3]:H:60
R:LH:H
N:RH:D4:E:70
N:RH:F4:E:70
N:RH:A4:E:70
N:RH:G4:E:70
N:RH:F4:E:70
N:RH:A4:E:70
N:RH:C5:E:70
N:RH:A4:E:70
BAR:34
C:LH:[D3,A3]:H:60
R:LH:H
N:RH:D4:E:70
N:RH:F4:E:70
N:RH:A4:E:70
N:RH:G4:E:70
N:RH:F4:Q:70
N:RH:E4:Q:65
BAR:35
C:LH:[G3,D4]:H:60
R:LH:H
N:RH:G4:E:70
N:RH:Bb4:E:70
N:RH:D5:E:70
N:RH:C5:E:70
N:RH:Bb4:E:70
N:RH:D5:E:70
N:RH:F5:E:70
N:RH:D5:E:70
BAR:36
C:LH:[G3,D4]:H:60
R:LH:H
N:RH:G4:E:70
N:RH:Bb4:E:70
N:RH:D5:E:70
N:RH:C5:E:70
N:RH:Bb4:Q:70
N:RH:A4:Q:65
BAR:37
C:LH:[C3,G3]:H:60
R:LH:H
N:RH:C4:E:70
N:RH:E4:E:70
N:RH:G4:E:70
N:RH:F4:E:70
N:RH:E4:E:70
N:RH:G4:E:70
N:RH:A4:E:70
N:RH:G4:E:70
BAR:38
C:LH:[C3,G3,E4]:H:60
R:LH:H
N:RH:C4:E:70
N:RH:E4:E:70
N:RH:G4:E:70
N:RH:F4:E:70
N:RH:E4:Q:70
N:RH:D4:Q:65
BAR:39
C:LH:[A2,E3]:H:60
R:LH:H
N:RH:A3:E:70
N:RH:C4:E:70
N:RH:E4:E:70
N:RH:D4:E:70
N:RH:C4:E:70
N:RH:E4:E:70
N:RH:G4:E:70
N:RH:E4:E:70
BAR:40
C:LH:[D3,A3,D4]:H:65
R:LH:H
C:RH:[D4,F4,A4]:H:75
R:RH:H

