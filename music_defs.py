# -*- coding: utf-8 -*-
"""
Contains definitions, constants, and mappings related to the symbolic music format
and MIDI standards used in the MidiMaker project.
"""

import re

# --- Symbolic Format Definition (for prompts and parsing) ---
SYMBOLIC_FORMAT_DEFINITION = """
Use this compact symbolic format ONLY. Each command must be on a new line. Do NOT include comments after the command parameters on the same line.
- `T:<BPM>`: Sets the tempo in Beats Per Minute (e.g., T:120). Must be a number.
- `TS:<N>/<D>`: Sets the time signature (e.g., TS:4/4). Denominator should ideally be a power of 2.
- `K:<Key>`: Sets the key signature (e.g., K:Cmin, K:Gmaj, K:Ddor). Use standard `pretty_midi` key names (Major: maj, Minor: min, Modes: dor, phr, lyd, mix, loc, Ion=Maj, Aeo=Min).
- `BAR:<Num>`: Marks the beginning of a bar (measure), starting from 1, strictly sequential. Timing calculations rely on this.
- `N:<InstrumentName>:<Pitch>:<Duration>:<Velocity>`: Represents a single Note event. 
- `C:<InstrumentName>:<Pitches>:<Duration>:<Velocity>`: Represents a Chord event (multiple notes starting simultaneously).
- `R:<InstrumentName>:<Duration>`: Represents a Rest (silence) event for a specific track.

<InstrumentName>: Use lowercase instrument names from the provided list.
    - If <InstrumentName> is recognized as a drum track name (e.g., 'drums', 'drumkit', 'percussion', 'elecdrums', '808drums'), the `<Pitch>` will be interpreted as a drum sound name (see below), and it will use MIDI channel 10. Case-insensitive matching for drum track IDs.
    - If <InstrumentName> is NOT a recognized drum track name, it's considered a melodic track. 
PitchNames (`<Pitch>`):
    - For Melodic Tracks: Standard notation (e.g., C4, F#5, Gb3). Middle C is C4.
    - For Drum Tracks: Use drum sound names like Kick, Snare, HHC, HHO, Crash, Ride, HT, MT, LT (case-insensitive). See mapping below. Do NOT use standard notes (C4) on drum tracks.
DurationSymbols (`<Duration>`): W (Whole), H (Half), Q (Quarter), E (Eighth), S (Sixteenth), T (Thirty-second). Append '.' for dotted notes (e.g., Q., E.). Must be one of these symbols.
Velocity (`<Velocity>`): MIDI velocity (0-127). Must be a number. Affects loudness.
Pitches (`<Pitches>`): Comma-separated list of pitch names within square brackets (e.g., [C3,Eb3,G3]). For chords on drum tracks, list drum sound names. No spaces inside brackets unless part of a pitch name (shouldn't be).
Please follow the format strictly. Use one line per command. Do not include comments after the command parameters on the same line. Use the provided mappings for instrument names and drum sounds. Do not use any other symbols or characters outside of the specified format.

Example Note (Melodic): N:synlead:G5:E:95
Example Chord (Melodic): C:piano:[C3,Eb3,G3]:H:60
Example Rest: R:bass:W
Example Drum Note: N:drums:Kick:Q:95

Example Sequence:
K:Amin
T:90
TS:4/4
BAR:1
C:pad:[A3,C4,E4]:W:55
N:bass:A2:Q.:100
N:bass:E2:E:100
N:bass:A2:H:100
N:drums:Kick:Q:95
N:drums:HHC:E:80
"""

# --- Music Data Structures and Mappings ---

PITCH_MAP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
ACCIDENTAL_MAP = {"#": 1, "S": 1, "B": -1, "": 0}  # Allow S for sharp

# General MIDI Instrument Program Numbers (Expanded Selection) - Lowercase keys for lookup
INSTRUMENT_PROGRAM_MAP = {
    # Piano
    "pno": 0, "piano": 0, "acoustic grand piano": 0, "bright acoustic piano": 1,
    "electric grand piano": 2, "honky-tonk piano": 3, "electric piano 1": 4,
    "rhodes piano": 4, "electric piano 2": 5, "epiano": 4,
    "chorused piano": 5,
    "harpsichord": 6,
    "clavinet":7,
    
    # Chromatic Percussion
    "celesta": 8, "glockenspiel": 9, "music box": 10, "vibraphone": 11,
    "marimba": 12, "xylophone": 13, "tubular bells": 14, "dulcimer": 15,
    # Organ
    "org": 16, "organ": 16, "drawbar organ": 16, "percussive organ": 17,
    "rock organ": 18, "church organ": 19, "reed organ": 20, "accordion": 21,
    "harmonica": 22, "tango accordion": 23,
    # Guitar
    "gtr": 25, "guitar": 25, "acoustic guitar": 25, "nylon guitar": 24,
    "steel guitar": 25, "electric guitar": 27, "jazz guitar": 26,
    "clean electric guitar": 27, "muted electric guitar": 28,
    "overdriven guitar": 29, "distortion guitar": 30, "guitar harmonics": 31,
    # Bass
    "bass": 33, "acoustic bass": 32, "electric bass": 33, "finger bass": 33,
    "pick bass": 34, "fretless bass": 35, "slap bass": 36, "slap bass 2": 37, "synth bass": 38, "synth bass 1": 38,
    "synthbass": 38, "synth bass 2": 39,
    # Strings
    "str": 48, "strings": 48, "violin": 40, "viola": 41, "cello": 42,
    "contrabass": 43, "tremolo strings": 44, "pizzicato strings": 45,
    "orchestral harp": 46, "timpani": 47, "string ensemble 1": 48,
    "string ensemble 2": 49, "synth strings 1": 50, "synth strings 2": 51, 	"choir aahs": 52,
    "voice": 52, "voice oohs": 53, "synth voice": 54, "orchestra hit": 55,
    # Brass
    "tpt": 56, "trumpet": 56, "trombone": 57, "tuba": 58, "muted trumpet": 59,
    "french horn": 60, "brass section": 61, "synth brass 1": 62,
    "synth brass 2": 63,
    # Reed
    "sax": 65, "soprano sax": 64, "alto sax": 65, "tenor sax": 66,
    "baritone sax": 67, "oboe": 68, "english horn": 69, "bassoon": 70,
    "clarinet": 71,
    # Pipe
    "flt": 73, "flute": 73, "piccolo": 72, "recorder": 74, "pan flute": 75,
    "blown bottle": 76, "shakuhachi": 77, "whistle": 78, "ocarina": 79,
    # Synth Lead
    "synlead": 81, "synth lead": 81, "lead 1 (square)": 80, "lead 2 (sawtooth)": 81, "synthlead" : 81,
    "lead 3 (calliope)": 82, "lead 4 (chiff)": 83, "lead 5 (charang)": 84,
    "lead 6 (voice)": 85, "lead 7 (fifths)": 86, "lead 8 (bass + lead)": 87,
    # Synth Pad
    "synpad": 89, "synth pad": 89, "pad 1 (new age)": 88, "pad 2 (warm)": 89, "synthpad": 89,
    "pad 3 (polysynth)": 90, "pad 4 (choir)": 91, "pad 5 (bowed)": 92,
    "pad 6 (metallic)": 93, "pad 7 (halo)": 94, "pad 8 (sweep)": 95,
    # Synth FX
    "fx 1 (rain)": 96, "fx 2 (soundtrack)": 97, "fx 3 (crystal)": 98,
    "fx 4 (atmosphere)": 99, "fx 5 (brightness)": 100, "fx 6 (goblins)": 101,
    "fx 7 (echoes)": 102, "fx 8 (sci-fi)": 103,
    # Ethnic
    "sitar": 104, "banjo": 105, "shamisen": 106, "koto": 107, "kalimba": 108,
    "bag pipe": 109, "fiddle": 110, "shanai": 111,
    # Percussive
    "tinkle bell": 112, "agogo": 113, "steel drums": 114, "woodblock": 115,
    "taiko drum": 116, "melodic tom": 117, "synth drum": 118, "reverse cymbal": 119,
    # Sound Effects
    "guitar fret noise": 120, "breath noise": 121, "seashore": 122, "bird tweet": 123,
    "telephone ring": 124, "helicopter": 125, "applause": 126, "gunshot": 127,
    # Arp (Mapped to a synth sound)
    "arp": 81, "arpeggiator": 81,
    # Drums are a special case (channel 10 / index 9) - Program 0 is conventional
    "drs": 0, "drums": 0, "drumkit": 0, "elecdrums": 0, "808drums": 0, "percussion": 0,
}

# Set of lowercase track IDs that should be treated as drum tracks
# Includes instrument names commonly used for drums AND common track IDs
DRUM_TRACK_IDS = {
    k.lower()
    for k, v in INSTRUMENT_PROGRAM_MAP.items()
    if v == 0 and ("dr" in k or "kit" in k or "perc" in k)
}
DRUM_TRACK_IDS.update(
    ["drums", "drum", "drumkit", "percussion", "elecdrums", "808drums"]
)

# Standard drum note map (MIDI channel 10 / index 9) - Keys MUST be lowercase for lookup (Expanded)
DRUM_PITCH_MAP = {
    # Bass Drum
    "kick": 36, "bd": 36, "bass drum 1": 36, "bass drum": 36, "acoustic bass drum": 35, "kick 2": 35,
    # Snare
    "snare": 38, "sd": 38, "acoustic snare": 38, "electric snare": 40,
    # Hi-Hat
    "hihatclosed": 42, "hhc": 42, "closed hi hat": 42, "closed hi-hat": 42,
    "hihatopen": 46, "hho": 46, "open hi hat": 46, "open hi-hat": 46,
    "hihatpedal": 44, "hhp": 44, "pedal hi hat": 44, "pedal hi-hat": 44,
    # Cymbals
    "crash": 49, "cr": 49, "crash cymbal 1": 49, "crash cymbal 2": 57,
    "ride": 51, "rd": 51, "ride cymbal 1": 51, "ride cymbal 2": 59,
    "ride bell": 53, "rb": 53, "splash cymbal": 55, "splash": 55,
    "chinese cymbal": 52, "chinese": 52, "reverse cymbal": 119,
    # Toms
    "high tom": 50, "ht": 50, "hi tom": 50, "mid tom": 47, "mt": 47,
    "hi-mid tom": 48, "low-mid tom": 47, "low tom": 43, "lt": 43,
    "high floor tom": 43, "low floor tom": 41, "floor tom": 41, "ft": 41,
    # Hand Percussion
    "rimshot": 37, "rs": 37, "side stick": 37, "clap": 39, "cp": 39, "hand clap": 39,
    "cowbell": 56, "cb": 56, "tambourine": 54, "tmb": 54, "vibraslap": 58,
    "high bongo": 60, "low bongo": 61, "mute high conga": 62, "open high conga": 63,
    "low conga": 64, "high timbale": 65, "low timbale": 66, "high agogo": 67,
    "low agogo": 68, "cabasa": 69, "maracas": 70, "short whistle": 71,
    "long whistle": 72, "short guiro": 73, "long guiro": 74, "claves": 75, "cl": 75,
    "high wood block": 76, "low wood block": 77, "mute cuica": 78, "open cuica": 79,
    "mute triangle": 80, "open triangle": 81, "shaker": 82,
}

# Basic validation pattern for Key Signatures (used in parser and pipeline)
KEY_SIGNATURE_PATTERN = re.compile(r"^[A-Ga-g][#sb]?(maj|min|dor|phr|lyd|mix|loc|aeo|ion)?$", re.IGNORECASE)

# Basic validation pattern for Time Signatures (used in parser and pipeline)
TIME_SIGNATURE_PATTERN = re.compile(r"^(\d+)\s*/\s*(\d+)$")

# Duration symbols map to relative quarter note values
DURATION_RELATIVE_MAP = {"W": 4.0, "H": 2.0, "Q": 1.0, "E": 0.5, "S": 0.25, "T": 0.125}

# Known Melodic Instrument Names (lowercase, derived from map)
KNOWN_MELODIC_INSTRUMENTS = sorted([k for k, v in INSTRUMENT_PROGRAM_MAP.items() if k not in DRUM_TRACK_IDS])

# Known Drum Sound Names (lowercase, derived from map)
KNOWN_DRUM_SOUNDS = sorted(list(DRUM_PITCH_MAP.keys()))