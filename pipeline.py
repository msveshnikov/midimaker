# -*- coding: utf-8 -*-
"""
Contains the core pipeline functions for generating music:
- Enriching the initial description.
- Generating the section plan.
- Generating symbolic music for individual sections.
"""

import json
import re

# Import from local modules
import config
import llm_interface
import music_defs


def enrich_music_description(description):
    """Step 1: Use LLM to enrich the initial music description."""
    print("\n--- Step 1: Enriching Description ---")
    prompt = f"""
Analyze the following music description. Extract or infer the key signature (using standard notation like 'Amin', 'F#maj', 'Gdor'), tempo (BPM), time signature (N/D), and suggest primary instrumentation.
If any parameter is explicitly mentioned, use that. If not, infer plausible values based on the description (genre, mood, artist references, etc.).
Also, identify the core mood and suggest a musical structure (like AABA, ABAC, Verse-Chorus-Bridge) if not already provided.

Output the parameters clearly at the start using these exact prefixes:
K: <key_signature>
T: <tempo_bpm>
TS: <numerator>/<denominator>
Instruments: <instrument1>, <instrument2>, ...
Mood: <mood_description>
Structure: <structure_suggestion>

Follow this with a brief summary elaborating on the musical style based on your analysis.

Music Description: "{description}"

Enriched Output:
"""
    enriched_result = llm_interface.call_llm(prompt)

    current_key = config.CONFIG["default_key"]
    current_tempo = config.CONFIG["default_tempo"]
    current_timesig = config.CONFIG["default_timesig"]
    primary_instruments = [config.CONFIG["default_instrument_name"]]
    structure_hint = "AABA"  # Default fallback structure
    enriched_summary = f"Could not enrich description. Using defaults. Original: {description}"

    if enriched_result:
        print(f"LLM Enrichment Output:\n{enriched_result}\n")
        enriched_summary = enriched_result # Use the full output as summary unless parsing fails

        key_match = re.search(r"^[Kk]\s*:\s*(.+)$", enriched_result, re.MULTILINE | re.IGNORECASE)
        tempo_match = re.search(r"^[Tt]\s*:\s*(\d+(?:\.\d+)?)$", enriched_result, re.MULTILINE | re.IGNORECASE)
        ts_match = re.search(r"^[Tt][Ss]\s*:\s*(\d+)\s*/\s*(\d+)$", enriched_result, re.MULTILINE | re.IGNORECASE)
        inst_match = re.search(r"^[Ii][Nn][Ss][Tt]\s*:\s*(.+)$", enriched_result, re.MULTILINE | re.IGNORECASE)
        struct_match = re.search(r"^[Ss]tructure\s*:\s*([\w\-]+)$", enriched_result, re.MULTILINE | re.IGNORECASE)

        if key_match:
            potential_key = key_match.group(1).strip()
            if music_defs.KEY_SIGNATURE_PATTERN.match(potential_key):
                current_key = potential_key
                print(f"Updated Default Key: {current_key}")
            else: print(f"Warning: Extracted key '{potential_key}' format unclear. Using default: {current_key}")
        if tempo_match:
            try:
                tempo_val = int(float(tempo_match.group(1)))
                if 5 <= tempo_val <= 400:
                    current_tempo = tempo_val
                    print(f"Updated Default Tempo: {current_tempo}")
                else: print(f"Warning: Ignoring extracted tempo {tempo_val} (out of range 5-400).")
            except ValueError: print(f"Warning: Could not parse extracted tempo '{tempo_match.group(1)}'.")
        if ts_match:
            try:
                ts_num, ts_den = int(ts_match.group(1)), int(ts_match.group(2))
                if ts_num > 0 and ts_den > 0:
                    current_timesig = (ts_num, ts_den)
                    print(f"Updated Default Time Signature: {ts_num}/{ts_den}")
                else: print(f"Warning: Ignoring extracted time signature {ts_num}/{ts_den} (non-positive values).")
            except ValueError: print(f"Warning: Could not parse extracted time signature '{ts_match.group(1)}/{ts_match.group(2)}'.")
        if inst_match:
            instruments_str = inst_match.group(1).strip()
            potential_instruments = [inst.strip().lower() for inst in instruments_str.split(',') if inst.strip()]
            valid_instruments = [inst for inst in potential_instruments if inst in music_defs.INSTRUMENT_PROGRAM_MAP]
            if valid_instruments:
                primary_instruments = valid_instruments
                print(f"Identified Primary Instrument Hints: {', '.join(primary_instruments)}")
            else: print(f"Warning: No valid instruments found in extracted list: '{instruments_str}'.")
        if struct_match:
            structure_hint = struct_match.group(1).upper()
            print(f"Identified Structure Hint: {structure_hint}")

        # Update global defaults in config module based on enrichment
        config.CONFIG["default_key"] = current_key
        config.CONFIG["default_tempo"] = current_tempo
        config.CONFIG["default_timesig"] = current_timesig

        return enriched_summary, structure_hint
    else:
        print("Failed to enrich description. Using initial description and defaults.")
        return description, structure_hint


def generate_section_plan(enriched_desc, structure_hint):
    """Step 2: Use LLM to generate a detailed section plan."""
    print("\n--- Step 2: Generating Section Plan ---")
    prompt = f"""
Based on the following enriched music description and suggested structure, create a detailed plan for the sections.
The plan MUST be a valid JSON object where keys are section names (e.g., "Intro", "A1", "B", "A2", "Bridge", "Outro") and values are objects containing:
1. "bars": An integer number of bars for the section (strictly between {config.CONFIG['min_section_bars']} and {config.CONFIG['max_section_bars']}).
2. "goal": A concise string describing the musical purpose or content of the section (e.g., "Introduce main theme softly on SynPad", "Develop theme with variation, add SynthBass and Drums", "Contrasting bridge section, new chords on Electric Piano", "Return to main theme, fuller texture with Arp", "Fade out main elements").

The total number of bars across all sections should ideally be around {config.CONFIG['max_total_bars'] // 2} to {config.CONFIG['max_total_bars']}, but strictly MUST NOT exceed {config.CONFIG['max_total_bars']}.
Use the suggested structure "{structure_hint}" as a guide, adapting it as needed (e.g., AABA -> Intro A1 B A2 Outro). Ensure section names are unique and descriptive (e.g., A1, A2 instead of just A, A). Use standard section names where applicable (Intro, Verse, Chorus, Bridge, Solo, Outro, Build, Drop, Breakdown etc.).

Enriched Description:
{enriched_desc}

Generate ONLY the JSON plan now, starting with {{ and ending with }}. Do not include ```json markers or any other text.
"""

    plan_json = llm_interface.call_llm(prompt, output_format="json")

    if plan_json and isinstance(plan_json, dict):
        validated_plan = {}
        total_bars = 0
        section_order = list(plan_json.keys())

        for name in section_order:
            if not isinstance(name, str) or not name.strip():
                print(f"Warning: Invalid section name type '{type(name)}' or empty name. Skipping.")
                continue

            section_name = name.strip()
            info = plan_json.get(name)

            if not isinstance(info, dict):
                 print(f"Warning: Section '{section_name}' value is not a dictionary: {info}. Skipping.")
                 continue

            bars_val = info.get("bars")
            goal_val = info.get("goal")

            if not isinstance(bars_val, int):
                try: bars_val = int(bars_val)
                except (ValueError, TypeError): print(f"Warning: Invalid 'bars' type for section '{section_name}': {bars_val}. Skipping."); continue

            if not isinstance(goal_val, str) or not goal_val.strip():
                 print(f"Warning: Invalid or empty 'goal' for section '{section_name}': {goal_val}. Skipping.")
                 continue

            if config.CONFIG["min_section_bars"] <= bars_val <= config.CONFIG["max_section_bars"]:
                if total_bars + bars_val <= config.CONFIG["max_total_bars"]:
                    info["bars"] = bars_val
                    info["goal"] = goal_val.strip()[:250]
                    validated_plan[section_name] = info
                    total_bars += bars_val
                else:
                    print(f"Warning: Section '{section_name}' ({bars_val} bars) exceeds max total bars ({config.CONFIG['max_total_bars']}). Truncating plan.")
                    break
            else:
                print(f"Warning: Section '{section_name}' bars ({bars_val}) out of range ({config.CONFIG['min_section_bars']}-{config.CONFIG['max_section_bars']}). Skipping.")

        if not validated_plan:
            print("ERROR: Failed to generate a valid section plan after validation. Cannot proceed.")
            return None

        print("Generated Section Plan:")
        final_section_order = list(validated_plan.keys())
        for name in final_section_order:
            info = validated_plan[name]
            print(f"  - {name} ({info['bars']} bars): {info['goal']}")
        print(f"Total Bars in Plan: {total_bars}")
        return {name: validated_plan[name] for name in final_section_order}
    else:
        print("ERROR: Failed to generate or parse section plan JSON from LLM. Cannot proceed.")
        if isinstance(plan_json, str): print(f"LLM Output (expected JSON):\n{plan_json[:500]}...")
        return None


def generate_symbolic_section(
    overall_desc,
    section_plan,
    section_name,
    current_bar,
    previous_section_summary=None,
):
    """Step 3: Generate symbolic music for one section using LLM."""
    print(
        f"\n--- Step 3: Generating Symbolic Section: {section_name} (Starting Bar: {current_bar}) ---"
    )
    section_info = section_plan[section_name]
    bars = section_info["bars"]
    goal = section_info["goal"]

    context_prompt = ""
    if previous_section_summary:
        prev_name = previous_section_summary.get("name", "Previous")
        prev_summary = previous_section_summary.get("summary", "No summary available")
        prev_key = previous_section_summary.get("key", config.CONFIG["default_key"])
        prev_tempo = previous_section_summary.get("tempo", config.CONFIG["default_tempo"])
        prev_ts = previous_section_summary.get(
            "time_sig", f"{config.CONFIG['default_timesig'][0]}/{config.CONFIG['default_timesig'][1]}"
        )
        context_prompt = (
            f"Context from previous section ({prev_name}): {prev_summary}\n"
            f"It ended around key {prev_key}, tempo {prev_tempo} BPM, and time signature {prev_ts}.\n"
            "Ensure a smooth musical transition if appropriate for the overall structure and goals.\n"
        )

    default_tempo = config.CONFIG["default_tempo"]
    default_timesig = config.CONFIG["default_timesig"]
    default_key = config.CONFIG["default_key"]

    prompt = f"""
You are a precise symbolic music generator. Your task is to generate ONLY the symbolic music notation for a specific section of a piece, following the provided format strictly.

Overall Music Goal: {overall_desc}
Full Section Plan: {json.dumps(section_plan, indent=2)}
{context_prompt}
Current Section to Generate: {section_name}
Target Bars: {bars} (Start this section exactly at BAR:{current_bar}, end *before* BAR:{current_bar + bars})
Section Goal: {goal}

Instructions:
1. Generate music ONLY for this section, starting *exactly* with `BAR:{current_bar}` unless initial T, TS, or K commands are needed for this specific section start.
2. If tempo (T), time signature (TS), key (K) need to be set or changed *at the very beginning* of this section (time = start of BAR:{current_bar}), include those commands *before* the `BAR:{current_bar}` marker. Otherwise, assume they carry over from the previous section context or use defaults (T:{default_tempo}, TS:{default_timesig[0]}/{default_timesig[1]}, K:{default_key}). 
3. Strictly adhere to the compact symbolic format defined below. Output ONLY the commands, each on a new line.
4. DO NOT include any other text, explanations, apologies, section titles, comments (#), or formatting like ```mus``` or ```.
5. Ensure musical coherence within the section and try to achieve the Section Goal. Use appropriate instrumentation and musical ideas based on the goal and overall description.
6. The total duration of notes/rests/chords within each bar MUST add up precisely according to the active time signature (e.g., 4 quarter notes in 4/4, 6 eighth notes in 6/8). Be precise. Use rests (R:<Track>:<Duration>) to fill empty time accurately for each active track within a bar. Ensure parallel tracks are synchronized at bar lines.
7. End the generation cleanly *after* the content for bar {current_bar + bars - 1} is complete. Do NOT include `BAR:{current_bar + bars}`.
8. Instrument Names. Use ONLY lowercase names from this list: {", ".join(music_defs.KNOWN_MELODIC_INSTRUMENTS)}.
9. Drum Pitch Names (for N/C commands on drum tracks): Use ONLY names from this list (case-insensitive): {", ".join(music_defs.KNOWN_DRUM_SOUNDS)}.

{music_defs.SYMBOLIC_FORMAT_DEFINITION}

Generate Section {section_name} symbolic music now (starting BAR:{current_bar}):
"""

    symbolic_text = llm_interface.call_llm(prompt)

    if symbolic_text:
        symbolic_text = symbolic_text.strip()
        lines = symbolic_text.split("\n")
        meaningful_lines = [
            line.strip()
            for line in lines
            if line.strip() and not line.strip().startswith("#")
        ]

        if not meaningful_lines:
            print(f"Warning: Generated text for {section_name} appears empty or only contains comments.")
            return "", None

        bar_marker = f"BAR:{current_bar}"
        first_bar_marker_found = False
        first_event_line_index = -1
        initial_commands = []

        for idx, line in enumerate(meaningful_lines):
            if re.match(r"^(T:|TS:|K:|INST:)", line, re.IGNORECASE):
                 initial_commands.append(line)
                 if first_event_line_index == -1: first_event_line_index = idx
            elif line.startswith(bar_marker):
                 first_bar_marker_found = True
                 if first_event_line_index == -1: first_event_line_index = idx
                 break
            elif first_event_line_index == -1:
                 print(f"Warning: Section {section_name} generation started with unexpected command '{line}' before initial settings or '{bar_marker}'. Attempting to use.")
                 first_event_line_index = idx
                 if line.startswith("BAR:"):
                     print(f"ERROR: Section {section_name} started with wrong bar number '{line}'. Expected '{bar_marker}'. Discarding section.")
                     return "", None
                 break

        if first_event_line_index == -1:
             print(f"ERROR: No meaningful commands found in generated text for {section_name}. Discarding section.")
             return "", None

        if not first_bar_marker_found:
             if meaningful_lines[first_event_line_index].startswith(bar_marker):
                 first_bar_marker_found = True
             else:
                 print(f"ERROR: Generated text for {section_name} does not contain the required start marker '{bar_marker}'. Discarding section.")
                 print(f"Received text (first 500 chars):\n{symbolic_text[:500]}...")
                 return "", None

        validated_symbolic_lines = meaningful_lines[first_event_line_index:]
        validated_symbolic_text = "\n".join(validated_symbolic_lines)

        print(f"Validated symbolic text for Section {section_name} (first 300 chars):\n{validated_symbolic_text[:300]}...\n")

        summary_info = {
            "name": section_name,
            "summary": f"Generated {bars} bars targeting goal: {goal}.",
            "key": default_key,
            "tempo": default_tempo,
            "time_sig": f"{default_timesig[0]}/{default_timesig[1]}",
        }
        last_k, last_t, last_ts = (
            summary_info["key"], summary_info["tempo"], summary_info["time_sig"]
        )
        for line in reversed(validated_symbolic_lines):
            line = line.strip()
            if line.startswith("K:"): last_k = line.split(":", 1)[1].strip(); break
        for line in reversed(validated_symbolic_lines):
            line = line.strip()
            if line.startswith("T:"):
                try: last_t = float(line.split(":", 1)[1].strip()); break
                except ValueError: pass
        for line in reversed(validated_symbolic_lines):
            line = line.strip()
            if line.startswith("TS:"): last_ts = line.split(":", 1)[1].strip(); break
        summary_info["key"], summary_info["tempo"], summary_info["time_sig"] = (last_k, last_t, last_ts)

        return validated_symbolic_text, summary_info
    else:
        print(f"Failed to generate symbolic text for Section {section_name}.")
        return "", None