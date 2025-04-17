# -*- coding: utf-8 -*-
"""
Main script to generate MIDI music using the MidiMaker pipeline.

Orchestrates the steps:
1. Load configuration.
2. Configure LLM interface.
3. Enrich the initial music description.
4. Generate a section plan.
5. Generate symbolic music section by section.
6. Parse the combined symbolic text.
7. Create the final MIDI file.
"""

import datetime
import os

# Import modules from the MidiMaker package/directory
import config
import llm_interface
import pipeline
import symbolic_parser
import midi_generator

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting MidiMaker Generator Pipeline...")
    print(f"Using LLM Backend: {'OpenAI' if config.CONFIG.get('use_openai') else 'Gemini'}")

    # Configure API clients (Gemini and OpenAI)
    llm_interface.configure_llm_clients()

    # Step 1: Enrich Description
    enriched_description, structure_hint = pipeline.enrich_music_description(
        config.CONFIG["initial_description"]
    )
    # Use enriched description if available, otherwise fallback to initial
    overall_description_for_generation = enriched_description if enriched_description else config.CONFIG["initial_description"]

    # Step 2: Generate Section Plan
    section_plan = pipeline.generate_section_plan(overall_description_for_generation, structure_hint)
    if not section_plan:
        print("Exiting due to failure in generating section plan.")
        exit(1)

    # Step 3: Generate Symbolic Sections Iteratively
    all_symbolic_text = ""
    current_bar_count = 1
    last_section_summary_info = None # Stores {name, summary, key, tempo, time_sig}
    generated_sections_count = 0
    total_planned_bars = sum(info.get("bars", 0) for info in section_plan.values())

    print(
        f"\n--- Step 3 (Iterative): Generating {len(section_plan)} Sections ({total_planned_bars} planned bars) ---"
    )
    section_order = list(section_plan.keys()) # Get the planned order
    for section_name in section_order:
        section_info = section_plan[section_name]
        # Basic check for valid section info from plan
        if not isinstance(section_info.get("bars"), int) or section_info["bars"] <= 0:
            print(
                f"ERROR: Invalid 'bars' ({section_info.get('bars')}) for section {section_name} in plan. Skipping."
            )
            continue

        symbolic_section, current_section_summary_info = pipeline.generate_symbolic_section(
            overall_description_for_generation,
            section_plan,
            section_name,
            current_bar_count,
            last_section_summary_info, # Pass summary of the *previous* section
        )

        # Check if generation was successful (returned text and summary)
        if symbolic_section is not None and current_section_summary_info is not None:
            symbolic_section_cleaned = symbolic_section.strip()
            if symbolic_section_cleaned: # Only add if not empty
                all_symbolic_text += symbolic_section_cleaned + "\n\n" # Add newline separator
                generated_sections_count += 1
                last_section_summary_info = current_section_summary_info # Update summary for next iteration
                current_bar_count += section_info["bars"] # Advance bar counter
            else:
                print(
                    f"Warning: Section {section_name} generated empty symbolic text after validation. Skipping concatenation."
                )
                # Decide if we should stop or continue? Let's continue for now.
                # Update last_section_summary_info with placeholder if needed? Or keep the previous one?
                # Keeping the previous one seems safer for context passing.
        else:
            print(
                f"ERROR: Failed to generate or validate section {section_name}. Stopping generation process."
            )
            print("Attempting to proceed with previously generated sections...")
            break # Stop generating more sections on critical failure

    # --- Post-Generation Processing ---
    if not all_symbolic_text.strip():
        print("\nERROR: No symbolic text was generated successfully. Cannot proceed.")
        exit(1)
    if generated_sections_count < len(section_plan):
        print(
            f"\nWarning: Only {generated_sections_count}/{len(section_plan)} sections were generated successfully."
        )

    # Save the combined symbolic text
    print("\n--- Saving Combined Symbolic Text ---")
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    symbolic_filename = os.path.join(
        config.CONFIG["output_dir"], f"symbolic_music_{timestamp_str}.txt"
    )
    os.makedirs(config.CONFIG["output_dir"], exist_ok=True)
    try:
        with open(symbolic_filename, "w", encoding="utf-8") as f:
            f.write(all_symbolic_text)
        print(f"Saved combined symbolic text to: {symbolic_filename}")
    except IOError as e:
        print(f"Error saving symbolic text file: {e}")
    print("-" * 36)

    # Step 4: Parse Symbolic Text
    (
        parsed_notes,
        instrument_definitions,
        tempo_changes,
        time_sig_changes,
        key_sig_changes,
        estimated_duration,
        final_key, # Get final state from parser
        final_tempo,
    ) = symbolic_parser.parse_symbolic_to_structured_data(all_symbolic_text)

    # Step 5: Create MIDI File
    if parsed_notes and instrument_definitions:
        output_filename = f"generated_music_{timestamp_str}.mid"
        midi_generator.create_midi_file(
            parsed_notes,
            instrument_definitions,
            tempo_changes,
            time_sig_changes,
            key_sig_changes,
            output_filename,
        )
    else:
        print("\nError: No valid notes or instruments parsed. MIDI file not created.")

    print("\n--- MidiMaker Generator Pipeline Finished ---")