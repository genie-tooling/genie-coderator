# Filename: stages/phase_2_build.py
# -*- coding: utf-8 -*-
# Description: Stage 2 - Phased Build and Debugging Logic. (Testing Decoupled)
# Version: 1.3.0 (Uses Explicit File Targeting from Plan)

import logging
import argparse
import json
import re
import os
import time
from typing import Dict, Any, Optional, Tuple, List, Set, Union

# Relative imports for sibling modules
from .. import config
from .. import models
from .. import prompts
from .. import llm_interface
from .. import file_utils
# from .. import testing # Removed testing import
from .. import utils

logger = logging.getLogger(__name__)

# File path matching regex (Fallback Only)
FILE_PATH_REGEX = re.compile(
    r'([\.\w\\/_-]+\.(?:' +
    '|'.join(re.escape(ext) for ext in config.LANGUAGE_EXTENSIONS.values()) +
    r'|json|yaml|yml|toml|ini|cfg|md|txt|dockerfile|makefile' +
    r'))\b',
    re.IGNORECASE
)
# Component name regex (Fallback Only)
COMPONENT_NAME_REGEX = re.compile(r'\b(?:class|function|method|component|module)\s+`?([\w\.]+)`?', re.IGNORECASE)


def _parse_phase_description_for_targets(
    phase_description: str,
    phase_target_mapping: Dict[str, List[str]],
    design: models.ProjectStructureOutput
) -> Set[str]:
    """
    Identifies target file paths for a build phase.

    PRIORITY 1: Uses the explicit mapping from the planning phase.
    FALLBACK: Uses regex/description matching if mapping is unavailable/invalid.

    Args:
        phase_description: The string description of the current build phase.
        phase_target_mapping: The explicit mapping from the PhasingPlanOutput.
        design: The overall project structure definition (used for fallback validation).

    Returns:
        A set of unique relative file paths identified for this phase.
    """
    target_files: Set[str] = set()
    phase_desc_lower = phase_description.lower()
    design_file_paths = {os.path.normpath(f.path) for f in design.file_structure} # For validation

    # --- PRIORITY 1: Use Explicit Mapping ---
    mapped_files = phase_target_mapping.get(phase_description)
    if mapped_files:
        logger.debug(f"Targeting: Using explicit mapping for phase '{phase_description}'. Found: {mapped_files}")
        valid_mapped_files = []
        for file_path in mapped_files:
            norm_path = os.path.normpath(file_path)
            if norm_path in design_file_paths:
                valid_mapped_files.append(file_path) # Keep original case from mapping if valid
            else:
                logger.warning(f"Targeting: Mapped file '{file_path}' for phase '{phase_description}' not found in project design. Ignoring.")
        if valid_mapped_files:
            target_files.update(valid_mapped_files)
            logger.info(f"Targeting via explicit map: {', '.join(target_files)}")
            return target_files # Successfully used mapping
        else:
            logger.warning(f"Targeting: Explicit mapping for phase '{phase_description}' yielded no valid files found in design. Proceeding to fallback.")
    else:
        logger.warning(f"Targeting: No explicit mapping found for phase '{phase_description}'. Proceeding to fallback.")

    # --- FALLBACK: Regex/Description Matching ---
    logger.info(f"Targeting: Using fallback regex/description matching for phase '{phase_description}'.")

    # 1. Find directly mentioned file paths in phase description
    found_paths = FILE_PATH_REGEX.findall(phase_description)
    for path in found_paths:
        normalized_path = os.path.normpath(path.strip('\'"` '))
        # Find original casing from design if matched
        for design_path_norm in design_file_paths:
             if design_path_norm == normalized_path:
                 # Find the FileStructure object to get the original path string
                 for file_def in design.file_structure:
                     if os.path.normpath(file_def.path) == design_path_norm:
                         target_files.add(file_def.path)
                         logger.debug(f"Targeting (Fallback): Found direct path reference '{file_def.path}'.")
                         break
                 break # Found match in design_file_paths

    # 2. Find mentioned component names and map to files (Optional Enhancement)
    found_components = COMPONENT_NAME_REGEX.findall(phase_description)
    if found_components:
        logger.debug(f"Targeting (Fallback): Found potential component names: {found_components}")
        for component_name in found_components:
            component_name_lower = component_name.lower()
            for file_def in design.file_structure:
                file_added = False
                # Check class names
                for class_def in file_def.classes:
                    if class_def.name.lower() == component_name_lower:
                        if file_def.path not in target_files:
                            target_files.add(file_def.path)
                            logger.debug(f"Targeting (Fallback): Mapped component '{component_name}' to file '{file_def.path}'.")
                            file_added = True
                        break # Found component in this file
                if file_added: continue # Already added file

                # Check function names
                for func_def in file_def.functions:
                    if func_def.name.lower() == component_name_lower:
                         if file_def.path not in target_files:
                             target_files.add(file_def.path)
                             logger.debug(f"Targeting (Fallback): Mapped component '{component_name}' to file '{file_def.path}'.")
                             file_added = True
                         break # Found component in this file
                if file_added: continue

                # Check method names (nested)
                for class_def in file_def.classes:
                    method_found_in_class = False
                    for method_def in class_def.methods:
                          # Check simple name match or qualified Class.method match
                          if method_def.name.lower() == component_name_lower or \
                             f"{class_def.name}.{method_def.name}".lower() == component_name_lower:
                              if file_def.path not in target_files:
                                   target_files.add(file_def.path)
                                   logger.debug(f"Targeting (Fallback): Mapped component '{component_name}' (method) to file '{file_def.path}'.")
                              method_found_in_class = True
                              break # Found component method in this class
                    if method_found_in_class:
                         file_added = True # Mark file as added/checked for this component
                         break # Move to next file_def
                if file_added: continue


    # 3. Fallback: General description match (Less reliable)
    if not target_files and not found_paths and not found_components: # Only if other methods failed
        logger.debug("Targeting (Fallback): No paths/components found. Using broad description matching.")
        for file_def in design.file_structure:
             path_match = file_def.path.lower() in phase_desc_lower
             desc_match = file_def.description and file_def.description.lower() in phase_desc_lower
             if path_match or desc_match:
                 if file_def.path not in target_files:
                      # Validate against design paths again (should be redundant but safe)
                      if os.path.normpath(file_def.path) in design_file_paths:
                          target_files.add(file_def.path)
                          logger.debug(f"Targeting (Fallback): Matched file '{file_def.path}' based on general description/path mention.")

    if target_files:
        logger.info(f"Targeting via fallback: {', '.join(target_files)}")
    else:
        logger.warning(f"Targeting: Could not identify any target files for phase '{phase_description}' using explicit map or fallback methods.")

    return target_files


# Corresponds to Plan v1.1 Section 3.D (Phased Build)
# Testing logic has been REMOVED and moved to orchestrator.py
# Test-failure driven debugging loop is also REMOVED.
def run_phase_2_build(args: argparse.Namespace,
                        final_requirements_json_str: Optional[str],
                        project_design_data: models.ProjectStructureOutput,
                        final_plan_data: models.PhasingPlanOutput,
                        initial_continuity_seed: Optional[str],
                        target_language: str) -> Tuple[bool, Set[str], Set[str]]:
    """
    Executes Phase 2: Phased Build/Debug loop for code generation.
    Aggregates dependencies. Writes final requirements file on success.
    Uses explicit file targeting from the plan.
    Testing is handled separately by the orchestrator.

    Args:
        args: Command line arguments (passed through, mainly for logging/flags if needed).
        final_requirements_json_str: Sanitized requirements JSON string (or placeholder if blueprint).
        project_design_data: The validated project structure.
        final_plan_data: The validated *aggregated* phasing plan (must include phase_target_mapping).
        initial_continuity_seed: Optional seed from planning.
        target_language: The target programming language.

    Returns:
        Tuple:
            - bool: True if all phases build successfully, False otherwise.
            - Set[str]: Aggregated pip requirements.
            - Set[str]: Aggregated system dependencies.
    """
    logger.info("\n" + "="*10 + " Phase 2: Phased Build & File Debugging (Using Explicit Targets) " + "="*10)
    current_continuity_seed = initial_continuity_seed or "N/A"
    overall_build_success = True
    requirements_filepath_abs = file_utils.get_requirements_filepath() # Needed for saving
    all_pip_requirements: Set[str] = set()
    all_system_dependencies: Set[str] = set()
    phase_failure_history: Dict[int, List[str]] = {} # Store history per phase

    # Validate that the plan includes the necessary mapping
    if not hasattr(final_plan_data, 'phase_target_mapping') or not final_plan_data.phase_target_mapping:
         logger.critical("FATAL: Build phase requires 'phase_target_mapping' in the detailed plan data. Aborting.")
         return False, set(), set()

    # Prepare context strings (handle potential None for reqs in blueprint mode)
    req_str_for_build = final_requirements_json_str or "{}"
    try:
        project_structure_json_str = project_design_data.model_dump_json()
        final_phase_plan_list_json_str = json.dumps(final_plan_data.final_phasing_plan)
    except Exception as e:
        logger.critical(f"FATAL: Failed to serialize context for build phase: {e}", exc_info=True)
        return False, set(), set() # Return failure status and empty dependency sets

    # Truncate shared context once for build phase
    logger.debug("Preparing and truncating shared context for build phase...")
    try:
        base_prompt_build_text = prompts.PROMPT_4_BUILD.format(
            final_requirements_json_str="", project_structure_json_str="",
            final_phase_plan_list_json_str="", current_part_description="",
            target_file_path="", current_file_content="",
            current_continuity_seed="", target_language=target_language
        )
        base_prompt_tokens = llm_interface.get_token_count(base_prompt_build_text)
        # Allocate slightly more buffer for build prompts which include code history
        remaining_tokens = config.MAX_CONTEXT_HISTORY_TOKENS - base_prompt_tokens - 2000

        if remaining_tokens <= 5000: # Increased threshold for warning
           logger.warning(f"Base build prompt ({base_prompt_tokens} tokens) leaves potentially insufficient space ({remaining_tokens} tokens) for context/history.")

        # Keep similar budget allocation, more for design
        design_budget = int(remaining_tokens * 0.5)
        plan_budget = int(remaining_tokens * 0.25)
        req_budget = remaining_tokens - design_budget - plan_budget
        logger.debug(f"Build Context Budget: Design={design_budget}, Plan={plan_budget}, Reqs={req_budget}")

        truncated_req_str = llm_interface.truncate_prompt_context(
            req_str_for_build, req_budget, "Build: Reqs JSON"
        )
        truncated_design_str = llm_interface.truncate_prompt_context(
            project_structure_json_str, design_budget, "Build: Design JSON"
        )
        truncated_plan_str = llm_interface.truncate_prompt_context(
            final_phase_plan_list_json_str, plan_budget, "Build: Plan List JSON"
        )
    except Exception as e:
        logger.critical(f"Failed to serialize/truncate shared context for build phase: {e}", exc_info=True)
        return False, set(), set()


    total_phases = len(final_plan_data.final_phasing_plan)

    # --- Loop Through Phases Defined in the Plan ---
    for phase_idx, current_part_description in enumerate(final_plan_data.final_phasing_plan):
        phase_number = phase_idx + 1
        logger.info(f"\n--- Starting Build Phase {phase_number}/{total_phases}: {current_part_description} ---")
        phase_succeeded = True # Assume success until failure
        phase_failure_history.setdefault(phase_number, []) # Initialize history for this phase

        # --- Identify Target Files using Explicit Map first ---
        target_files_in_phase: Set[str] = _parse_phase_description_for_targets(
            phase_description=current_part_description,
            phase_target_mapping=final_plan_data.phase_target_mapping, # Pass the explicit map
            design=project_design_data
        )

        if not target_files_in_phase:
            logger.warning(f"No valid target file(s) identified for phase '{current_part_description}'. Skipping code generation for this phase.")
            continue # Skip to next phase

        target_files_list = sorted(list(target_files_in_phase)) # Process in consistent order
        # logger.info(f"Target file(s) for Phase {phase_number}: {', '.join(target_files_list)}") # Logged by parser now

        phase_pip_reqs_delta: Set[str] = set()
        phase_sys_deps_delta: Set[str] = set()
        all_files_generated_ok_this_phase = True

        # --- Loop Through Files within the Phase (Generation/Debug Cycles) ---
        for target_file_rel_path in target_files_list:
            logger.info(f"\n-- Processing File: {target_file_rel_path} (Phase {phase_number}) --")
            file_cycle_succeeded = False
            file_specific_failure_history: List[str] = [] # History just for this file in this phase
            last_attempt_output_model: Optional[models.CodeOutput] = None
            last_llm_response_text: Optional[str] = None

            for cycle in range(1, config.MAX_CYCLES_PER_PHASE + 1):
                logger.info(f"-- File '{target_file_rel_path}', Generation Cycle {cycle}/{config.MAX_CYCLES_PER_PHASE} --")

                current_file_content = file_utils.read_code_artifact(target_file_rel_path) or ""
                # Increased truncation limit for build context
                max_file_content_len = 30000
                truncated_current_content = current_file_content
                if len(current_file_content) > max_file_content_len:
                     # Keep more from the end
                     chars_to_keep = int(max_file_content_len * 0.8)
                     chars_from_start = max_file_content_len - chars_to_keep
                     logger.warning(f"Truncating existing content of '{target_file_rel_path}' for prompt (keeping start {chars_from_start}, end {chars_to_keep}).")
                     truncated_current_content = f"{current_file_content[:chars_from_start]}\n\n# ... [Content Truncated] ...\n\n{current_file_content[-chars_to_keep:]}"


                placeholder_map: Dict[str, Any] = {
                    "final_requirements_json_str": truncated_req_str,
                    "project_structure_json_str": truncated_design_str,
                    "final_phase_plan_list_json_str": truncated_plan_str,
                    "current_part_description": current_part_description,
                    "target_file_path": target_file_rel_path,
                    "current_file_content": truncated_current_content,
                    "current_continuity_seed": current_continuity_seed,
                    "target_language": target_language,
                }

                prompt_template = None
                prompt_identifier = ""
                if cycle == 1:
                    prompt_template = prompts.PROMPT_4_BUILD
                    prompt_identifier = f"BUILD_C{cycle}"
                else:
                    prompt_template = prompts.PROMPT_4_DEBUG
                    prompt_identifier = f"GEN_DEBUG_C{cycle}"
                    # Ensure previous attempt data is available for debug prompt
                    if last_attempt_output_model:
                        prev_code = last_attempt_output_model.solution_code
                        prev_test = last_attempt_output_model.test_code or "# No tests prev"
                        prev_reqs = last_attempt_output_model.requirements_content_delta or "# None prev"
                        prev_sys_deps = last_attempt_output_model.system_dependencies_delta or []
                    else: # Fallback if previous model invalid or missing
                         prev_code = last_llm_response_text or "# Error: Prev code unavailable (LLM Error?)"
                         prev_test = "# N/A"
                         prev_reqs = "# N/A"
                         prev_sys_deps = []

                    try: prev_sys_deps_json = json.dumps(prev_sys_deps)
                    except Exception: prev_sys_deps_json = "[]"

                    last_error = file_specific_failure_history[-1] if file_specific_failure_history else "N/A"

                    # Add detail to history for validation failures
                    if not last_attempt_output_model and last_llm_response_text:
                         failure_detail = f"LLM/Validation Error (C{cycle-1}):\n{last_error}\nRaw Response Snippet:\n{last_llm_response_text[:500]}..."
                         if not file_specific_failure_history or not file_specific_failure_history[-1].startswith("LLM/Validation Error"):
                             file_specific_failure_history.append(failure_detail)
                             last_error = failure_detail

                    placeholder_map.update({
                        "cycle_number": cycle,
                        "previous_cycle_number": cycle - 1,
                        "previous_solution_code": prev_code,
                        "previous_test_code": prev_test,
                        "previous_requirements_content_delta": prev_reqs,
                        "previous_system_dependencies_delta_json": prev_sys_deps_json,
                        "test_output_from_last_run": last_error, # Note: Legacy key name
                        "failure_history_formatted": utils.format_failure_history(file_specific_failure_history, "Generation/Validation"),
                    })

                try:
                    prompt_to_run = prompt_template.format(**placeholder_map)
                except KeyError as e:
                    logger.critical(f"FATAL: Prompt Key Error '{e}' prep prompt for '{target_file_rel_path}' {prompt_identifier}. Aborting.")
                    return False, all_pip_requirements, all_system_dependencies
                except Exception as e:
                    logger.critical(f"FATAL: Prompt Format Error for '{target_file_rel_path}' {prompt_identifier}: {e}", exc_info=True)
                    return False, all_pip_requirements, all_system_dependencies

                response_text = llm_interface.call_gemini_api(
                    prompt_to_run,
                    temperature=config.CODE_GENERATION_TEMPERATURE,
                    output_schema=models.CodeOutput
                )
                last_llm_response_text = response_text

                parsed_code_output: Optional[Union[models.CodeOutput, models.ErrorOutput]] = llm_interface.parse_llm_response(response_text, models.CodeOutput)

                if isinstance(parsed_code_output, models.ErrorOutput) or not parsed_code_output:
                    error_msg = getattr(parsed_code_output, 'error', f'LLM/Parsing/Validation Error for {target_file_rel_path}')
                    logger.error(f"LLM/Parse/Validate Failed for '{target_file_rel_path}' ({prompt_identifier}). Error: {error_msg}")
                    failure_reason = f"[{prompt_identifier}] LLM/Parse/Validate Error: {error_msg}"
                    file_specific_failure_history.append(failure_reason)
                    phase_failure_history[phase_number].append(f"{target_file_rel_path}: {failure_reason}")
                    last_attempt_output_model = None

                    if cycle < config.MAX_CYCLES_PER_PHASE:
                        logger.warning(f"Attempting generation debug cycle {cycle + 1} for '{target_file_rel_path}'.")
                        time.sleep(1)
                        continue
                    else:
                        logger.error(f"File '{target_file_rel_path}' failed: LLM/Parse/Validate error on final generation cycle {cycle}.")
                        all_files_generated_ok_this_phase = False
                        break
                else: # Successfully parsed CodeOutput
                    last_attempt_output_model = parsed_code_output

                    # Validate target_file_path consistency
                    if parsed_code_output.target_file_path != target_file_rel_path:
                         mismatch_error = f"LLM returned code for wrong file! Expected '{target_file_rel_path}', Got '{parsed_code_output.target_file_path}'."
                         logger.error(mismatch_error)
                         failure_reason = f"[{prompt_identifier}] LLM Error: {mismatch_error}"
                         file_specific_failure_history.append(failure_reason)
                         phase_failure_history[phase_number].append(f"{target_file_rel_path}: {failure_reason}")
                         last_attempt_output_model = None # Treat as invalid
                         if cycle < config.MAX_CYCLES_PER_PHASE:
                             logger.warning(f"Attempting generation debug cycle {cycle + 1} for '{target_file_rel_path}'.")
                             time.sleep(1)
                             continue
                         else:
                             logger.error(f"File '{target_file_rel_path}' failed: LLM wrong file path on final generation cycle.")
                             all_files_generated_ok_this_phase = False
                             break

                    # --- Save successful code and tests ---
                    if not file_utils.save_code_artifact(target_file_rel_path, parsed_code_output.solution_code):
                        logger.critical(f"FATAL: Failed to save code for '{target_file_rel_path}'. Aborting build.")
                        return False, all_pip_requirements, all_system_dependencies

                    if parsed_code_output.test_code and target_language == 'python':
                        # Basic logic to place test file alongside source or in parallel 'tests' dir
                        base_name, _ = os.path.splitext(os.path.basename(target_file_rel_path))
                        test_file_rel_path_parts = target_file_rel_path.split(os.path.sep)
                        if 'src' in test_file_rel_path_parts:
                             src_index = test_file_rel_path_parts.index('src')
                             test_file_rel_path = os.path.join("tests", *test_file_rel_path_parts[src_index+1:])
                        else:
                             # If no 'src', place in 'tests' mirroring structure from root
                             test_file_rel_path = os.path.join("tests", target_file_rel_path)
                        # Construct filename like test_*.py
                        test_file_rel_path = os.path.join(os.path.dirname(test_file_rel_path), f"test_{base_name}.py")
                        test_file_rel_path = os.path.normpath(test_file_rel_path)

                        try:
                            test_dir_abs = os.path.dirname(file_utils.get_sandbox_path(test_file_rel_path))
                            os.makedirs(test_dir_abs, exist_ok=True)
                            if not file_utils.save_code_artifact(test_file_rel_path, parsed_code_output.test_code):
                                logger.warning(f"Could not save test file '{test_file_rel_path}'.")
                            else:
                                logger.info(f"Saved/Updated test file: {test_file_rel_path}")
                        except Exception as e_test_save:
                            logger.error(f"Error saving test file '{test_file_rel_path}': {e_test_save}")

                    logger.info(f"-- File '{target_file_rel_path}' generated/updated successfully (Cycle {cycle}). --")
                    file_cycle_succeeded = True

                    # --- Aggregate dependencies from successful cycle ---
                    if parsed_code_output.requirements_content_delta:
                        for req_line in parsed_code_output.requirements_content_delta.strip().splitlines():
                            clean_line = req_line.strip()
                            if clean_line and not clean_line.startswith('#'):
                                phase_pip_reqs_delta.add(clean_line)
                    if parsed_code_output.system_dependencies_delta:
                         phase_sys_deps_delta.update(parsed_code_output.system_dependencies_delta)

                    break # Success for this file

            # --- End Generation Cycle Loop ---
            if not file_cycle_succeeded:
                logger.critical(f"Failed to generate/fix file '{target_file_rel_path}' after {config.MAX_CYCLES_PER_PHASE} generation cycles.")
                all_files_generated_ok_this_phase = False
                # Continue processing other files in the phase for now

        # --- End Loop Through Files in Phase ---

        # --- Check if Phase Succeeded (Generation Only) ---
        if not all_files_generated_ok_this_phase:
             logger.error(f"Build Phase {phase_number} ('{current_part_description}') failed: One or more files failed generation.")
             phase_succeeded = False
             # No testing block here anymore

        # --- Handle Phase Success or Failure ---
        if not phase_succeeded:
             logger.error(f"Build Phase {phase_number} ('{current_part_description}') failed generation checks.")
             overall_build_success = False
             logger.info("Aborting remaining build phases due to failure.")
             break # Stop processing subsequent phases
        else:
            # Update aggregated dependencies after successful phase generation
            all_pip_requirements.update(phase_pip_reqs_delta)
            all_system_dependencies.update(phase_sys_deps_delta)
            logger.info(f"--- Finished Build Phase {phase_number} Successfully ---")
            # Update continuity seed for the *next* phase
            current_continuity_seed = utils.generate_continuity_seed(project_design_data, final_plan_data)
            logger.debug(f"Generated continuity seed for next phase (truncated): {current_continuity_seed[:100]}...")

    # --- End Loop Through Phases ---

    # Final save of aggregated requirements if overall build succeeded
    # This is now crucial because testing happens *after* this function returns
    if overall_build_success:
         logger.info("Build process finished successfully. Saving final aggregated requirements...")
         try:
              req_content_final = "\n".join(sorted(list(all_pip_requirements)))
              if file_utils.save_code_artifact(config.REQUIREMENTS_FILENAME, req_content_final):
                   logger.info(f"Final aggregated requirements saved to '{requirements_filepath_abs}'.")
              else:
                   logger.error("Failed to save final aggregated requirements file. Testing may fail.")
                   # Should this make overall build fail? Arguably yes, as testing depends on it.
                   overall_build_success = False
         except Exception as e:
              logger.error(f"Error writing final aggregated requirements file: {e}. Testing may fail.")
              overall_build_success = False
    else:
         logger.warning("Build process failed. Final requirements file may be incomplete or inconsistent.")

    return overall_build_success, all_pip_requirements, all_system_dependencies
