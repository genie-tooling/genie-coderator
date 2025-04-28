# Filename: orchestrator.py
# -*- coding: utf-8 -*-
# Description: Main orchestration logic for the Gemini Code Orchestrator.
# Version: 1.10.0 (Added State Management & Resumability T005)

import argparse
import logging
import os
import json
import re
from typing import List, Optional, Tuple, Dict, Any, Set, Union
import time


from . import config
from . import models
from . import file_utils
from . import utils
from . import testing # Import testing module directly
# Import stage functions using the package structure
from .stages import phase_0_summary
from .stages import phase_0_requirements
from .stages import phase_1a_design
from .stages import phase_1b_chunking
from .stages import phase_1c_planning
from .stages import phase_2_build # Renamed import
from .stages import phase_3_docs

logger = logging.getLogger(__name__)

# Define phase functions and their corresponding state indices
# Note: Phase 0 requires special handling due to blueprint mode split
PHASE_EXECUTORS = [
    # Phase Index 0: Requirements (Standard) / Summary (Blueprint)
    # Handled conditionally at the start
    # Phase Index 1: Design (Standard Gen / Blueprint Extract)
    phase_1a_design.run_phase_1a_extract_or_generate_design,
    # Phase Index 2: Plan Chunking
    phase_1b_chunking.run_phase_1b_plan_chunking,
    # Phase Index 3: Planning per Chunk
    phase_1c_planning.run_phase_1c_planning_per_chunk,
    # Phase Index 4: Build
    phase_2_build.run_phase_2_build,
    # Phase Index 5: Final Testing (handled separately after build)
    # Phase Index 6: Documentation
    phase_3_docs.run_phase_3_documentation,
]
PHASE_DESCRIPTIONS = [
    "Requirements/Summary", # Index 0
    "Design",               # Index 1
    "Plan Chunking",        # Index 2
    "Planning per Chunk",   # Index 3
    "Build",                # Index 4
    "Testing",              # Index 5 (Handled separately)
    "Documentation",        # Index 6
]

# --- Blueprint Detection (Keep in main orchestrator) ---
def detect_blueprint(supporting_docs: Optional[str]) -> bool:
    """Checks if the supporting_docs seem to contain a detailed blueprint."""
    if not supporting_docs:
        return False

    hits = 0
    # Use case-insensitive matching
    doc_lower = supporting_docs.lower()
    for indicator in config.BLUEPRINT_INDICATORS:
        if indicator.lower() in doc_lower:
            hits += 1
            # Check for phasing markers more thoroughly
            if indicator.startswith("#P"):
                 if re.search(r'#P\d+(\.\d+)?', supporting_docs): # Match #P1, #P3.5 etc.
                      # Allow a single phasing marker hit to count strongly
                      hits += 1

    logger.debug(f"Blueprint indicator check: Found {hits} hits (threshold: {config.BLUEPRINT_DETECTION_THRESHOLD}).")
    return hits >= config.BLUEPRINT_DETECTION_THRESHOLD

# --- State Saving Helper ---
def save_current_state(
    current_phase_index: int, # Index of the phase *about* to run (or just completed + 1)
    completed_descriptions: List[str],
    task_file: str,
    base_output_dir: str,
    code_dir: str,
    docs_dir: str,
    language: str,
    is_blueprint: bool,
    docs_req: Optional[List[str]],
    artifacts: Dict[str, Optional[str]],
    pip_reqs: Set[str],
    sys_deps: Set[str]
) -> bool:
    """Helper function to create and save the pipeline state."""
    # Task T005: Save state after successful phase completion
    state_to_save = models.PipelineState(
        current_phase_index=current_phase_index,
        completed_phase_descriptions=completed_descriptions,
        task_file_path=task_file,
        base_output_dir=base_output_dir,
        code_dir=code_dir,
        docs_dir=docs_dir,
        target_language=language,
        is_blueprint_mode=is_blueprint,
        docs_requested=docs_req,
        artifact_references=artifacts,
        aggregated_pip_requirements=pip_reqs,
        aggregated_system_dependencies=sys_deps,
    )
    if not file_utils.save_pipeline_state(state_to_save):
        logger.warning(f"Failed to save pipeline state after completing phase index {current_phase_index - 1}.")
        return False
    return True

# --- Main Orchestration Logic ---

def main(args: argparse.Namespace):
    """Main orchestration flow, executing stages sequentially by calling stage functions."""
    start_time = time.time()
    # Define script_version within main for its internal logging
    script_version = "1.10.0 (Added State Management & Resumability T005)"
    logger.info(f"Starting Gemini Code Orchestrator v{script_version} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"CLI Arguments: {vars(args)}")

    # --- Initialize State Variables ---
    loaded_state: Optional[models.PipelineState] = None
    start_phase_index: int = 0 # Default start
    # Artifacts and dependencies will be loaded/initialized based on resume status
    final_req_data: Optional[models.RequirementsOutput] = None
    final_req_json_str: Optional[str] = None
    extracted_summary_str: Optional[str] = None # Variable to hold summary
    project_design_data: Optional[models.ProjectStructureOutput] = None
    plan_chunk_data: Optional[models.PlanChunkOutput] = None
    final_plan_data: Optional[models.PhasingPlanOutput] = None
    initial_continuity_seed: Optional[str] = None
    aggregated_pip_reqs: Set[str] = set()
    aggregated_sys_deps: Set[str] = set()
    artifact_paths: Dict[str, Optional[str]] = { # Relative paths
        'requirements': None, 'summary': None, 'design': None,
        'chunk_list': None, 'plan': None, 'state': config.STATE_FILENAME # state filename is relative to base dir
    }
    target_language: str = 'python' # Default
    is_blueprint_mode: bool = False # Default
    initial_task_data: Optional[Dict[str, Any]] = None
    phase_times = {}
    final_status_code = 1 # Default to failure
    completed_phase_descriptions: List[str] = [] # Track completed phases

    # --- Setup & Potential Resume (T005) ---
    try:
        # Configure paths based on output_dir arg OR loaded state
        # Load state *before* configuring paths if resuming
        if args.resume:
            state_load_dir = args.output_dir or config.DEFAULT_BASE_OUTPUT_DIR
            loaded_state = file_utils.load_pipeline_state() # Uses configure_paths implicitly if base dir not set
            if loaded_state:
                logger.info(f"--- RESUMING RUN from Phase Index {loaded_state.current_phase_index} ---")
                # Restore state BEFORE configuring paths from args
                config.BASE_OUTPUT_DIR = loaded_state.base_output_dir
                config.CODE_DIR = loaded_state.code_dir
                config.DOCS_DIR = loaded_state.docs_dir
                args.output_dir = config.BASE_OUTPUT_DIR # Ensure args reflect loaded state
                args.task_file = loaded_state.task_file_path # Ensure args reflect loaded state
                args.language = loaded_state.target_language # Ensure args reflect loaded state
                args.docs = loaded_state.docs_requested # Ensure args reflect loaded state

                target_language = loaded_state.target_language
                is_blueprint_mode = loaded_state.is_blueprint_mode
                aggregated_pip_reqs = loaded_state.aggregated_pip_requirements
                aggregated_sys_deps = loaded_state.aggregated_system_dependencies
                artifact_paths = loaded_state.artifact_references
                start_phase_index = loaded_state.current_phase_index
                completed_phase_descriptions = loaded_state.completed_phase_descriptions

                logger.info(f"Restored State: Lang='{target_language}', Blueprint={is_blueprint_mode}, Output='{config.BASE_OUTPUT_DIR}'")
                logger.info(f"Resuming execution from: {PHASE_DESCRIPTIONS[start_phase_index] if start_phase_index < len(PHASE_DESCRIPTIONS) else 'End'}")
                logger.debug(f"Restored Artifacts: {artifact_paths}")
                logger.debug(f"Restored Pip Reqs: {aggregated_pip_reqs}")
                logger.debug(f"Restored Sys Deps: {aggregated_sys_deps}")

                # Re-load key artifact models needed for later phases
                if start_phase_index > 1 and artifact_paths.get('design'): # Need design for chunking onwards
                    project_design_data = file_utils.load_project_structure() # Assumes file exists
                    if not project_design_data: logger.warning("Could not reload Project Structure on resume.")
                if start_phase_index > 2 and artifact_paths.get('chunk_list'): # Need chunks for planning
                    plan_chunk_data = file_utils.load_plan_chunk_list()
                    if not plan_chunk_data: logger.warning("Could not reload Plan Chunk List on resume.")
                if start_phase_index > 3 and artifact_paths.get('plan'): # Need plan for build
                    final_plan_data = file_utils.load_detailed_plan()
                    if final_plan_data:
                        initial_continuity_seed = final_plan_data.initial_continuity_seed
                    else:
                        logger.warning("Could not reload Detailed Plan on resume.")
                # Reload requirements JSON string if needed (less critical for later stages, but possible)
                if start_phase_index > 0 and artifact_paths.get('requirements'):
                    final_req_json_str = file_utils.read_code_artifact(artifact_paths['requirements'])
                    if not final_req_json_str: logger.warning("Could not reload Requirements JSON on resume.")


            else:
                logger.warning("Resume requested (--resume) but state file not found or invalid. Starting a fresh run.")
                # Ensure paths are configured from args if state load failed
                file_utils.configure_paths(args.output_dir)
        else:
            # Not resuming, standard setup
            file_utils.configure_paths(args.output_dir)

        # Save conceptual plan reference (do this on fresh run or successful resume start)
        # Path is relative to sandbox
        conceptual_plan_path_rel = config.CONCEPTUAL_PLAN_FILENAME
        conceptual_plan_content = f"# Conceptual Plan: Orchestrator v{script_version}\n# Features: State Management (T005), Blueprint Mode, Stage Refactor, JSON MIME Type Request, Validation Retry (All Phases), Plan Chunking, Test Debug Loop (Removed post-build), Blueprint Summary Extraction, Decoupled Testing.\n"
        if not os.path.exists(file_utils.get_sandbox_path(conceptual_plan_path_rel)):
             if not file_utils.save_code_artifact(conceptual_plan_path_rel, conceptual_plan_content):
                 logger.warning("Could not save conceptual plan reference file.")

    except Exception as e:
        logger.critical(f"FATAL error during setup/resume: {e}", exc_info=True)
        return 1

    # Load task (always needed, path taken from args potentially updated by resume)
    initial_task_data = file_utils.load_yaml_task(args.task_file)
    if not initial_task_data:
        logger.critical(f"FATAL: Invalid or missing task file '{args.task_file}'. Exiting.")
        return 1

    # Determine language (if not resuming or state load failed)
    if not loaded_state:
        target_language = args.language or initial_task_data.get('language') or 'python'
        target_language = target_language.lower().strip()
        logger.info(f"Target language determined: '{target_language}'")
        if target_language not in config.LANGUAGE_EXTENSIONS:
            logger.warning(f"Target language '{target_language}' not in known extensions.")

        # Detect Blueprint (only on fresh run)
        is_blueprint_mode = detect_blueprint(initial_task_data.get('supporting_docs'))
        if is_blueprint_mode:
            logger.info("Blueprint indicators detected in supporting_docs. Running in Blueprint Extraction Mode.")
        else:
            logger.info("No significant blueprint indicators detected. Running in Standard Generation Mode.")

    # Log effective runtime configuration
    logger.info(f"Using Model: {config.GEMINI_MODEL_NAME}")
    logger.info(f"Code Temp: {config.CODE_GENERATION_TEMPERATURE}, Planning Temp: {config.PLANNING_TEMPERATURE}, Default Temp: {config.DEFAULT_API_TEMPERATURE or 'API Default'}")
    logger.info(f"Skip Human Validation: {args.skip_human_validation}, Skip Testing: {args.skip_testing}, Generate Docs: {args.docs if args.docs else 'No'}")
    logger.info(f"Max Gen Cycles/File: {config.MAX_CYCLES_PER_PHASE}, Max Test Debug/Phase: {config.MAX_TEST_DEBUG_ATTEMPTS}")
    logger.info(f"Resuming: {'Yes (Phase Idx ' + str(start_phase_index) + ')' if loaded_state else 'No'}")

    # --- Execute Phases (Conditional Workflow & Resume Logic) ---
    build_success: bool = True # Assume success if skipped or already done
    test_success: bool = True # Assume success if skipped or already done
    docs_success: bool = True # Assume success if skipped or already done

    current_phase_idx = 0 # Track the current phase being processed

    try:
        # --- Phase 0: Requirements / Summary (Index 0) ---
        current_phase_idx = 0
        phase_desc = PHASE_DESCRIPTIONS[current_phase_idx]
        if current_phase_idx >= start_phase_index:
            logger.info(f"\n>>> Executing Phase {current_phase_idx}: {phase_desc} <<<")
            phase_start_time = time.time()
            phase0_success = False
            if is_blueprint_mode:
                extracted_summary_str = phase_0_summary.run_phase_0_extract_summary(initial_task_data)
                phase0_success = extracted_summary_str is not None # Consider success if it runs without error
                if phase0_success: artifact_paths['summary'] = config.EXTRACTED_SUMMARY_FILENAME
                phase_times['0_Blueprint_Summary'] = time.time() - phase_start_time
            else: # Standard Mode
                final_req_data, final_req_json_str = phase_0_requirements.run_phase_0_requirements_and_sanitization(args, initial_task_data, target_language)
                phase0_success = final_req_data is not None and final_req_json_str is not None
                if phase0_success: artifact_paths['requirements'] = config.FINAL_SPEC_FILENAME
                phase_times['0_Requirements'] = time.time() - phase_start_time

            if not phase0_success:
                 logger.critical(f"Phase {current_phase_idx} ({phase_desc}) failed. Aborting.")
                 return 1
            completed_phase_descriptions.append(phase_desc)
            # Save state after Phase 0
            if not save_current_state(current_phase_idx + 1, completed_phase_descriptions, args.task_file, config.BASE_OUTPUT_DIR, config.CODE_DIR, config.DOCS_DIR, target_language, is_blueprint_mode, args.docs, artifact_paths, aggregated_pip_reqs, aggregated_sys_deps):
                 return 1 # Exit if state saving fails

        # --- Phase 1: Design (Index 1) ---
        current_phase_idx = 1
        phase_desc = PHASE_DESCRIPTIONS[current_phase_idx]
        if current_phase_idx >= start_phase_index:
            logger.info(f"\n>>> Executing Phase {current_phase_idx}: {phase_desc} <<<")
            phase_start_time = time.time()
            # Re-load input from file if resuming and data not already loaded
            if not final_req_json_str and not is_blueprint_mode and artifact_paths.get('requirements'):
                 final_req_json_str = file_utils.read_code_artifact(artifact_paths['requirements'])

            project_design_data = phase_1a_design.run_phase_1a_extract_or_generate_design(
                blueprint_mode=is_blueprint_mode,
                initial_task_data=initial_task_data,
                final_requirements_json_str=final_req_json_str, # Used only if not blueprint
                target_language=target_language
            )
            phase_times[f'{current_phase_idx}_{phase_desc}'] = time.time() - phase_start_time
            if not project_design_data:
                 logger.critical(f"Phase {current_phase_idx} ({phase_desc}) failed. Aborting.")
                 return 1
            artifact_paths['design'] = config.PROJECT_STRUCTURE_FILENAME
            completed_phase_descriptions.append(phase_desc)
            # Save state after Phase 1
            if not save_current_state(current_phase_idx + 1, completed_phase_descriptions, args.task_file, config.BASE_OUTPUT_DIR, config.CODE_DIR, config.DOCS_DIR, target_language, is_blueprint_mode, args.docs, artifact_paths, aggregated_pip_reqs, aggregated_sys_deps):
                 return 1

        # --- Phase 2: Plan Chunking (Index 2) ---
        current_phase_idx = 2
        phase_desc = PHASE_DESCRIPTIONS[current_phase_idx]
        if current_phase_idx >= start_phase_index:
            logger.info(f"\n>>> Executing Phase {current_phase_idx}: {phase_desc} <<<")
            phase_start_time = time.time()
            if not project_design_data and artifact_paths.get('design'):
                 project_design_data = file_utils.load_project_structure() # Reload if needed
            if not project_design_data:
                 logger.critical(f"Cannot run Phase {current_phase_idx} ({phase_desc}) without Project Design data. Aborting.")
                 return 1

            plan_chunk_data = phase_1b_chunking.run_phase_1b_plan_chunking(
                project_design_data=project_design_data,
                target_language=target_language
            )
            phase_times[f'{current_phase_idx}_{phase_desc}'] = time.time() - phase_start_time
            if not plan_chunk_data:
                logger.critical(f"Phase {current_phase_idx} ({phase_desc}) failed. Aborting.")
                return 1
            artifact_paths['chunk_list'] = config.PLAN_CHUNK_LIST_FILENAME
            completed_phase_descriptions.append(phase_desc)
            # Save state after Phase 2
            if not save_current_state(current_phase_idx + 1, completed_phase_descriptions, args.task_file, config.BASE_OUTPUT_DIR, config.CODE_DIR, config.DOCS_DIR, target_language, is_blueprint_mode, args.docs, artifact_paths, aggregated_pip_reqs, aggregated_sys_deps):
                 return 1

        # --- Phase 3: Planning per Chunk (Index 3) ---
        current_phase_idx = 3
        phase_desc = PHASE_DESCRIPTIONS[current_phase_idx]
        if current_phase_idx >= start_phase_index:
            logger.info(f"\n>>> Executing Phase {current_phase_idx}: {phase_desc} <<<")
            phase_start_time = time.time()
            # Reload inputs if needed
            if not project_design_data and artifact_paths.get('design'):
                 project_design_data = file_utils.load_project_structure()
            if not plan_chunk_data and artifact_paths.get('chunk_list'):
                 plan_chunk_data = file_utils.load_plan_chunk_list()
            if not final_req_json_str and not is_blueprint_mode and artifact_paths.get('requirements'):
                 final_req_json_str = file_utils.read_code_artifact(artifact_paths['requirements'])
            if not project_design_data or not plan_chunk_data:
                 logger.critical(f"Cannot run Phase {current_phase_idx} ({phase_desc}) without Design and Chunk List data. Aborting.")
                 return 1

            final_plan_data, initial_continuity_seed = phase_1c_planning.run_phase_1c_planning_per_chunk(
                plan_chunk_data=plan_chunk_data,
                project_design_data=project_design_data,
                final_requirements_json_str=final_req_json_str,
                target_language=target_language
            )
            phase_times[f'{current_phase_idx}_{phase_desc}'] = time.time() - phase_start_time
            if not final_plan_data:
                logger.critical(f"Phase {current_phase_idx} ({phase_desc}) failed. Aborting.")
                return 1
            artifact_paths['plan'] = config.DETAILED_PLAN_FILENAME
            completed_phase_descriptions.append(phase_desc)
            # Save state after Phase 3
            if not save_current_state(current_phase_idx + 1, completed_phase_descriptions, args.task_file, config.BASE_OUTPUT_DIR, config.CODE_DIR, config.DOCS_DIR, target_language, is_blueprint_mode, args.docs, artifact_paths, aggregated_pip_reqs, aggregated_sys_deps):
                 return 1

        # --- Phase 4: Build (Index 4) ---
        current_phase_idx = 4
        phase_desc = PHASE_DESCRIPTIONS[current_phase_idx]
        if current_phase_idx >= start_phase_index:
            logger.info(f"\n>>> Executing Phase {current_phase_idx}: {phase_desc} <<<")
            phase_start_time = time.time()
            # Reload inputs if needed
            if not project_design_data and artifact_paths.get('design'):
                 project_design_data = file_utils.load_project_structure()
            if not final_plan_data and artifact_paths.get('plan'):
                 final_plan_data = file_utils.load_detailed_plan()
                 if final_plan_data: initial_continuity_seed = final_plan_data.initial_continuity_seed
            if not final_req_json_str and not is_blueprint_mode and artifact_paths.get('requirements'):
                 final_req_json_str = file_utils.read_code_artifact(artifact_paths['requirements'])
            if not project_design_data or not final_plan_data:
                 logger.critical(f"Cannot run Phase {current_phase_idx} ({phase_desc}) without Design and Plan data. Aborting.")
                 return 1

            # Call the build function, capture success and aggregated dependencies
            build_success, phase_pip_reqs, phase_sys_deps = phase_2_build.run_phase_2_build(
                args, # Pass args through
                final_req_json_str,
                project_design_data,
                final_plan_data,
                initial_continuity_seed,
                target_language
            )
            # Update aggregated dependencies *during* the build phase run
            aggregated_pip_reqs.update(phase_pip_reqs)
            aggregated_sys_deps.update(phase_sys_deps)
            phase_times[f'{current_phase_idx}_{phase_desc}'] = time.time() - phase_start_time
            if not build_success:
                 logger.critical(f"Phase {current_phase_idx} ({phase_desc}) failed. Aborting subsequent steps.")
                 return 1 # Exit if build failed
            artifact_paths['requirements'] = config.REQUIREMENTS_FILENAME # Build saves this
            completed_phase_descriptions.append(phase_desc)
            # Save state after Phase 4
            if not save_current_state(current_phase_idx + 1, completed_phase_descriptions, args.task_file, config.BASE_OUTPUT_DIR, config.CODE_DIR, config.DOCS_DIR, target_language, is_blueprint_mode, args.docs, artifact_paths, aggregated_pip_reqs, aggregated_sys_deps):
                 return 1

        # --- Phase 5: Final Testing (Index 5 - Handled Separately) ---
        current_phase_idx = 5
        phase_desc = PHASE_DESCRIPTIONS[current_phase_idx]
        test_exit_code = 0 # Default to success (or skipped)
        test_output_log = "Testing Skipped or Not Applicable."
        if current_phase_idx >= start_phase_index:
            logger.info(f"\n>>> Executing Phase {current_phase_idx}: {phase_desc} <<<")
            phase_start_time = time.time()
            if not args.skip_testing and target_language == 'python':
                # Ensure requirements file path is correct
                requirements_filepath_abs = file_utils.get_requirements_filepath()
                if not os.path.exists(requirements_filepath_abs):
                     # Attempt to save again if missing (e.g., resuming before test phase)
                     logger.warning(f"Requirements file missing before testing ('{requirements_filepath_abs}'). Attempting to save aggregated.")
                     req_content_final = "\n".join(sorted(list(aggregated_pip_reqs)))
                     if not file_utils.save_code_artifact(config.REQUIREMENTS_FILENAME, req_content_final):
                         logger.critical("FATAL: Failed to save requirements file before testing! Aborting.")
                         return 1

                # Run tests
                test_exit_code, test_output_log = testing.run_tests_or_skip(
                    args,
                    target_language,
                    requirements_filepath_abs,
                    list(aggregated_sys_deps) # Pass final aggregated system deps
                )
                if test_exit_code == 0:
                    logger.info("✅ Final Tests PASSED (or skipped/no tests found).")
                    test_success = True
                else:
                    logger.error(f"❌ Final Tests FAILED (Exit Code: {test_exit_code}).")
                    logger.info(f"Test Output Log:\n{test_output_log}")
                    test_success = False
                    final_status_code = 1
                    logger.critical("Aborting due to test failures.")
                    return 1 # Stop if tests fail

            else: # Testing skipped
                logger.info("\n" + "="*10 + " Phase 5: Final Testing (Skipped) " + "="*10)
                if args.skip_testing:
                     logger.info("Reason: Testing explicitly skipped via --skip-testing flag.")
                elif target_language != 'python':
                     logger.info(f"Reason: Automated testing only implemented for Python, not '{target_language}'.")
                test_success = True # Mark as successful if skipped
            phase_times[f'{current_phase_idx}_{phase_desc}'] = time.time() - phase_start_time

            # Save state after Phase 5 (even if skipped, mark as complete)
            if test_success:
                completed_phase_descriptions.append(phase_desc)
                if not save_current_state(current_phase_idx + 1, completed_phase_descriptions, args.task_file, config.BASE_OUTPUT_DIR, config.CODE_DIR, config.DOCS_DIR, target_language, is_blueprint_mode, args.docs, artifact_paths, aggregated_pip_reqs, aggregated_sys_deps):
                     return 1

        # --- Phase 6: Documentation (Index 6) ---
        current_phase_idx = 6
        phase_desc = PHASE_DESCRIPTIONS[current_phase_idx]
        if current_phase_idx >= start_phase_index:
            logger.info(f"\n>>> Executing Phase {current_phase_idx}: {phase_desc} <<<")
            phase_start_time = time.time()
            if args.docs:
                # Only run docs if build and tests succeeded (or tests were skipped)
                if build_success and test_success:
                    # Reload required inputs if resuming at this phase
                    if not project_design_data and artifact_paths.get('design'):
                        project_design_data = file_utils.load_project_structure()
                    if not final_plan_data and artifact_paths.get('plan'):
                        final_plan_data = file_utils.load_detailed_plan()
                    if not final_req_json_str and not is_blueprint_mode and artifact_paths.get('requirements'):
                        final_req_json_str = file_utils.read_code_artifact(artifact_paths['requirements'])
                    if not project_design_data or not final_plan_data:
                        logger.error(f"Cannot run Phase {current_phase_idx} ({phase_desc}) without Design and Plan data. Skipping docs.")
                        docs_success = False
                    else:
                        docs_success = phase_3_docs.run_phase_3_documentation(
                            args,
                            final_req_json_str,
                            project_design_data,
                            final_plan_data,
                            target_language
                        )
                        if not docs_success:
                             logger.warning("Phase 3 (Documentation) reported errors, but proceeding.")
                             # Non-fatal for overall success, but noted
                else:
                    logger.warning("\n" + "="*10 + " Phase 6: Documentation Generation (Skipped due to Build/Test Failure) " + "="*10)
                    docs_success = True # Not a failure of docs phase itself if skipped due to upstream issues
            else:
                 # Skip message handled within phase_3_docs if args.docs is None
                 docs_success = True
            phase_times[f'{current_phase_idx}_{phase_desc}'] = time.time() - phase_start_time

            # Save state after Phase 6 (final state)
            if docs_success:
                completed_phase_descriptions.append(phase_desc)
                if not save_current_state(current_phase_idx + 1, completed_phase_descriptions, args.task_file, config.BASE_OUTPUT_DIR, config.CODE_DIR, config.DOCS_DIR, target_language, is_blueprint_mode, args.docs, artifact_paths, aggregated_pip_reqs, aggregated_sys_deps):
                    return 1 # Or maybe just warn?


        # If we reached here and build/test succeeded.
        final_status_code = 0 # Mark overall success

    except Exception as e:
        logger.critical(f"FATAL: Unhandled exception during phase execution: {type(e).__name__} - {e}", exc_info=True)
        # Attempt to save failure state
        save_current_state(current_phase_idx, completed_phase_descriptions, args.task_file, config.BASE_OUTPUT_DIR, config.CODE_DIR, config.DOCS_DIR, target_language, is_blueprint_mode, args.docs, artifact_paths, aggregated_pip_reqs, aggregated_sys_deps)
        return 1 # Return failure code

    # --- Final Summary ---
    total_time = time.time() - start_time
    logger.info("\n" + "="*20 + " Orchestration Finished " + "="*20)
    logger.info(f"Execution Mode: {'Blueprint Extraction' if is_blueprint_mode else 'Standard Generation'}")
    logger.info(f"Orchestrator Version: {script_version}")
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    logger.info("Phase Durations:")
    for name, duration in phase_times.items():
        logger.info(f"- {name}: {duration:.2f}s")

    # Final status reporting based on overall success code
    base_output_dir = config.BASE_OUTPUT_DIR # Should be set correctly now

    if final_status_code == 0:
        logger.info("✅ Orchestration Completed Successfully!")
        logger.info(f"Build Status: SUCCESSFUL")
        test_report = "PASSED" if test_success else ("SKIPPED" if args.skip_testing else "N/A")
        if not test_success and not args.skip_testing and target_language=='python': test_report = "FAILED (Check Logs)" # Should not happen if final_status_code is 0
        logger.info(f"Testing Status ({target_language}): {test_report}")
        logger.info(f"Sandbox Directory: {config.CODE_DIR}")
        # Update artifact paths before logging
        final_artifact_paths = artifact_paths.copy()
        final_artifact_paths['requirements'] = config.REQUIREMENTS_FILENAME
        final_artifact_paths['summary'] = config.EXTRACTED_SUMMARY_FILENAME if is_blueprint_mode else None
        final_artifact_paths['design'] = config.PROJECT_STRUCTURE_FILENAME
        final_artifact_paths['chunk_list'] = config.PLAN_CHUNK_LIST_FILENAME
        final_artifact_paths['plan'] = config.DETAILED_PLAN_FILENAME
        # conceptual plan is always saved
        final_artifact_paths['conceptual_plan'] = config.CONCEPTUAL_PLAN_FILENAME
        # Remove state file from this list
        final_artifact_paths.pop('state', None)


        logger.info("Key artifacts (relative to sandbox):")
        for name, rel_path in final_artifact_paths.items():
            if rel_path:
                 artifact_abs_path = os.path.join(config.CODE_DIR, rel_path)
                 if os.path.exists(artifact_abs_path):
                     logger.info(f"- {rel_path} ({name})")

        if args.docs:
             docs_dir_exists = os.path.exists(config.DOCS_DIR)
             docs_dir_has_content = docs_dir_exists and len(os.listdir(config.DOCS_DIR)) > 0

             if docs_success and docs_dir_has_content:
                 logger.info(f"Docs Directory: {config.DOCS_DIR}")
             elif docs_success:
                 logger.warning("Documentation requested, but no files generated/saved (check logs).")
             else:
                 logger.warning("Documentation requested, but errors occurred during generation (check logs).")
        logger.info(f"\nReview generated code/tests in sandbox.")
        logger.info(f"Base output directory: {base_output_dir}")
        logger.info(f"State file saved to: {file_utils.get_state_filepath()}")
    else:
        logger.error("❌ Orchestration Failed.")
        logger.error(f"Build Status: {'SUCCESSFUL' if build_success else 'FAILED'}")
        test_report = "FAILED (Check Logs)" if not test_success else ("SKIPPED" if args.skip_testing else "N/A")
        # Only report test status if relevant phase was reached
        if current_phase_idx >= 5:
             logger.error(f"Testing Status ({target_language}): {test_report}")
        logger.error(f"Failed during phase index {current_phase_idx}: {PHASE_DESCRIPTIONS[current_phase_idx]}")
        logger.error("Review logs for details on the failure point.")
        logger.info(f"Partial output may be in sandbox: {config.CODE_DIR}")
        logger.info(f"Base output directory: {base_output_dir}")
        logger.info(f"Partial state may be saved to: {file_utils.get_state_filepath()}")

    return final_status_code


# --- Script Entry Point ---
if __name__ == "__main__":

    # --- Determine script_version FIRST ---
    script_version = "1.10.0" # Default - Keep this simple or use the logic below

    # Fallback: Read version string from main() function definition in this file
    try:
        with open(__file__, 'r') as f:
            content = f.read()
            # Updated regex to find version within main()
            match = re.search(r'def main\(.*\):\s+.*\s+script_version = "([^"]+)"', content)
            if match:
                script_version = match.group(1)
            else: # Fallback to checking the version header in config.py if needed
                # Corrected regex to search within the read content, not config.py directly
                # Use config.py version as ultimate fallback if main() version not found
                match_config_header = re.search(r'^# Version: ([^\s]+)', content, re.MULTILINE) # Check orchestrator first
                if not match_config_header:
                     # If not in orchestrator, check config.py's version header
                     try:
                         import config as cfg_module
                         # Assuming config.py has '# Version: X.Y.Z' at the top
                         # This is fragile, reading the file might be better
                         # For simplicity, let's assume config.__version__ might exist or parse it
                         cfg_version_str = getattr(cfg_module, '__version__', None) # Ideal
                         if not cfg_version_str:
                             # Try parsing config.py content if possible (less ideal)
                             pass # Keep default for now if parsing is too complex here
                     except ImportError:
                         pass # Keep default if config can't be imported yet
                # else: use script_version from default

    except Exception:
        pass # Ignore errors reading self, keep default

    # --- Now Setup Argument Parser ---
    parser = argparse.ArgumentParser(
        # Use the determined script_version here
        description=f"Gemini Code Orchestrator v{script_version} - Phased code generation supporting standard tasks and blueprint extraction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--task-file", type=str, default=config.DEFAULT_TASK_FILE, help="Path to YAML task file.")
    parser.add_argument("--output-dir", type=str, default=None, help="Base directory for output (sandbox/docs/state subdirs). Defaults to CWD.")
    parser.add_argument("--language", type=str, default=None, help="Target language. Overrides task file. Defaults to 'python'.")
    parser.add_argument(
        "--docs", nargs='?', const=','.join(config.DEFAULT_DOC_TYPES), type=str, default=None,
        help=f"Generate docs. Optional comma-separated types (e.g., 'readme,api'). Defaults to: {','.join(config.DEFAULT_DOC_TYPES)}."
    )
    parser.add_argument("--skip-testing", action='store_true', help="Skip automated testing step after build phases.")
    parser.add_argument("--skip-human-validation", action='store_true', help="Skip interactive human review (Phase 0, Standard Mode only).")
    parser.add_argument("--resume", action='store_true', help="Attempt to resume from the last saved state found in the output directory.") # T005
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG level logging.")

    args = parser.parse_args()

    # Configure Logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Adjust logging level for noisy libraries if needed
    # logging.getLogger("google.api_core").setLevel(logging.WARNING)
    # logging.getLogger("urllib3").setLevel(logging.INFO)

    logger.info(f"Logging level set to {logging.getLevelName(log_level)}")

    # Post-process --docs argument
    if isinstance(args.docs, str):
        requested_docs = [d.strip().lower() for d in args.docs.split(',') if d.strip()]
        parsed_docs = []
        for doc in requested_docs:
            # Allow any requested doc type for flexibility
            if doc not in parsed_docs:
                 parsed_docs.append(doc)
        if not parsed_docs:
             logger.warning("No valid documentation types requested via --docs. Disabling.")
             args.docs = None
        else:
             args.docs = parsed_docs
             logger.info(f"Will attempt to generate docs: {', '.join(args.docs)}")


    # Check API Key LAST before calling main
    if not config.settings.gemini_api_key:
        logger.critical("FATAL: GEMINI_API_KEY not found. Set environment variable or .env file.")
        exit(1)
    else:
        logger.debug(f"Effective runtime arguments (before potential resume override): {vars(args)}")
        # Rebuild models involving ForwardRef (good practice)
        try:
            models.RequirementItem.model_rebuild()
        except Exception as e:
            logger.error(f"Error rebuilding Pydantic models: {e}")

        exit_code = main(args)
        exit(exit_code)
