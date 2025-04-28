# Filename: stages/phase_0_requirements.py
# -*- coding: utf-8 -*-
# Description: Stage 0 - Requirements Generation and Sanitization Logic (Standard Mode).
# Version: 1.0

import logging
import argparse
import json
from typing import Dict, Any, Optional, Tuple, Union

# Relative imports for sibling modules
from .. import config
from .. import models
from .. import prompts
from .. import llm_interface
from .. import file_utils
from .. import utils

logger = logging.getLogger(__name__)

# Corresponds to Plan v1.1 Section 3.B (Requirements Refinement & Sanitization)
# Used ONLY if blueprint is NOT detected.
def run_phase_0_requirements_and_sanitization(args: argparse.Namespace,
                                                initial_task_data: Dict[str, Any],
                                                target_language: str) -> Tuple[Optional[models.RequirementsOutput], Optional[str]]:
    """
    Executes Phase 0: Requirements generation, optional user review, LLM sanitization
    (WITHOUT structured output request due to $ref issues), and validation retry loop.
    Uses `args.skip_human_validation`. Implements Plan v1.1 Section 3.B + enhancements.

    Returns:
        Tuple: (Validated RequirementsOutput model, requirements as JSON string) or (None, None) on failure/abort.
    """
    logger.info("\n" + "="*10 + " Phase 0: Requirements Refinement & Sanitization (Standard Mode) " + "="*10)

    # Display human summary if provided in task YAML
    if initial_task_data.get('human_summary'):
        print(f"\n[INFO] Task Summary: {initial_task_data['human_summary']}\n")
    else:
        logger.info("No 'human_summary' provided in task file.")

    # State variables for the loop
    user_reviewed_text: Optional[str] = None # Holds text going into sanitization/correction
    current_llm_response_text: Optional[str] = None # Holds the latest raw text from LLM
    final_req_data: Optional[models.RequirementsOutput] = None # Stores the validated final output
    review_cycle_attempt = 0
    max_review_cycles = 3 # Max times to go through USER REVIEW -> SANITIZE/CORRECT
    last_validation_error_for_user: Optional[str] = None # Store error for user review prompt

    # Initial Generation (Prompt 1) - No structured output needed here usually
    logger.info("Generating initial requirements specification (Prompt 1)...")
    prompt1_text = prompts.PROMPT_1_REQUIREMENTS.format(
        user_task_description_from_yaml=initial_task_data['description'],
        supporting_docs_from_yaml=initial_task_data.get('supporting_docs', 'N/A'),
        target_language=target_language,
        docs_requested=bool(args.docs),
    )
    # Use standard API call for initial generation - NO SCHEMA
    current_llm_response_text = llm_interface.call_gemini_api(
        prompt1_text,
        temperature=config.DEFAULT_API_TEMPERATURE
        # output_schema=None # Explicitly None
    )
    # If initial generation fails completely, exit early
    if current_llm_response_text is None:
         logger.critical("FATAL: Initial requirements generation failed. Aborting.")
         return None, None
    # Check if LLM returned a structured error immediately
    parsed_initial = utils.safe_json_loads(current_llm_response_text)
    if isinstance(parsed_initial, dict) and "error" in parsed_initial:
        logger.critical(f"FATAL: Initial requirements generation returned error: {parsed_initial['error']}. Aborting.")
        return None, None

    user_reviewed_text = current_llm_response_text # Start with initial generation output

    # --- User Review -> Sanitize/Correct Loop ---
    while review_cycle_attempt < max_review_cycles:
        review_cycle_attempt += 1
        logger.info(f"\n--- Requirements Review/Sanitize Cycle {review_cycle_attempt}/{max_review_cycles} ---")
        final_req_data = None # Reset validation result for this cycle

        # 1. Interactive Review (Optional)
        if not args.skip_human_validation:
            logger.info("Proceeding with interactive human review...")
            # Pass the latest text (from initial gen or previous failed sanitize/correct)
            # Also pass the last validation error for context in the review prompt
            user_edited_text_from_review = file_utils.run_interactive_review(
                user_reviewed_text,
                config.REVIEW_FILE_PATH,
                failed_validation_info=last_validation_error_for_user # Pass error info
            )
            last_validation_error_for_user = None # Reset after showing it

            if user_edited_text_from_review is None:
                logger.critical("Interactive review aborted or failed. Exiting Phase 0.")
                return None, None
            user_reviewed_text = user_edited_text_from_review # Update text with user edits
            logger.info("Human review completed.")

            # Confirmation before proceeding with potentially costly LLM calls
            print("\n--- Review Confirmation ---")
            print(f"Content from review is ready for LLM processing (Sanitization/Correction).")
            try:
                user_choice = input("Proceed with LLM processing? (yes/no/edit): ").strip().lower()
            except EOFError:
                logger.warning("EOFError during confirmation. Assuming 'no'.")
                user_choice = 'no'

            if user_choice in ['yes', 'y']:
                pass # Proceed to sanitization/correction
            elif user_choice == 'edit':
                logger.info("User chose 'edit'. Restarting review cycle with current text.")
                # user_reviewed_text already holds the text to be re-edited
                continue # Go back to the start of the while loop for another review cycle
            else:
                logger.critical("Aborting: User declined LLM processing after review.")
                return None, None
        else:
            logger.info("Skipping human validation step (--skip-human-validation).")
            # Use the text directly from the previous step


        # 2. LLM Sanitization & Validation Loop (with Retries)
        # Use global max attempts config
        last_validation_error = "Initial sanitization attempt" # For correction prompt

        for val_attempt in range(config.MAX_VALIDATION_ATTEMPTS):
            logger.info(f"-- Processing Attempt {val_attempt + 1}/{config.MAX_VALIDATION_ATTEMPTS} (Review Cycle {review_cycle_attempt}) --")

            # Determine which prompt to use: Sanitize first, then Correct
            prompt_to_use_str = None
            api_temp = config.DEFAULT_API_TEMPERATURE # Default temp
            prompt_name = "" # For logging
            if val_attempt == 0:
                # First attempt in validation loop: Use Sanitization prompt
                prompt_name = "Prompt 1.5 (Sanitization)"
                logger.info(f"Running LLM {prompt_name}...")
                prompt_to_use_str = prompts.PROMPT_1_5_SANITIZATION.format(
                     user_edited_requirements_and_scenarios_text=user_reviewed_text, # Use latest reviewed text
                     target_language=target_language,
                     docs_requested=bool(args.docs),
                )
            else:
                # Subsequent attempts: Use Correction prompt
                prompt_name = "Prompt 1.6 (Correction)"
                logger.info(f"Running LLM {prompt_name}...")
                prompt_to_use_str = prompts.PROMPT_1_6_CORRECTION.format(
                    failed_json_output=current_llm_response_text, # Use the output that failed validation
                    validation_error_message=last_validation_error, # Pass the error from previous attempt
                    target_language=target_language
                )
                # Optionally use lower temp for correction
                api_temp = config.CODE_GENERATION_TEMPERATURE

            # Call LLM, indicate schema expected for JSON MIME type, but don't pass schema
            current_llm_response_text = llm_interface.call_gemini_api(
                prompt_to_use_str,
                temperature=api_temp,
                output_schema=models.RequirementsOutput # Indicate JSON expected
            )

            # Check for immediate API call failure
            if current_llm_response_text is None:
                # API call failed definitively after retries in call_gemini_api
                logger.error(f"LLM API call failed during {prompt_name}. Cannot proceed with validation.")
                last_validation_error = "LLM API call failed completely."
                # Break validation loop, proceed to check if user wants to retry review cycle
                break

            # Check if LLM returned structured error
            parsed_for_error = utils.safe_json_loads(current_llm_response_text)
            if isinstance(parsed_for_error, dict) and "error" in parsed_for_error:
                llm_error = parsed_for_error['error']
                logger.error(f"LLM returned a structured error during {prompt_name}: {llm_error}")
                last_validation_error = f"LLM Returned Error: {llm_error}"
                # If LLM returns its own error, don't retry validation, break inner loop
                break

            # Attempt to parse and validate the TEXT response using Pydantic
            logger.info("Parsing and validating LLM response against RequirementsOutput schema...")
            parsed_response: Optional[Union[models.RequirementsOutput, models.ErrorOutput]] = llm_interface.parse_llm_response(
                current_llm_response_text,
                models.RequirementsOutput # Use the full model for validation here
            )

            if not isinstance(parsed_response, models.ErrorOutput) and parsed_response:
                # Validation Succeeded!
                final_req_data = parsed_response
                logger.info(f"Requirements successfully processed and validated (Attempt {val_attempt + 1}).")
                break # Exit validation loop successfully
            else:
                # Validation failed or other parsing error
                # Extract the error message for the next correction attempt
                last_validation_error = getattr(parsed_response, 'error', f'Unknown validation/parsing error on attempt {val_attempt + 1}')
                logger.warning(f"Validation failed (Attempt {val_attempt + 1}/{config.MAX_VALIDATION_ATTEMPTS}). Error: {last_validation_error}")
                logger.debug(f"Raw Response causing validation error:\n{current_llm_response_text[:1000]}...") # Log more context
                # Loop continues to the correction attempt if val_attempt < config.MAX_VALIDATION_ATTEMPTS - 1

        # --- End of Inner Validation Loop ---

        # Check if validation succeeded within the inner loop
        if final_req_data:
            # Validation was successful for this review cycle
            proceed_to_next_phase = False
            if not args.skip_human_validation:
                # Final confirmation if interactive
                print("\n--- Sanitization/Correction Complete ---")
                print("Final specification generated and validated.")
                try:
                     print("Example Requirement: ", final_req_data.requirements[0].text[:100] + "..." if final_req_data.requirements else "N/A")
                     final_confirm = input("Proceed to Planning/Design (Phase 1)? (yes/no/edit): ").strip().lower()
                except EOFError:
                     logger.warning("EOFError during final confirmation. Assuming 'no'.")
                     final_confirm = 'no'

                if final_confirm in ['yes', 'y']:
                    proceed_to_next_phase = True
                elif final_confirm == 'edit':
                     logger.info("User chose 'edit'. Restarting review cycle with the *validated* output.")
                     try:
                         # Serialize the successful data for the next review
                         user_reviewed_text = final_req_data.model_dump_json(indent=2)
                         continue # Go back to start of outer review cycle loop
                     except Exception as dump_err:
                         logger.error(f"Failed to serialize validated requirements for editing: {dump_err}")
                         user_reviewed_text = json.dumps({"error": "Serialization error for review."})
                         continue # Go back to review cycle with error text
                else:
                    logger.critical("Aborting: User declined to proceed after final review.")
                    return None, None # Abort orchestration
            else:
                # Automatic progression if skipping validation
                logger.info("Automatically proceeding to Planning/Design phase.")
                proceed_to_next_phase = True

            if proceed_to_next_phase:
                try:
                    # Serialize final data for saving and return
                    final_req_json_str_save = final_req_data.model_dump_json(indent=2)
                    final_req_json_str_return = final_req_data.model_dump_json() # Non-indented

                    # Save artifact
                    spec_path_relative = config.FINAL_SPEC_FILENAME
                    if file_utils.save_code_artifact(spec_path_relative, final_req_json_str_save):
                        logger.info(f"Final specification saved to sandbox: {spec_path_relative}")
                    else:
                        logger.warning(f"Could not save final specification file '{spec_path_relative}'.")

                    # Return the successful result
                    return final_req_data, final_req_json_str_return

                except Exception as e:
                    logger.critical(f"FATAL: Failed to serialize final requirements: {e}.", exc_info=True)
                    return None, None

        else:
            # Validation failed after all attempts in the inner loop for this review cycle
            logger.error(f"Failed to obtain valid specification after {config.MAX_VALIDATION_ATTEMPTS} sanitization/correction attempts (Review Cycle {review_cycle_attempt}).")
            # Update user_reviewed_text with the last failing output for the next potential review cycle
            user_reviewed_text = current_llm_response_text or json.dumps({"error": f"Validation failed after all retries: {last_validation_error}"})
            # Store the error message to show user if they review again
            last_validation_error_for_user = last_validation_error


            if args.skip_human_validation:
                 # If skipping review, failure here is terminal for the phase
                 logger.critical("Aborting Phase 0: Validation failed automatically.")
                 return None, None
            else:
                 # Ask user if they want to edit the failed output and retry the whole review cycle
                 try:
                     retry_choice = input(f"Processing failed. Edit the last output and retry review cycle? (yes/no): ").strip().lower()
                 except EOFError:
                     logger.warning("EOFError during retry prompt. Assuming 'no'.")
                     retry_choice = 'no'

                 if retry_choice in ['yes', 'y']:
                     logger.info("User chose to edit based on processing failure. Restarting review cycle.")
                     continue # Go back to start of outer review cycle loop
                 else:
                     logger.critical("Aborting: User chose not to retry after processing failure.")
                     return None, None

    # --- End of Outer Review Cycle Loop ---

    # If loop finishes without returning successfully (e.g., exceeded max_review_cycles)
    logger.critical(f"FATAL: Phase 0 (Requirements) did not complete successfully after {max_review_cycles} review cycles.")
    return None, None
