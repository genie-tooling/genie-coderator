# Filename: stages/phase_1c_planning.py
# -*- coding: utf-8 -*-
# Description: Stage 1c - Planning per Chunk, Size Estimation, Refinement, and Explicit File Targeting Logic.
# Version: 1.1.0 (Added phase_target_mapping logic)

import logging
import json
from typing import Dict, Any, Optional, Tuple, Union, List # Added List

# Relative imports for sibling modules
from .. import config
from .. import models
from .. import prompts
from .. import llm_interface
from .. import file_utils
from .. import utils

logger = logging.getLogger(__name__)

# --- Planning per Chunk ---
def run_phase_1c_planning_per_chunk(
    plan_chunk_data: models.PlanChunkOutput,
    project_design_data: models.ProjectStructureOutput,
    final_requirements_json_str: Optional[str], # May be None in blueprint mode
    target_language: str
) -> Tuple[Optional[models.PhasingPlanOutput], Optional[str]]:
    """
    Generates detailed phasing plan per chunk, estimates sizes, refines,
    creates explicit phase-to-file mapping, and aggregates results.
    Includes validation/correction loops. Relies on JSON MIME type request + post-validation.

    Args:
        plan_chunk_data: Validated plan chunks.
        project_design_data: Validated project structure.
        final_requirements_json_str: Requirements JSON (if available).
        target_language: Target programming language.

    Returns:
        Tuple (Aggregated PhasingPlanOutput, initial_continuity_seed) or (None, None).
    """
    logger.info("\n" + "="*10 + " Phase 1c: Size Estimation, Phasing Plan & File Targeting per Chunk " + "="*10)
    # Initialize with correct types, including the new mapping field
    aggregated_final_plan = models.PhasingPlanOutput(
        estimated_output_bytes_per_phase=[], # List[PhaseEstimate]
        threshold_check_result="Aggregated result:", # str
        final_phasing_plan=[], # List[str]
        phase_target_mapping={}, # Dict[str, List[str]]
        initial_continuity_seed=None # Optional[str]
    )
    all_chunk_plans_succeeded = True

    # Truncate shared inputs once
    logger.debug("Preparing shared inputs for chunked planning...")
    try:
        project_structure_json_str = project_design_data.model_dump_json()
        req_str_for_planning = final_requirements_json_str or "{}" # Use empty JSON if no reqs

        # Estimate base prompt size for chunked planning (Use updated prompt)
        base_prompt_est_text = prompts.PROMPT_3_5_PLAN_ESTIMATE_REFINE.format(
            current_date=config.CURRENT_DATE_STR, target_language=target_language, model_name=config.GEMINI_MODEL_NAME,
            output_byte_limit=0, target_output_bytes=0, target_bytes_factor=0,
            final_requirements_json_str="{req_placeholder}",
            project_structure_json_str="{design_placeholder}",
            current_chunk_description="{chunk_placeholder}",
        )
        base_prompt_tokens = llm_interface.get_token_count(base_prompt_est_text)
        remaining_tokens = config.MAX_CONTEXT_HISTORY_TOKENS - base_prompt_tokens - 100 # Extra buffer

        if remaining_tokens <= 100:
            logger.critical(f"FATAL: Base planning prompt ({base_prompt_tokens} tokens) leaves insufficient space ({remaining_tokens} tokens) for reqs/design/chunk.")
            return None, None

        # Allocate budget (giving more to design)
        req_token_budget = int(remaining_tokens * 0.3)
        design_token_budget = remaining_tokens - req_token_budget
        logger.debug(f"Chunked Planning input budget: Requirements={req_token_budget}, Design={design_token_budget}")

        truncated_req_str = llm_interface.truncate_prompt_context(
            req_str_for_planning, req_token_budget, "Requirements JSON"
        )
        truncated_design_str = llm_interface.truncate_prompt_context(
            project_structure_json_str, design_token_budget, "Design JSON"
        )
    except Exception as e:
        logger.critical(f"Failed to serialize/truncate shared inputs for chunked planning: {e}", exc_info=True)
        return None, None

    # --- Loop through Chunks ---
    num_chunks = len(plan_chunk_data.plan_chunks)
    for chunk_idx, chunk_desc in enumerate(plan_chunk_data.plan_chunks):
        logger.info(f"\n-- Planning for Chunk {chunk_idx + 1}/{num_chunks}: '{chunk_desc}' --")
        chunk_plan_data: Optional[models.PhasingPlanOutput] = None
        planning_attempt = 0
        last_planning_error: str = f"Initial planning attempt for chunk {chunk_idx+1}"
        last_planning_response_text: Optional[str] = None

        # --- Planning Loop for Current Chunk ---
        while planning_attempt < config.MAX_VALIDATION_ATTEMPTS:
            planning_attempt += 1
            logger.info(f"Generating phasing plan for chunk (Attempt {planning_attempt}/{config.MAX_VALIDATION_ATTEMPTS})...")

            # Determine output size limits for target model
            model_base = 'default'
            for key in config.OUTPUT_BYTE_LIMITS:
                if key != 'default' and key in config.GEMINI_MODEL_NAME:
                    model_base = key
                    break
            output_byte_limit = config.OUTPUT_BYTE_LIMITS.get(model_base, config.OUTPUT_BYTE_LIMITS['default'])
            target_output_bytes = int(output_byte_limit * config.TARGET_BYTES_FACTOR)
            logger.info(f"Planning constraints: Model Base='{model_base}', Output Limit={output_byte_limit}, Target Bytes/Phase={target_output_bytes}")

            prompt_to_use_str = None
            format_args = {}
            if planning_attempt == 1:
                prompt_to_use_str = prompts.PROMPT_3_5_PLAN_ESTIMATE_REFINE # Use updated prompt
                format_args = {
                    "current_date": config.CURRENT_DATE_STR,
                    "target_language": target_language,
                    "model_name": config.GEMINI_MODEL_NAME,
                    "output_byte_limit": output_byte_limit,
                    "target_output_bytes": target_output_bytes,
                    "target_bytes_factor": config.TARGET_BYTES_FACTOR,
                    "final_requirements_json_str": truncated_req_str,
                    "project_structure_json_str": truncated_design_str,
                    "current_chunk_description": chunk_desc, # Pass current chunk desc
                }
            else: # Correction attempt
                 logger.info(f"Running LLM Correction (Prompt 3.6) for Chunk {chunk_idx+1} Planning...")
                 prompt_to_use_str = prompts.PROMPT_3_6_PLAN_CORRECTION # Use updated prompt
                 format_args = {
                     "failed_json_output": last_planning_response_text,
                     "validation_error_message": last_planning_error,
                     "target_language": target_language,
                     "model_name": config.GEMINI_MODEL_NAME,
                     "output_byte_limit": output_byte_limit, # Pass constraints
                     "target_output_bytes": target_output_bytes,
                     # Add context back for correction
                     "final_requirements_json_str": truncated_req_str,
                     "project_structure_json_str": truncated_design_str,
                     "current_chunk_description": chunk_desc,
                 }

            try:
                 prompt_text = prompt_to_use_str.format(**format_args)
            except KeyError as e:
                logger.critical(f"FATAL: Prompt Key Error '{e}' preparing chunk planning prompt. Aborting chunk.")
                last_planning_error = f"Prompt Key Error: {e}"
                # Break inner loop, outer loop will catch failure
                break
            except Exception as e:
                logger.critical(f"FATAL: Prompt Format Error for chunk planning: {e}", exc_info=True)
                last_planning_error = f"Prompt Format Error: {e}"
                break

            # Indicate schema expected (for JSON MIME type), but don't pass schema itself
            response3_5_text = llm_interface.call_gemini_api(
                prompt_text,
                temperature=config.PLANNING_TEMPERATURE,
                output_schema=models.PhasingPlanOutput # Indicate JSON expected
            )
            last_planning_response_text = response3_5_text

            if response3_5_text is None:
                logger.error(f"LLM API call failed definitively during chunk planning (Attempt {planning_attempt}).")
                last_planning_error = "LLM API call failed completely."
                if planning_attempt < config.MAX_VALIDATION_ATTEMPTS: continue
                else: break

            parsed_plan_response: Optional[Union[models.PhasingPlanOutput, models.ErrorOutput]] = llm_interface.parse_llm_response(
                response3_5_text,
                 models.PhasingPlanOutput
            )

            if isinstance(parsed_plan_response, models.ErrorOutput) or not parsed_plan_response:
                error_msg = getattr(parsed_plan_response, 'error', f'Unknown planning/estimation error for chunk {chunk_idx+1} (Attempt {planning_attempt})')
                logger.error(f"Phasing plan/estimation failed. Error: {error_msg}")
                logger.debug(f"Raw Planning Response (Chunk {chunk_idx+1}):\n{response3_5_text[:1000]}...")
                last_planning_error = error_msg
            else:
                # Planning succeeded for this chunk!
                chunk_plan_data = parsed_plan_response
                # Add extra validation for empty plan or mapping
                validation_passed_extra = True
                if not chunk_plan_data.final_phasing_plan:
                    logger.error(f"Chunk planning validation passed, but 'final_phasing_plan' list is empty (Chunk {chunk_idx+1}, Attempt {planning_attempt}).")
                    last_planning_error = "LLM returned an empty list for 'final_phasing_plan'."
                    validation_passed_extra = False
                elif not chunk_plan_data.phase_target_mapping:
                     logger.error(f"Chunk planning validation passed, but 'phase_target_mapping' dict is empty (Chunk {chunk_idx+1}, Attempt {planning_attempt}).")
                     last_planning_error = "LLM returned an empty dict for 'phase_target_mapping'."
                     validation_passed_extra = False
                # Check consistency (handled by model validator now, but good to log)
                # elif set(chunk_plan_data.final_phasing_plan) != set(chunk_plan_data.phase_target_mapping.keys()):
                #      logger.error(f"Mismatch between final_phasing_plan and phase_target_mapping keys (Chunk {chunk_idx+1}, Attempt {planning_attempt}).")
                #      last_planning_error = "Keys in 'phase_target_mapping' do not match 'final_phasing_plan'."
                #      validation_passed_extra = False

                if validation_passed_extra:
                    logger.info(f"Planning & Estimation successful for Chunk {chunk_idx + 1}.")
                    break # Exit planning loop for this chunk
                else:
                    chunk_plan_data = None # Treat as failure for retry logic
                    # Loop continues if attempts remain

        # --- End Planning Loop for Current Chunk ---

        if not chunk_plan_data:
            logger.error(f"Failed to generate phasing plan for Chunk {chunk_idx + 1} after {config.MAX_VALIDATION_ATTEMPTS} attempts.")
            all_chunk_plans_succeeded = False
            break # Stop processing further chunks if one fails

        # Aggregate results from successful chunk plan
        aggregated_final_plan.estimated_output_bytes_per_phase.extend(chunk_plan_data.estimated_output_bytes_per_phase)
        aggregated_final_plan.final_phasing_plan.extend(chunk_plan_data.final_phasing_plan)
        # Update mapping dictionary safely
        for phase_desc, targets in chunk_plan_data.phase_target_mapping.items():
            if phase_desc in aggregated_final_plan.phase_target_mapping:
                # This shouldn't happen if chunks are distinct, but handle defensively
                logger.warning(f"Duplicate phase description '{phase_desc}' found while aggregating chunk plans. Overwriting mapping.")
            if targets: # Only add if target list is not empty
                aggregated_final_plan.phase_target_mapping[phase_desc] = targets
            else:
                logger.warning(f"Empty target list found for phase '{phase_desc}' in chunk {chunk_idx + 1}. Skipping mapping entry.")

        aggregated_final_plan.threshold_check_result += f" Chunk {chunk_idx+1}: {chunk_plan_data.threshold_check_result};"
        if chunk_idx == 0: # Take seed from the first chunk only
             aggregated_final_plan.initial_continuity_seed = chunk_plan_data.initial_continuity_seed

    # --- End Loop Through Chunks ---

    if not all_chunk_plans_succeeded:
        logger.critical("FATAL: Planning failed for one or more chunks. Cannot proceed.")
        return None, None # Failed plan

    # --- Finalize Aggregated Plan ---
    logger.info("\n--- Aggregated Planning & Estimation Complete ---")
    try:
         # Validate final aggregated plan structure consistency
         if set(aggregated_final_plan.final_phasing_plan) != set(aggregated_final_plan.phase_target_mapping.keys()):
             logger.error("CRITICAL: Mismatch between aggregated final plan phases and target mapping keys after processing all chunks!")
             # Attempt to fix by removing extra keys from mapping
             extra_keys = set(aggregated_final_plan.phase_target_mapping.keys()) - set(aggregated_final_plan.final_phasing_plan)
             if extra_keys:
                 logger.warning(f"Removing extra keys from mapping: {extra_keys}")
                 for key in extra_keys:
                     del aggregated_final_plan.phase_target_mapping[key]
             # If phases are missing from mapping, it's harder to fix, log error and proceed cautiously
             missing_keys = set(aggregated_final_plan.final_phasing_plan) - set(aggregated_final_plan.phase_target_mapping.keys())
             if missing_keys:
                  logger.error(f"CRITICAL: Phases missing from final mapping: {missing_keys}. Build targeting may fail.")

         estimated_bytes_log = json.dumps(
             [p.model_dump() for p in aggregated_final_plan.estimated_output_bytes_per_phase], indent=2
         )
         logger.info(f"Aggregated Estimated Bytes per Initial Phase:\n{estimated_bytes_log}")
    except Exception as log_err:
         logger.error(f"Error logging planning results: {log_err}")

    logger.info(f"Aggregated Threshold Check Result: {aggregated_final_plan.threshold_check_result}")
    logger.info(f"Final Aggregated Phasing Plan ({len(aggregated_final_plan.final_phasing_plan)} parts):")
    for i, desc in enumerate(aggregated_final_plan.final_phasing_plan):
         targets = aggregated_final_plan.phase_target_mapping.get(desc, ["<Mapping Error!>"])
         print(f"  {i+1}. {desc} -> Targets: {', '.join(targets)}") # Show mapping
    final_continuity_seed = aggregated_final_plan.initial_continuity_seed
    logger.info(f"Final Initial Continuity Seed: {final_continuity_seed or 'None'}")

    # Save final aggregated plan artifact
    plan_path_relative = config.DETAILED_PLAN_FILENAME
    try:
        plan_json_str_save = aggregated_final_plan.model_dump_json(indent=2)

        # Post-validation size check for the final plan itself
        plan_bytes = len(plan_json_str_save.encode('utf-8'))
        plan_limit = config.OUTPUT_BYTE_LIMITS.get('gemini-2.5', 65536) # Use a reasonable limit
        if plan_bytes > plan_limit * 0.9: # Check if > 90% of limit
            logger.warning(f"Final aggregated plan size ({plan_bytes} bytes) is large (>90% of {plan_limit} bytes). May impact context for build phase.")

        if file_utils.save_code_artifact(plan_path_relative, plan_json_str_save):
            logger.info(f"Detailed aggregated phasing plan saved to sandbox: {plan_path_relative}")
        else:
            logger.warning(f"Could not save detailed aggregated plan file '{plan_path_relative}'.")
    except Exception as e:
        logger.error(f"Failed to serialize/save detailed aggregated plan: {e}")

    return aggregated_final_plan, final_continuity_seed # Success
