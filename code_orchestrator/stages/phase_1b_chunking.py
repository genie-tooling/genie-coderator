# Filename: stages/phase_1b_chunking.py
# -*- coding: utf-8 -*-
# Description: Stage 1b - Plan Chunking Logic.
# Version: 1.0

import logging
import json
from typing import Dict, Any, Optional, Union

# Relative imports for sibling modules
from .. import config
from .. import models
from .. import prompts
from .. import llm_interface
from .. import file_utils
from .. import utils

logger = logging.getLogger(__name__)

# --- Plan Chunking ---
def run_phase_1b_plan_chunking(
    project_design_data: models.ProjectStructureOutput,
    target_language: str
) -> Optional[models.PlanChunkOutput]:
    """
    Generates high-level plan chunks based on the design.
    Includes validation/correction loop. Relies on JSON MIME type request + post-validation.

    Args:
        project_design_data: The validated project structure (generated or extracted).
        target_language: Target programming language.

    Returns:
        Validated PlanChunkOutput model or None on failure.
    """
    logger.info("\n" + "="*10 + " Phase 1b: Plan Chunking " + "="*10)
    plan_chunk_data: Optional[models.PlanChunkOutput] = None
    attempt = 0
    last_error: str = "Initial chunking attempt"
    last_response_text: Optional[str] = None

    try:
        project_structure_json_str = project_design_data.model_dump_json()
    except Exception as e:
        logger.critical(f"FATAL: Failed to serialize design for chunking prompt: {e}", exc_info=True)
        return None

    # --- Chunking Loop (with Correction) ---
    while attempt < config.MAX_VALIDATION_ATTEMPTS:
        attempt += 1
        logger.info(f"Generating plan chunks (Attempt {attempt}/{config.MAX_VALIDATION_ATTEMPTS})...")

        prompt_to_use_str = None
        format_args = {}
        if attempt == 1:
            prompt_to_use_str = prompts.PROMPT_3_0_CHUNK_PLAN
            format_args = {
                "project_structure_json_str": project_structure_json_str,
                "target_language": target_language
            }
        else: # Correction attempt
             logger.info("Running LLM Correction (Prompt 3.1) for Plan Chunking...")
             prompt_to_use_str = prompts.PROMPT_3_1_CHUNK_CORRECTION
             format_args = {
                 "failed_json_output": last_response_text,
                 "validation_error_message": last_error,
                 "target_language": target_language,
                 "project_structure_json_str": project_structure_json_str # Provide context again
             }

        try:
            prompt_text = prompt_to_use_str.format(**format_args)
        except KeyError as e:
            logger.critical(f"FATAL: Prompt Key Error '{e}' preparing chunking prompt. Aborting.")
            return None
        except Exception as e:
            logger.critical(f"FATAL: Prompt Format Error for chunking: {e}", exc_info=True)
            return None

        # Indicate schema expected (for JSON MIME type), but don't pass schema itself
        response_chunk_text = llm_interface.call_gemini_api(
            prompt_text,
            temperature=config.PLANNING_TEMPERATURE,
            output_schema=models.PlanChunkOutput # Indicate JSON expected
        )
        last_response_text = response_chunk_text

        if response_chunk_text is None:
            logger.error(f"LLM API call failed definitively during chunk generation (Attempt {attempt}).")
            last_error = "LLM API call failed completely."
            if attempt < config.MAX_VALIDATION_ATTEMPTS: continue
            else: break

        parsed_chunk_response: Optional[Union[models.PlanChunkOutput, models.ErrorOutput]] = llm_interface.parse_llm_response(
            response_chunk_text,
            models.PlanChunkOutput
        )

        if isinstance(parsed_chunk_response, models.ErrorOutput) or not parsed_chunk_response:
            error_msg = getattr(parsed_chunk_response, 'error', f'Unknown chunk generation/validation error (Attempt {attempt})')
            logger.error(f"Chunk generation/validation failed. Error: {error_msg}")
            logger.debug(f"Raw Chunking Response:\n{response_chunk_text[:1000]}...")
            last_error = error_msg
        else:
            plan_chunk_data = parsed_chunk_response
            if not plan_chunk_data.plan_chunks:
                 logger.error(f"Chunking validation passed, but the 'plan_chunks' list is empty (Attempt {attempt}).")
                 last_error = "LLM returned an empty list for 'plan_chunks'."
                 plan_chunk_data = None # Treat as failure
                 # Continue loop for correction
            else:
                logger.info("Plan chunks generated and validated successfully.")
                # Save chunk list artifact
                chunk_path_relative = config.PLAN_CHUNK_LIST_FILENAME
                try:
                    chunk_json_str_save = plan_chunk_data.model_dump_json(indent=2)
                    if file_utils.save_code_artifact(chunk_path_relative, chunk_json_str_save):
                        logger.info(f"Plan chunk list saved to sandbox: {chunk_path_relative}")
                    else:
                        logger.warning(f"Could not save plan chunk list file '{chunk_path_relative}'.")
                except Exception as e:
                    logger.error(f"Failed to serialize/save plan chunk list: {e}")
                break # Exit chunking loop on success

    if not plan_chunk_data:
        logger.critical(f"FATAL: Failed to generate plan chunks after {config.MAX_VALIDATION_ATTEMPTS} attempts.")
        return None

    return plan_chunk_data
