# Filename: stages/phase_1a_design.py
# -*- coding: utf-8 -*-
# Description: Stage 1a - Design Generation or Extraction Logic.
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

# --- Blueprint Extraction and Design Generation ---
# Replaces original Phase 1a if blueprint detected, or performs Phase 1a if not.
def run_phase_1a_extract_or_generate_design(
    blueprint_mode: bool,
    initial_task_data: Dict[str, Any],
    final_requirements_json_str: Optional[str], # Only used in standard mode
    target_language: str
) -> Optional[models.ProjectStructureOutput]:
    """
    Either extracts design from blueprint OR generates design from requirements.
    Includes validation/correction loops. Relies on JSON MIME type request + post-validation.

    Args:
        blueprint_mode: True if blueprint was detected in input.
        initial_task_data: Loaded task data (for supporting_docs in blueprint mode).
        final_requirements_json_str: Requirements JSON (standard mode only).
        target_language: Target programming language.

    Returns:
        Validated ProjectStructureOutput model or None on failure.
    """

    project_design_data: Optional[models.ProjectStructureOutput] = None
    attempt = 0
    last_error: str = "Initial attempt"
    last_response_text: Optional[str] = None

    if blueprint_mode:
        logger.info("\n" + "="*10 + " Phase 1a: Extract Design from Blueprint " + "="*10)
        phase_name = "Design Extraction"
        creation_prompt_template = prompts.PROMPT_1A_EXTRACT_DESIGN
        correction_prompt_template = prompts.PROMPT_1A_EXTRACT_DESIGN_CORRECT
        creation_format_args = {
            "supporting_docs_blueprint": initial_task_data.get('supporting_docs', '# Blueprint not found in task data!'),
            "target_language": target_language
        }
    else: # Standard generation mode
        logger.info("\n" + "="*10 + " Phase 1a: Generate Design from Requirements " + "="*10)
        phase_name = "Design Generation"
        if not final_requirements_json_str:
            logger.critical("FATAL: Cannot generate design in standard mode without requirements JSON.")
            return None
        creation_prompt_template = prompts.PROMPT_2_DESIGN
        correction_prompt_template = prompts.PROMPT_2_1_CORRECTION
        creation_format_args = {
            "final_requirements_json_str": final_requirements_json_str,
            "target_language": target_language
        }

    # --- Generation/Extraction Loop (with Correction) ---
    while attempt < config.MAX_VALIDATION_ATTEMPTS:
        attempt += 1
        logger.info(f"{phase_name} (Attempt {attempt}/{config.MAX_VALIDATION_ATTEMPTS})...")

        prompt_to_use_str = None
        format_args = {}
        if attempt == 1:
            prompt_to_use_str = creation_prompt_template
            format_args = creation_format_args
        else: # Correction attempt
            logger.info(f"Running LLM Correction for {phase_name}...")
            prompt_to_use_str = correction_prompt_template
            format_args = {
                "failed_json_output": last_response_text,
                "validation_error_message": last_error,
                "target_language": target_language
            }
            # Add blueprint context back if in blueprint mode for correction
            if blueprint_mode:
                 format_args["supporting_docs_blueprint"] = initial_task_data.get('supporting_docs', '# Blueprint not found in task data!')


        try:
            prompt_text = prompt_to_use_str.format(**format_args)
        except KeyError as e:
            logger.critical(f"FATAL: Prompt Key Error '{e}' preparing {phase_name} prompt. Aborting.")
            return None
        except Exception as e:
            logger.critical(f"FATAL: Prompt Format Error for {phase_name}: {e}", exc_info=True)
            return None

        # Indicate schema expected (for JSON MIME type), but don't pass schema itself
        response_text = llm_interface.call_gemini_api(
            prompt_text,
            temperature=config.PLANNING_TEMPERATURE,
            output_schema=models.ProjectStructureOutput # Indicate JSON expected
        )
        last_response_text = response_text # Store for potential correction

        # Check for immediate API call failure
        if response_text is None:
            logger.error(f"LLM API call failed definitively during {phase_name} (Attempt {attempt}).")
            last_error = "LLM API call failed completely."
            if attempt < config.MAX_VALIDATION_ATTEMPTS: continue # Allow retry if API failed
            else: break # Exit loop if max attempts reached

        # Parsing handles potential ErrorOutput from API/validation
        parsed_response: Optional[Union[models.ProjectStructureOutput, models.ErrorOutput]] = llm_interface.parse_llm_response(
            response_text,
            models.ProjectStructureOutput
        )

        if isinstance(parsed_response, models.ErrorOutput) or not parsed_response:
            error_msg = getattr(parsed_response, 'error', f'Unknown {phase_name}/validation error (Attempt {attempt})')
            logger.error(f"{phase_name}/validation failed. Error: {error_msg}")
            logger.debug(f"Raw {phase_name} Response:\n{response_text[:1000]}...")
            last_error = error_msg # Store error for correction prompt
            # Loop continues if attempts remain
        else:
            project_design_data = parsed_response
            logger.info(f"{phase_name} successful and validated.")
            # Save the design artifact
            design_path_relative = config.PROJECT_STRUCTURE_FILENAME
            try:
                design_json_str_save = project_design_data.model_dump_json(indent=2)
                if file_utils.save_code_artifact(design_path_relative, design_json_str_save):
                    logger.info(f"Project structure saved to sandbox: {design_path_relative}")
                else:
                    logger.warning(f"Could not save project structure file '{design_path_relative}'.")
            except Exception as e:
                logger.error(f"Failed to serialize/save project design: {e}")
            break # Exit loop on success
    # End of generation/extraction loop

    if not project_design_data:
        logger.critical(f"FATAL: Failed {phase_name} after {config.MAX_VALIDATION_ATTEMPTS} attempts.")
        return None

    return project_design_data
