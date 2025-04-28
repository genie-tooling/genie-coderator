# Filename: stages/phase_0_summary.py
# -*- coding: utf-8 -*-
# Description: Stage 0 - Blueprint Summary Extraction Logic.
# Version: 1.0

import logging
from typing import Dict, Any, Optional, Union

# Relative imports for sibling modules
from .. import config
from .. import models
from .. import prompts
from .. import llm_interface
from .. import file_utils
from .. import utils

logger = logging.getLogger(__name__)

def run_phase_0_extract_summary(
    initial_task_data: Dict[str, Any]
) -> Optional[str]:
    """
    Extracts a concise summary from the blueprint text using an LLM call.
    Includes a simple retry loop for validation. Saves the artifact.

    Args:
        initial_task_data: Loaded task data containing 'supporting_docs'.

    Returns:
        The extracted summary string or None on failure.
    """
    logger.info("\n" + "="*10 + " Phase 0: Extract Summary from Blueprint (Optional Step) " + "="*10)
    blueprint_text = initial_task_data.get('supporting_docs')
    if not blueprint_text:
        logger.warning("No supporting_docs found in task data. Cannot extract summary.")
        return None

    extracted_summary: Optional[str] = None
    last_error = "Initial summary extraction attempt."
    last_response_text: Optional[str] = None

    # Simple retry loop (using validation attempts constant for consistency)
    for attempt in range(config.MAX_VALIDATION_ATTEMPTS):
        logger.info(f"Attempting blueprint summary extraction (Attempt {attempt + 1}/{config.MAX_VALIDATION_ATTEMPTS})...")

        # TODO: Add correction prompt for summary if needed, similar to other phases.
        # For now, just retry the main extraction prompt.
        prompt_text = prompts.PROMPT_0_EXTRACT_SUMMARY.format(
            supporting_docs_blueprint=blueprint_text
            # Note: No correction prompt logic added here yet
        )

        # Call API, indicate schema expected (for JSON MIME type) but don't pass it
        response_text = llm_interface.call_gemini_api(
            prompt_text,
            temperature=config.PLANNING_TEMPERATURE, # Use planning temp for extraction
            output_schema=models.BlueprintSummaryOutput # Indicate JSON expected
        )
        last_response_text = response_text # Store for logging if needed

        if response_text is None:
            logger.error(f"LLM API call failed definitively during summary extraction (Attempt {attempt + 1}).")
            last_error = "LLM API call failed completely."
            if attempt < config.MAX_VALIDATION_ATTEMPTS - 1: continue
            else: break # Exit loop if max attempts reached

        parsed_response: Optional[Union[models.BlueprintSummaryOutput, models.ErrorOutput]] = llm_interface.parse_llm_response(
            response_text,
            models.BlueprintSummaryOutput
        )

        if isinstance(parsed_response, models.ErrorOutput) or not parsed_response:
            error_msg = getattr(parsed_response, 'error', f'Unknown summary extraction/validation error (Attempt {attempt + 1})')
            logger.error(f"Blueprint summary extraction/validation failed. Error: {error_msg}")
            logger.debug(f"Raw Summary Extraction Response:\n{response_text[:1000]}...")
            last_error = error_msg # Store error (though not used in correction yet)
            # Loop continues if attempts remain
        else:
            # Successfully extracted and validated
            extracted_summary = parsed_response.extracted_summary
            logger.info("Blueprint summary extracted and validated successfully.")
            # Save the summary artifact
            summary_path_relative = config.EXTRACTED_SUMMARY_FILENAME
            if file_utils.save_code_artifact(summary_path_relative, extracted_summary):
                logger.info(f"Extracted summary saved to sandbox: {summary_path_relative}")
            else:
                logger.warning(f"Could not save extracted summary file '{summary_path_relative}'.")
            break # Exit loop on success

    if not extracted_summary:
        logger.warning(f"Failed to extract blueprint summary after {config.MAX_VALIDATION_ATTEMPTS} attempts. Continuing without summary artifact.")
        return None

    return extracted_summary
