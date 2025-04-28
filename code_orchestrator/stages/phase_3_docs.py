# Filename: stages/phase_3_docs.py
# -*- coding: utf-8 -*-
# Description: Stage 3 - Documentation Generation Logic.
# Version: 1.0

import logging
import argparse
import json
import os
from typing import Dict, Any, Optional, List, Union

# Relative imports for sibling modules
from .. import config
from .. import models
from .. import prompts
from .. import llm_interface
from .. import file_utils
from .. import utils

logger = logging.getLogger(__name__)

# Corresponds to Plan v1.1 Section 3.E (Documentation Generation)
def run_phase_3_documentation(args: argparse.Namespace,
                                final_requirements_json_str: Optional[str], # Optional
                                project_design_data: models.ProjectStructureOutput,
                                final_plan_data: models.PhasingPlanOutput,
                                target_language: str) -> bool:
    """
    Executes Phase 3: Documentation generation. Relies on JSON MIME type request + post-validation.

    Args:
        args: Command line arguments.
        final_requirements_json_str: Sanitized requirements JSON string (or placeholder).
        project_design_data: The validated project structure.
        final_plan_data: The validated *aggregated* phasing plan.
        target_language: The target programming language.

    Returns:
        True if documentation generation was successful or skipped, False on error.
    """
    if not args.docs:
        logger.info("\n" + "="*10 + " Phase 3: Documentation Generation (Skipped - Not Requested) " + "="*10)
        return True

    logger.info("\n" + "="*10 + " Phase 3: Documentation Generation " + "="*10)
    doc_types_to_generate: List[str] = args.docs
    logger.info(f"Requested documentation types: {', '.join(doc_types_to_generate)}")

    # 1. Gather Context
    logger.info("Gathering final codebase and context for documentation...")
    final_codebase_content_parts: List[str] = []
    files_processed = 0
    relevant_files = 0
    req_str_for_docs = final_requirements_json_str or "{}" # Placeholder if None

    try:
        for file_def in project_design_data.file_structure:
            lang_ext = config.LANGUAGE_EXTENSIONS.get(target_language)
            is_target_lang = lang_ext and file_def.path.endswith(f".{lang_ext}")
            # Include common config/doc files in context
            is_common_config_or_doc = file_def.path.lower().endswith(
                ('.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.md', '.txt', 'dockerfile', 'makefile')
            )
            # Exclude venv files explicitly if somehow included in design
            if config.VENV_DIR_NAME in file_def.path.split(os.path.sep):
                continue

            if is_target_lang or is_common_config_or_doc:
                 relevant_files += 1
                 content = file_utils.read_code_artifact(file_def.path)
                 if content is not None:
                     # Add file markers for clarity in the prompt context
                     final_codebase_content_parts.append(f"\n--- START FILE: {file_def.path} ---\n{content}\n--- END FILE: {file_def.path} ---\n")
                     files_processed += 1
                 else:
                     logger.warning(f"Could not read file '{file_def.path}' for documentation context.")
                     final_codebase_content_parts.append(f"\n--- FILE: {file_def.path} (Content Unavailable) ---\n")

        if files_processed == 0 and relevant_files > 0:
            logger.error("No source code or relevant file content could be read. Cannot generate documentation.")
            return False
        elif relevant_files == 0:
             logger.warning("No relevant source or config files found in design. Documentation context will be limited.")

        full_codebase_str = "".join(final_codebase_content_parts)
        # Allocate slightly less for code to leave more room for specs/design/plan
        docs_code_token_budget = int(config.MAX_CONTEXT_HISTORY_TOKENS * 0.60)
        truncated_codebase_str = llm_interface.truncate_prompt_context(
            full_codebase_str, docs_code_token_budget, "Final Codebase"
        )
        logger.info(f"Prepared codebase context ({files_processed}/{relevant_files} files read).")

        # Serialize other contexts safely and truncate if needed
        try: project_structure_json_str = project_design_data.model_dump_json()
        except Exception: project_structure_json_str = '{"error": "serialize failed"}'

        try: final_plan_list = final_plan_data.final_phasing_plan
        except Exception: final_plan_list = ["error serializing plan"]
        final_phase_plan_list_json_str = json.dumps(final_plan_list)


        # Estimate remaining tokens and budget for other contexts
        base_prompt_tokens = llm_interface.get_token_count(prompts.PROMPT_5_DOCS.format(
             doc_type="{doc_type}", target_language=target_language,
             final_requirements_json_str="", project_structure_json_str="",
             final_phase_plan_list_json_str="", final_codebase_content_str="",
             final_requirements_content="", final_system_dependencies_json=""
        ))
        code_tokens = llm_interface.get_token_count(truncated_codebase_str)
        remaining_tokens = config.MAX_CONTEXT_HISTORY_TOKENS - base_prompt_tokens - code_tokens - 100 # Buffer

        if remaining_tokens <= 0:
            logger.warning("Insufficient token budget for non-code context in documentation prompt. Context will be limited.")
            truncated_req_str = "{# Truncated due to budget #}"
            truncated_design_str = "{# Truncated due to budget #}"
            truncated_plan_str = "{# Truncated due to budget #}"
        else:
             # Allocate remaining budget (e.g., 40% reqs, 40% design, 20% plan)
             req_budget = int(remaining_tokens * 0.4)
             design_budget = int(remaining_tokens * 0.4)
             plan_budget = remaining_tokens - req_budget - design_budget

             truncated_req_str = llm_interface.truncate_prompt_context(
                 req_str_for_docs, req_budget, "Docs: Reqs JSON"
             )
             truncated_design_str = llm_interface.truncate_prompt_context(
                 project_structure_json_str, design_budget, "Docs: Design JSON"
             )
             truncated_plan_str = llm_interface.truncate_prompt_context(
                 final_phase_plan_list_json_str, plan_budget, "Docs: Plan JSON"
             )

        # Read requirements file content (usually small)
        final_req_content = file_utils.read_code_artifact(config.REQUIREMENTS_FILENAME) or "# Requirements file unavailable."
        # System deps not persisted, use placeholder
        final_sys_deps_list = ["# System dependencies info not available in docs phase"]
        try: final_sys_deps_json = json.dumps(final_sys_deps_list)
        except Exception: final_sys_deps_json = '["# Error serializing system deps"]'

    except Exception as e:
        logger.error(f"Error gathering context for documentation: {e}", exc_info=True)
        return False

    # 2. Ensure Docs Directory Exists
    try:
        # get_docs_path creates the dir if needed
        file_utils.get_docs_path("placeholder.tmp")
        # Attempt removal, ignore error if it fails (it might not exist if dir already existed)
        try: os.remove(file_utils.get_docs_path("placeholder.tmp"))
        except OSError: pass
        logger.info(f"Ensured documentation directory exists: {config.DOCS_DIR}")
    except OSError as e:
        logger.error(f"Cannot create/access docs directory '{config.DOCS_DIR}': {e}. Skipping documentation.")
        return False

    # 3. Loop Through Doc Types and Generate
    docs_generation_overall_success = True
    for doc_type in doc_types_to_generate:
        logger.info(f"-- Generating documentation type: '{doc_type}'... --")
        # Use truncated context strings
        doc_format_args = {
            "doc_type": doc_type,
            "target_language": target_language,
            "final_requirements_json_str": truncated_req_str,
            "project_structure_json_str": truncated_design_str,
            "final_phase_plan_list_json_str": truncated_plan_str,
            "final_codebase_content_str": truncated_codebase_str,
            "final_requirements_content": final_req_content,
            "final_system_dependencies_json": final_sys_deps_json,
        }

        try:
            prompt_doc_text = prompts.PROMPT_5_DOCS.format(**doc_format_args)
        except KeyError as e:
            logger.error(f"Docs prompt key error for '{doc_type}': {e}. Skipping.")
            docs_generation_overall_success = False
            continue
        except Exception as e:
            logger.error(f"Docs prompt format error for '{doc_type}': {e}. Skipping.", exc_info=True)
            docs_generation_overall_success = False
            continue

        # Call LLM for docs
        # Indicate schema expected (for JSON MIME type), but don't pass schema itself
        doc_attempt = 0
        parsed_doc_data = None
        last_doc_error = "Initial doc generation attempt"
        last_doc_response_text = None

        while doc_attempt < config.MAX_VALIDATION_ATTEMPTS:
             doc_attempt += 1
             logger.info(f"Calling LLM for doc '{doc_type}' (Attempt {doc_attempt}/{config.MAX_VALIDATION_ATTEMPTS})")

             # TODO: Add correction prompt for docs if needed (PROMPT_5_CORRECT)
             # For now, just retrying the main prompt.
             prompt_to_use_str = prompt_doc_text # Always use main prompt for now

             response_doc_text = llm_interface.call_gemini_api(
                 prompt_to_use_str,
                 temperature=config.DEFAULT_API_TEMPERATURE,
                 output_schema=models.DocsOutput # Indicate JSON expected
             )
             last_doc_response_text = response_doc_text

             if response_doc_text is None:
                 logger.error(f"LLM API call failed definitively for doc '{doc_type}' (Attempt {doc_attempt}).")
                 last_doc_error = "LLM API call failed completely."
                 if doc_attempt < config.MAX_VALIDATION_ATTEMPTS: continue
                 else: break

             parsed_doc_data_attempt: Optional[Union[models.DocsOutput, models.ErrorOutput]] = llm_interface.parse_llm_response(response_doc_text, models.DocsOutput)

             if isinstance(parsed_doc_data_attempt, models.ErrorOutput) or not parsed_doc_data_attempt:
                 error_msg = getattr(parsed_doc_data_attempt, 'error', f'LLM/Parse/Validate Error for doc type {doc_type} (Attempt {doc_attempt})')
                 logger.error(f"Documentation generation/validation failed for '{doc_type}'. Error: {error_msg}")
                 logger.debug(f"Raw documentation response ('{doc_type}'):\n{response_doc_text[:1000]}...")
                 last_doc_error = error_msg
                 # Loop continues if attempts remain
             else:
                 parsed_doc_data = parsed_doc_data_attempt
                 logger.info(f"Documentation '{doc_type}' generated and validated successfully.")
                 break # Exit retry loop

        # --- End Doc Retry Loop ---

        if not parsed_doc_data:
             logger.error(f"Failed to generate valid documentation for '{doc_type}' after {config.MAX_VALIDATION_ATTEMPTS} attempts.")
             docs_generation_overall_success = False
             continue # Try next doc type
        else:
            # Success for this doc type
            returned_doc_type = parsed_doc_data.documentation_type.lower()
            if returned_doc_type != doc_type.lower():
                logger.warning(f"LLM generated type '{returned_doc_type}' instead of requested '{doc_type}'. Saving as requested type.")
                # Use requested type for filename consistency

            doc_filename = f"{doc_type.upper()}.md"
            header_type = returned_doc_type if returned_doc_type != doc_type.lower() else doc_type
            # Use the inline f-string logic
            doc_content_with_header = (
                 f"# Documentation: {header_type.replace('_', ' ').title()}\n\n"
                 f"*Generated on:* {config.CURRENT_DATE_STR}\n"
                 f"*Target Language:* {target_language}\n"
                 # Add reference back to task file for context
                 f"*Project Task:* See relevant task file (e.g., {config.DEFAULT_TASK_FILE} if default was used)\n"
                 f"---\n\n"
                 f"{parsed_doc_data.content}"
             )

            if not file_utils.save_code_artifact(doc_filename, doc_content_with_header, is_doc=True):
                logger.error(f"Failed to save documentation file '{doc_filename}' to docs dir.")
                docs_generation_overall_success = False
                # Continue trying other doc types

    logger.info("--- Documentation Generation Finished ---")
    return docs_generation_overall_success
