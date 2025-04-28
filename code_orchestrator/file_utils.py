# -*- coding: utf-8 -*-
# Filename: file_utils.py
# Description: Utilities for file system operations, path management, state persistence, and user interaction.
# Version: 1.4 (Added State Management T005)

import os
import yaml
import logging
import shutil
import subprocess
import json
from typing import Optional, Dict, Any, List

# Import config (needs to be adjusted based on execution context)
# Assuming standard package structure where config is accessible
from . import config
from . import models # For loading design/plan models and PipelineState
from pydantic import ValidationError # Import specifically

logger = logging.getLogger(__name__)

# --- Path Configuration and Management ---

def configure_paths(output_dir_arg: Optional[str]):
    """
    Sets the global BASE_OUTPUT_DIR, CODE_DIR and DOCS_DIR paths based on user args or defaults.
    Creates the CODE_DIR. DOCS_DIR is created later if needed.

    Args:
        output_dir_arg: The base output directory path from command line args.
    """
    global BASE_OUTPUT_DIR, CODE_DIR, DOCS_DIR # Allow modification of global variables

    if output_dir_arg:
        # Use the user-specified base directory
        config.BASE_OUTPUT_DIR = os.path.abspath(output_dir_arg)
    else:
        # Use defaults relative to the current working directory
        config.BASE_OUTPUT_DIR = os.path.abspath(config.DEFAULT_BASE_OUTPUT_DIR)

    # Derive sandbox and docs dirs from the base dir
    config.CODE_DIR = os.path.join(config.BASE_OUTPUT_DIR, config.DEFAULT_SANDBOX_SUBDIR)
    config.DOCS_DIR = os.path.join(config.BASE_OUTPUT_DIR, config.DEFAULT_DOCS_SUBDIR)

    logger.info(f"Using output base directory: '{config.BASE_OUTPUT_DIR}'")
    logger.info(f"Sandbox directory set to: '{config.CODE_DIR}'")
    logger.info(f"Docs directory set to:    '{config.DOCS_DIR}'")

    # Always ensure the sandbox directory exists
    try:
        os.makedirs(config.CODE_DIR, exist_ok=True)
        logger.debug(f"Ensured sandbox directory exists: {config.CODE_DIR}")
    except OSError as e:
        logger.critical(f"FATAL: Could not create sandbox directory '{config.CODE_DIR}': {e}. Exiting.")
        exit(1)

def _is_safe_path(base_dir: str, target_path: str) -> bool:
    """Checks if the target path is safely within the base directory."""
    try:
        # Resolve symbolic links and normalize case for robust comparison
        abs_base_dir = os.path.realpath(os.path.normcase(os.path.abspath(base_dir)))
        abs_target_path = os.path.realpath(os.path.normcase(os.path.abspath(target_path)))

        # Check if the target path starts with the base directory path + separator
        # Handles case where target is the base directory itself.
        return abs_target_path.startswith(abs_base_dir + os.path.sep) or abs_target_path == abs_base_dir
    except Exception as e:
        logger.error(f"Error during path safety check (Base: '{base_dir}', Target: '{target_path}'): {e}")
        return False

def get_sandbox_path(relative_path: str) -> str:
    """Constructs an absolute path within the configured sandbox (CODE_DIR)."""
    # Clean the relative path to prevent escaping the sandbox (basic check)
    clean_relative_path = os.path.normpath(relative_path).lstrip('.').lstrip('/')
    if not config.CODE_DIR:
         logger.critical("FATAL: CODE_DIR not configured. Call configure_paths first.")
         exit(1)
    return os.path.join(config.CODE_DIR, clean_relative_path)

def get_docs_path(relative_path: str) -> str:
    """Constructs an absolute path within the configured docs (DOCS_DIR)."""
    clean_relative_path = os.path.normpath(relative_path).lstrip('.').lstrip('/')
    if not config.DOCS_DIR:
         # Create docs dir on first attempt to use it if not explicitly created yet
         try:
              os.makedirs(config.DOCS_DIR, exist_ok=True)
              logger.info(f"Created documentation directory: {config.DOCS_DIR}")
         except OSError as e:
              logger.critical(f"FATAL: Could not create docs directory '{config.DOCS_DIR}': {e}. Exiting.")
              exit(1)
    return os.path.join(config.DOCS_DIR, clean_relative_path)

def get_state_filepath() -> str:
    """Gets the absolute path to the pipeline state file within the BASE_OUTPUT_DIR."""
    if not config.BASE_OUTPUT_DIR:
        logger.critical("FATAL: BASE_OUTPUT_DIR not configured. Call configure_paths first.")
        exit(1)
    return os.path.join(config.BASE_OUTPUT_DIR, config.STATE_FILENAME)

def get_requirements_filepath() -> str:
    """Gets the absolute path to the requirements file within the sandbox."""
    return get_sandbox_path(config.REQUIREMENTS_FILENAME)

def get_venv_dir() -> str:
    """Gets the absolute path to the virtual environment directory within the sandbox."""
    return get_sandbox_path(config.VENV_DIR_NAME)

# --- File Content Operations ---

def save_code_artifact(relative_path: str, content: str, is_doc: bool = False):
    """
    Saves content to a specified relative path within the sandbox or docs dir.
    Creates necessary subdirectories.

    Args:
        relative_path: The path relative to CODE_DIR (or DOCS_DIR if is_doc=True).
        content: The string content to write.
        is_doc: If True, save relative to DOCS_DIR, otherwise relative to CODE_DIR.
    Returns:
        True if successful, False otherwise.
    """
    try:
        absolute_path = ""
        base_dir = ""
        if is_doc:
            # Get path, potentially creating DOCS_DIR if needed
            absolute_path = get_docs_path(relative_path)
            base_dir = config.DOCS_DIR
        else:
            if not config.CODE_DIR:
                raise ValueError("CODE_DIR not configured.")
            absolute_path = get_sandbox_path(relative_path)
            base_dir = config.CODE_DIR

        # Ensure the calculated path is still within the intended base directory
        if not _is_safe_path(base_dir, absolute_path):
            logger.error(f"Security Risk: Attempted to write outside designated directory: '{absolute_path}' (Relative: '{relative_path}', Base: '{base_dir}')")
            return False

        # Create subdirectories if they don't exist
        os.makedirs(os.path.dirname(absolute_path), exist_ok=True)

        # Write the file content
        with open(absolute_path, "w", encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Saved artifact: {absolute_path} ({len(content)} bytes)")
        return True
    except ValueError as ve: # Specific error for config issues
         logger.critical(f"{ve}")
         return False
    except IOError as e:
        logger.error(f"Failed to save artifact '{absolute_path}': {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving artifact '{absolute_path}': {e}", exc_info=True)
        return False

def read_code_artifact(relative_path: str) -> Optional[str]:
    """
    Reads content from a specified relative path within the sandbox (CODE_DIR).
    Args:
        relative_path: The path relative to CODE_DIR.
    Returns:
        The file content as a string, or None if the file doesn't exist or an error occurs.
    """
    try:
        if not config.CODE_DIR:
            raise ValueError("CODE_DIR not configured.")
        absolute_path = get_sandbox_path(relative_path)

        # Security check
        if not _is_safe_path(config.CODE_DIR, absolute_path):
            logger.error(f"Security Risk: Attempted to read outside sandbox: '{absolute_path}' (Relative: '{relative_path}', Base: '{config.CODE_DIR}')")
            return None

        if not os.path.exists(absolute_path):
            logger.debug(f"Artifact not found (normal for first write): {absolute_path}")
            return None # Return None if file doesn't exist yet

        with open(absolute_path, "r", encoding='utf-8') as f:
            content = f.read()
        # logger.debug(f"Read artifact: {absolute_path} ({len(content)} bytes)")
        return content
    except ValueError as ve:
         logger.critical(f"{ve}")
         return None
    except IOError as e:
        logger.error(f"Failed to read artifact '{absolute_path}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error reading artifact '{absolute_path}': {e}", exc_info=True)
        return None

def load_yaml_task(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Loads the initial task definition from a YAML file.
    Includes fields required by Plan v1.1: human_summary, supporting_docs, language.
    """
    logger.info(f"Loading task definition from: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Basic validation
        if not isinstance(data, dict):
            logger.error(f"Invalid YAML format in '{filepath}'. Expected a dictionary (key-value pairs).")
            return None
        if 'description' not in data or not data['description']:
            logger.error(f"Task file '{filepath}' is missing required 'description' field or it's empty.")
            return None

        # Set defaults for optional fields as per Plan v1.1
        data.setdefault('human_summary', None)
        data.setdefault('language', None) # Will be overridden by CLI if provided
        data.setdefault('supporting_docs', '') # Default to empty string if missing

        logger.debug(f"YAML task loaded successfully: Keys={list(data.keys())}")
        return data
    except FileNotFoundError:
        logger.error(f"Task file not found: {filepath}")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file '{filepath}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading task file '{filepath}': {e}", exc_info=True)
        return None

def load_json_artifact(relative_path: str) -> Optional[Dict[str, Any]]:
    """Loads and parses a JSON artifact from the sandbox directory."""
    content = read_code_artifact(relative_path)
    if content is None:
        # read_code_artifact logs if file not found or permission error
        return None

    try:
        # Use the utility function which handles potential markdown blocks
        from . import utils # Local import to avoid circular dependency issues at top level
        data = utils.safe_json_loads(content)

        if data is None:
             logger.error(f"Failed to decode JSON from artifact '{relative_path}'. safe_json_loads provided details.")
             return None # safe_json_loads already logged the error details

        if not isinstance(data, dict):
             logger.error(f"Invalid JSON structure in '{relative_path}'. Expected a dictionary, got {type(data).__name__}.")
             return None
        logger.debug(f"Successfully loaded and parsed JSON artifact: {relative_path}")
        return data
    # safe_json_loads handles JSONDecodeError, so we only need generic catch here
    except Exception as e:
        logger.error(f"Unexpected error loading JSON artifact '{relative_path}' after parsing: {e}", exc_info=True)
        return None

def load_project_structure() -> Optional[models.ProjectStructureOutput]:
    """Loads the project structure JSON artifact from the sandbox."""
    logger.info("Loading project structure definition...")
    json_data = load_json_artifact(config.PROJECT_STRUCTURE_FILENAME)
    if not json_data:
        # load_json_artifact logs the reason
        return None
    try:
        # Use model_validate for Pydantic v2
        return models.ProjectStructureOutput.model_validate(json_data)
    except ValidationError as e:
        logger.error(f"Project structure validation failed: {e}")
        try:
            invalid_json_str = json.dumps(json_data, indent=2)
            logger.debug(f"Invalid Structure JSON received:\n{invalid_json_str}")
        except TypeError:
             logger.debug(f"Could not serialize invalid structure JSON for debug log: {json_data}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error validating project structure: {e}", exc_info=True)
        return None

def load_detailed_plan() -> Optional[models.PhasingPlanOutput]:
    """Loads the detailed phasing plan JSON artifact from the sandbox."""
    logger.info("Loading detailed phasing plan...")
    json_data = load_json_artifact(config.DETAILED_PLAN_FILENAME)
    if not json_data:
        # load_json_artifact logs the reason
        return None
    try:
        # Use model_validate for Pydantic v2
        return models.PhasingPlanOutput.model_validate(json_data)
    except ValidationError as e:
        logger.error(f"Detailed plan validation failed: {e}")
        try:
            invalid_json_str = json.dumps(json_data, indent=2)
            logger.debug(f"Invalid Plan JSON received:\n{invalid_json_str}")
        except TypeError:
             logger.debug(f"Could not serialize invalid plan JSON for debug log: {json_data}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error validating detailed plan: {e}", exc_info=True)
        return None

def load_plan_chunk_list() -> Optional[models.PlanChunkOutput]:
    """Loads the plan chunk list JSON artifact from the sandbox."""
    logger.info("Loading plan chunk list definition...")
    json_data = load_json_artifact(config.PLAN_CHUNK_LIST_FILENAME)
    if not json_data:
        # load_json_artifact logs the reason
        return None
    try:
        # Use model_validate for Pydantic v2
        return models.PlanChunkOutput.model_validate(json_data)
    except ValidationError as e:
        logger.error(f"Plan chunk list validation failed: {e}")
        try:
            invalid_json_str = json.dumps(json_data, indent=2)
            logger.debug(f"Invalid Plan Chunk JSON received:\n{invalid_json_str}")
        except TypeError:
             logger.debug(f"Could not serialize invalid plan chunk JSON for debug log: {json_data}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error validating plan chunk list: {e}", exc_info=True)
        return None

# --- State Management (T005) ---

def save_pipeline_state(state: models.PipelineState) -> bool:
    """
    Saves the current pipeline state to the designated state file.
    Args:
        state: The PipelineState object to save.
    Returns:
        True if successful, False otherwise.
    """
    state_filepath = get_state_filepath()
    logger.info(f"Saving pipeline state to: {state_filepath}")
    try:
        # Ensure BASE_OUTPUT_DIR exists (configure_paths should have done this)
        os.makedirs(config.BASE_OUTPUT_DIR, exist_ok=True)

        # Serialize the Pydantic model to JSON
        # Use model_dump_json for Pydantic v2
        state_json = state.model_dump_json(indent=2)

        # Write JSON to the state file
        with open(state_filepath, "w", encoding="utf-8") as f:
            f.write(state_json)
        logger.debug(f"Pipeline state saved successfully (Phase Index: {state.current_phase_index}).")
        return True
    except ValidationError as e: # Should not happen if state object is valid
        logger.error(f"Pydantic validation error during state serialization (should not occur): {e}")
        return False
    except TypeError as e: # For json.dumps issues if model_dump_json fails
        logger.error(f"Failed to serialize pipeline state to JSON: {e}", exc_info=True)
        return False
    except IOError as e:
        logger.error(f"Failed to write pipeline state file '{state_filepath}': {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving pipeline state: {e}", exc_info=True)
        return False

def load_pipeline_state() -> Optional[models.PipelineState]:
    """
    Loads the pipeline state from the designated state file.
    Returns:
        The loaded PipelineState object, or None if the file doesn't exist or is invalid.
    """
    state_filepath = get_state_filepath()
    if not os.path.exists(state_filepath):
        logger.info(f"State file '{state_filepath}' not found. Starting fresh run.")
        return None

    logger.info(f"Attempting to load pipeline state from: {state_filepath}")
    try:
        with open(state_filepath, "r", encoding="utf-8") as f:
            state_json_str = f.read()

        # Use safe_json_loads to handle potential markdown/errors initially
        state_data = utils.safe_json_loads(state_json_str)
        if not isinstance(state_data, dict):
            logger.error(f"Invalid format in state file '{state_filepath}'. Expected JSON object.")
            return None

        # Validate the loaded dictionary against the Pydantic model
        # Use model_validate for Pydantic v2
        loaded_state = models.PipelineState.model_validate(state_data)
        logger.info(f"Pipeline state loaded successfully (Resuming from phase index {loaded_state.current_phase_index}).")
        return loaded_state

    except json.JSONDecodeError as e: # Should be caught by safe_json_loads, but defense in depth
        logger.error(f"Failed to decode JSON from state file '{state_filepath}': {e}")
        return None
    except ValidationError as e:
        logger.error(f"State file validation failed: {e}")
        logger.debug(f"Invalid state file content ({state_filepath}):\n{state_json_str[:1000]}...")
        return None
    except IOError as e:
        logger.error(f"Failed to read state file '{state_filepath}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading pipeline state: {e}", exc_info=True)
        return None

# --- User Interaction ---

def find_editor() -> Optional[str]:
    """
    Finds a suitable text editor based on Plan v1.1 specified order: nano -> vi.
    """
    # Strict fallback order as specified in Plan v1.1 (Section 3.B)
    preferred_editors = ['nano', 'vi']
    for editor in preferred_editors:
        path = shutil.which(editor)
        if path:
            logger.info(f"Found preferred editor: {path}")
            return path

    logger.warning(f"Preferred editors ('nano', 'vi') not found in PATH.")
    # Optional: Check for VISUAL/EDITOR environment variables as a common convention
    editor_env = os.environ.get('VISUAL') or os.environ.get('EDITOR')
    if editor_env:
        path_env = shutil.which(editor_env)
        if path_env:
            logger.info(f"Found editor from environment variable (VISUAL/EDITOR): {path_env}")
            return path_env
        else:
             logger.warning(f"Editor specified in VISUAL/EDITOR ('{editor_env}') not found in PATH.")

    # Fallback for Windows (if nano/vi aren't available, notepad is common)
    if os.name == 'nt':
         notepad_path = shutil.which('notepad')
         if notepad_path:
              logger.info("Falling back to Notepad on Windows.")
              return notepad_path

    logger.warning("No suitable text editor found automatically.")
    return None

def run_interactive_review(initial_text: str,
                           review_filepath_in_cwd: str,
                           failed_validation_info: Optional[str] = None) -> Optional[str]:
    """
    Writes text to a file, opens it in an editor (nano -> vi -> fallback input),
    waits, and reads it back.
    Includes validation failure info if provided.

    Args:
        initial_text: The text to write to the review file (e.g., LLM proposal or failed JSON).
        review_filepath_in_cwd: The path to the review file (relative to CWD).
        failed_validation_info: Optional string explaining previous validation failure.
    Returns:
        The content of the file after editing, or None on error or abort.
    """
    # Ensure review file path is absolute based on CWD for clarity
    abs_review_filepath = os.path.abspath(review_filepath_in_cwd)
    logger.info(f"Writing LLM output/proposal to review file: '{abs_review_filepath}'")

    # Standard instructions for the review file
    validation_header = ""
    if failed_validation_info:
        validation_header = f"""
# === ATTENTION: VALIDATION FAILED ===
# The previous LLM attempt to generate/correct the specification failed validation.
# Error: {failed_validation_info}
# Please review the content below AND the error message, then make corrections.
# ======================================

"""

    instructions = f"""{validation_header}
# === Review & Edit Specification Below ===
# Instructions:
# 1. Carefully review the LLM-generated requirements/scenarios below the '--- Start ---' marker.
#    (If validation failed previously, the content below is the *invalid* output).
# 2. Make any necessary corrections, additions, or clarifications directly in this text.
#    Focus on WHAT the system should do (specification), not HOW (implementation).
#    If fixing validation errors, ensure the structure matches the required JSON format.
# 3. If you have specific instructions for the *next* LLM step (Sanitization & Formatting),
#    add them below the '--- USER CLARIFICATION ---' marker.
# 4. Save this file and close the editor when you are finished.
# =========================================

# --- Start of LLM Proposal / Failed Output ---

"""
    end_marker = """

# --- End of LLM Proposal / Failed Output ---

# --- USER CLARIFICATION (Optional) ---
# Add specific instructions for the LLM Sanitization & Formatting step here.
# Example: "Ensure all user authentication requirements are grouped under section 2."
# Example: "Fix the missing 'id' field in the second test scenario."
"""
    # Write the initial content with instructions to the review file
    try:
        with open(abs_review_filepath, "w", encoding='utf-8') as f:
            f.write(instructions + initial_text + end_marker)
    except IOError as e:
        logger.error(f"I/O error writing review file '{abs_review_filepath}': {e}")
        return None

    # Attempt to open the file in a text editor using the specified fallback
    editor_path = find_editor()
    editor_launched = False
    if editor_path:
        editor_name = os.path.basename(editor_path)
        print(f"\n>>> Opening '{os.path.basename(abs_review_filepath)}' in editor '{editor_name}'.")
        print(">>> Please review/edit the file, then save & close the editor.")
        try:
            # Use Popen and wait to ensure editor closes before script continues
            # Prefer list format if possible.
            process = subprocess.Popen([editor_path, abs_review_filepath])
            process.wait() # Wait for the editor process to terminate
            editor_launched = True # Mark that we attempted and waited for the editor
            if process.returncode != 0:
                # Log non-zero exit code but proceed, user might have saved anyway
                logger.warning(f"Editor '{editor_name}' exited with code {process.returncode}.")
        except OSError as e:
            logger.error(f"Failed to launch editor '{editor_path}': {e}. Falling back to manual editing prompt.")
            # editor_path = None # No need to nullify, just use editor_launched flag
        except Exception as e:
            logger.error(f"An unexpected error occurred while running the editor '{editor_path}': {e}", exc_info=True)
            # editor_path = None

    # Fallback to input prompt if editor couldn't be found or launched
    # This implements the nano -> vi -> input() fallback from Plan v1.1 Sec 3.B
    if not editor_launched:
        # find_editor logs if it can't find nano/vi, Popen logs if launch fails.
        print(f"\n>>> Could not automatically open a preferred text editor (nano/vi).")
        print(f">>> Please manually open and edit the file: '{abs_review_filepath}'")
        try:
             input(">>> After saving your changes, press Enter here to continue...")
        except EOFError: # Handle cases where input stream is closed (e.g., CI environment)
             logger.warning("EOFError received on input prompt. Assuming non-interactive environment. Cannot proceed with interactive review.")
             # Clean up the temp file?
             try:
                 os.remove(abs_review_filepath)
             except OSError:
                 pass
             return None # Indicate failure to review

    # Read the potentially modified file content back
    try:
        with open(abs_review_filepath, "r", encoding='utf-8') as f:
           final_text = f.read()
        logger.info(f"Successfully read review file '{abs_review_filepath}' after user review.")

        # Clean up the review file after successful read
        try:
            os.remove(abs_review_filepath)
            logger.debug(f"Removed temporary review file: {abs_review_filepath}")
        except OSError as e:
           logger.warning(f"Could not remove temporary review file '{abs_review_filepath}': {e}")

        return final_text
    except IOError as e:
        logger.error(f"I/O error reading review file '{abs_review_filepath}' after edit: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error reading review file '{abs_review_filepath}' after edit: {e}", exc_info=True)
        return None
