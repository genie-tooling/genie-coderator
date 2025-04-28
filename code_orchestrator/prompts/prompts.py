# Filename: prompts.py
# -*- coding: utf-8 -*-
# Description: Loads LLM prompt templates from JSON files in the prompts/ directory.
# Version: 2.0.0 (Refactored to load from JSON)

import os
import json
import logging
import pathlib # Using pathlib for easier path manipulation

logger = logging.getLogger(__name__)

# --- Dynamic Prompt Loading ---

_prompt_data = {} # Optional: Store loaded prompts internally if needed later

try:
    # Determine the directory containing this script file
    _current_dir = pathlib.Path(__file__).parent.resolve()
    # Assume 'prompts' directory is a sibling to this 'prompts.py' file
    _prompts_dir = _current_dir / "prompts"

    if not _prompts_dir.is_dir():
        # Fallback logic: If not found relative to the script, maybe try relative to CWD?
        # This might happen if the execution structure is different.
        _cwd_prompts_dir = pathlib.Path("./prompts").resolve()
        if _cwd_prompts_dir.is_dir():
            _prompts_dir = _cwd_prompts_dir
            logger.warning(f"Prompts directory not found relative to prompts.py ({_current_dir / 'prompts'}), using relative to CWD ({_prompts_dir}).")
        else:
            logger.error(f"Prompts directory not found at expected location relative to script ({_current_dir / 'prompts'}) or CWD ({_cwd_prompts_dir}).")
            # Raise an error or define prompts as None/empty strings? Raising seems safer.
            raise FileNotFoundError(f"Prompts directory not found: tried {_current_dir / 'prompts'} and {_cwd_prompts_dir}")

    logger.info(f"Loading prompts from JSON files in: {_prompts_dir}")

    _loaded_prompt_names = set() # Keep track of what was loaded

    for filepath in _prompts_dir.glob("*.json"):
        prompt_variable_name = None # Define before try block
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            prompt_variable_name = data.get("name") # Get name from JSON metadata
            prompt_template = data.get("template")

            if prompt_variable_name and isinstance(prompt_variable_name, str) and prompt_template is not None:
                # Basic validation of the name format (optional)
                if not prompt_variable_name.isidentifier() or not prompt_variable_name.isupper():
                     logger.warning(f"Skipping prompt file {filepath.name}: 'name' key '{prompt_variable_name}' is not a valid uppercase Python identifier.")
                     continue

                # Assign to globals() to make them available as module variables like prompts.PROMPT_NAME
                globals()[prompt_variable_name] = prompt_template
                _prompt_data[prompt_variable_name] = prompt_template # Store internally
                _loaded_prompt_names.add(prompt_variable_name)
                logger.debug(f"Loaded prompt: {prompt_variable_name} from {filepath.name}")
            else:
                logger.warning(f"Skipping invalid prompt file {filepath.name}: Missing or invalid 'name' or 'template' key.")

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {filepath.name}: {e}")
        except Exception as e:
            logger.error(f"Error processing prompt file {filepath.name} (Prompt Variable: {prompt_variable_name or 'Unknown'}): {e}", exc_info=True)

    # Optional: Check if any prompts were loaded at all
    if not _loaded_prompt_names:
        logger.warning(f"No prompts were successfully loaded from '{_prompts_dir}'. Prompt variables will be undefined.")

except FileNotFoundError as e:
     # Logged above, re-raise or handle? Let's log and continue, variables won't exist.
     logger.critical(f"Could not load prompts: {e}. Prompt variables will be undefined.")
except Exception as e:
    logger.critical(f"Unexpected error during prompt loading: {e}", exc_info=True)
    # Ensure prompt variables might not exist if loading fails

# --- End Dynamic Prompt Loading ---

# Clean up helper variables used only during loading
# These are module-scoped anyway, but explicit deletion can sometimes help clarity
try:
    del _current_dir, _prompts_dir, _cwd_prompts_dir
    del filepath, data, prompt_variable_name, prompt_template
    del _loaded_prompt_names
except NameError:
    pass # Ignore if variables weren't assigned due to errors


# Example check for use in other modules (optional):
# def get_prompt(prompt_name: str) -> str:
#     """Safely retrieves a loaded prompt."""
#     if prompt_name not in globals():
#         logger.error(f"Required prompt '{prompt_name}' failed to load or does not exist.")
#         # Return a default error message or raise an exception
#         return f"# ERROR: Prompt '{prompt_name}' not loaded! #"
#     return globals()[prompt_name]

