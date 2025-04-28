# Filename: prompts.py
# -*- coding: utf-8 -*-
# Description: Loads LLM prompt templates from plain text files (.prompt) in the prompts/ directory.
# Version: 3.0.0 (Refactored to load from .prompt files)

import os
import logging
import pathlib # Using pathlib for easier path manipulation

logger = logging.getLogger(__name__)

# --- Dynamic Prompt Loading from .prompt files ---

_prompt_data = {} # Optional: Store loaded prompts internally if needed later

try:
    # Determine the directory containing this script file
    _current_dir = pathlib.Path(__file__).parent.resolve()
    # Assume 'prompts' directory is a sibling to this 'prompts.py' file
    _prompts_dir = _current_dir / "prompts"

    if not _prompts_dir.is_dir():
        # Fallback logic: If not found relative to the script, maybe try relative to CWD?
        _cwd_prompts_dir = pathlib.Path("./prompts").resolve()
        if _cwd_prompts_dir.is_dir():
            _prompts_dir = _cwd_prompts_dir
            logger.warning(f"Prompts directory not found relative to prompts.py ({_current_dir / 'prompts'}), using relative to CWD ({_prompts_dir}).")
        else:
            logger.error(f"Prompts directory not found at expected location relative to script ({_current_dir / 'prompts'}) or CWD ({_cwd_prompts_dir}).")
            raise FileNotFoundError(f"Prompts directory not found: tried {_current_dir / 'prompts'} and {_cwd_prompts_dir}")

    logger.info(f"Loading prompts from .prompt files in: {_prompts_dir}")

    _loaded_prompt_names = set() # Keep track of what was loaded
    prompt_file_extension = ".prompt"

    for filepath in _prompts_dir.glob(f"*{prompt_file_extension}"):
        prompt_variable_name = None # Define before try block
        try:
            # Extract variable name from filename
            prompt_variable_name = filepath.stem # Gets filename without extension

            # Validate variable name (must be valid Python identifier, typically uppercase)
            if not prompt_variable_name.isidentifier() or not prompt_variable_name.isupper():
                 logger.warning(f"Skipping prompt file {filepath.name}: Filename '{prompt_variable_name}' is not a valid uppercase Python identifier.")
                 continue

            # Read the raw content of the prompt file
            with open(filepath, 'r', encoding='utf-8') as f:
                prompt_content = f.read()

            # Assign to globals() to make them available as module variables like prompts.PROMPT_NAME
            globals()[prompt_variable_name] = prompt_content
            _prompt_data[prompt_variable_name] = prompt_content # Store internally if needed
            _loaded_prompt_names.add(prompt_variable_name)
            logger.debug(f"Loaded prompt: {prompt_variable_name} from {filepath.name}")

        except IOError as e:
            logger.error(f"Error reading prompt file {filepath.name}: {e}")
        except Exception as e:
            logger.error(f"Error processing prompt file {filepath.name} (Variable: {prompt_variable_name or 'Unknown'}): {e}", exc_info=True)

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
try:
    del _current_dir, _prompts_dir, _cwd_prompts_dir, prompt_file_extension
    del filepath, prompt_variable_name, prompt_content
    del _loaded_prompt_names
except NameError:
    pass # Ignore if variables weren't assigned due to errors

# Example check (optional):
# def get_prompt(prompt_name: str) -> str:
#     """Safely retrieves a loaded prompt."""
#     prompt = globals().get(prompt_name)
#     if prompt is None:
#         logger.error(f"Required prompt '{prompt_name}' failed to load or does not exist.")
#         return f"# ERROR: Prompt '{prompt_name}' not loaded! #"
#     return prompt
