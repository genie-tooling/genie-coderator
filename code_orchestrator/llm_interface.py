# Filename: llm_interface.py
# -*- coding: utf-8 -*-
# Description: Handles interactions with the LLM API (Google Gemini).
# Version: 1.4.0 (Removed response_schema usage due to SDK incompatibility)

import time
import logging
import json
import google.generativeai as genai
# Correct imports based on common library structure and previous errors
from google.generativeai.types import GenerationConfig, ContentDict
# FinishReason might be directly on the candidate object or under genai.types
# Let's try accessing via candidate first, and use genai.types.FinishReason for comparison constants


try:
    import tiktoken
except ImportError:
    print("ERROR: tiktoken not found. Please install: pip install tiktoken")
    exit(1)

from typing import Optional, Dict, Any, Union, Type, List
from pydantic import BaseModel, ValidationError

# Import config and models using relative imports
from . import config
from . import models
from . import utils # For safe_json_loads


logger = logging.getLogger(__name__)

# --- Tiktoken Setup ---
try:
    # Using a standard encoder, as specific model mapping can be fragile
    tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
    logger.info("Using tiktoken 'cl100k_base' encoder.")
except Exception as e_tiktoken:
    logger.warning(f"Could not get tiktoken 'cl100k_base' encoder ({e_tiktoken}), falling back to 'gpt2'. Token counts may be approximate.")
    try:
        tiktoken_encoding = tiktoken.get_encoding("gpt2")
    except Exception as e_gpt2:
        logger.critical(f"FATAL: Could not get any tiktoken encoder (cl100k_base or gpt2): {e_gpt2}. Tokenization will fail.")
        tiktoken_encoding = None # Ensure it's None if fails


# --- Core LLM Interaction Functions ---

def call_gemini_api(prompt_text: str,
                    model_name: str = config.GEMINI_MODEL_NAME,
                    temperature: Optional[float] = config.DEFAULT_API_TEMPERATURE,
                    output_schema: Optional[Type[BaseModel]] = None, # Kept for logging/intent
                    attempt: int = 1) -> Optional[str]:
    """
    Calls the Gemini API with specified prompt, model, temperature.
    Requests JSON output if output_schema is specified, but DOES NOT pass the
    schema directly to the API due to SDK compatibility issues.
    Handles retries, error logging, and basic response validation.
    Extracts text content from response parts.

    Args:
        prompt_text: The complete prompt string.
        model_name: The name of the Gemini model to use.
        temperature: The sampling temperature (None for API default).
        output_schema: If provided, indicates JSON output is expected. NOT passed to API.
        attempt: The current retry attempt number.

    Returns:
        The extracted text content from the LLM response as a string (likely JSON if schema used),
        or a JSON string representing an error, or None if retries fail badly.
    """
    prompt_tokens = get_token_count(prompt_text)
    temp_str = f" (Temp: {temperature})" if temperature is not None else " (Temp: Default)"
    # Log intent even if schema not passed to API
    schema_str = f" (Expecting Schema: {output_schema.__name__})" if output_schema else ""
    logger.info(
        f"Calling Gemini ({model_name}, Attempt {attempt}/{config.API_MAX_RETRIES})"
        f"{temp_str}{schema_str}... Input Tokens(est): {prompt_tokens}"
    )
    logger.debug(f"Prompt start (first 500 chars):\n{prompt_text[:500]}...")


    # Check token limit before API call
    effective_limit = config.MAX_CONTEXT_HISTORY_TOKENS
    if prompt_tokens > effective_limit:
         error_msg = f"Estimated prompt tokens ({prompt_tokens}) exceed configured limit ({effective_limit}). Aborting call."
         logger.error(error_msg)
         # Return error in consistent JSON format
         return json.dumps({"error": error_msg})

    try:
        # Ensure API key is configured
        if not config.settings.gemini_api_key:
             raise ValueError("Gemini API key not configured.")

        # Pass the list of dictionaries directly from config
        safety_settings_to_pass = config.SAFETY_SETTINGS
        logger.debug(f"Using safety settings: {safety_settings_to_pass}")

        # --- Configure Generation Based on Output Expectation ---
        gen_config_params = {}
        if temperature is not None:
            # Ensure temperature is within valid range [0.0, 1.0]
            gen_config_params['temperature'] = max(0.0, min(temperature, 1.0))

        # *** MODIFICATION START: Remove response_schema usage ***
        if output_schema:
            # If a schema is expected by the caller, request JSON output type.
            logger.debug(f"Requesting JSON output (schema validation happens post-call).")
            gen_config_params['response_mime_type'] = "application/json"
            # DO NOT ADD: gen_config_params['response_schema'] = output_schema
        else:
            # Default to text if no schema provided by caller
             gen_config_params['response_mime_type'] = "text/plain"
        # *** MODIFICATION END ***


        # Use GenerationConfig directly
        # Filter out None values explicitly if GenerationConfig doesn't handle them well
        filtered_gen_config_params = {k: v for k, v in gen_config_params.items() if v is not None}
        generation_config_obj = GenerationConfig(**filtered_gen_config_params) if filtered_gen_config_params else None
        logger.debug(f"GenerationConfig params used: {filtered_gen_config_params}")

        model = genai.GenerativeModel(
            model_name,
            safety_settings=safety_settings_to_pass,
            # generation_config=generation_config_obj # Passed to generate_content
        )


        # --- API Call ---
        # Pass generation_config directly to generate_content
        response = model.generate_content(
            prompt_text,
            generation_config=generation_config_obj
        )


        # --- Process Response ---
        # Handle blocked prompt first
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            block_reason = response.prompt_feedback.block_reason
            # Use getattr safely for name attribute
            reason_name = getattr(block_reason, 'name', str(block_reason))
            logger.error(f"API blocked prompt. Reason: {reason_name}")
            if hasattr(response.prompt_feedback, 'safety_ratings'):
                logger.debug(f"Safety Ratings (Prompt): {response.prompt_feedback.safety_ratings}")
            return json.dumps({"error": f"API Blocked Prompt: {reason_name}"})

        if not response.candidates:
             logger.error("API Error: No candidates returned in response.")
             logger.debug(f"Full API Response (raw): {response}")
             # Raise specific error for retry logic
             raise ValueError("No candidates received in API response")

        candidate = response.candidates[0]
        # Ensure finish_reason exists before accessing name
        finish_reason_enum = getattr(candidate, 'finish_reason', None)
        finish_reason_name = getattr(finish_reason_enum, 'name', 'UNKNOWN') if finish_reason_enum else 'UNKNOWN'


        # Use try-except for robust enum access
        try:
            FinishReason = genai.types.FinishReason
            STOP_REASON = FinishReason.STOP
            SAFETY_REASON = FinishReason.SAFETY
            MAX_TOKENS_REASON = FinishReason.MAX_TOKENS
            RECITATION_REASON = FinishReason.RECITATION
            OTHER_REASON = FinishReason.OTHER # Added OTHER for completeness
        except AttributeError:
            logger.warning("Could not access genai.types.FinishReason enum constants. Falling back to string comparison for finish reasons.")
            FinishReason = None # Indicate enum is not available
            STOP_REASON = "STOP"
            SAFETY_REASON = "SAFETY"
            MAX_TOKENS_REASON = "MAX_TOKENS"
            RECITATION_REASON = "RECITATION"
            OTHER_REASON = "OTHER"

        # Choose comparison value based on enum availability
        current_reason_compare = finish_reason_enum if FinishReason else finish_reason_name

        # Check finish reason (STOP is the expected success case)
        if current_reason_compare != STOP_REASON:
            logger.warning(f"API Warning: Generation finished unexpectedly. Reason: {finish_reason_name}")
            if current_reason_compare == SAFETY_REASON:
                logger.error("API blocked response content due to safety filters.")
                if hasattr(candidate, 'safety_ratings'):
                    logger.debug(f"Safety Ratings (Candidate): {candidate.safety_ratings}")
                return json.dumps({"error": "API Blocked Response Content (Safety)"})
            elif current_reason_compare == MAX_TOKENS_REASON:
                logger.error("API stopped due to reaching maximum output tokens.")
                # Let partial text extraction proceed, maybe partial JSON is usable
                pass
            elif current_reason_compare == RECITATION_REASON:
                 logger.error("API stopped due to potential recitation.")
                 return json.dumps({"error": "API Stopped: Recitation"})
            elif current_reason_compare == OTHER_REASON:
                 logger.error("API stopped due to 'OTHER' reason.")
                 # Potentially raise or return error, as this is ambiguous
                 # Raise ValueError to trigger retry maybe?
                 raise ValueError(f"API generation finished with 'OTHER' reason. Response: {response}")
            # Handle other potential non-STOP reasons if necessary

        # Extract text content from parts
        response_text = None
        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts:
            text_parts = []
            for part in candidate.content.parts:
                if hasattr(part, 'text') and part.text is not None: # Check text is not None
                     text_parts.append(part.text)
                # Log if a part exists but has no text (might happen with non-text modalities)
                elif hasattr(part, 'text') and part.text is None:
                    logger.debug(f"Candidate part had 'text' attribute but it was None: {part}")
                else:
                    logger.debug(f"Candidate part did not contain 'text' attribute or was empty: {part}")

            if text_parts:
                 response_text = "".join(text_parts).strip()
                 if len(text_parts) > 1:
                      logger.debug(f"Joined {len(text_parts)} text parts from API response.")
            else:
                 logger.warning("API response candidate had content parts, but none contained usable text.")
        elif not hasattr(candidate, 'content') or not hasattr(candidate.content, 'parts'):
             logger.warning("API response candidate structure missing 'content' or 'parts' attribute.")

        # Check for empty content after normal finish reason
        is_normal_stop = (current_reason_compare == STOP_REASON)
        if not response_text and is_normal_stop:
             # If we requested JSON, an empty response on STOP is likely an error.
             if output_schema: # Check caller's intent
                  logger.error("API returned no text content despite normal 'STOP' finish reason when JSON output was expected.")
                  # Raise error to potentially trigger retry?
                  raise ValueError("API returned empty content despite STOP finish, when JSON output was expected.")
             else:
                  # For text generation, empty might be valid in some contexts, but log warning.
                  logger.warning("API returned no usable text content despite normal 'STOP' finish reason.")
                  # Return empty string? Or specific error? Returning error JSON seems safer.
                  return json.dumps({"error": "API returned empty content despite normal finish."})


        # Log output size if text exists
        if response_text:
            # Estimate bytes - useful for non-JSON too
            output_bytes = len(response_text.encode('utf-8'))
            logger.info(f"Response Output Bytes(est): {output_bytes}")
            # Check against configured limits (optional, but good practice)
            check_output_size(output_bytes, model_name)

        # Before returning, check if the response *itself* is a structured error JSON
        # This might be returned even if the API call itself didn't fail initially.
        if response_text:
             parsed_json_for_error_check = utils.safe_json_loads(response_text)
             if isinstance(parsed_json_for_error_check, dict) and "error" in parsed_json_for_error_check:
                 # Check if it ONLY contains the error key or maybe a few standard keys
                 if len(parsed_json_for_error_check) <= 2: # Allow maybe 'error' and 'details'
                    llm_returned_error = parsed_json_for_error_check['error']
                    logger.error(f"LLM returned structured error JSON in response content: {llm_returned_error}")
                    # Return the structured error JSON as is
                    return response_text
                 else:
                    logger.debug("Response contained an 'error' key but also other keys, treating as data.")


        return response_text # Return the raw text (likely JSON if schema was used)

    # Exception Handling (includes retries)
    except Exception as e:
        # Log different levels based on retry attempt
        log_level = logging.ERROR if attempt == config.API_MAX_RETRIES else logging.WARNING
        logger.log(log_level, f"API call failed (Attempt {attempt}/{config.API_MAX_RETRIES}): {e}", exc_info=(attempt == config.API_MAX_RETRIES))

        if attempt < config.API_MAX_RETRIES:
            # Increase delay for certain errors (like schema issues which might resolve with API updates?)
            # Keep standard delay for now
            retry_delay = config.API_RETRY_DELAY * (2**(attempt - 1)) # Exponential backoff
            logger.info(f"Retrying API call in {retry_delay} seconds...")
            time.sleep(retry_delay)
            # Recursive call for retry
            return call_gemini_api(
                prompt_text,
                model_name,
                temperature,
                output_schema, # Pass caller's intent along for retry logging
                attempt + 1
            )
        else:
            logger.error("API call failed after maximum retries.")
            # Return structured error
            return json.dumps({"error": f"API call failed after {config.API_MAX_RETRIES} retries: {e}"})


def parse_llm_response(response_text: Optional[str],
                         model_type: Type[BaseModel]) -> Optional[Union[BaseModel, models.ErrorOutput]]:
    """
    Parses the LLM response string, validates it against a Pydantic model.
    Handles JSON decoding errors, Pydantic validation errors, and
    checks if the response itself is a structured ErrorOutput.
    Provides more detailed error messages for correction prompts.

    Args:
        response_text: The string response from the LLM API call (expected to be JSON).
        model_type: The Pydantic model class to validate against.

    Returns:
        An instance of the validated Pydantic model if successful,
        an instance of ErrorOutput if parsing/validation fails or
        if the LLM returned a structured error, or None if input is None.
    """
    if response_text is None:
        logger.error(f"Cannot parse None response for {model_type.__name__}.")
        # Return structured error instead of None
        return models.ErrorOutput(error="Empty or None API response received.")

    # Use safe_json_loads which handles Markdown blocks and initial JSON errors
    json_data = utils.safe_json_loads(response_text)

    if json_data is None:
        # safe_json_loads already logged the JSONDecodeError details
        logger.error(f"Failed to decode or extract valid JSON from response text for {model_type.__name__}.")
        # Provide context in the error output
        error_detail = f"Invalid JSON response format received from LLM. Raw start: '{response_text[:200]}...'"
        return models.ErrorOutput(error=error_detail)

    # Check if the successfully parsed JSON is actually an error structure
    # (This check is potentially redundant if call_gemini_api already returns it, but provides defense in depth)
    if isinstance(json_data, dict) and "error" in json_data:
        # Maybe check len(json_data) == 1 to be more specific?
        llm_error_msg = json_data.get('error', 'Unknown LLM error structure returned')
        logger.warning(f"LLM response content appears to be a structured error for {model_type.__name__}: {llm_error_msg}")
        # Return the error wrapped in our standard ErrorOutput model
        return models.ErrorOutput(error=f"LLM Processing Error: {llm_error_msg}")

    try:
        # Validate the parsed JSON data against the expected Pydantic model
        validated_data = model_type.model_validate(json_data)
        logger.debug(f"Successfully parsed and validated response for {model_type.__name__}.")
        return validated_data
    except ValidationError as e:
        # Pydantic validation failed
        logger.error(f"Pydantic validation failed for {model_type.__name__}: {e}")

        # --- Enhanced Error Message Generation ---
        error_summary = f"Pydantic validation failed for {model_type.__name__}. Details:\n"
        try:
            # Extract details from Pydantic's error structure
            error_details = e.errors() # Get list of error dicts
            for err in error_details:
                loc = " -> ".join(map(str, err.get('loc', ('N/A',)))) # Field path
                msg = err.get('msg', 'Unknown error')
                err_type = err.get('type', 'N/A')
                # input_snippet = str(err.get('input', 'N/A'))[:100] # Get snippet of invalid input
                error_summary += f"- Location: '{loc}', Type: '{err_type}', Message: '{msg}'\n"
        except Exception as format_err:
            logger.error(f"Could not format detailed Pydantic error: {format_err}")
            error_summary = f"Pydantic validation failed for {model_type.__name__}: {e}" # Fallback

        logger.debug(f"Validation Error Summary for Correction Prompt:\n{error_summary}")
        # --- End Enhanced Error Message Generation ---

        # Log the problematic JSON data structure for debugging
        try:
            invalid_json_str = json.dumps(json_data, indent=2)
            logger.debug(f"Invalid JSON data structure received (failed Pydantic validation):\n{invalid_json_str}")
        except TypeError:
             # Handle cases where json_data might not be serializable (less likely here)
             logger.debug(f"Invalid (non-serializable) JSON data structure received: {json_data}")

        # Return a structured error with the enhanced details
        return models.ErrorOutput(error=error_summary) # Use the detailed summary

    except Exception as e:
        # Catch unexpected errors during the validation process itself
        logger.error(f"Unexpected error during Pydantic validation for {model_type.__name__}: {e}", exc_info=True)
        return models.ErrorOutput(error=f"Unexpected validation error for {model_type.__name__}: {e}")

# --- Token Counting and Truncation ---

def get_token_count(text: str) -> int:
    """Estimates token count using the globally defined tiktoken encoder."""
    if not text:
        return 0
    if tiktoken_encoding is None:
        logger.error("Tiktoken encoder not initialized. Cannot count tokens accurately. Using char/4 approximation.")
        # Use integer division
        return len(text) // 4
    try:
        return len(tiktoken_encoding.encode(text))
    except Exception as e:
        logger.error(f"Tiktoken encoding failed: {e}. Falling back to char count / 4 approximation.")
        # Use integer division
        return len(text) // 4

def truncate_prompt_context(context_string: str, token_budget: int, context_name: str = "Context") -> str:
    """
    Truncates a generic text string (like code, JSON, or requirements)
    to fit within a specified token budget. Prioritizes keeping the end.

    Args:
        context_string: The text string to truncate.
        token_budget: The maximum number of tokens allowed.
        context_name: A descriptive name for logging purposes (e.g., "Code", "Requirements JSON").

    Returns:
        The truncated string, possibly with a marker comment/prefix.
    """
    if token_budget <= 0:
        logger.warning(f"{context_name} token budget is zero or negative ({token_budget}). Returning empty context marker.")
        # Return a clear marker indicating truncation due to budget
        return f"# [{context_name} Truncated: Zero/Negative Budget]"

    if not context_string:
        # Return empty string if input is empty
        return context_string

    current_token_count = get_token_count(context_string)
    if current_token_count <= token_budget:
         logger.debug(f"{context_name} fits within token budget ({current_token_count} <= {token_budget} tokens).")
         return context_string

    # If using approximation, apply directly
    if tiktoken_encoding is None:
        logger.warning(f"Using character approximation for {context_name} truncation.")
        estimated_chars_per_token = 4
        char_budget = token_budget * estimated_chars_per_token
        original_len = len(context_string)
        truncated_context = context_string[-char_budget:]
        removed_chars = original_len - len(truncated_context)
        logger.warning(
            f"{context_name} truncated (char fallback): Original {original_len} chars -> "
            f"Kept approx {len(truncated_context)} chars (Budget: {token_budget} tokens ~ {char_budget} chars, Removed: {removed_chars} chars)."
        )
        return f"# [{context_name} Truncated (Character Fallback) - {removed_chars} chars removed from start]\n...\n" + truncated_context

    # Use tiktoken for accurate truncation
    try:
        tokens = tiktoken_encoding.encode(context_string)
        original_token_count = len(tokens) # Should match current_token_count if no error

        # Keep the tokens from the end
        truncated_tokens = tokens[-token_budget:]
        kept_token_count = len(truncated_tokens)
        removed_count = original_token_count - kept_token_count
        try:
            # Decode the truncated tokens back to string
            truncated_context = tiktoken_encoding.decode(truncated_tokens)
            logger.warning(
                f"{context_name} truncated: Original {original_token_count} tokens -> "
                f"Kept {kept_token_count} tokens (Budget: {token_budget}, Removed: {removed_count} tokens)."
            )
            # Add a clear marker
            return f"# [{context_name} Truncated - {removed_count} tokens removed from start]\n...\n" + truncated_context
        except Exception as decode_error:
            # Handle potential errors during decoding (rare)
            logger.error(f"Tiktoken decode error after {context_name} truncation: {decode_error}. Returning simple marker.")
            # Return a very basic marker indicating failure after truncation
            return f"# [{context_name} Heavily Truncated due to decode error: Original {original_token_count} tokens, Budget {token_budget}]"

    except Exception as e:
        # Handle errors during the tokenization itself
        logger.error(f"Error during tiktoken {context_name} tokenization/truncation: {e}. Falling back to char slicing.", exc_info=True)
        # Use the same character fallback as when encoder is None
        estimated_chars_per_token = 4
        char_budget = token_budget * estimated_chars_per_token
        if len(context_string) > char_budget:
             logger.warning(f"{context_name} truncated (char fallback after encode error): Original {len(context_string)} chars -> Kept approx {char_budget} chars.")
             return f"# [{context_name} Truncated (Character Fallback after Error)]\n...\n" + context_string[-char_budget:]
        else:
             # Fits within fallback estimate
             return context_string


def check_output_size(output_bytes: int, model_name: str):
    """Checks if the output size exceeds the configured limit for the model."""
    try:
        # Find the most specific matching key in OUTPUT_BYTE_LIMITS
        model_base_key = 'default'
        # Iterate through keys (excluding default) to find a match in the model name
        for key in config.OUTPUT_BYTE_LIMITS:
             if key != 'default' and key in model_name:
                 model_base_key = key
                 # Use the first specific match found
                 break

        # Get the byte limit for the determined key, fallback to default
        byte_limit = config.OUTPUT_BYTE_LIMITS.get(model_base_key, config.OUTPUT_BYTE_LIMITS['default'])

        # Compare and log
        if output_bytes > byte_limit:
            logger.warning(
                f"Response output bytes ({output_bytes}) EXCEEDS estimated limit ({byte_limit}) "
                f"for model base '{model_base_key}' (derived from '{model_name}')!"
            )
        else:
            logger.debug(f"Output size {output_bytes} bytes is within limit {byte_limit} for model base '{model_base_key}'.")

    except Exception as e:
        # Catch errors during the check itself
        logger.error(f"Error checking output size against limits: {e}", exc_info=True)
