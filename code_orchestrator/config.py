# -*- coding: utf-8 -*-
# Filename: config.py
# Description: Configuration settings and constants for the Gemini Code Orchestrator.
# Version: 1.8.0 (Added State Management)

import os
import time
import logging
from typing import List, Dict, Optional
try:
    import google.generativeai as genai
except ImportError:
    print("ERROR: google-generativeai not found. Please install: pip install google-generativeai")
    exit(1)
try:
    from pydantic import Field
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    print("ERROR: pydantic or pydantic-settings not found. Please install: pip install pydantic pydantic-settings python-dotenv")
    exit(1)

logger = logging.getLogger(__name__)

# --- Pydantic Settings Model ---
class Settings(BaseSettings):
    """Loads settings from environment variables or .env file."""
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    gemini_api_key: str = Field(..., validation_alias="GEMINI_API_KEY")

# --- Load Settings ---
try:
    settings = Settings()
    # Configure the Gemini client globally on import
    genai.configure(api_key=settings.gemini_api_key)
    logger.info("Gemini API key loaded and configured.")
except Exception as e:
    # Catching broad exception during initial setup is critical
    logger.critical(f"FATAL: Configuration error during settings load or genai config: {e}", exc_info=True)
    exit(1)

# --- Core Model and API Configuration ---
# Set Gemini 2.5 as the default model name
GEMINI_MODEL_NAME: str = 'gemini-2.5-pro-exp-03-25'
CODE_GENERATION_TEMPERATURE: Optional[float] = 0.3   # Lower temp for precision in code generation/debugging
PLANNING_TEMPERATURE: Optional[float] = 0.6        # Moderate temp for planning/design/extraction
DEFAULT_API_TEMPERATURE: Optional[float] = None      # Use API default for other tasks (reqs, docs) - often higher

# Set context limit for Gemini 2.5
MAX_CONTEXT_HISTORY_TOKENS: int = 1000000 # Max tokens for history/context (Gemini 2.5 specific)

SAFETY_SETTINGS: List[Dict[str, str]] = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
API_MAX_RETRIES: int = 3
API_RETRY_DELAY: int = 5 # Base seconds delay for retries

# --- Output Size Configuration ---
# Note: These align with Plan v1.1's need for output size management during Planning.
# Estimated limits (verify based on actual model documentation)
OUTPUT_BYTE_LIMITS: Dict[str, int] = {
    # Specific entry for Gemini 2.5
    'gemini-2.5': 65536, # Keeping lower for safety during planning/design generation/extraction
    'gemini-1.5': 131072, # Keeping 1.5 limit for reference if needed
    'gemini-1.0': 8192,
    'default': 8192 # Fallback default
}
# Target output size as fraction of limit - Lowered for safety margin
TARGET_BYTES_FACTOR: float = 0.55 # Aim for ~35k for Gemini 2.5 during planning estimates

# --- File Paths and Naming Conventions ---
# These are defaults; `configure_paths` in file_utils sets the runtime values
DEFAULT_BASE_OUTPUT_DIR: str = "." # Default place to create sandbox/docs
DEFAULT_SANDBOX_SUBDIR: str = "sandbox"
DEFAULT_DOCS_SUBDIR: str = "docs" # As per Plan v1.1 Phase 3.E
STATE_FILENAME: str = "_pipeline_state.json" # State file (T005)

# File names used *within* the sandbox
REQUIREMENTS_FILENAME: str = "requirements.txt"
VENV_DIR_NAME: str = ".venv"
# File names for saving orchestration artifacts *within* the sandbox
# Aligning filenames with Plan v1.1 concepts where applicable
FINAL_SPEC_FILENAME: str = "_final_specification.json" # Output of Phase 0 (if run)
EXTRACTED_SUMMARY_FILENAME: str = "_extracted_summary.txt" # Output of blueprint summary extraction (NEW)
PROJECT_STRUCTURE_FILENAME: str = "_project_structure.json" # Output of Phase 1a (gen or extract)
PLAN_CHUNK_LIST_FILENAME: str = "_plan_chunk_list.json" # Output of initial plan chunking
CONCEPTUAL_PLAN_FILENAME: str = "_conceptual_plan_reference.txt" # Meta-info file
DETAILED_PLAN_FILENAME: str = "_final_phasing_plan.json" # Output of Phase 1c (aggregated)

# Other file names (used outside sandbox)
DEFAULT_TASK_FILE: str = "task.yaml"
REVIEW_FILE_PATH: str = "review_task.txt" # Temp file in CWD for user review (Plan Sec 3.B)

# --- Orchestration Constants ---
MAX_CYCLES_PER_PHASE: int = 5 # Cycles for build/debug loop per file
MAX_TEST_DEBUG_ATTEMPTS: int = 2 # Max attempts to fix test failures per phase
MAX_VALIDATION_ATTEMPTS: int = 2 # Max attempts for LLM JSON correction (per phase)
CONTINUITY_SEED_MAX_LEN: int = 400 # Increased slightly for structure summary

# --- Blueprint Detection ---
# Keywords/patterns to detect if supporting_docs contains a blueprint
BLUEPRINT_INDICATORS: List[str] = [
    "FILE STRUCTURE", "CORE COMPONENTS", "SCHEMAS",
    "api_srv.py", "orchestrator.py", "/services/", "/repositories/",
    "#P0", "#P1", "#P2", "#P3", "#P4", "#P5" # Presence of phasing markers
]
BLUEPRINT_DETECTION_THRESHOLD: int = 3 # How many indicators must be present

# --- Miscellaneous ---
CURRENT_DATE_STR: str = time.strftime("%Y-%m-%d")
LANGUAGE_EXTENSIONS: Dict[str, str] = {
    "python": "py", "javascript": "js", "typescript": "ts", "java": "java",
    "go": "go", "ruby": "rb", "csharp": "cs", "cpp": "cpp", "c": "c",
    "swift": "swift", "kotlin": "kt", "php": "php", "html": "html", "css": "css",
    # Add more as needed
}
# Default types for --docs flag as per Plan v1.1 Phase 3.E
DEFAULT_DOC_TYPES: List[str] = ['readme', 'api', 'architecture', 'todo', 'bugs']

# --- Global Runtime Variables (Initialized by file_utils.configure_paths) ---
# These hold the *absolute* paths used during runtime
BASE_OUTPUT_DIR: str = "" # Absolute path to the base output directory (T005)
CODE_DIR: str = "" # Absolute path to the sandbox directory
DOCS_DIR: str = "" # Absolute path to the docs directory
