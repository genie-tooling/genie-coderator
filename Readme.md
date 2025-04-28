# Genie Code-erator (Code Orchestrator)

This project implements a multi-stage AI pipeline for automated code generation using Google's Gemini models. It follows a structured approach involving requirements analysis, design, planning, phased code generation with debugging, automated testing (for Python), and documentation generation.

## Features

* **Task Definition:** Accepts project tasks via YAML files.
* **Requirements Refinement:** Generates detailed specifications and test scenarios from the task description (Standard Mode).
* **Blueprint Mode:** Can extract project structure and requirements directly from a detailed blueprint provided in the task's supporting documents.
* **Structured Design:** Generates or extracts a project structure definition (files, classes, functions, signatures).
* **Phased Planning:** Breaks down the design into manageable chunks and phases, estimating output size to manage context windows.
* **Phased Code Generation:** Implements code file-by-file based on the plan, using iterative correction loops for LLM validation errors.
* **Automated Testing (Python):** Sets up a virtual environment, installs dependencies, and runs `pytest` to validate generated Python code.
* **Documentation Generation:** Creates project documentation (Readme, API docs, etc.) based on the final code and design artifacts.
* **State Management & Resumability (T005):** Saves pipeline progress after each major phase to `_pipeline_state.json` in the output directory. Allows resuming interrupted runs using the `--resume` flag.
* **Configuration:** Uses `.env` for API keys and `config.py` for pipeline constants.
* **Logging:** Provides detailed logging throughout the orchestration process.

## Usage

1.  **Setup:**
    * Install required Python packages:
        ```bash
        pip install google-generativeai pydantic pydantic-settings python-dotenv PyYAML tiktoken
        # For testing Python code:
        pip install pytest
        ```
    * Create a `.env` file in the project root with your Gemini API key:
        ```
        GEMINI_API_KEY="YOUR_API_KEY_HERE"
        ```

2.  **Define Task:**
    * Create a YAML task file (e.g., `task.yaml`) describing the project requirements. See examples in the `tasks/` directory.

3.  **Run Orchestrator:**
    ```bash
     python -m code_orchestrator.orchestrator --help
    ```
    * To run one of the examples:
    ```bash
    python -m code_orchestrator.orchestrator --skip-testing --task-file example_tasks/cli_fact_calc.yaml
    ```

    **Common Options:**
    * `--task-file PATH`: Path to the YAML task file (default: `task.yaml`).
    * `--output-dir PATH`: Base directory for output (sandbox, docs, state file). Defaults to current directory.
    * `--language LANG`: Target language (e.g., `python`, `javascript`). Overrides task file.
    * `--docs [TYPES]`: Generate documentation. Optionally specify comma-separated types (e.g., `readme,api`). Defaults to `readme,api,architecture,todo,bugs`.
    * `--skip-testing`: Skip the automated testing phase (useful for non-Python or quick runs).
    * `--skip-human-validation`: Skip the interactive review step during requirements refinement (Phase 0, Standard Mode).
    * `--resume`: Attempt to resume the pipeline from the last saved state found in the output directory (using `_pipeline_state.json`).
    * `-v`, `--verbose`: Enable DEBUG level logging.

## Resuming Runs (`--resume`)

If the orchestrator is interrupted (e.g., Ctrl+C, system crash), you can attempt to resume it from the last successfully completed phase.

1.  Ensure the output directory (`--output-dir` or the default CWD) from the interrupted run still exists and contains the `_pipeline_state.json` file along with the `sandbox/` subdirectory.
2.  Run the orchestrator again with the **same `--output-dir`** (if used originally) and add the `--resume` flag.
    ```bash
    # Example resuming run that used default output dir
    python -m code_orchestrator --resume

    # Example resuming run that used a specific output dir
    python -m code_orchestrator --output-dir ./my_project_output --resume
    ```
3.  The orchestrator will load the state, skip the already completed phases, and continue execution from where it left off.

**Note:** The state file (`_pipeline_state.json`) is saved in the *base output directory*, not inside the `sandbox/` directory.

## Project Structure

* `code_orchestrator/`: Main package directory.
    * `__main__.py`: Entry point for `python -m code_orchestrator`.
    * `orchestrator.py`: Core pipeline orchestration logic.
    * `config.py`: Configuration constants and settings loading.
    * `models.py`: Pydantic models for data validation.
    * `prompts.py`: LLM prompt templates for different stages.
    * `llm_interface.py`: Handles interaction with the Gemini API.
    * `file_utils.py`: File system operations, path management, state persistence.
    * `testing.py`: Automated test execution logic.
    * `utils.py`: General utility functions.
    * `stages/`: Modules for each pipeline phase logic.
        * `phase_0_summary.py`
        * `phase_0_requirements.py`
        * `phase_1a_design.py`
        * `phase_1b_chunking.py`
        * `phase_1c_planning.py`
        * `phase_2_build.py`
        * `phase_3_docs.py`
* `example_tasks/`: Example task definition YAML files.
* `docs/`: Generated documentation output (if requested).
* `.env`: Stores the Gemini API key (create this file).
* `sandbox/`: Default output directory for generated code and artifacts (created during run).
* `_pipeline_state.json`: Saved state file for resumability (created in output dir during run).
* `TODO.md`: Tracks planned features and improvements.
* `bugs.md`: Tracks known issues.
