# Filename: testing.py
# Description: Handles execution of tests within a sandboxed virtual environment.
# Version: 1.1 (Reflecting Plan v1.1 Integration - conditional logic verified)

import os
import subprocess
import logging
import shutil
import argparse
from typing import Optional, List, Tuple

# Import config and file_utils
from . import config
from . import file_utils

logger = logging.getLogger(__name__)

def run_tests_or_skip(args: argparse.Namespace,
                        target_language: str,
                        requirements_filepath_abs: str, # Expect absolute path
                        system_dependencies: Optional[List[str]]) -> Tuple[int, str]:
    """
    Conditionally sets up a venv, installs deps, and runs tests (currently Python/pytest only).
    Implements conditional logic from Plan v1.1 Section 3.D.

    Args:
        args: Command line arguments (contains `skip_testing` flag).
        target_language: The programming language of the project.
        requirements_filepath_abs: Absolute path to the requirements.txt file.
        system_dependencies: List of system dependencies (logged for info only).

    Returns:
        A tuple containing:
         - exit_code (int): 0 for success or skipped, non-zero for test failure or setup error.
        - output_log (str): Combined log of setup and test execution or skip reason.
    """
    # 1. Check Skip Conditions (Plan v1.1 Sec 3.D)
    if args.skip_testing:
        skip_reason = "Skipped: Testing disabled via --skip-testing flag."
        logger.info(skip_reason)
        return 0, skip_reason
    if target_language != 'python':
        skip_reason = f"Skipped: Automated testing only implemented for 'python', not '{target_language}'."
        logger.info(skip_reason)
        return 0, skip_reason # Treat unsupported language as a successful skip

    # Check if sandbox dir is configured (should be by orchestrator)
    if not config.CODE_DIR or not os.path.isdir(config.CODE_DIR):
         error_msg = "FAIL: Testing cannot proceed, CODE_DIR (sandbox path) is not configured or does not exist."
         logger.error(error_msg)
         return -1, error_msg

    # Proceed with Python testing setup
    venv_dir = file_utils.get_venv_dir() # Gets absolute path to .venv within sandbox
    logger.info(f"Attempting Python test execution using virtual environment: {venv_dir}")
    # Accumulate logs for setup and execution, returned on success/failure
    combined_log = f"--- Test Setup & Execution Log ({target_language}) ---\n"
    combined_log += f"Sandbox: {config.CODE_DIR}\nVenv Dir: {venv_dir}\n"
    combined_log += f"Requirements File: {requirements_filepath_abs}\n"

    # 2. Setup Virtual Environment
    venv_python: Optional[str] = None
    venv_pip: Optional[str] = None
    try:
        # Ensure parent directory exists (sandbox dir should already exist)

        # Remove old venv if it exists to ensure clean state
        if os.path.exists(venv_dir):
            logger.info(f"Removing existing virtual environment: {venv_dir}")
            combined_log += f"INFO: Removing existing venv: {venv_dir}\n"
            try:
                # Use robust removal for Windows compatibility issues
                if os.name == 'nt':
                     # Simple rmtree might fail on Windows due to long paths/permissions
                     rm_cmd = ['cmd', '/c', 'rmdir', '/s', '/q', venv_dir]
                     subprocess.run(rm_cmd, check=False, capture_output=True) # Don't check, just attempt
                     if os.path.exists(venv_dir): # Check if it still exists
                          logger.warning(f"Standard rmdir failed, attempting shutil.rmtree again for {venv_dir}")
                          shutil.rmtree(venv_dir, ignore_errors=True) # Try shutil again, ignore errors this time
                else:
                    shutil.rmtree(venv_dir)

                # Final check
                if os.path.exists(venv_dir):
                     logger.error(f"Failed to completely remove existing venv: {venv_dir}")
                     combined_log += f"ERROR: Failed to remove existing venv: {venv_dir}\n"
                     # Decide whether to proceed or fail - let's try proceeding carefully

            except OSError as rm_err:
                 # Handle potential errors during removal (e.g., permissions)
                 logger.error(f"Error removing existing venv '{venv_dir}': {rm_err}. Attempting to continue...")
                 combined_log += f"WARN: Error removing existing venv: {rm_err}\n"


        # Find python executable (prefer python3 if available)
        python_executable = shutil.which("python3") or shutil.which("python")
        if not python_executable:
            raise EnvironmentError("Could not find 'python3' or 'python' executable in system PATH.")
        logger.info(f"Using Python executable for venv creation: {python_executable}")
        combined_log += f"INFO: Using host Python: {python_executable}\n"

        # Create venv using the found python executable
        venv_cmd = [python_executable, "-m", "venv", venv_dir]
        logger.debug(f"Running venv creation command: {' '.join(venv_cmd)}")
        # Increased timeout for potentially slow venv creation on some systems
        venv_result = subprocess.run(
            venv_cmd, capture_output=True, text=True, check=True, timeout=180,
            # Prevent shell injection vulnerability
            shell=False
        )
        combined_log += f"INFO: Venv created successfully.\n"
        if venv_result.stdout: combined_log += f"Venv stdout:\n{venv_result.stdout}\n"
        if venv_result.stderr: combined_log += f"Venv stderr:\n{venv_result.stderr}\n"
        logger.info("Virtual environment created successfully.")

        # Determine platform-specific paths within the created venv
        if os.name == 'nt': # Windows
            venv_scripts_dir = os.path.join(venv_dir, "Scripts")
            venv_python = os.path.join(venv_scripts_dir, "python.exe")
            venv_pip = os.path.join(venv_scripts_dir, "pip.exe")
        else: # POSIX (Linux, macOS, etc.)
            venv_bin_dir = os.path.join(venv_dir, "bin")
            venv_python = os.path.join(venv_bin_dir, "python")
            venv_pip = os.path.join(venv_bin_dir, "pip")

        # Verify the determined executable paths exist
        if not os.path.exists(venv_python) or not os.path.exists(venv_pip):
            raise EnvironmentError(f"Python ({venv_python}) or Pip ({venv_pip}) missing after venv creation.")
        logger.debug(f"Venv Python path: {venv_python}")
        logger.debug(f"Venv Pip path: {venv_pip}")

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, EnvironmentError, Exception) as e:
        error_type = type(e).__name__
        output = ""
        # Safely access stdout/stderr if they exist on the exception object
        if hasattr(e, 'stdout') and e.stdout: output += f"STDOUT:\n{e.stdout}\n"
        if hasattr(e, 'stderr') and e.stderr: output += f"STDERR:\n{e.stderr}\n"
        # Log specific error type and message
        logger.error(f"FAIL: Virtual environment creation failed ({error_type}): {e}\n{output}", exc_info=isinstance(e, Exception) and not isinstance(e, (subprocess.CalledProcessError, subprocess.TimeoutExpired, EnvironmentError)))
        combined_log += f"ERROR: Venv creation failed ({error_type}): {e}\n{output}---\n"
        # Return error code and the log containing failure details
        return -1, f"FAIL: Venv creation failed ({error_type}).\nLog:\n{combined_log}"

    # 3. Log System Dependencies (Informational Only - Plan v1.1 doesn't require execution)
    combined_log += "---\nINFO: System Dependencies (Logged, Not Executed):\n"
    if system_dependencies:
        logger.info("System dependencies specified by LLM (for information only):")
        for i, cmd in enumerate(system_dependencies):
            logger.info(f"  - Dep {i+1}: {cmd}")
            combined_log += f"  - {cmd}\n"
    else:
        logger.info("No system dependencies were specified by LLM.")
        combined_log += "  (None Specified)\n"
    combined_log += "---\n"

    # 4. Install Pip Requirements
    pip_installed_ok = False
    if os.path.exists(requirements_filepath_abs) and os.path.getsize(requirements_filepath_abs) > 0:
        logger.info(f"Installing pip requirements from '{requirements_filepath_abs}' into venv...")
        combined_log += f"INFO: Installing pip requirements from {requirements_filepath_abs}\n"
        try:
            # Ensure pip, setuptools, wheel are up-to-date within the venv first
            pip_upgrade_cmd = [venv_pip, "install", "--upgrade", "pip", "setuptools", "wheel"]
            logger.debug(f"Running pip upgrade command: {' '.join(pip_upgrade_cmd)}")
            pip_upgrade_result = subprocess.run(pip_upgrade_cmd, capture_output=True, text=True, check=True, timeout=120, shell=False)
            combined_log += f"INFO: Pip upgrade OK.\n"
            if pip_upgrade_result.stdout: combined_log += f"Pip Upgrade stdout:\n{pip_upgrade_result.stdout}\n"
            if pip_upgrade_result.stderr: combined_log += f"Pip Upgrade stderr:\n{pip_upgrade_result.stderr}\n"

            # Install from the requirements file
            # Add --no-cache-dir to potentially avoid issues with corrupted caches
            pip_install_cmd = [venv_pip, "install", "--no-cache-dir", "-r", requirements_filepath_abs]
            logger.debug(f"Running pip install command: {' '.join(pip_install_cmd)}")
            result_pip = subprocess.run(pip_install_cmd, capture_output=True, text=True, check=True, timeout=600, shell=False) # Allow 10 mins for complex installs
            logger.info("Pip requirements installed successfully.")
            combined_log += f"INFO: Pip install from requirements file OK.\n"
            if result_pip.stdout: combined_log += f"Pip Install stdout:\n{result_pip.stdout}\n"
            if result_pip.stderr: combined_log += f"Pip Install stderr:\n{result_pip.stderr}\n"
            pip_installed_ok = True

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, Exception) as e:
            error_type = type(e).__name__
            output = ""
            if hasattr(e, 'stdout') and e.stdout: output += f"STDOUT:\n{e.stdout}\n"
            if hasattr(e, 'stderr') and e.stderr: output += f"STDERR:\n{e.stderr}\n"
            logger.error(f"FAIL: Pip install failed ({error_type}): {e}\n{output}", exc_info=isinstance(e, Exception) and not isinstance(e, (subprocess.CalledProcessError, subprocess.TimeoutExpired)))
            combined_log += f"ERROR: Pip install failed ({error_type}): {e}\n{output}---\n"
            return -1, f"FAIL: Pip install failed ({error_type}).\nLog:\n{combined_log}"
    else:
        logger.info(f"Requirements file '{requirements_filepath_abs}' not found or is empty. Skipping pip install step.")
        combined_log += "INFO: Skipped pip install (requirements file missing or empty).\n"
        pip_installed_ok = True # Considered OK if nothing needed to be installed

    # Safety check - should not be reachable if setup errors occurred above, but good practice
    if not pip_installed_ok:
         err_msg = "FAIL: Cannot proceed to testing due to previous pip installation failure."
         logger.error(err_msg)
         combined_log += f"ERROR: {err_msg}\n"
         return -1, f"{err_msg}\nLog:\n{combined_log}"

    # 5. Ensure Pytest is Installed and Run Tests
    logger.info(f"Running tests using pytest in virtual environment...")
    combined_log += "---\nINFO: Running pytest...\n"
    try:
        # Attempt to install pytest within the venv if not already present via requirements
        # Use '--disable-pip-version-check' and '-q' for cleaner output unless debugging
        install_pytest_cmd = [venv_pip, "install", "pytest", "--disable-pip-version-check", "-q"]
        logger.debug(f"Ensuring pytest is installed: {' '.join(install_pytest_cmd)}")
        try:
            # Don't capture output unless needed, check=True handles errors
            subprocess.run(install_pytest_cmd, check=True, timeout=120, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, shell=False)
            combined_log += "INFO: Pytest installation check/run successful.\n"
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e_pytest_install:
             stderr = e_pytest_install.stderr if hasattr(e_pytest_install, 'stderr') else 'No stderr captured.'
             logger.error(f"FAIL: Failed to install pytest in virtual environment: {e_pytest_install}\nStderr:\n{stderr}", exc_info=False)
             combined_log += f"ERROR: Could not install pytest in venv: {e_pytest_install}\nStderr:\n{stderr}\n---\n"
             return -1, f"FAIL: Pytest installation failed.\nLog:\n{combined_log}"

        # Execute pytest command within the CODE_DIR (sandbox root)
        # Pytest will discover tests (e.g., files named test_*.py or *_test.py)
        pytest_cmd = [venv_python, "-m", "pytest", "-vv"] # Use verbose pytest output
        logger.debug(f"Executing test command in '{config.CODE_DIR}': {' '.join(pytest_cmd)}")
        result = subprocess.run(
            pytest_cmd,
            capture_output=True, text=True, # Capture output
            cwd=config.CODE_DIR, # Run from sandbox root
            timeout=600, # 10 min timeout for potentially larger test suites
            shell=False # Security: Do not use shell=True
        )

        # Combine setup and test output into the final log string
        final_output_log = (
            f"{combined_log}"
            f"--- Pytest Execution Result (Return Code: {result.returncode}) ---\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}\n"
            f"--- End Log ---"
        )

        # Check pytest exit code
        if result.returncode == 0:
            logger.info(f"Pytest PASSED (Return Code: {result.returncode}).")
            logger.debug(f"Full test execution log:\n{final_output_log}")
        elif result.returncode == 5:
            # Pytest exit code 5: No tests were collected.
            logger.warning(f"Pytest finished with code 5: No tests were collected. Check test file naming (test_*.py / *_test.py) and contents.")
            # Treat "no tests found" as a PASS for the build phase, but log clearly.
            # Return 0 but include the log indicating no tests were found.
            return 0, f"PASSED (No tests found).\n{final_output_log}"
        else:
            # Any other non-zero exit code indicates test failures.
            logger.error(f"Pytest FAILED (Return Code: {result.returncode}).")
            # Log the full output at INFO level for visibility of failures
            logger.info(f"Full test execution log:\n{final_output_log}")

        # Return the pytest exit code and the detailed log
        return result.returncode, final_output_log

    except subprocess.TimeoutExpired as e:
        # Handle test execution timeout
        output = f"STDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}" if hasattr(e, 'stdout') else "Timeout - No output captured."
        logger.error(f"FAIL: Pytest execution timed out after 600 seconds: {e}\nOutput:\n{output}", exc_info=False)
        final_output_log = f"{combined_log}--- Pytest TIMEOUT ---\nFAIL: Test execution timed out.\n{output}\n--- End Log ---"
        return -1, final_output_log # Return error code
    except Exception as e:
        # Handle any other unexpected errors during test execution
        logger.error(f"FAIL: Pytest execution encountered an unexpected error: {e}", exc_info=True)
        final_output_log = f"{combined_log}--- Pytest Execution Error ---\nFAIL: Unexpected error during test run: {e}\n--- End Log ---"
        return -1, final_output_log # Return error code

# --- T005: Placeholder for Resume Tests ---
# Actual tests for the --resume functionality should be added here or in a dedicated test file.
# These tests would typically involve mocking file_utils.save_pipeline_state and
# file_utils.load_pipeline_state, creating mock state files, and running the
# orchestrator's main function with the --resume flag to verify phase skipping
# and correct state restoration.

# Example (Conceptual - Requires mocking framework like unittest.mock or pytest-mock):
#
# def test_resume_skips_completed_phases(mocker):
#     # Mock load_pipeline_state to return a state indicating Phase 0, 1 are done
#     mock_state = models.PipelineState(...) # Create mock state object
#     mocker.patch('your_module.file_utils.load_pipeline_state', return_value=mock_state)
#     mocker.patch('your_module.file_utils.save_pipeline_state', return_value=True)
#
#     # Mock phase execution functions (e.g., phase_0_requirements, phase_1a_design)
#     mock_phase0 = mocker.patch('your_module.stages.phase_0_requirements.run_phase_0_requirements_and_sanitization')
#     mock_phase1 = mocker.patch('your_module.stages.phase_1a_design.run_phase_1a_extract_or_generate_design')
#     mock_phase2 = mocker.patch('your_module.stages.phase_1b_chunking.run_phase_1b_plan_chunking', return_value=...) # Needs valid return
#     # ... mock other phases as needed ...
#
#     # Create mock args with --resume
#     mock_args = argparse.Namespace(..., resume=True, output_dir='mock_output')
#
#     # Run the orchestrator's main function
#     exit_code = your_module.orchestrator.main(mock_args)
#
#     # Assertions
#     assert exit_code == 0 # Should succeed if remaining phases work
#     mock_phase0.assert_not_called() # Phase 0 should be skipped
#     mock_phase1.assert_not_called() # Phase 1 should be skipped
#     mock_phase2.assert_called_once() # Phase 2 should be the first one called
#     # ... add more assertions for state saving, etc.
