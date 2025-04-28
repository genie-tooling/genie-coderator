# TODO Checklist - Gemini Code Orchestrator Improvements (Consolidated)

## Priority 1: Core Logic, State & Context Management (High Impact)

* **State Management & Resumability:** (T005 - Partially Complete)
    * `- [x]` Design State Model: Defined %%%PipelineState%%% model (%%%models.py%%%).
    * `- [x]` Create State Manager: Implemented save/load functions in %%%file_utils.py%%%.
    * `- [x]` Refactor Orchestrator: Modified %%%orchestrator.py%%% to add %%%--resume%%%, load/save state, and skip phases.
    * `- [ ]` Refactor Stages: Update stage functions (%%%stages/*.py%%%) to interact with the %%%StateManager%%% (Not Done - current implementation uses orchestrator state).
    * `- [x]` Implement Persistence: Implemented save/load to %%%_pipeline_state.json%%% via %%%file_utils.py%%% and %%%orchestrator.py%%%.
    * `- [ ]` Add Specific Resume Tests: Implement dedicated tests in %%%testing.py%%% for the %%%--resume%%% functionality (See Priority 4).

* **Advanced Context Management:**
    * `- [ ]` Design Summarization Prompt: Create %%%PROMPT_CONTEXT_SUMMARIZER%%% for summarizing prior stage outputs (design, reqs, code). (%%%prompts.py%%%)
    * `- [ ]` Integrate Summarization: Add logic to call the summarizer (e.g., via %%%llm_interface.py%%%) within %%%orchestrator.py%%% before stages needing extensive context. (%%%orchestrator.py%%%, %%%llm_interface.py%%%)
    * `- [ ]` Update Prompts for Summaries: Modify planning, build, and docs prompts (%%%PROMPT_3_5...%%%, %%%PROMPT_4_...%%%, %%%PROMPT_5_...%%%) to accept and effectively utilize summarized context. (%%%prompts.py%%%)
    * `- [ ]` *Explore* Vector Memory (Advanced): Design schema and implement embedding/retrieval logic for more targeted context retrieval if simple summarization proves insufficient for very large projects. (New %%%components/memory_manager.py%%% or %%%utils.py%%%, %%%models.py%%%)

* **Refine Phase 2 File Targeting Logic:** (Complete)
    * `- [x]` Analyze Current Logic: Review how target files are currently identified in %%%_parse_phase_description_for_targets%%% within %%%stages/phase_2_build.py%%%. (Done as part of implementation)
    * `- [x]` Enhance Planning Output: Modified Phase 1c (Planning) (%%%stages/phase_1c_planning.py%%%, %%%prompts.py%%%, %%%models.py%%%) to explicitly map final plan phases to specific file paths.
    * `- [x]` Update Build Stage Parsing: Updated %%%_parse_phase_description_for_targets%%% (%%%stages/phase_2_build.py%%%) to prioritize using the explicit mapping from the plan data.

## Priority 2: Planning, Feedback & Validation (Medium-High Impact)

* **Adaptive Planning:**
    * `- [ ]` Feedback Mechanism: Modify %%%phase_2_build.py%%% to return specific status indicating failure due to estimated complexity/size limits of a plan phase. (%%%stages/phase_2_build.py%%%)
    * `- [ ]` Orchestrator Handling: Update %%%orchestrator.py%%% to catch this feedback. (%%%orchestrator.py%%%)
    * `- [ ]` Re-Planning Trigger: Implement logic in %%%orchestrator.py%%% to potentially recall %%%phase_1c_planning.py%%% targeting the failed chunk/phase with instructions to subdivide it. (%%%orchestrator.py%%%, %%%stages/phase_1c_planning.py%%%)
    * `- [ ]` Update Planning Prompts: Ensure planning/correction prompts (%%%PROMPT_3_5...%%%, %%%PROMPT_3_6...%%%) can handle re-planning requests. (%%%prompts.py%%%)

* **Cross-Stage LLM Validation:**
    * `- [ ]` Design vs. Requirements Prompt: Create %%%PROMPT_VALIDATE_DESIGN_VS_REQS%%%. (%%%prompts.py%%%)
    * `- [ ]` Design Validation Step: Add step in %%%orchestrator.py%%% post-Phase 1a to call this prompt. (%%%orchestrator.py%%%, %%%llm_interface.py%%%)
    * `- [ ]` Code vs. Design Prompt: Create %%%PROMPT_VALIDATE_CODE_VS_DESIGN%%%. (%%%prompts.py%%%)
    * `- [ ]` Code Validation Step: Integrate calls post-file generation in %%%phase_2_build.py%%% or %%%orchestrator.py%%%. (%%%stages/phase_2_build.py%%% or %%%orchestrator.py%%%, %%%llm_interface.py%%%)

* **Refine Test Debugging Identification & Handling:**
    * `- [ ]` Parse Test Output: Enhance %%%orchestrator.py%%% (where testing is now called) to attempt parsing %%%pytest%%% tracebacks from %%%testing.py%%% output to identify likely source file(s) causing test failures. (%%%orchestrator.py%%%, %%%testing.py%%%)
    * `- [ ]` Update Debug Prompt Input: Modify the logic calling the debug prompt (%%%PROMPT_4_DEBUG%%% or a reinstated %%%PROMPT_4_TEST_DEBUG%%%) to potentially pass only the identified failing file(s) instead of all phase files. (%%%orchestrator.py%%% or %%%stages/phase_2_build.py%%%, %%%prompts.py%%%)
    * `- [ ]` Define Test Code Correction Strategy: Formalize rules/prompting for when/how the LLM should correct %%%test_code%%% itself versus source code during debugging. (%%%prompts.py%%%, %%%orchestrator.py%%% or %%%stages/phase_2_build.py%%%)

## Priority 3: Prompt Engineering & Correction Loops (Medium Impact)

* **Add Missing Correction Prompts:**
    * `- [ ]` Blueprint Summary Correction: Implement %%%PROMPT_0_EXTRACT_SUMMARY_CORRECT%%% and add retry logic using it in %%%stages/phase_0_summary.py%%%. (%%%prompts.py%%%, %%%stages/phase_0_summary.py%%%)
    * `- [ ]` Documentation Correction: Implement %%%PROMPT_5_DOCS_CORRECT%%% and add retry logic using it in %%%stages/phase_3_docs.py%%%. (%%%prompts.py%%%, %%%stages/phase_3_docs.py%%%)

* **Prompt Engineering - Reasoning & Constraints:**
    * `- [ ]` Add Reasoning Field: Add optional %%%reasoning: str%%% field to decision-making models (Design, Plan). (%%%models.py%%%)
    * `- [ ]` Update Prompts for Reasoning: Modify relevant prompts (%%%PROMPT_2...%%%, %%%PROMPT_3...%%%) to request rationale. (%%%prompts.py%%%)
    * `- [ ]` Log Reasoning: Ensure reasoning is logged if present. (%%%orchestrator.py%%%, %%%stages/*.py%%%)
    * `- [ ]` Add Negative Constraints: Identify and add "DO NOT..." constraints to prompts (especially Build) based on observed issues. (%%%prompts.py%%%)

## Priority 4: Testing Framework & Modularity (Lower Immediate Impact)

* **Intermediate Testing Integration:**
    * `- [ ]` Re-evaluate Build Stage Testing: Decide whether to reintegrate calls to %%%testing.py%%% *within* %%%phase_2_build.py%%% after logical file groups, potentially based on flags. (%%%stages/phase_2_build.py%%%, %%%testing.py%%%, %%%orchestrator.py%%%)
    * `- [ ]` Targeted Test Execution: Enhance %%%testing.py%%% to run tests relevant only to specified files/modules. (%%%testing.py%%%)
    * `- [ ]` Add Specific Resume Tests (Related to T005): Implement dedicated tests in %%%testing.py%%% for the %%%--resume%%% functionality. (%%%testing.py%%%)

* **Expand Testing Language Support:**
    * `- [ ]` Add JS/Node Testing: Implement setup (npm install) and execution (jest/mocha) logic in %%%testing.py%%%. (%%%testing.py%%%)
    * `- [ ]` Add Java Testing: Implement setup (maven/gradle build) and execution (JUnit) logic in %%%testing.py%%%. (%%%testing.py%%%)
    * `- [ ]` (Future): Add support for other requested languages. (%%%testing.py%%%)

* **Modular Tool Definition Review:**
    * `- [ ]` Review Tool Simulation Prompts: Re-examine prompts involving simulated external interactions (e.g., dependency checks) for clarity. (%%%prompts.py%%%)
    * `- [ ]` Robust Tool Parsing: Strengthen parsing logic for simulated tool outputs (e.g., dependency parsing in %%%phase_2_build.py%%%). (%%%stages/phase_2_build.py%%%)