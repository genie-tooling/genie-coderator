# Filename: models.py
# -*- coding: utf-8 -*-
# Description: Pydantic models for data validation throughout the orchestration process.
# Version: 1.7.0 (Enhanced PhasingPlanOutput for File Targeting)

# Using Pydantic v2 syntax (BaseModel, Field, model_validator)
import os
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from typing import List, Optional, Dict, Any, Union, ForwardRef, Set # Added Set for T005

# --- Phase 0: Blueprint Summary Extraction ---
class BlueprintSummaryOutput(BaseModel):
    """Output structure for blueprint summary extraction."""
    model_config = {'extra': 'ignore'}

    extracted_summary: str = Field(..., description="Concise summary extracted from the blueprint text, focusing on the core purpose and high-level components.")

# --- Phase 0/1: Requirements Models (Plan Sec 3.B) ---
# Use ForwardRef for recursive structures like sub_requirements
RequirementItemRef = ForwardRef('RequirementItem')

class RequirementItem(BaseModel):
    model_config = {'extra': 'ignore'}

    id: str = Field(..., description="Unique hierarchical identifier (e.g., '1', '1.2', '1.2.1a').")
    text: str = Field(..., description="Detailed requirement description (WHAT, not HOW).")
    sub_requirements: Optional[List[RequirementItemRef]] = Field(None, description="Nested list of sub-requirements.") # Use None as default

# Update the model's references after definition is complete
RequirementItem.model_rebuild()

class TestScenario(BaseModel):
    model_config = {'extra': 'ignore'}

    id: str = Field(..., description="Unique identifier for the test scenario (e.g., 'T1', 'T1.1').")
    name: str = Field(..., description="Concise name describing the test scenario.")
    given: str = Field(..., description="Initial state or context before the action (Given...).")
    when: str = Field(..., description="The action or event being tested (When...).")
    then: str = Field(..., description="The expected, verifiable outcome (Then...).")

class RequirementsOutput(BaseModel):
    """
    Output structure for Phase 0 (Requirements Refinement) & Phase 0.5 (Sanitization).
    """
    model_config = {'extra': 'ignore'}

    clarified_task_description: str = Field(..., description="Final, concise description of the task's scope.")
    requirements: List[RequirementItem] = Field(..., description="Structured, potentially nested list of functional requirements.")
    test_scenarios: List[TestScenario] = Field(..., description="List of test scenarios verifying requirements.")


# --- Phase 1: Planning / Design Models ---

class FunctionSignature(BaseModel):
    """Defines a function's signature within the project design."""
    model_config = {'extra': 'ignore'}

    name: str = Field(..., description="Function or method name.")
    parameters: List[str] = Field(default_factory=list, description="List of parameter definitions as strings (e.g., 'user_id: int', 'data: dict').") # Keep default_factory
    return_type: str = Field(..., description="Return type annotation as a string (e.g., 'bool', 'str', 'UserModel', 'None').")
    description: Optional[str] = Field(None, description="Concise description of the function's purpose.")

class ClassStructure(BaseModel):
    """Defines a class structure within the project design."""
    model_config = {'extra': 'ignore'}

    name: str = Field(..., description="Class name.")
    description: Optional[str] = Field(None, description="Concise description of the class's purpose.")
    methods: List[FunctionSignature] = Field(default_factory=list, description="List of methods defined in this class.") # Keep default_factory

class FileStructure(BaseModel):
    """Defines the structure and contents of a single file in the design."""
    model_config = {'extra': 'ignore'}

    path: str = Field(..., description="Relative path within the sandbox (e.g., 'src/models/user.py'). Should not start with '/'")
    description: Optional[str] = Field(None, description="Purpose of this file within the project.")
    classes: List[ClassStructure] = Field(default_factory=list, description="List of classes defined in this file.") # Keep default_factory
    functions: List[FunctionSignature] = Field(default_factory=list, description="List of top-level functions defined in this file.") # Keep default_factory
    imports: List[str] = Field(default_factory=list, description="Suggested necessary import statements for this file.") # Keep default_factory

    @field_validator('path')
    @classmethod
    def path_must_be_relative(cls, v: str) -> str:
        if os.path.isabs(v) or v.startswith('/'):
            raise ValueError('File path must be relative within the sandbox')
        if '..' in v.split(os.path.sep):
             raise ValueError('File path cannot contain ".." components')
        return v

class ProjectStructureOutput(BaseModel):
    """
    Output structure for the Design step (Phase 1a - generated or extracted).
    """
    model_config = {'extra': 'ignore'}

    design_summary: str = Field(..., description="High-level description of the proposed architecture and structure.")
    file_structure: List[FileStructure] = Field(..., description="List defining each file and its core contents.")
    shared_interfaces: Optional[Dict[str, str]] = Field(None, description="Key shared data models or interface definitions (e.g., Pydantic models as strings). Key is name, value is definition.")


# --- Phase 1: Planning / Chunking Model ---
class PlanChunkOutput(BaseModel):
    """Output structure for the initial plan chunking step (Phase 1b)."""
    model_config = {'extra': 'ignore'}

    plan_chunks: List[str] = Field(..., description="List of high-level phase/chunk descriptions for breaking down the project.")


# --- Phase 1: Planning / Size Estimation Models ---

class PhaseEstimate(BaseModel):
    """Represents the estimated output size for a single phase."""
    model_config = {'extra': 'ignore'}

    phase_description: str = Field(..., description="Description of the phase, aligned with design components.")
    estimated_output_bytes: int = Field(..., description="Estimated total JSON output bytes for implementing this phase's code.")

class PhasingPlanOutput(BaseModel):
    """
    Output structure for Phase 1c (Size Estimation & Refinement).
    Generated PER CHUNK and then aggregated. Includes explicit file targeting map.
    """
    model_config = {'extra': 'ignore'}

    estimated_output_bytes_per_phase: List[PhaseEstimate] = Field(..., description="List of estimated byte sizes for each initially proposed phase *within this chunk*.")
    threshold_check_result: str = Field(..., description="Summary of whether any phase *within this chunk* exceeded the target output size.")
    final_phasing_plan: List[str] = Field(..., description="Final list of phase descriptions *for this chunk*, potentially refined. Descriptions MUST clearly reference the main target file/component.")
    # New field for explicit file targeting
    phase_target_mapping: Dict[str, List[str]] = Field(..., description="Maps each 'final_phasing_plan' description string (key) to a list of relative file paths (values) from the design that it primarily targets.")
    initial_continuity_seed: Optional[str] = Field(None, description="Concise seed text for the build phase, or null/None.")

    @field_validator('initial_continuity_seed', mode='before')
    @classmethod
    def seed_handle_na_or_empty(cls, v):
        if isinstance(v, str):
            v_stripped = v.strip()
            if v_stripped.upper() == "N/A" or v_stripped.lower() == "null" or not v_stripped:
                return None
        return v

    @model_validator(mode='after') # Use model_validator for cross-field checks in Pydantic v2
    def check_mapping_keys_match_plan(self):
        """Ensures keys in phase_target_mapping match final_phasing_plan."""
        plan_phases = set(self.final_phasing_plan)
        mapping_phases = set(self.phase_target_mapping.keys())
        if plan_phases != mapping_phases:
            missing_in_mapping = plan_phases - mapping_phases
            extra_in_mapping = mapping_phases - plan_phases
            error_parts = []
            if missing_in_mapping:
                error_parts.append(f"Phases in 'final_phasing_plan' but missing from 'phase_target_mapping' keys: {missing_in_mapping}")
            if extra_in_mapping:
                error_parts.append(f"Keys in 'phase_target_mapping' not found in 'final_phasing_plan': {extra_in_mapping}")
            raise ValueError("; ".join(error_parts))
        return self


# --- Phase 2: Build Models ---

class CodeOutput(BaseModel):
    """
    Expected JSON output from the LLM during the build/debug phase (Phase 2).
    """
    model_config = {'extra': 'ignore'}

    target_file_path: str = Field(..., description="The relative path of the file this content belongs to (must match the requested path).")
    solution_code: str = Field(..., description="The complete, functional code content for the specified target_file_path.")
    test_code: Optional[str] = Field(None, description="Associated test code for the generated solution_code, if applicable for this file.")
    language: str
    requirements_content_delta: Optional[str] = Field(None, description="Pip requirements specific to *this file's* code (used for aggregation). Format: 'package==version\\npackage2'. Null if none.")
    system_dependencies_delta: Optional[List[str]] = Field(None, description="System dependencies (e.g., apt install commands) specific to *this file's* code. Null or empty list if none.")

    @field_validator('language')
    @classmethod
    def check_language(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Language field cannot be empty")
        return v.lower().strip()


# --- Phase 3: Documentation Model ---

class DocsOutput(BaseModel):
    """Output structure for Phase 3 (Documentation)."""
    model_config = {'extra': 'ignore'}

    documentation_type: str = Field(..., description="Type of documentation generated (e.g., 'readme', 'api'). Should match request.")
    content: str = Field(..., description="The generated documentation content in Markdown format.")

    @field_validator('documentation_type')
    @classmethod
    def check_doc_type(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Documentation type cannot be empty")
        return v.lower().strip()

# --- General Error Model ---
class ErrorOutput(BaseModel):
    """Standard structure for reporting errors from LLM parsing/validation or structured LLM errors."""
    model_config = {'extra': 'ignore'}

    error: str = Field(..., description="Description of the error encountered.")

# --- State Management Model (T005) ---
class PipelineState(BaseModel):
    """Represents the saveable state of the orchestration pipeline."""
    model_config = {'extra': 'ignore'}

    # Phase tracking
    current_phase_index: int = Field(..., description="Index of the *next* phase to be executed (0-based).")
    completed_phase_descriptions: List[str] = Field(..., description="List of descriptions for phases already completed.")

    # Configuration / Context
    task_file_path: str = Field(..., description="Path to the original task file.")
    base_output_dir: str = Field(..., description="Absolute path to the base output directory.")
    code_dir: str = Field(..., description="Absolute path to the sandbox directory.")
    docs_dir: str = Field(..., description="Absolute path to the docs directory.")
    target_language: str = Field(..., description="Target programming language for the project.")
    is_blueprint_mode: bool = Field(..., description="Whether the orchestrator ran in blueprint extraction mode.")
    docs_requested: Optional[List[str]] = Field(None, description="List of documentation types requested via CLI.")

    # References to key artifacts within the sandbox (relative paths)
    artifact_references: Dict[str, Optional[str]] = Field(default_factory=dict, description="Dictionary mapping artifact type to its relative path in the sandbox (e.g., {'requirements': 'requirements.txt', 'design': '_project_structure.json'}).")

    # Aggregated dependencies
    aggregated_pip_requirements: Set[str] = Field(default_factory=set, description="Set of pip requirements aggregated so far.")
    aggregated_system_dependencies: Set[str] = Field(default_factory=set, description="Set of system dependencies aggregated so far.")

    # Add other relevant state info as needed (e.g., failure history?)
