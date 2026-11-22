"""Dataset QA: release registry validation and read-only C10 episode checks.

Episode API: dataset_qa.episode.check_dataset. Registry validity alone does not
establish recording quality. training_readiness consumes a fresh C10 report.
"""

from oa_embodied_ai.dataset_qa.validator import (
    find_schema,
    load_registry_file,
    load_schema,
    validate_registry,
)
from oa_embodied_ai.dataset_qa.handoff import training_readiness

__all__ = ["find_schema", "load_registry_file", "load_schema", "validate_registry", "training_readiness"]
