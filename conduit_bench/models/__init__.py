"""Model registry - imported from Conduit.

All model registry code now lives in Conduit.
This module re-exports for backward compatibility.
"""

from conduit.models import (
    DEFAULT_REGISTRY,
    PRICING,
    create_model_registry,
    filter_models,
    get_model_by_id,
    get_models_by_provider,
    get_registry_stats,
)

__all__ = [
    "DEFAULT_REGISTRY",
    "PRICING",
    "create_model_registry",
    "get_model_by_id",
    "get_models_by_provider",
    "filter_models",
    "get_registry_stats",
]
