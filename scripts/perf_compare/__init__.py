"""
perf_compare package

This package contains the modules required to:
    - Load backend-agnostic meta-config JSON files ("logical runs").
    - Generate backend-specific JSON configs (scalar + neon, etc.).
    - Run the ray tracer under perf (to be implemented).
    - Parse and compare high-level and microarchitectural metrics.
    - Emit comparison reports in text and optional CSV formats.

The current implementation covers:
    - CLI parsing.
    - Logical run modeling.
    - Meta-config loading and label handling.
    - Backend-specific config generation.

Further functionality will be added in separate modules to keep the overall
design modular and maintainable.
"""
