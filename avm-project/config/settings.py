"""
Compatibility shim for notebooks and scripts that import `config.settings`.
Re-exports settings from `src.config.settings` (single source of truth).
"""

try:
    # Import and re-export uppercase names from the src package settings
    from src.config.settings import *  # noqa: F401,F403
except Exception:
    # If for any reason the src package isn't importable, provide minimal
    # fallbacks so importing `config.settings` does not crash unhelpfully.
    import os
    ENVIRONMENT = os.getenv("ENVIRONMENT", "local")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    APP_NAME = os.getenv("APP_NAME", "avm-project")
