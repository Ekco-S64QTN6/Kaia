import functools
import logging
import requests
import config
from typing import Any, Dict, List, Optional, Tuple, Union


logger = logging.getLogger(__name__)

# Cache for Ollama model availability checks
_model_cache: Dict[str, Tuple[str, Optional[str]]] = {}

# Color Percentage Calculation
def get_color_for_percentage(percent: Union[int, float]) -> str:
    """Returns an ANSI color code based on a percentage value."""
    if not isinstance(percent, (int, float)):
        return config.COLOR_RESET

    if percent <= 70:
        return config.COLOR_GREEN
    elif 70 < percent <= 80:
        return config.COLOR_YELLOW
    else:
        return config.COLOR_RED

# Ollama Model Availability Check
@functools.lru_cache(maxsize=32)
def check_ollama_model_availability(model_name: str, fallback_model: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Checks if a model is available in Ollama, with fallback options.
    Caches results to avoid repeated API calls.
    """
    cache_key = (model_name, fallback_model)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=config.TIMEOUT_SECONDS)
        response.raise_for_status()
        available_models = {m['name'] for m in response.json().get('models', [])}

        if model_name in available_models:
            _model_cache[cache_key] = (model_name, None)
            return model_name, None

        logger.warning(f"Configured Ollama model '{model_name}' not found. Available: {list(available_models)}")

        if fallback_model and fallback_model in available_models:
            logger.warning(f"Using fallback model '{fallback_model}'.")
            _model_cache[cache_key] = (fallback_model, None)
            return fallback_model, None

        # Generic fallbacks
        generic_fallbacks = ['llama2:7b-chat', 'mistral:instruct']
        for fb in generic_fallbacks:
            if fb in available_models:
                logger.warning(f"Using generic fallback model '{fb}'.")
                _model_cache[cache_key] = (fb, None)
                return fb, None

        error_msg = f"No suitable Ollama model found. None of the configured, fallback, or default models are available."
        logger.error(error_msg)
        _model_cache[cache_key] = ("", error_msg)
        return "", error_msg

    except requests.exceptions.ConnectionError:
        error_msg = "Could not connect to Ollama server. Please ensure Ollama is running."
        logger.error(error_msg)
        _model_cache[cache_key] = ("", error_msg)
        return "", error_msg
    except requests.exceptions.Timeout:
        error_msg = "Ollama server connection timed out."
        logger.error(error_msg)
        _model_cache[cache_key] = ("", error_msg)
        return "", error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred while checking Ollama models: {e}"
        logger.error(error_msg, exc_info=True)
        _model_cache[cache_key] = ("", error_msg)
        return "", error_msg
