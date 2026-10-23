"""Fail early on broken optional-library imports, before downloads or worker startup."""


def verify_model_imports():
    try:
        from transformer_lens.model_bridge import TransformerBridge
        from transformers import AutoModelForCausalLM
    except Exception as error:
        raise RuntimeError(
            "Model-library import failed before any model loading. Run "
            "kaggle/setup_environment.py and use the isolated stcci-venv interpreter "
            "for every command; see KAGGLE_GUIDE.md. The original exception follows."
        ) from error
    return TransformerBridge, AutoModelForCausalLM
