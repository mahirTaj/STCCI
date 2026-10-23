"""Installed TransformerLens 4.2.0 API, with mandatory runtime validation."""
import gc

import torch
from transformer_lens.model_bridge import TransformerBridge
from transformers import AutoModelForCausalLM

from models.intervention_hooks import batch_tensors, bounded_score, capture_final, hook_name, patch_head


def load_bridge(config, device, dtype):
    model = TransformerBridge.boot_transformers("gpt2", revision=config["model_revision"],
                                                device=device, dtype=dtype)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.cfg.n_layers != 12 or model.cfg.n_heads != 12 or model.cfg.d_head != 64:
        raise ValueError("Model architecture differs from frozen GPT-2 Small")
    return model


@torch.inference_mode()
def validate_bridge(model, records, config, heads, device, dtype):
    precision = "fp16" if dtype == torch.float16 else "fp32"
    atol, rtol = config[f"validation_atol_{precision}"], config[f"validation_rtol_{precision}"]
    tensors = batch_tensors(records, device)
    clean, corrupt, mask, positions = tensors[:4]
    reference = AutoModelForCausalLM.from_pretrained(
        config["model_id"], revision=config["model_revision"], dtype=dtype,
        attn_implementation="eager").to(device).eval()
    diagnostics = {"precision": precision, "atol": atol, "rtol": rtol,
                   "registered_hooks": sorted(model.hook_dict), "comparisons": []}
    layers = sorted({layer for layer, head in heads})
    for label, tokens in (("clean", clean), ("corrupt", corrupt)):
        expected = reference(tokens, attention_mask=mask, use_cache=False).logits.float()
        actual, activations = capture_final(model, tokens, mask, positions, layers)
        torch.testing.assert_close(actual.float(), expected, atol=atol, rtol=rtol)
        diagnostics["comparisons"].append({"context": label,
            "hf_max_abs_logit_error": float((actual.float() - expected).abs().max()),
            "record_ids": [record["record_id"] for record in records],
            "clean_target_bounded_scores": bounded_score(actual, positions, tensors[4], tensors[5]).tolist()})
        # Every requested head is checked on both supplied contexts.
        for layer, head in heads:
            patched = model.run_with_hooks(tokens, fwd_hooks=[
                (hook_name(layer), patch_head(activations[layer], positions, head))],
                attention_mask=mask, prepend_bos=False, use_cache=False)
            torch.testing.assert_close(patched.float(), actual.float(), atol=atol, rtol=rtol)
            diagnostics["comparisons"].append({"context": label, "head": [layer, head],
                "noop_max_abs_logit_error": float((patched.float() - actual.float()).abs().max())})
        diagnostics["hook_shapes"] = {str(layer): list(value.shape) for layer, value in activations.items()}
        del expected, actual, activations, patched
    del reference
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return diagnostics
