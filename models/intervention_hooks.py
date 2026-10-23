"""Exact replacement of one pre-output-projection attention head at one token."""
import torch


def hook_name(layer):
    return f"blocks.{layer}.attn.hook_z"


def patch_head(source, positions, head):
    def replace(z, hook):
        if z.ndim != 4 or z.shape[2:] != (12, 64):
            raise ValueError(f"Unexpected hook_z dimensions: {tuple(z.shape)}")
        if source.shape != (z.shape[0], 12, 64):
            raise ValueError("Source head activation dimensions do not match")
        result = z.clone()
        rows = torch.arange(z.shape[0], device=z.device)
        result[rows, positions, head, :] = source[:, head, :]
        return result
    return replace


def bounded_score(logits, positions, target, distractor):
    rows = torch.arange(logits.shape[0], device=logits.device)
    final = logits[rows, positions].float()
    return torch.tanh((final[rows, target] - final[rows, distractor]) / 2)


def batch_tensors(records, device):
    lengths = [len(record["clean_tokens"]) for record in records]
    width = max(lengths)
    # Right padding is masked and all scoring/patching indexes true final tokens.
    tensors = []
    for field in ("clean_tokens", "corrupt_tokens"):
        tensors.append(torch.tensor([r[field] + [50256] * (width - len(r[field])) for r in records],
                                    dtype=torch.long, device=device))
    mask = torch.arange(width, device=device)[None, :] < torch.tensor(lengths, device=device)[:, None]
    return (*tensors, mask.long(), torch.tensor(lengths, device=device) - 1,
            torch.tensor([r["target_token"] for r in records], device=device),
            torch.tensor([r["distractor_token"] for r in records], device=device))


def capture_final(model, tokens, mask, positions, layers):
    captured = {}

    def capture(layer):
        def callback(z, hook):
            if z.ndim != 4 or z.shape[2:] != (12, 64):
                raise ValueError(f"Unexpected hook_z shape {tuple(z.shape)}")
            captured[layer] = z[torch.arange(z.shape[0], device=z.device), positions].detach().clone()
        return callback

    logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name(l), capture(l)) for l in layers],
                                  attention_mask=mask, prepend_bos=False, use_cache=False)
    if set(captured) != set(layers):
        raise ValueError("Required hook did not fire")
    return logits, captured


@torch.inference_mode()
def effect_batch(model, records, heads, device):
    clean, corrupt, mask, positions, target, distractor = batch_tensors(records, device)
    _, source = capture_final(model, clean, mask, positions, sorted({l for l, h in heads}))
    reference = bounded_score(model(corrupt, attention_mask=mask, prepend_bos=False, use_cache=False),
                              positions, target, distractor)
    values = []
    for layer, head in heads:
        logits = model.run_with_hooks(corrupt, fwd_hooks=[
            (hook_name(layer), patch_head(source[layer], positions, head))],
            attention_mask=mask, prepend_bos=False, use_cache=False)
        values.append(bounded_score(logits, positions, target, distractor) - reference)
    result = torch.stack(values, dim=1)
    if not torch.isfinite(result).all() or (result.abs() > 2).any():
        raise ValueError("Non-finite or unbounded intervention effects")
    return result.cpu().numpy()
