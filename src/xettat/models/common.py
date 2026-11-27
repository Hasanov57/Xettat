import torch


def _out(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """[B,T,1] -> [B,T] for binary (v1) models; [B,T,C] unchanged for v2 (keep/convert/delete)."""
    return logits.squeeze(-1) if num_classes == 1 else logits
