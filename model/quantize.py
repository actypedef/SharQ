import torch


@torch.no_grad()
def quantize_int_group(tensor: torch.Tensor, nbits: int, group_size: int) -> torch.Tensor:
    saved_shape = tensor.shape
    tensor = tensor.reshape(-1, group_size)

    tensor_max = tensor.amax(dim=-1, keepdim=True)
    tensor_min = tensor.amin(dim=-1, keepdim=True)
    q_max = (2**nbits) - 1
    q_min = 0

    scales = (tensor_max - tensor_min).clamp(min=1e-5) / q_max
    zero_point = torch.round(-tensor_min / scales).clamp_(min=q_min, max=q_max)
    tensor = (torch.clamp(torch.round(tensor / scales) + zero_point, q_min, q_max) - zero_point) * scales
    return tensor.reshape(saved_shape)


__all__ = ["quantize_int_group"]
