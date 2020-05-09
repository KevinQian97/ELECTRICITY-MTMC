import numpy as np
import torch


def pack_tensor(tensor, dim_size):
    rounded_size = np.ceil(tensor.shape[0] / dim_size).astype(int) * dim_size
    packed_tensor = torch.zeros((rounded_size, *tensor.shape[1:]),
                                dtype=tensor.dtype, device=tensor.device)
    packed_tensor[:tensor.shape[0]] = tensor
    packed_tensor = packed_tensor.reshape((dim_size, -1, *tensor.shape[1:]))
    return packed_tensor
