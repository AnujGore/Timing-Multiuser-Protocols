import torch

def flip_extend_list(lst):
    return lst + lst[::-1]

def check_if_tensor(arr):
    if type(arr) == torch.Tensor:
        return arr
    else:
        return torch.tensor(arr, dtype = torch.float32)
    