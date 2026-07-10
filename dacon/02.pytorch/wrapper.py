import torch

# Device 자동 선택 wrapper
def get_device():
    """MPS > CUDA > CPU 순서로 자동 선택"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# 모델/텐서 이동 헬퍼
def to_device(obj, device):
    """모델, 배치 한 번에 device로 이동"""
    if isinstance(obj, torch.nn.Module):
        return obj.to(device)
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, (list, tuple)):
        return [to_device(item, device) for item in obj]
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj