`nn.Parameter`는 “이 텐서는 학습 대상이다”라고 모델에 알려주는 용도입니다.

핵심:
1. `nn.Module` 안에 `nn.Parameter`로 선언하면 자동으로 파라미터로 등록됩니다.
2. 그래서 `model.parameters()`에 포함되어 옵티마이저가 업데이트합니다.
3. 일반 `torch.Tensor`는 같은 위치에 둬도 자동 등록되지 않습니다.

즉, 직접 학습될 가중치/편향을 만들 때 `nn.Parameter`를 씁니다.