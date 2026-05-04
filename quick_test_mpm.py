import timm
import torch
from mpm import apply_patch
from time import perf_counter_ns

model = timm.create_model("vit_base_patch16_384", pretrained=True).eval()

x = torch.randn(1, 3, 384, 384)
s1 = perf_counter_ns()
with torch.inference_mode():
    y = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
e1 = perf_counter_ns()
print(f"Time without MPM: {e1 - s1}ns")

apply_patch(model, mpm_layers=(2, 5))

s2 = perf_counter_ns()
with torch.inference_mode():
    y = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
e2 = perf_counter_ns()
print(f"Time with MPM: {e2 - s2}ns")

print(f"MPM Speedup: {(e1 - s1) / (e2 - s2):.3f}x faster")
