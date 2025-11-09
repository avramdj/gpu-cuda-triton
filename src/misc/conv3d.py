import torch
import triton.testing
from torch.nn.functional import conv3d

device = torch.device("cuda")

input_shape = (8, 384, 3, 34, 34)
weight_shape = (384, 384, 3, 3, 3)
B, C, T, H, W = 8, 384, 3, 34, 34

conv3d_compiled = torch.compile(conv3d, mode="max-autotune")


def make_tensors(dtype):
    input = torch.randn(input_shape, device=device, dtype=dtype)
    kernel = torch.randn(weight_shape, device=device, dtype=dtype)
    bias = torch.randn(C, device=device, dtype=dtype)
    return input, kernel, bias


def fn(input, kernel, bias):
    return conv3d(input, kernel, bias=bias, padding=0, dilation=1, stride=1)


def fn_compiled(input, kernel, bias):
    return conv3d_compiled(input, kernel, bias=bias, padding=0, dilation=1, stride=1)


print(
    f"float32: {triton.testing.do_bench(lambda: fn(*make_tensors(torch.float32)), warmup=15, rep=100)}"
)
print(
    f"bfloat16: {triton.testing.do_bench(lambda: fn(*make_tensors(torch.bfloat16)), warmup=15, rep=100)}"
)
print(
    f"float16: {triton.testing.do_bench(lambda: fn(*make_tensors(torch.float16)), warmup=15, rep=100)}"
)

print(
    f"float32 compiled: {triton.testing.do_bench(lambda: fn_compiled(*make_tensors(torch.float32)), warmup=15, rep=100)}"
)
print(
    f"bfloat16 compiled: {triton.testing.do_bench(lambda: fn_compiled(*make_tensors(torch.bfloat16)), warmup=15, rep=100)}"
)
print(
    f"float16 compiled: {triton.testing.do_bench(lambda: fn_compiled(*make_tensors(torch.float16)), warmup=15, rep=100)}"
)
