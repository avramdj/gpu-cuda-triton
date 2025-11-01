import torch
import triton
import triton.language as tl
from utils import get_device

DEVICE = get_device()


@triton.jit
def vadd_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    vadd_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)  # type: ignore
    return output


class VAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor):
        ctx.save_for_backward(x, y)
        return add(x, y)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, y = ctx.saved_tensors
        return grad_output, grad_output


if __name__ == "__main__":

    def fmt(x: float):
        return f"{x:.2e}"

    def bench(size: int):
        x = torch.randn(size, device=DEVICE)
        y = torch.randn(size, device=DEVICE)
        triton_ms = triton.testing.do_bench(lambda: VAdd.apply(x, y))
        torch_ms = triton.testing.do_bench(lambda: torch.add(x, y))
        return triton_ms, torch_ms

    print("Triton vs. Torch")
    print("Size\t\tTriton (ms)\tTorch (ms)\tRatio")
    for size in [1024, 1024 * 1024, 1024 * 1024 * 1024]:
        triton_ms, torch_ms = bench(size)
        print(
            f"{size:<10}\t{fmt(triton_ms):<10}\t{fmt(torch_ms):<10}\t{((triton_ms - torch_ms) / max(triton_ms, torch_ms)):<10}"
        )
