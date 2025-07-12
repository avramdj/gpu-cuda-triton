import math
from typing import Any, Callable

import torch
import triton
import triton.testing


def get_device():
    return triton.runtime.driver.active.get_active_torch_device()


def select_perf_unit(
    x_vals: list[Any],
    x_names: list[str],
    args: dict[str, Any],
    input_factory: Callable,
    func_torch: Callable,
    quantiles: list[float],
) -> tuple[str, float]:
    perf_unit = "GB/s"
    perf_divisor = 1e9
    if not x_vals:
        return perf_unit, perf_divisor

    sample_kwargs = args.copy()
    median_x_val = x_vals[len(x_vals) // 2]

    if len(x_names) == 1:
        sample_kwargs[x_names[0]] = median_x_val
    else:
        for i, name in enumerate(x_names):
            sample_kwargs[name] = median_x_val[i]

    sample_inputs = input_factory(**sample_kwargs)
    output = func_torch(*sample_inputs)
    total_bytes = sum(
        t.numel() * t.element_size()
        for t in (*sample_inputs, output)
        if isinstance(t, torch.Tensor)
    )

    sample_result = triton.testing.do_bench(
        lambda: func_torch(*sample_inputs), quantiles=quantiles
    )

    if sample_result is not None and sample_result[0] > 0:
        sample_ms = sample_result[0]
        bytes_per_sec = total_bytes / (sample_ms * 1e-3)

        units = ["B/s", "KB/s", "MB/s", "GB/s", "TB/s"]
        divisors = [1, 1e3, 1e6, 1e9, 1e12]

        if bytes_per_sec > 0:
            power = math.floor(math.log(bytes_per_sec, 1000))
            unit_idx = int(power) if power >= 0 else 0
            if unit_idx >= len(units):
                unit_idx = len(units) - 1
            perf_unit = units[unit_idx]
            perf_divisor = divisors[unit_idx]

    return perf_unit, perf_divisor


def get_bench(
    plot_name: str,
    x_names: list[str],
    x_vals: list[Any],
    func_triton: Callable,
    func_torch: Callable,
    input_factory: Callable | None = None,
    num_inputs: int | None = None,
    quantiles: list[float] | None = None,
    x_log: bool = True,
    line_arg: str = "provider",
    line_vals: list[str] = ["triton", "torch"],
    line_names: list[str] = ["Triton", "Torch"],
    styles: list[tuple[str, str]] = [("blue", "-"), ("green", "-")],
    ylabel: str = "",
    args: dict[str, Any] = {},
):
    if (input_factory is None and num_inputs is None) or (
        input_factory is not None and num_inputs is not None
    ):
        raise ValueError(
            "Must provide either 'input_factory' or 'num_inputs', but not both."
        )

    final_input_factory: Callable
    if num_inputs is not None:
        device = get_device()

        def default_input_factory(**kwargs):
            if "size" not in kwargs:
                raise ValueError(
                    "Default input factory requires 'size' as one of the x_names."
                )
            size = kwargs["size"]
            return tuple(
                torch.rand(size, device=device, dtype=torch.float32)
                for _ in range(num_inputs)
            )

        final_input_factory = default_input_factory
    elif input_factory is not None:
        final_input_factory = input_factory
    else:
        raise ValueError("Should not be reached")

    if quantiles is None:
        quantiles = [0.5, 0.2, 0.8]

    perf_unit, perf_divisor = select_perf_unit(
        x_vals=x_vals,
        x_names=x_names,
        args=args,
        input_factory=final_input_factory,
        func_torch=func_torch,
        quantiles=quantiles,
    )
    if not ylabel:
        ylabel = perf_unit

    perf_report_decorator = triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=x_names,
            x_vals=x_vals,
            x_log=x_log,
            line_arg=line_arg,
            line_vals=line_vals,
            line_names=line_names,
            styles=styles,
            ylabel=ylabel,
            plot_name=plot_name,
            args=args,
        )
    )

    def benchmark(provider, **kwargs):
        inputs = final_input_factory(**kwargs)
        output = func_torch(*inputs)
        total_bytes = sum(
            t.numel() * t.element_size()
            for t in (*inputs, output)
            if isinstance(t, torch.Tensor)
        )

        bench_fn = None
        if provider == "torch":
            bench_fn = lambda: func_torch(*inputs)  # noqa: E731
        elif provider == "triton":
            bench_fn = lambda: func_triton(*inputs)  # noqa: E731

        if bench_fn is None:
            ms, min_ms, max_ms = -1.0, -1.0, -1.0
        else:
            result = triton.testing.do_bench(bench_fn, quantiles=quantiles)
            if result is None:
                ms, min_ms, max_ms = -1.0, -1.0, -1.0
            else:
                ms, min_ms, max_ms = result

        def perf_calculator(ms: float) -> float:
            if ms <= 0:
                return 0
            bytes_per_sec = total_bytes / (ms * 1e-3)
            return bytes_per_sec / perf_divisor

        return perf_calculator(ms), perf_calculator(max_ms), perf_calculator(min_ms)

    return perf_report_decorator(benchmark)


if __name__ == "__main__":
    print(get_device())
