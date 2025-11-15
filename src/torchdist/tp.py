from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)

from utils import barrier, dist_context, print0


class Model(nn.Module):
    def __init__(self, hidden_size: int = 1000):
        super().__init__()
        self.layer1 = nn.Linear(1000, hidden_size)
        self.activation = nn.SiLU()
        self.layer2 = nn.Linear(hidden_size, 1000)

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.activation(x)
        return self.layer2(x)


def main():
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError("This TP demo assumes world_size == 2")

    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=['tp'])


    torch.manual_seed(0)

    model = Model(hidden_size=2_500_000).cuda()

    model = parallelize_module(
        model,
        mesh["tp"],
        {
            "layer1": ColwiseParallel(),
            "layer2": RowwiseParallel(),
        },
    )

    # model = fully_shard(
    #     model,
    #     mesh=mesh["fsdp"],
    #     reshard_after_forward=True,
    # )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    inputs = torch.randn(4, 1000, device="cuda")
    targets = torch.randn(4, 1000, device="cuda")

    for step in range(2):
        outputs = model(inputs)

        loss = F.mse_loss(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print0(f"step {step} loss: {loss.item():.4f}")

    barrier()
    print0("tp mesh:", mesh["tp"])


if __name__ == "__main__":
    with dist_context():
        main()
