import torch

x = torch.randn(3,4,5,6)

print(f"{x.shape=}, {x.stride()=}")

x = x.transpose(1,2)

print(f"{x.shape=}, {x.stride()=}")

torch.save(x, "x.pt")

x = torch.load("x.pt", weights_only=True)

print(f"{x.shape=}, {x.stride()=}")

import os
os.system("rm x.pt")