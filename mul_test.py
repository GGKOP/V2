import torch

from util.save_tensor import save_tensor

device = torch.device("cpu")

if __name__ == "__main__":
    a = torch.randn(10, 512, dtype=torch.float32, device=device)
    b = torch.randn(512, 10, dtype=torch.float32, device=device)
    c = a @ b
    print(a)
    print(b)
    print(c)
    save_tensor(a, "a")
    save_tensor(b, "b")
    save_tensor(c, "c")
