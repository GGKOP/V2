import os
import io

import torch

dir_ = "./data"


def save_tensor(tensor: torch.Tensor, filename: str):
    if not os.path.exists(dir_):
        os.mkdir(dir_)
    tensor.to(torch.float32)
    if len(tensor.shape) == 3:
        tensor = tensor[0]
    f = io.BytesIO()
    torch.save(tensor, f, _use_new_zipfile_serialization=True)
    print(filename, tensor.shape)
    with open(f"{dir_}/{filename}.pt", "wb") as file:
        file.write(f.getbuffer())


def load_tensor(filename: str):
    with open(f"{dir_}/{filename}", "rb") as file:
        buffer = io.BytesIO(file.read())
    return torch.load(buffer)


if __name__ == "__main__":
    print(load_tensor("source.pt").shape)
