import torch

rand = torch.tensor([[0, 0], [1, 1]])

map = torch.tensor([[1, 2, 3, 4, 5, 6],
                    [7, 8, 9, 10, 11, 12]])

print(map[rand[..., 1], rand[..., 0]].shape)

