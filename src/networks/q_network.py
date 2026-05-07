import torch

class QNetwork(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: list):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_layers:
            layers.append(torch.nn.Linear(last, h))
            layers.append(torch.nn.ReLU())
            last = h
        layers.append(torch.nn.Linear(last, output_dim))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.model(x)


if __name__ == "__main__":
    pass
