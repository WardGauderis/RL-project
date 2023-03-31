import torch


class Policy(torch.nn.Module):
    def __init__(self, inputs, outputs, hidden_dim=128):
        super(Policy, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(inputs, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, outputs),
            torch.nn.Tanh(),
        )

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, state):
        return self.net(state)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
