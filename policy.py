import torch


class Policy(torch.nn.Module):
    def __init__(self, inputs: int, outputs: int, hidden_dim: int = 128):
        """
        Create a policy network
        :param inputs: number of inputs
        :param outputs: number of outputs
        :param hidden_dim: dimension of the hidden layer
        """
        super(Policy, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(inputs, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, outputs),
            torch.nn.Tanh(),
        )

        # cpu is faster than gpu for this small network
        self.device = "cpu"

        # no backprop is required for our methods
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, state: torch.Tensor):
        """
        Determine which action to take based on the state
        :param state: observed state
        :return: action to take
        """
        state = torch.FloatTensor(state).to(self.device)
        return self.net(state)

    def save(self, path: str):
        """
        Save the policy
        :param path: path to file
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """
        Load a saved policy
        :param path: path to file
        """
        self.load_state_dict(torch.load(path))
