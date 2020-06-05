import torch
from torch import nn, optim
from torch.nn import functional as F

NETWORK_INPUT_SIZE = 342
NETWORK_OUTPUT_SIZE = 2


class Network(nn.Module):
	def __init__(self, input_size=NETWORK_INPUT_SIZE, output_size=NETWORK_OUTPUT_SIZE):
		super(Network, self).__init__()
		self.fc1 = nn.Linear(input_size, 256)
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 64)
		self.fc4 = nn.Linear(64, 32)
		self.fc5 = nn.Linear(32, 16)
		self.fc6 = nn.Linear(16, 8)
		self.fc7 = nn.Linear(8, 4)
		self.fc8 = nn.Linear(4, output_size)

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.fc3(x)
		x = F.relu(x)
		x = self.fc4(x)
		x = F.relu(x)
		x = self.fc5(x)
		x = F.relu(x)
		x = self.fc6(x)
		x = F.relu(x)
		x = self.fc7(x)
		x = F.relu(x)
		x = self.fc8(x)
		return x


def normal_init(m, mean, std):
	if isinstance(m, nn.Linear):
		m.weight.data.normal_(mean, std)
		m.bias.data.zero_()
