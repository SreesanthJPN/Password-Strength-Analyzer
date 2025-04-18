import torch.nn as nn
import torch.nn.functional as F

class PasswordNet(nn.Module):
    def __init__(self, input_size, output_size=3):
        super(PasswordNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 64)
        self.out = nn.Linear(64, output_size)

        
    def forward(self, x):
        x = F.relu(self.fc1(x))      # Hidden layer 1
        x = F.relu(self.fc2(x))      # Hidden layer 2
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))              # Output layer (raw logits)
        x = F.softmax(self.out(x))
        return x                     # Use softmax during evaluation if needed
