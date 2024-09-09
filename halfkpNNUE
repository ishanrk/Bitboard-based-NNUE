import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class HalfKPNetwork(nn.Module):
    def __init__(self, num_features=41024, hidden_size=256):
        super(HalfKPNetwork, self).__init__()
        # Input size is num_features (HalfKP input size)
        self.fc1 = nn.Linear(num_features, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)  # Output is a scalar evaluation

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# dataset loader
  class HalfKPDataset(Dataset):
    def __init__(self, data_file, num_features=41024, transform=None):
        self.samples = []
        self.num_features = num_features
        self.transform = transform
        self.load_data(data_file)

    def load_data(self, data_file):
        with open(data_file, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                feature_indices = list(map(int, tokens[:-1]))
                target = float(tokens[-1])
                self.samples.append((feature_indices, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feature_indices, target = self.samples[idx]
        features = torch.zeros(self.num_features, dtype=torch.float32)
        features[feature_indices] = 1.0  # One-hot encoding for active features

        if self.transform:
            features = self.transform(features)

        return features, torch.tensor([target], dtype=torch.float32)
