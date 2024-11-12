import torch
import torch.nn as nn
import os

# The model architecture
class HalfKPNetwork(nn.Module):
    def __init__(self, num_features=41024, hidden_size=256):
        super(HalfKPNetwork, self).__init__()
        self.fc1 = nn.Linear(num_features, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def model_fn(model_dir):
    # Load the model from the saved file
    model = HalfKPNetwork()
    model_path = os.path.join(model_dir, 'model.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def predict_fn(input_data, model):
    # Convert the input to a torch tensor and perform inference
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    output = model(input_tensor)
    return output.detach().numpy()
