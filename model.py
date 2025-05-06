
import torch
import torch.nn as nn

# Define the Classifier model
class Classifier(nn.Module):
    def __init__(self, input_dim=2048, num_classes=4):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Function to load the model from a file
def load_classifier(path):
    model = Classifier()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model
