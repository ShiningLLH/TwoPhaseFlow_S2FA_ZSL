import torch
import torch.nn as nn

class Fea_encoder(nn.Module):
    def __init__(self):
        super(Fea_encoder, self).__init__()
        self.Net1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),  # Batch Normalization
            nn.LeakyReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=10),
            nn.Dropout(0.5)
        )

    def forward(self, input):
        input = input.unsqueeze(1)
        x = self.Net1(input)
        return x

# pre-trained attribute embedding
class Attribute_encoder(nn.Module):
    def __init__(self):
        super(Attribute_encoder, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 13)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.dropout(x)
        return x

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.Fea_encoder = Fea_encoder()
        self.Attribute_encoder = Attribute_encoder()

    def forward_once(self, x):
        output = self.Fea_encoder(x)
        output = torch.flatten(output, 1)
        return output

    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        output_attri = self.Attribute_encoder(output1)
        return output1, output2, output3, output_attri