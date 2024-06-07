import torch
import torch.nn as nn
import torch.nn.functional as F


class GaitAnalysisModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.5):
        super(GaitAnalysisModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        last_output = out[:, -1, :]
        out = self.dropout(last_output)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class CNN(nn.Module):
    def __init__(self, input_dim=500, vid_length=16, conv_dim=32, filter_length=8, dropout_amount=0.5,
                 l2_lambda=10 ** (-3.5),
                 last_layer_dim=10):
        super(CNN, self).__init__()

        # Define layers
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=conv_dim, kernel_size=filter_length, padding='same')
        self.bn1 = nn.BatchNorm1d(conv_dim)

        self.conv2 = nn.Conv1d(in_channels=conv_dim, out_channels=conv_dim, kernel_size=filter_length, padding='same')
        self.bn2 = nn.BatchNorm1d(conv_dim)

        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.drop1 = nn.Dropout(dropout_amount)

        self.conv3 = nn.Conv1d(in_channels=conv_dim, out_channels=conv_dim, kernel_size=filter_length, padding='same')
        self.bn3 = nn.BatchNorm1d(conv_dim)

        self.conv4 = nn.Conv1d(in_channels=conv_dim, out_channels=conv_dim, kernel_size=filter_length, padding='same')
        self.bn4 = nn.BatchNorm1d(conv_dim)

        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.drop2 = nn.Dropout(dropout_amount)

        self.conv5 = nn.Conv1d(in_channels=conv_dim, out_channels=conv_dim, kernel_size=filter_length, padding='same')
        self.bn5 = nn.BatchNorm1d(conv_dim)

        self.conv6 = nn.Conv1d(in_channels=conv_dim, out_channels=conv_dim, kernel_size=filter_length, padding='same')
        self.bn6 = nn.BatchNorm1d(conv_dim)

        self.pool3 = nn.MaxPool1d(kernel_size=3)
        self.drop3 = nn.Dropout(dropout_amount)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(conv_dim * (vid_length // 12),
                             last_layer_dim)  # Calculate the output size after convolutions and pooling
        self.fc2 = nn.Linear(last_layer_dim, 2)

        # Apply L2 regularization to conv layers
        for layer in [self.conv3, self.conv4, self.conv5, self.conv6]:
            layer.weight = nn.Parameter(layer.weight + l2_lambda * layer.weight)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
