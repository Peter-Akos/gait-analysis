import pickle
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import wandb

from model.models import GaitAnalysisModel, CNN

X_FILENAME = "data/X_100.pkl"
SPLIT_PATH = "data/split.pkl"
Y_FILENAME = "data/y.csv"
PREDICTED_VALUE = "cadence"
BATCH_SIZE = 8
NR_EPOCHS = 10
OPTIMIZER = "RMSprop"
LEARNING_RATE = 0.001

MODEL_ARCHITECTURE = "CNN"

# CONV1D PARAMETERS
INPUT_DIM = 500
VID_LENGTH = 16
CONV_DIM = 32
FILTER_LENGTH = 8
DROPOUT_AMOUNT = 0.5
L2_LAMBDA = 10 ** (-3.5)
LAST_LAYER_DIM = 10

# LSTM PARAMETERS
NUM_LSTM_LAYERS = 2
HIDDEN_SIZE = 128
DROPOUT_PROB = 0.5

hyperparameter_defaults = dict(
    # All
    nr_epochs=NR_EPOCHS,
    learning_rate=LEARNING_RATE,
    model_architecture=MODEL_ARCHITECTURE,
    batch_size=BATCH_SIZE,
    x_filename=X_FILENAME,
    y_filename=Y_FILENAME,
    split_path=SPLIT_PATH,
    predicted_value=PREDICTED_VALUE,
    optimizer=OPTIMIZER,
    # LSTM
    num_lstm_layers=NUM_LSTM_LAYERS,
    hidden_size=HIDDEN_SIZE,
    dropout_prob=DROPOUT_PROB,
    # CNN
    input_dim=INPUT_DIM,
    vid_length=VID_LENGTH,
    conv_dim=CONV_DIM,
    filter_length=FILTER_LENGTH,
    dropout_amount=DROPOUT_AMOUNT,
    l2_lambda=L2_LAMBDA,
    last_layer_dim=LAST_LAYER_DIM
)

wandb.init(config=hyperparameter_defaults, project="gait-analysis")
config = wandb.config


def get_split(X, y_df, video_ids):
    res_X = []
    res_y = []
    scaler = StandardScaler()
    for video_id, X_data in X:
        if video_id in video_ids:
            scaled_X = scaler.fit_transform(X_data)
            curr_y = y_df[y_df["video_id"] == int(video_id)].sort_values(by=['side'])[PREDICTED_VALUE].tolist()
            if len(curr_y) == 2 and len(scaled_X) == 500:
                res_X.append(scaled_X)
                res_y.append(curr_y)

    X_numpy = np.array(res_X)
    y_numpy = np.array(res_y)
    X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
    y_tensor = torch.tensor(y_numpy, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader


def prepare_data(test_only=False):
    with open(X_FILENAME, 'rb') as file:
        X = pickle.load(file)

    with open(SPLIT_PATH, 'rb') as file:
        split = pickle.load(file)

    y_df = pd.read_csv(Y_FILENAME)

    input_size = X[0][1].shape[1]

    if test_only:
        test_loader = get_split(X, y_df, split["test"])
        return test_loader, input_size

    train_loader = get_split(X, y_df, split["train"])
    validation_loader = get_split(X, y_df, split["validation"])
    test_loader = get_split(X, y_df, split["test"])
    return train_loader, validation_loader, test_loader, input_size


def train():
    train_loader, validation_loader, test_loader, input_size = prepare_data()
    if config.model_architecture == "LSTM":
        model = GaitAnalysisModel(input_size, config.hidden_size, config.num_lstm_layers, 2, config.dropout_prob)
    elif config.model_architecture == "CNN":
        model = CNN(
            input_dim=config.input_dim,
            vid_length=config.vid_length,
            conv_dim=config.conv_dim,
            filter_length=config.filter_length,
            dropout_amount=config.dropout_amount,
            l2_lambda=config.l2_lambda,
            last_layer_dim=config.last_layer_dim
        )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    best_model_state_dict = None
    best_val_loss = float('inf')

    print("Starting training...")

    # Training loop
    for epoch in range(config.nr_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Zero the parameter gradients

            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize the weights

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{config.nr_epochs}, Loss: {epoch_loss:.4f}')

        # Validation step
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        val_loss = val_loss / len(validation_loader.dataset)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict = model.state_dict()
            print("New best model found!")
        scheduler.step()
        print(f'Validation Loss: {val_loss:.4f}')

        wandb.log({"train_loss": epoch_loss, "validation_loss": val_loss})

    # Testing the best model
    model.load_state_dict(best_model_state_dict)
    # Save the best model

    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

    test_loss = test_loss / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')
    torch.save(best_model_state_dict, f"models/{config.model_architecture}_{config.nr_epochs}_{test_loss}.pth")
    wandb.log({"test_loss": test_loss, "best_validation_loss": best_val_loss})


if __name__ == "__main__":
    train()
