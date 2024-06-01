import pickle
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import wandb


from model.models import GaitAnalysisModel

X_FILENAME = "data/X_100.pkl"
SPLIT_PATH = "data/split.pkl"
Y_FILENAME = "data/y.csv"
PREDICTED_VALUE = "speed"
BATCH_SIZE = 8
NR_EPOCHS = 50
DROPOUT_PROB = 0.5
HIDDEN_SIZE = 128
NUM_LSTM_LAYERS = 2
MODEL_ARCHITECTURE = "LSTM"


def get_split(X, y_df, video_ids):
    res_X = []
    res_y = []
    for video_id, X_data in X:
        if video_id in video_ids:
            curr_X = X_data
            curr_y = y_df[y_df["video_id"] == int(video_id)][PREDICTED_VALUE].tolist()
            if len(curr_y) == 2 and len(curr_X) == 500:
                res_X.append(X_data)
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


def setup_wandb():
    wandb.init(
        # set the wandb project where this run will be logged
        project="gait-analysis",

        # track hyperparameters and run metadata
        config={
            "predicted_value": PREDICTED_VALUE,
            "architecture": MODEL_ARCHITECTURE,
            "dataset": X_FILENAME,
            "epochs": NR_EPOCHS,
        }
    )


def train():
    train_loader, validation_loader, test_loader, input_size = prepare_data()
    model = GaitAnalysisModel(input_size, HIDDEN_SIZE, NUM_LSTM_LAYERS, 2, DROPOUT_PROB)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_model_state_dict = None
    best_val_loss = float('inf')

    setup_wandb()

    print("Starting training...")

    # Training loop
    for epoch in range(NR_EPOCHS):
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
        print(f'Epoch {epoch + 1}/{NR_EPOCHS}, Loss: {epoch_loss:.4f}')

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
    torch.save(model, f"models/{MODEL_ARCHITECTURE}_{NR_EPOCHS}_{test_loss}.pth")
    wandb.log({"test_loss": test_loss})


if __name__ == "__main__":
    train()
