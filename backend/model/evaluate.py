import torch
from torch import nn

from model.models import GaitAnalysisModel
from model.train import prepare_data

MODEL_PATH = "model.pth"


def evaluate():
    train_loader, validation_loader, test_loader, input_size = prepare_data()
    model = GaitAnalysisModel(input_size, 128, 2, 2)
    model.load_state_dict(torch.load("model.pth"))

    criterion = nn.MSELoss()
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

    test_loss = test_loss / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')

if __name__ == "__main__":
    evaluate()
