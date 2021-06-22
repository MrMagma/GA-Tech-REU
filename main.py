import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import csv

# Download training data from open datasets.
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, size):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

loss_fn = nn.CrossEntropyLoss()

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return (correct, test_loss)

def run(size):
    performance = []
    model = NeuralNetwork(size).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    t = 0
    while t < 10 or not (len(performance) > 1 and performance[len(performance) - 1][0] == performance[len(performance) - 2][0]):
        print(f"Epoch {t+1} Size {size}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        performance.append(test(test_dataloader, model))
        t = t + 1
    print("Done!")

    fname = f"./models/model-{size}.pth"

    torch.save(model.state_dict(), fname)

    return [p[0] for p in performance]

performances = []

with open("data.csv", "r+") as data:
    r = csv.reader(data)
    next(r)

    neurons = 0
    for n, *p in r:
        neurons = int(n)
    neurons = neurons + 1

    w = csv.writer(data)
    while True:
        performance = run(neurons)
        w.writerow([neurons] + performance)
        print(f"Model with {neurons} neurons converged with accuracy {performance[len(performance) - 1]}")
        neurons = neurons + 1