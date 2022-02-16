from torch import nn
from torch.utils.data import Dataset


class Model(nn.Module):
    def __init__(self, inp_size):
        super().__init__()
        self.inp_size = inp_size
        layers = []
        layers += [
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32, affine=False),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64, affine=False),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(inp_size // (2 ** 5) * 128, 32),
            nn.Linear(32, 1),
            nn.Sigmoid()]
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        res = self.blocks(x)
        return res


class MelDataset(Dataset):
    def __init__(self, mel, labels):
        self.labels = labels
        self.mel = mel

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        label = self.labels[idx]
        mel = self.mel[idx]
        return mel, label
