import torch as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda import amp
from torchvision import transforms
from typing_extensions import Self
from typing import Callable
import pandas as pd
import numpy as np
from tqdm import tqdm


class MNISTDataset(Dataset):
    def __init__(
        self: Self,
        filename: str = "data.csv",
        transforms: Callable = None,
        target_transforms: Callable = None,
    ) -> None:
        super().__init__()
        self.data = pd.read_csv(filename).to_numpy()
        self.transforms = transforms
        self.target_transforms = target_transforms

    def __len__(self: Self) -> int:
        return len(self.data)

    def __getitem__(self: Self, idx: int) -> tuple[T.Tensor, int]:
        label = self.data[idx, 0]
        img = self.data[idx, 1:]

        if self.transforms is not None:
            img = self.transforms(img)
        if self.target_transforms is not None:
            label = self.target_transforms(label)

        return img, label


class Network(nn.Module):
    def __init__(self: Self) -> None:
        super().__init__()
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(1, 512, 3, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(),
            nn.Conv2d(512, 1024, 2, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(),
        )
        self.fully_connected_layers = nn.Sequential(
            nn.Linear(36864, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 10),
        )

    def forward(self: Self, x: T.Tensor) -> T.Tensor:
        x = self.convolutional_layers(x)
        x = x.view(-1, 36864)
        x = self.fully_connected_layers(x)
        return x


def main() -> None:
    device = T.device("cuda" if T.cuda.is_available() else "cpu")

    t = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.reshape(28, 28)),
            transforms.Lambda(lambda x: x.astype(np.float32)),
            transforms.Lambda(lambda x: x / 255),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.to(device)),
        ]
    )
    target_t = transforms.Lambda(lambda x: T.tensor(x, device=device))
    dataset = MNISTDataset(transforms=t, target_transforms=target_t)
    train_set, test_set = random_split(dataset, [0.95, 0.05])
    train_loader = DataLoader(train_set, 32)
    test_loader = DataLoader(test_set, len(test_set))

    network = Network().to(device)
    optimizer = optim.Adam(network.parameters(), lr=1e-3)
    scaler = amp.GradScaler()
    criterion = nn.CrossEntropyLoss()

    epochs = 100
    network.train()
    try:
        for epoch in range(1, epochs + 1):
            num_correct: int = 0
            count: int = 0
            for batch, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
                with amp.autocast():
                    y = network(batch)
                    loss = criterion(y, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                num_correct += T.sum(T.argmax(y.detach(), 1) == labels)
                count += labels.numel()

            print(f"Accuracy: {num_correct / (count):.2%}")
    except KeyboardInterrupt as e:
        T.save(network.state_dict(), "trained_model.pt")


if __name__ == "__main__":
    main()
