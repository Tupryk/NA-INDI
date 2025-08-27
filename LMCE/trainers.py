import torch
from torch import nn
from tqdm import trange
from torch.utils.data import DataLoader


def train(model: nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          epochs: int = 32,
          lr: float = 1e-3,
          scheduler_step_size: int=5,
          scheduler_gamma: float=.01,
          device: str = "cpu") -> nn.Module:

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    loss_fn = nn.L1Loss()
    # loss_fn = nn.MSELoss()

    pbar = trange(epochs, unit="epoch")
    for t in pbar:
        model.train()

        # Train step
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            out = model(X)
            loss = loss_fn(out, y)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader)

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = loss_fn(out, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step()

        pbar.set_postfix(train_loss=train_loss, val_loss=val_loss, LR=scheduler.get_last_lr()[0])

    print("Done!")
    return model
