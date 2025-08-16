import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms as T

from models.ttt_model import TTTModel
from utils import train_transform, make_rotations

import random

class TrainCfg:
    epochs = 15
    lr = 1e-3
    weight_decay = 1e-4
    lambda_aux = 0.5
    batch_size = 128
    num_workers = 2
    seed = 42

    #num_workers controls how many separate CPU processes PyTorch will use to load and preprocess batches from the dataset in parallel

def main():
    # Repro
    random.seed(TrainCfg.seed)
    torch.manual_seed(TrainCfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_set = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=TrainCfg.batch_size, shuffle=True,
                              num_workers=TrainCfg.num_workers, pin_memory=True)

    # Model/optim
    model = TTTModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=TrainCfg.lr, weight_decay=TrainCfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TrainCfg.epochs)

    # Train
    for ep in range(1, TrainCfg.epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Main loss
            logits_main, _ = model(x)
            loss_main = F.cross_entropy(logits_main, y)

            # Aux loss on rotated batch
            x_rot, z_rot = make_rotations(x)
            _, logits_aux = model(x_rot)
            loss_aux = F.cross_entropy(logits_aux, z_rot)

            loss = loss_main + TrainCfg.lambda_aux * loss_aux

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            loss_sum += loss.item() * x.size(0)
            correct += (logits_main.argmax(1) == y).sum().item()
            total += x.size(0)

        sched.step()
        print(f"Epoch {ep:02d} | Train Acc: {correct/total*100:.2f}% | Loss: {loss_sum/total:.4f}")

    # Save weights
    torch.save(model.state_dict(), "ttt_cifar10.pt")
    print("✅ Saved model to ttt_cifar10.pt")

if __name__ == "__main__":
    main()
