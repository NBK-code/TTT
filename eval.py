import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from models.ttt_model import TTTModel
from utils import test_clean_transform, test_shift_transform, eval_no_ttt, ttt_adapt_one_sample, predict_main

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loaders
    test_clean = CIFAR10(root="./data", train=False, download=True, transform=test_clean_transform)
    test_shift = CIFAR10(root="./data", train=False, download=False, transform=test_shift_transform)

    clean_loader = DataLoader(test_clean, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    shift_loader = DataLoader(test_shift, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    # Load model
    model = TTTModel().to(device)
    model.load_state_dict(torch.load("ttt_cifar10.pt", map_location=device))
    print("✅ Loaded ttt_cifar10.pt")

    # Baselines (no TTT)
    acc_clean_no = eval_no_ttt(model, clean_loader, device, desc="Clean Test")
    acc_shift_no = eval_no_ttt(model, shift_loader, device, desc="Shifted Test")

    # TTT evaluation on shifted test (per-sample)
    # Iterate the underlying dataset with batch_size=1 for per-sample adaptation
    base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    total, correct = 0, 0
    for idx in range(len(test_shift)):
        x, y = test_shift[idx]
        x = x.unsqueeze(0).to(device)
        y = torch.tensor([y], device=device)

        # reset model to base before each sample
        model.load_state_dict(base_state, strict=True)

        # adapt on aux loss (2 quick steps)
        ttt_adapt_one_sample(model, x, steps=2, lr=5e-4, bn_update=True, adapt_last_block_only=True)

        # predict
        pred = predict_main(model, x)
        correct += (pred == y).sum().item()
        total += 1

        if (idx + 1) % 1000 == 0:
            print(f"[TTT] processed {idx+1}/{len(test_shift)}")

    acc_shift_ttt = correct / total * 100
    print("\n=== Summary ===")
    print(f"Clean (No-TTT):   {acc_clean_no:.2f}%")
    print(f"Shifted (No-TTT): {acc_shift_no:.2f}%")
    print(f"Shifted (TTT):    {acc_shift_ttt:.2f}%  (gain: {acc_shift_ttt - acc_shift_no:+.2f} pts)")

if __name__ == "__main__":
    main()
