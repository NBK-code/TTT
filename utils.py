import torch
import torch.nn.functional as F
from torchvision import transforms as T

# Train-time transforms
train_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
])

# Clean test transform
test_clean_transform = T.ToTensor()

# Shifted/corrupted test transform (simple, reproducible)
test_shift_transform = T.Compose([
    T.ColorJitter(brightness=0.6),
    T.GaussianBlur(kernel_size=3, sigma=(0.5, 1.0)),
    T.ToTensor(),
])

def make_rotations(x):
    """
    Given a batch x [B,C,H,W], returns:
      x_rot [4B,C,H,W] and rotation labels z_rot in {0,1,2,3}.
    """
    xs, ys = [], []
    device = x.device
    for k in range(4):  # 0, 90, 180, 270
        xs.append(torch.rot90(x, k=k, dims=(-2, -1)))
        ys.append(torch.full((x.size(0),), k, dtype=torch.long, device=device))
    return torch.cat(xs, 0), torch.cat(ys, 0)

@torch.no_grad()
def eval_no_ttt(model, loader, device, desc=""):
    model.eval().to(device)
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    acc = correct / total * 100
    print(f"[No-TTT] {desc} Acc: {acc:.2f}%")
    return acc

def ttt_adapt_one_sample(model, x_test, steps=2, lr=5e-4, bn_update=True, adapt_last_block_only=True):
    """
    Do a few gradient steps on the aux rotation loss using (x_test, z_rot),
    updating a subset of parameters to keep it light/robust.
    """
    model.train()
    # Freeze all
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze last blocks (common, stable choice)
    if adapt_last_block_only:
        for m in [model.encoder.b3, model.encoder.b2]:
            for p in m.parameters():
                p.requires_grad = True
    else:
        for p in model.encoder.parameters():
            p.requires_grad = True

    # BN affine helps
    if bn_update:
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                if m.weight is not None: m.weight.requires_grad = True
                if m.bias  is not None: m.bias.requires_grad  = True

    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=lr, momentum=0.9)

    x_rot, z_rot = make_rotations(x_test)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        _, logits_aux = model(x_rot)
        loss_aux = F.cross_entropy(logits_aux, z_rot)
        loss_aux.backward()
        opt.step()

    model.eval()
    return model

@torch.no_grad()
def predict_main(model, x):
    model.eval()
    logits, _ = model(x)
    return logits.argmax(1)