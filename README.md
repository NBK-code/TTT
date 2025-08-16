# Test-Time Training (TTT) on CIFAR-10

This project is a clean, minimal implementation of **Test-Time Training (TTT)** as introduced in  
*Sun et al., "Test-Time Training with Self-Supervision for Generalization under Distribution Shifts" (ICML 2020)*.

TTT is a method to improve robustness to domain shifts:
- During training, the model learns **both the main task** (classification) and an **auxiliary self-supervised task** (e.g., rotation prediction).
- At test time, the model **adapts to each test sample** using the auxiliary loss (since its labels are self-generated).
- This improves representations of unseen/shifted inputs, leading to **better performance on the main task**.


---

## Project Structure
```
TTT
├── models
│   └── ttt_model.py
├── data
├── train.py
├── eval.py
├── utils.py
├── requirements.txt
└── README.md
```


## Setup

### 1. Create a new conda environment
```bash
conda create -n tttenv python=3.10 -y
conda activate tttenv
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Training
Train the model on CIFAR-10 with both classification and auxiliary rotation loss:
```bash
python train.py
```

This will:

Download CIFAR-10 automatically (if not present).

Train for 15 epochs by default.

Save the trained model checkpoint in `./checkpoints/`.


### 4. Evaluation

Evaluate on clean test set and a shifted test set (with distribution shift), with and without TTT:
```bash
python eval.py
```

Example output:
```bash
=== Summary ===
Clean (No-TTT):   87.16%
Shifted (No-TTT): 67.59%
Shifted (TTT):    69.98%  (gain: +2.39 pts)
```

## Results


| Setting            | Accuracy (%) |
| ------------------ | ------------ |
| Clean (No-TTT)     | 87.16        |
| Shifted (No-TTT)   | 67.59        |
| Shifted (With TTT) | 69.98 (+2.39)|


---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


**Reference**

Sun, Y., Wang, X., Liu, Z., Miller, J., Efros, A. A., & Hardt, M. (2020).  
*Test-Time Training with Self-Supervision for Generalization under Distribution Shifts.*  
ICML 2020. [Paper Link](https://arxiv.org/abs/1909.13231)

