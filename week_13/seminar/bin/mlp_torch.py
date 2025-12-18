#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import lib

# Reuse Tinygrad's visual utilities for comparison
from tqdm.auto import trange


# =============================================================================
# Modules
# =============================================================================


class NLinear(nn.Module):
    """
    N independent linear layers.
    Input: (Batch, N, In_Features)
    Output: (Batch, N, Out_Features)
    """

    def __init__(self, n: int, in_features: int, out_features: int):
        super().__init__()
        # Shape: (N, In, Out)
        self.weight = nn.Parameter(torch.empty(n, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(n, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # Glorot/Xavier Uniform
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, N, In)
        # To match Tinygrad logic exactly (transpose -> matmul -> transpose)
        # This exploits batch matrix multiplication logic
        x = x.transpose(0, 1)  # (N, Batch, In)
        x = torch.bmm(x, self.weight)  # (N, Batch, Out)
        x = x.transpose(0, 1)  # (Batch, N, Out)
        return x + self.bias


class PiecewiseLinearEmbeddings(nn.Module):
    """
    Piecewise Linear Embeddings with 'Sandwich' layout for vectorization.
    """

    def __init__(self, bins: list[np.ndarray], d_embedding: int):
        super().__init__()
        n_features = len(bins)
        n_bins_per_feat = [len(x) - 1 for x in bins]
        max_n_bins = max(n_bins_per_feat)

        # 1. Prepare raw weights (Sandwich Layout)
        w_np = np.zeros((n_features, max_n_bins), dtype=np.float32)
        b_np = np.zeros((n_features, max_n_bins), dtype=np.float32)
        sbm_np = np.array(n_bins_per_feat) == 1

        for i, edges in enumerate(bins):
            bin_width = np.diff(edges)
            w = 1.0 / bin_width
            b = -edges[:-1] / bin_width

            # Sandwich: Last Bin -> End
            w_np[i, -1] = w[-1]
            b_np[i, -1] = b[-1]

            # Sandwich: Other Bins -> Start
            if n_bins_per_feat[i] > 1:
                w_np[i, : n_bins_per_feat[i] - 1] = w[:-1]
                b_np[i, : n_bins_per_feat[i] - 1] = b[:-1]

        # Register buffers (non-trainable tensors)
        self.register_buffer("ple_weight", torch.from_numpy(w_np))
        self.register_buffer("ple_bias", torch.from_numpy(b_np))

        # Single bin mask
        self.has_single_bins = np.any(sbm_np)
        if self.has_single_bins:
            # Shape (1, N, 1) for broadcast
            mask = torch.from_numpy(sbm_np).reshape(1, n_features, 1)
            self.register_buffer("single_bin_mask", mask)
        else:
            self.single_bin_mask = None

        # 2. Linear Projection (NLinear)
        self.linear = NLinear(n_features, max_n_bins, d_embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Feat) -> (Batch, Feat, 1)
        # Linear transform: weight * x + bias
        # torch.addcmul is efficient for: output = bias + tensor1 * tensor2
        x_enc = torch.addcmul(self.ple_bias, self.ple_weight, x.unsqueeze(-1))

        # Activation (Clamp/Slice)
        # 1. First Bin: (-inf, 1.0] -> clamp_max
        p1 = x_enc[..., :1].clamp_max(1.0)

        # 2. Middle Bins: [0.0, 1.0] -> clamp
        p2 = x_enc[..., 1:-1].clamp(0.0, 1.0)

        # 3. Last Bin: [0.0, inf) or Raw
        p3_raw = x_enc[..., -1:]

        if self.single_bin_mask is not None:
            # For single bin features, keep raw, otherwise clamp min
            p3 = torch.where(self.single_bin_mask, p3_raw, p3_raw.clamp_min(0.0))
        else:
            p3 = p3_raw.clamp_min(0.0)

        x_ple = torch.cat([p1, p2, p3], dim=-1)  # (Batch, Feat, MaxBins)

        # Projection -> ReLU
        x_emb = self.linear(x_ple)
        return torch.relu(x_emb)


class MLP(nn.Module):
    def __init__(
        self,
        n_features_in: int,
        n_classes_out: int,
        bins: list[np.ndarray],
        d_embedding: int,
        hidden_layers: list[int],
    ):
        super().__init__()
        # Numerical Embeddings
        self.embeddings = PiecewiseLinearEmbeddings(bins, d_embedding)

        # Backbone MLP
        input_dim = n_features_in * d_embedding
        layers = []
        curr_dim = input_dim

        for h_dim in hidden_layers:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            curr_dim = h_dim

        layers.append(nn.Linear(curr_dim, n_classes_out))
        self.backbone = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(x)  # (Batch, Feat, Emb)
        x = x.flatten(1)  # (Batch, Feat * Emb)
        return self.backbone(x)


# =============================================================================
# Main
# =============================================================================


def train(config: dict, exp_path: Path) -> dict:
    # 0. Setup Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Load Data
    print(f"Loading {config['dataset']}...")
    ds = lib.Dataset.load(config["dataset"])
    ds = lib.preprocess(ds, num_method="standard")

    # 2. Compute Bins (using the lib function which returns numpy)
    print("Computing bins...")
    X_train_np = ds.data["x_num"]["train"]
    bins = lib.compute_quantile_bins(X_train_np, config["model"]["n_bins"])

    # 3. Model Setup
    model = MLP(
        n_features_in=ds.n_num_features,
        n_classes_out=1
        if ds.task.is_regression or ds.task.is_binclass
        else ds.task.compute_n_classes(),
        bins=bins,
        d_embedding=config["model"]["d_embedding"],
        hidden_layers=config["model"]["layers"],
    ).to(device)

    # Optional: Torch Compile
    # Note: On MPS, backend support varies. 'default' tries inductor.
    if config.get("compile", False):
        print("Compiling model...")
        model = torch.compile(model)

    # 4. Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])

    # 5. Data to Tensor/Device
    X_train = torch.from_numpy(ds.data["x_num"]["train"]).to(device)
    Y_train = torch.from_numpy(ds.data["y"]["train"]).to(device)

    X_val = torch.from_numpy(ds.data["x_num"]["val"]).to(device)
    Y_val = torch.from_numpy(ds.data["y"]["val"]).to(device)

    if ds.task.is_regression:
        Y_train, Y_val = Y_train.unsqueeze(1), Y_val.unsqueeze(1)

    batch_size = config["train"]["batch_size"]

    # 6. Training Functions
    def compute_loss(out, y):
        if ds.task.is_regression:
            return torch.sqrt(torch.nn.functional.mse_loss(out, y))  # RMSE
        elif ds.task.is_binclass:
            return torch.nn.functional.binary_cross_entropy_with_logits(out, y)
        else:
            return torch.nn.functional.cross_entropy(out, y)

    def train_step():
        opt.zero_grad()
        # Random sampling
        idx = torch.randint(0, X_train.shape[0], (batch_size,), device=device)
        pred = model(X_train[idx])
        loss = compute_loss(pred, Y_train[idx])
        loss.backward()
        opt.step()
        return loss

    @torch.no_grad()
    def val_step():
        model.eval()
        val_losses = []
        # simple slice loop
        for i in range(0, X_val.shape[0], batch_size):
            end = min(i + batch_size, X_val.shape[0])
            x_batch = X_val[i:end]
            y_batch = Y_val[i:end]

            pred = model(x_batch)
            loss = compute_loss(pred, y_batch)
            val_losses.append(loss.item())

        model.train()
        return np.mean(val_losses)

    # 7. Loop
    steps = config["train"]["steps"]
    print(f"Training for {steps} steps...")

    best_loss = float("inf")

    # Using tinygrad's trange for consistent visualization
    for i in (t := trange(steps)):
        loss = train_step()

        if i % 4000 == 0 or i == steps - 1:
            curr_val_loss = val_step()
            if curr_val_loss < best_loss:
                best_loss = curr_val_loss

            t.set_description(
                f"loss: {loss.item():.4f} | val_loss: {curr_val_loss:.4f}"
            )

    return {"metrics": {"val": {"loss": best_loss}}, "time": 0}


if __name__ == "__main__":
    # 1. Config (identical to Tinygrad version)
    default_config = {
        "seed": 42,
        "dataset": "diamond",
        "model": {"d_embedding": 16, "n_bins": 32, "layers": [256, 256]},
        "train": {"batch_size": 256, "steps": 10_000, "lr": 1e-4},
        "compile": True,  # Set to True to test torch.compile
    }

    # Setup experiment dir
    exp_dir = Path("exp/test_mlp_torch")
    lib.create_exp(exp_dir, default_config, force=True)

    # Run
    lib.run(train, exp_dir, force=True)
