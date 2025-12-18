#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import lib
from tinygrad import Tensor, TinyJit, nn, GlobalCounters
from tqdm.auto import trange

# =============================================================================
# Modules
# =============================================================================


class NLinear:
    """
    N independent linear layers.
    Input: (Batch, N, In_Features)
    Output: (Batch, N, Out_Features)
    """

    def __init__(self, n: int, in_features: int, out_features: int):
        self.weight = Tensor.glorot_uniform(n, in_features, out_features)
        self.bias = Tensor.zeros(n, out_features)

    def __call__(self, x: Tensor) -> Tensor:
        # x: (Batch, N, In) -> (N, Batch, In)
        x = x.transpose(0, 1)
        # (N, Batch, In) @ (N, In, Out) -> (N, Batch, Out)
        x = x.matmul(self.weight)
        # (N, Batch, Out) -> (Batch, N, Out)
        x = x.transpose(0, 1)
        return x.add(self.bias)


class PiecewiseLinearEmbeddings:
    """
    Piecewise Linear Embeddings with 'Sandwich' layout for vectorization.
    """

    def __init__(self, bins: list[np.ndarray], d_embedding: int):
        n_features = len(bins)
        n_bins_per_feat = [len(x) - 1 for x in bins]
        max_n_bins = max(n_bins_per_feat)

        # 1. Prepare raw weights for PLE encoding (Sandwich Layout)
        # Layout: [Bin 0, Bin 1, ..., Bin K, PADDING, Last Bin]
        w_np = np.zeros((n_features, max_n_bins), dtype=np.float32)
        b_np = np.zeros((n_features, max_n_bins), dtype=np.float32)
        sbm_np = np.array(n_bins_per_feat) == 1

        for i, edges in enumerate(bins):
            bin_width = np.diff(edges)
            w = 1.0 / bin_width
            b = -edges[:-1] / bin_width

            # Place Last Bin at the very end (-1)
            w_np[i, -1] = w[-1]
            b_np[i, -1] = b[-1]

            # Place Leading Bins at the start
            if n_bins_per_feat[i] > 1:
                w_np[i, : n_bins_per_feat[i] - 1] = w[:-1]
                b_np[i, : n_bins_per_feat[i] - 1] = b[:-1]

        # Register as Tensors
        self.ple_weight = Tensor(w_np)
        self.ple_bias = Tensor(b_np)

        # Single bin mask logic
        self.has_single_bins = np.any(sbm_np)
        if self.has_single_bins:
            self.single_bin_mask = Tensor(sbm_np).reshape(1, n_features, 1)

        # 2. Linear Projection (NLinear)
        # Projects from max_n_bins to d_embedding for each feature independently
        self.linear = NLinear(n_features, max_n_bins, d_embedding)

    def __call__(self, x: Tensor) -> Tensor:
        # --- 1. Encoding (0 to 1 expansion) ---
        # x: (Batch, Feat) -> (Batch, Feat, 1)
        x_enc = x.reshape(*x.shape, 1)
        x_enc = x_enc.mul(self.ple_weight).add(self.ple_bias)

        # Activation (Clamp/Clip)
        # Slice: First Bin
        p1 = x_enc[:, :, :1].clip(None, 1.0)
        # Slice: Middle Bins (strictly 0..1)
        p2 = x_enc[:, :, 1:-1].clip(0.0, 1.0)
        # Slice: Last Bin (Logic varies if it's the only bin or not)
        p3_raw = x_enc[:, :, -1:]

        if self.has_single_bins:
            # If single bin: behave like MinMax (no clamp min), else clamp min 0
            p3 = self.single_bin_mask.where(p3_raw, p3_raw.clip(0.0, None))
        else:
            p3 = p3_raw.clip(0.0, None)

        x_ple = p1.cat(p2, p3, dim=-1)  # (Batch, Feat, MaxBins)

        # --- 2. Embedding (Projection) ---
        x_emb = self.linear(x_ple)  # (Batch, Feat, d_embedding)
        return x_emb.relu()


class MLP:
    def __init__(
        self,
        n_features_in: int,
        n_classes_out: int,
        bins: list[np.ndarray],
        d_embedding: int,
        hidden_layers: list[int],
    ):
        # Numerical Embeddings
        self.embeddings = PiecewiseLinearEmbeddings(bins, d_embedding)

        # Backbone MLP
        # Input dim = n_features * d_embedding
        dims = [n_features_in * d_embedding] + hidden_layers + [n_classes_out]

        self.layers = []
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                self.layers.append(Tensor.relu)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.embeddings(x)  # (Batch, Feat, Emb)
        x = x.flatten(1)  # (Batch, Feat * Emb)
        return x.sequential(self.layers)


# =============================================================================
# Main
# =============================================================================


def train(config: dict, exp_path: Path) -> dict:
    seed = config.get("seed", 42)
    Tensor.manual_seed(seed)
    np.random.seed(seed)

    # 1. Load Data
    print(f"Loading {config['dataset']}...")
    ds = lib.Dataset.load(config["dataset"])
    ds = lib.preprocess(ds, num_method="standard")  # Standardize helps quantization

    # 2. Compute Bins (Quantile)
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
    )

    # 4. Optimizer
    # Note: Using Adam is standard for tabular. Muon is great but specific.
    opt = nn.optim.Adam(nn.state.get_parameters(model), lr=config["train"]["lr"])

    # 5. Data Tensors
    # Move entire dataset to GPU/Tensor for speed if it fits, otherwise batch loader needed
    # For tabular, usually fits in VRAM.
    X_train = Tensor(ds.data["x_num"]["train"])
    Y_train = Tensor(ds.data["y"]["train"])

    X_val = Tensor(ds.data["x_num"]["val"])
    Y_val = Tensor(ds.data["y"]["val"])

    if ds.task.is_regression:
        Y_train, Y_val = Y_train.unsqueeze(1), Y_val.unsqueeze(1)

    batch_size = config["train"]["batch_size"]

    # 6. Training Functions
    def compute_loss(out, y):
        if ds.task.is_regression:
            return (out - y).square().mean().sqrt()  # RMSE
        elif ds.task.is_binclass:
            return out.binary_crossentropy_logits(y.unsqueeze(1))
        else:
            return out.sparse_categorical_crossentropy(y)

    # 1. Change train_step to accept samples as an argument
    @TinyJit
    @Tensor.train()
    def train_step() -> Tensor:
        opt.zero_grad()
        # samples is now an input, not created internally
        samples = Tensor.randint(batch_size, high=X_train.shape[0])
        pred = model(X_train[samples])
        loss = compute_loss(pred, Y_train[samples])
        loss.backward()
        opt.step()
        return loss

    @Tensor.train(False)
    def val_step() -> np.float64:
        val_losses = []
        # simple slice loop
        for i in range(0, X_val.shape[0], batch_size):
            # Clip the end index
            end = min(i + batch_size, X_val.shape[0])
            x_batch = X_val[i:end]
            y_batch = Y_val[i:end]

            pred = model(x_batch)
            # We need to weigh the loss by batch size if the last batch is smaller
            # but for simple monitoring, mean is usually fine.
            loss = compute_loss(pred, y_batch)
            val_losses.append(loss.item())

        return np.mean(val_losses)

    # 7. Loop
    print(f"Training for {config['train']['steps']} steps...")

    for _ in trange(config["train"]["steps"]):
        GlobalCounters.reset()
        loss = train_step()
        loss.item()  # sync

    return {
        "metrics": {"val": {"loss": val_step()}},
        "time": 0,  # Placeholder
    }


if __name__ == "__main__":
    # 1. Create a dummy config if running directly
    # In a real scenario, you might pass a config.toml path via args
    default_config = {
        "seed": 42,
        "dataset": "california",  # Ensure you have this dataset in ~/new-data/california
        "model": {"d_embedding": 16, "n_bins": 32, "layers": [256, 256]},
        "train": {"batch_size": 256, "steps": 10_000, "lr": 1e-4},
    }

    # Setup experiment dir
    exp_dir = Path("exp/test_mlp")
    lib.create_exp(exp_dir, default_config, force=True)

    # Run
    lib.run(train, exp_dir, force=True)
