import os
import argparse
from contextlib import nullcontext
from dataclasses import dataclass

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch.profiler import profile, schedule, tensorboard_trace_handler
from torchvision.transforms import RandomHorizontalFlip

from tqdm import tqdm

from src import TinyViT, make_dataloaders


# ---------------------------
# Detect GPU-CPU synchronizations
# ---------------------------

@contextmanager
def cuda_sync_detect(mode="error"):
    """
    Context manager to temporarily set torch CUDA sync debug mode.

    Args:
        mode (str): one of {"default", "warn", "error"}.
    """
    # Save the old mode
    old_mode = torch.cuda.get_sync_debug_mode()

    try:
        torch.cuda.set_sync_debug_mode(mode)
        yield
    finally:
        # Restore the old mode
        torch.cuda.set_sync_debug_mode(old_mode)

# ---------------------------
# Train loop with optional Profiler & sync
# ---------------------------

@dataclass
class TrainConfig:
    device: str = "cuda"
    epochs: int = 1
    lr: float = 1e-3
    compile: bool = False
    debug_gpu_cpu_sync: bool = False
    profile_enable: bool = True
    prof_wait: int = 1
    prof_warmup: int = 1
    prof_active: int = 3
    prof_repeat: int = 1
    prof_dir: str = "./tb_logs"
    attn_backend: str = "sdpa"
    d_model: int = 256
    depth: int = 4
    n_heads: int = 8
    batch_size: int = 256
    workers: int = 2
    pin_memory: bool = True
    data_dir: str = "./data"
    limit_steps: int| None = 20,  # short epoch for benchmarking,
    no_tqdm: bool = False,
    use_faster_dataset: bool = False,


def train(cfg: TrainConfig):
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    device = torch.device(cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")

    model = TinyViT(
        num_classes=10,
        img_size=32,
        patch=4,
        d_model=cfg.d_model,
        depth=cfg.depth,
        n_heads=cfg.n_heads,
        mlp_ratio=4.0,
        backend=cfg.attn_backend,
    ).to(device)

    if cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model, dynamic=True)

    dl = make_dataloaders(cfg.data_dir, cfg.batch_size, cfg.workers, cfg.pin_memory, cfg.use_faster_dataset)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # Profiler schedule
    if cfg.profile_enable:
        os.makedirs(cfg.prof_dir, exist_ok=True)
        sched = schedule(wait=cfg.prof_wait, warmup=cfg.prof_warmup, active=cfg.prof_active, repeat=cfg.prof_repeat)
        prof_ctx = profile(
            schedule=sched,
            on_trace_ready=tensorboard_trace_handler(cfg.prof_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        print(f"[Profiler] writing traces to: {cfg.prof_dir}")
    else:
        prof_ctx = nullcontext()

    # Training
    model.train()
    losses = []


    cuda_sync_detect_ctx = cuda_sync_detect('error') if cfg.debug_gpu_cpu_sync else nullcontext()

    n_steps = len(dl) * cfg.epochs if cfg.limit_steps is None else cfg.limit_steps

    progress_bar = tqdm(total=n_steps, desc='Training...') if not cfg.no_tqdm else nullcontext()


    transforms = RandomHorizontalFlip()
    with prof_ctx as prof, progress_bar:
        for epoch in range(cfg.epochs):
            exited = False
            for step, (x, y) in enumerate(dl):
                x = x.to(device, non_blocking=True)

                if cfg.use_faster_dataset:  # For faster dataset, we need to perform ToTensor() transformation on GPU by ourselves
                    x = (x / 255.0).permute(0, 3, 1, 2) # to tensor BxHxWxC --> BxCxHxW
                    x = transforms(x)  # random horizontal flip

                y = y.to(device, non_blocking=True)

                opt.zero_grad()
                with cuda_sync_detect_ctx:
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                    loss.backward()
                    opt.step()

                    # losses.append(loss.item()) # causes gpu-cpu sync
                    losses.append(loss.detach()) # causes gpu-cpu sync

                if cfg.profile_enable:
                    prof.step()

                if step % 10 == 0:
                    if not cfg.no_tqdm:
                        progress_bar.set_postfix(
                            {"epoch": epoch, "step": step, "loss": round(loss.item(), 4)}
                        )

                if not cfg.no_tqdm:
                    progress_bar.update()

                if cfg.limit_steps is not None and step > cfg.limit_steps:
                    exited = True
                    break

            if exited:
                break

    print("Done.")


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark Transformer attention backends with optional torch.profiler.")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--compile", action="store_true", help="Enable torch.compile on the model.")
    p.add_argument("--debug_gpu_cpu_sync", action="store_true", help="Throw an error when gpu-cpu sync is occured") # NOTE use this to track GPU-CPU synchronization points
    p.add_argument("--no-tqdm", action="store_true", help="Disable tqdm")

    # profiler knobs
    p.add_argument("--no-profiler", action="store_true", help="Disable torch.profiler.")
    p.add_argument("--prof-wait", type=int, default=2)
    p.add_argument("--prof-warmup", type=int, default=2)
    p.add_argument("--prof-active", type=int, default=3)
    p.add_argument("--prof-repeat", type=int, default=1)
    p.add_argument("--prof-dir", type=str, default="./tb_logs")

    # model
    p.add_argument("--attn", type=str, default="sdpa", choices=["sdpa", "looped", "batched"])

    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)

    # data
    p.add_argument("--use-faster-dataset", action="store_true", help="Switch to a bit more optimized dataset logic")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true", help="Enable pin_memory in DataLoader.")
    p.add_argument("--data-dir", type=str, default="./data")

    p.add_argument("--limit-steps", type=int, help="Batches per epoch to run (short benchmark). Default is None, e.g. full dataloader is traversed")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        compile=args.compile,
        debug_gpu_cpu_sync=args.debug_gpu_cpu_sync,
        profile_enable=not args.no_profiler,
        prof_wait=args.prof_wait,
        prof_warmup=args.prof_warmup,
        prof_active=args.prof_active,
        prof_repeat=args.prof_repeat,
        prof_dir=args.prof_dir,
        attn_backend=args.attn,
        d_model=args.d_model,
        depth=args.depth,
        n_heads=args.heads,
        batch_size=args.batch_size,
        workers=args.workers,
        pin_memory=args.pin_memory,
        data_dir=args.data_dir,
        limit_steps=args.limit_steps,
        no_tqdm=args.no_tqdm,
        use_faster_dataset=args.use_faster_dataset,
    )
    # make TF32 controllable from env; default off for cleaner comparisons. Set CUBLAS determinism
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    train(cfg)


if __name__ == "__main__":
    main()
