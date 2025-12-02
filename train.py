"""
Training script for HOPE model.

Implements training loop with:
- Memory state management across batches
- Continuum Memory System online learning
- Gradient accumulation
- Logging and checkpointing

Reference: Nested Learning paper, Section 4
"""

import os
import math
import time
import argparse
from typing import Optional, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.config import (
    HopeConfig,
    HopeSmallConfig,
    HopeBaseConfig,
    HopeLargeConfig,
)
from src.model import Hope, HopeForCausalLM, createHopeModel
from src.optimizers import (
    DeepMomentumGD,
    DeltaRuleOptimizer,
    AdamWithDeltaRule,
)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    model_size: str = "small"
    vocab_size: int = 32000

    # Training
    batch_size: int = 8
    seq_len: int = 512
    num_epochs: int = 10
    max_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1

    # Optimizer
    optimizer: str = "adamw"  # adamw, adam_delta, sgd_delta, muon
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    max_grad_norm: float = 1.0

    # Learning rate schedule
    warmup_steps: int = 1000
    lr_scheduler: str = "cosine"  # cosine, linear, constant

    # Memory
    reset_memory_every: int = 0  # 0 = never reset
    enable_cms_online_learning: bool = False

    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000

    # Paths
    output_dir: str = "./output"
    checkpoint_path: Optional[str] = None

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"  # float32, float16, bfloat16


class RandomTextDataset(Dataset):
    """
    Random token dataset for testing.
    Replace with actual dataset for real training.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        num_samples: int = 10000,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random tokens
        tokens = torch.randint(0, self.vocab_size, (self.seq_len + 1,))
        input_ids = tokens[:-1]
        labels = tokens[1:]
        return {"input_ids": input_ids, "labels": labels}


def getOptimizer(model: nn.Module, config: TrainingConfig):
    """Create optimizer based on config."""
    params = model.parameters()

    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adam_delta":
        return AdamWithDeltaRule(
            params,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            delta_beta=0.01,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd_delta":
        return DeltaRuleOptimizer(
            params,
            lr=config.learning_rate,
            momentum=config.beta1,
            beta=0.1,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "deep_momentum":
        return DeepMomentumGD(
            params,
            lr=config.learning_rate,
            momentum=config.beta1,
            weight_decay=config.weight_decay,
            memory_depth=2,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def getLrScheduler(optimizer, config: TrainingConfig, num_training_steps: int):
    """Create learning rate scheduler."""
    warmup_steps = config.warmup_steps

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        if config.lr_scheduler == "constant":
            return 1.0
        elif config.lr_scheduler == "linear":
            progress = float(current_step - warmup_steps) / float(
                max(1, num_training_steps - warmup_steps)
            )
            return max(0.0, 1.0 - progress)
        elif config.lr_scheduler == "cosine":
            progress = float(current_step - warmup_steps) / float(
                max(1, num_training_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        else:
            return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def trainStep(
    model: HopeForCausalLM,
    batch: Dict[str, torch.Tensor],
    memory_states: Optional[list],
    config: TrainingConfig,
) -> tuple:
    """
    Single training step.

    Returns:
        loss: Scalar loss value
        memory_states: Updated memory states
    """
    input_ids = batch["input_ids"].to(config.device)
    labels = batch["labels"].to(config.device)

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        labels=labels,
        memory_states=memory_states,
        return_memory=True,
    )

    loss = outputs["loss"]
    new_memory_states = outputs.get("memory_states", None)

    return loss, new_memory_states


def evaluate(
    model: HopeForCausalLM,
    eval_dataloader: DataLoader,
    config: TrainingConfig,
    max_batches: int = 100,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    memory_states = None

    with torch.no_grad():
        for batch in eval_dataloader:
            if num_batches >= max_batches:
                break

            input_ids = batch["input_ids"].to(config.device)
            labels = batch["labels"].to(config.device)

            outputs = model(
                input_ids=input_ids,
                labels=labels,
                memory_states=memory_states,
                return_memory=True,
            )

            total_loss += outputs["loss"].item()
            memory_states = outputs.get("memory_states", None)
            num_batches += 1

    model.train()

    avg_loss = total_loss / max(1, num_batches)
    perplexity = math.exp(min(avg_loss, 20))  # Clip to avoid overflow

    return {
        "eval_loss": avg_loss,
        "eval_perplexity": perplexity,
    }


def saveCheckpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    config: TrainingConfig,
    metrics: Dict[str, float],
):
    """Save training checkpoint."""
    os.makedirs(config.output_dir, exist_ok=True)

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "config": config,
        "metrics": metrics,
    }

    path = os.path.join(config.output_dir, f"checkpoint_{step}.pt")
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def loadCheckpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> int:
    """Load training checkpoint. Returns the step number."""
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("step", 0)


def train(config: TrainingConfig):
    """Main training function."""
    print(f"Training config: {config}")
    print(f"Device: {config.device}")

    # Set dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(config.dtype, torch.float32)

    # Create model
    print(f"Creating {config.model_size} model...")
    model = createHopeModel(
        model_size=config.model_size,
        vocab_size=config.vocab_size,
    )
    model = HopeForCausalLM(model.config)
    model = model.to(config.device)

    if dtype == torch.float16:
        model = model.half()
    elif dtype == torch.bfloat16:
        model = model.to(torch.bfloat16)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create datasets
    print("Creating datasets...")
    train_dataset = RandomTextDataset(
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        num_samples=100000,
    )
    eval_dataset = RandomTextDataset(
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        num_samples=1000,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Calculate training steps
    steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
    num_training_steps = steps_per_epoch * config.num_epochs
    if config.max_steps:
        num_training_steps = min(num_training_steps, config.max_steps)

    print(f"Training steps: {num_training_steps}")

    # Create optimizer and scheduler
    optimizer = getOptimizer(model, config)
    scheduler = getLrScheduler(optimizer, config, num_training_steps)

    # Load checkpoint if specified
    start_step = 0
    if config.checkpoint_path:
        print(f"Loading checkpoint from {config.checkpoint_path}")
        start_step = loadCheckpoint(config.checkpoint_path, model, optimizer, scheduler)
        print(f"Resuming from step {start_step}")

    # Training loop
    model.train()
    memory_states = None
    global_step = start_step
    accumulated_loss = 0.0

    print("Starting training...")
    start_time = time.time()

    for epoch in range(config.num_epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            # Check if we've reached max steps
            if config.max_steps and global_step >= config.max_steps:
                break

            # Reset memory if configured
            if (
                config.reset_memory_every > 0
                and global_step % config.reset_memory_every == 0
            ):
                memory_states = None

            # Forward pass
            loss, memory_states = trainStep(model, batch, memory_states, config)
            loss = loss / config.gradient_accumulation_steps
            accumulated_loss += loss.item()

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.max_grad_norm
                    )

                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Logging
                if global_step % config.log_interval == 0:
                    elapsed = time.time() - start_time
                    lr = scheduler.get_last_lr()[0]
                    avg_loss = accumulated_loss / config.log_interval
                    ppl = math.exp(min(avg_loss, 20))

                    print(
                        f"Step {global_step} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"PPL: {ppl:.2f} | "
                        f"LR: {lr:.2e} | "
                        f"Time: {elapsed:.1f}s"
                    )
                    accumulated_loss = 0.0

                # Evaluation
                if global_step % config.eval_interval == 0:
                    eval_metrics = evaluate(model, eval_dataloader, config)
                    print(
                        f"Eval @ Step {global_step} | "
                        f"Loss: {eval_metrics['eval_loss']:.4f} | "
                        f"PPL: {eval_metrics['eval_perplexity']:.2f}"
                    )

                # Save checkpoint
                if global_step % config.save_interval == 0:
                    metrics = {"train_loss": accumulated_loss}
                    saveCheckpoint(
                        model, optimizer, scheduler, global_step, config, metrics
                    )

        # End of epoch
        print(f"Epoch {epoch + 1} completed")

        if config.max_steps and global_step >= config.max_steps:
            break

    # Final save
    print("Training completed!")
    metrics = {"train_loss": accumulated_loss}
    saveCheckpoint(model, optimizer, scheduler, global_step, config, metrics)

    return model


def main():
    parser = argparse.ArgumentParser(description="Train HOPE model")

    # Model
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        choices=["small", "base", "large", "xl"],
    )
    parser.add_argument("--vocab_size", type=int, default=32000)

    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # Optimizer
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adam_delta", "sgd_delta", "deep_momentum"],
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Schedule
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "constant"],
    )

    # Memory
    parser.add_argument("--reset_memory_every", type=int, default=0)

    # Logging
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=5000)

    # Paths
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--checkpoint_path", type=str, default=None)

    # Device
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )

    args = parser.parse_args()

    # Create config
    config = TrainingConfig(
        model_size=args.model_size,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        lr_scheduler=args.lr_scheduler,
        reset_memory_every=args.reset_memory_every,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=args.dtype,
    )

    train(config)


if __name__ == "__main__":
    main()
