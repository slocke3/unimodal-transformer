import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TrainerConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 50
    patience: int = 10
    grad_clip: float = 1.0
    log_every: int = 100
    eval_every: int = 1
    save_dir: str = "outputs/checkpoints"
    device: str = "auto"

    def resolve_device(self):
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)


class Trainer:
    def __init__(self, model, train_loader, val_loader,
                 config=None, run_name="run"):
        self.config = config or TrainerConfig()
        self.device = self.config.resolve_device()
        self.run_name = run_name

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.max_epochs
        )
        self.criterion = nn.CrossEntropyLoss()

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.best_epoch = 0
        os.makedirs(self.config.save_dir, exist_ok=True)

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss, n_batches = 0.0, 0
        t0 = time.time()

        for batch_idx, (context, target, _) in enumerate(self.train_loader):
            context = context.to(self.device)
            target  = target.to(self.device)

            self.optimizer.zero_grad()
            loss = self.criterion(self.model(context), target)
            loss.backward()

            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % self.config.log_every == 0:
                print(f"  Epoch {epoch} | batch {batch_idx+1}/{len(self.train_loader)} "
                      f"| loss {loss.item():.4f} | {time.time()-t0:.1f}s")

        return total_loss / n_batches

    @torch.no_grad()
    def _eval(self, loader):
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        for context, target, _ in loader:
            context = context.to(self.device)
            target  = target.to(self.device)
            total_loss += self.criterion(self.model(context), target).item()
            n_batches += 1
        return total_loss / n_batches

    def train(self):
        print(f"Device: {self.device} | Parameters: {self.model.count_parameters():,}")
        print(f"Train batches: {len(self.train_loader)} | Val batches: {len(self.val_loader)}")
        print("-" * 60)

        for epoch in range(1, self.config.max_epochs + 1):
            t_epoch = time.time()
            train_loss = self._train_epoch(epoch)
            self.train_losses.append(train_loss)

            if epoch % self.config.eval_every == 0:
                val_loss = self._eval(self.val_loader)
                self.val_losses.append(val_loss)
                improved = val_loss < self.best_val_loss
                marker = " *" if improved else ""

                print(f"Epoch {epoch:3d}/{self.config.max_epochs} | "
                      f"train {train_loss:.4f} | val {val_loss:.4f} | "
                      f"{time.time()-t_epoch:.1f}s{marker}")

                if improved:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.epochs_without_improvement = 0
                    self._save_checkpoint("best")
                else:
                    self.epochs_without_improvement += 1

                if self.epochs_without_improvement >= self.config.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            self.scheduler.step()

        self._save_checkpoint("final")
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
        }

    def _save_checkpoint(self, tag):
        path = os.path.join(self.config.save_dir, f"{self.run_name}_{tag}.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.best_epoch,
            "val_loss": self.best_val_loss,
        }, path)

    def load_best(self):
        path = os.path.join(self.config.save_dir, f"{self.run_name}_best.pt")
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best checkpoint from epoch {ckpt['epoch']} "
              f"(val loss {ckpt['val_loss']:.4f})")