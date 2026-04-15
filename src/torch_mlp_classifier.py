"""Sklearn-like binary classifier backed by PyTorch."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split


class TorchMLPClassifier:
    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (64, 32),
        alpha: float = 1e-4,
        max_iter: int = 50,
        batch_size: int = 2048,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        optimizer: str = "sgd",
        activation: str = "relu",
        early_stopping: bool = True,
        n_iter_no_change: int = 10,
        verbose: bool = False,
        random_state: int | None = 42,
        prefer_gpu: bool = True,
        require_gpu: bool = False,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.optimizer = optimizer
        self.activation = activation
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.verbose = verbose
        self.random_state = random_state
        self.prefer_gpu = prefer_gpu
        self.require_gpu = require_gpu

        self.classes_ = np.array([0, 1])
        self.input_dim_ = None
        self._state_dict = None
        self._device_name = "cpu"

    @staticmethod
    def _build_model(nn, input_dim: int, hidden_layer_sizes: Tuple[int, ...], activation: str):
        layers = []
        in_dim = input_dim
        activation_layers = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "identity": nn.Identity,
            "none": nn.Identity,
        }
        act_cls = activation_layers.get(str(activation).lower(), nn.ReLU)
        for h in hidden_layer_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_cls())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    def _pick_device(self, torch):
        if self.prefer_gpu:
            try:
                import torch_directml  # type: ignore

                dml_device = torch_directml.device()
                self._device_name = "directml"
                return dml_device
            except Exception:
                pass

            if torch.cuda.is_available():
                self._device_name = "cuda"
                return torch.device("cuda")

        if self.require_gpu:
            raise RuntimeError(
                "GPU is required but neither DirectML nor CUDA backend is available."
            )

        self._device_name = "cpu"
        return torch.device("cpu")

    def get_params(self, deep=True):
        return {
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "optimizer": self.optimizer,
            "activation": self.activation,
            "early_stopping": self.early_stopping,
            "n_iter_no_change": self.n_iter_no_change,
            "verbose": self.verbose,
            "random_state": self.random_state,
            "prefer_gpu": self.prefer_gpu,
            "require_gpu": self.require_gpu,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self.input_dim_ = X_np.shape[1]

        if self.early_stopping:
            X_train, X_val, y_train, y_val = train_test_split(
                X_np,
                y_np,
                test_size=0.1,
                random_state=self.random_state if self.random_state is not None else 42,
                stratify=y_np.ravel().astype(int),
            )
        else:
            X_train, y_train = X_np, y_np
            X_val, y_val = None, None

        device = self._pick_device(torch)
        if self.verbose:
            print(f"  [MLP] device={self._device_name}  samples={len(X_train):,}")

        model = self._build_model(nn, self.input_dim_, self.hidden_layer_sizes, self.activation).to(device)
        if str(self.optimizer).lower() == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.alpha,
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                nesterov=True,
                weight_decay=self.alpha,
            )
        criterion = nn.MSELoss()

        train_ds = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        if X_val is not None:
            X_val_t = torch.from_numpy(X_val).to(device)
            y_val_t = torch.from_numpy(y_val).to(device)

        best_state = None
        best_val = float("inf")
        no_improve = 0

        for epoch in range(self.max_iter):
            model.train()
            total_loss = 0.0
            n_batches = 0

            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                probs = torch.sigmoid(model(xb))
                loss = criterion(probs, yb)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                n_batches += 1

            avg_train_loss = total_loss / max(n_batches, 1)

            if X_val is None:
                if self.verbose:
                    print(f"Iteration {epoch + 1}, loss = {avg_train_loss:.8f}")
                continue

            model.eval()
            with torch.no_grad():
                val_probs = torch.sigmoid(model(X_val_t))
                val_loss = float(criterion(val_probs, y_val_t).item())

            preds = (val_probs >= 0.5).float()
            val_acc = float((preds == y_val_t).float().mean().item())

            if self.verbose:
                print(f"Iteration {epoch + 1}, loss = {avg_train_loss:.8f}")
                print(f"Validation score: {val_acc:.6f}")

            if val_loss + 1e-6 < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if self.early_stopping and no_improve >= self.n_iter_no_change:
                if self.verbose:
                    print(
                        "Validation loss did not improve for "
                        f"{self.n_iter_no_change} consecutive epochs. Stopping."
                    )
                break

        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        self._state_dict = best_state
        return self

    def _predict_scores(self, X):
        import torch
        import torch.nn as nn

        if self._state_dict is None or self.input_dim_ is None:
            raise RuntimeError("Model is not fitted.")

        device = self._pick_device(torch)
        model = self._build_model(nn, self.input_dim_, self.hidden_layer_sizes, self.activation).to(device)
        model.load_state_dict(self._state_dict)
        model.eval()

        X_np = np.asarray(X, dtype=np.float32)
        X_t = torch.from_numpy(X_np).to(device)

        with torch.no_grad():
            logits = model(X_t)
            probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
        return probs

    def predict_proba(self, X):
        probs = self._predict_scores(X)
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X):
        probs = self._predict_scores(X)
        return (probs >= 0.5).astype(int)
