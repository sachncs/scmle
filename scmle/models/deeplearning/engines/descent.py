"""Optimizer updates: the batch plan, the learning rate and one epoch."""

import math
from typing import Any

import numpy

from scmle.contracts import errors
from scmle.models.deeplearning.engines.network import Network
from scmle.models.deeplearning.engines.network import Plan
import scmle.models.deeplearning.training


class Descent(Network):
    """Cuts epochs into optimizer updates and runs one epoch of them."""

    def apportion(self, training_rows: int) -> Plan:
        """Works out how the epochs are cut into optimizer updates.

        Args:
            training_rows: Number of rows used for training.

        Returns:
            The batch size, accumulation count and step counts. L-BFGS uses
            one full batch and no accumulation.
        """
        second_order = self.optimizer == "lbfgs"
        batch_rows = (
            training_rows
            if second_order
            else scmle.models.deeplearning.training.batching(
                self.batch_size, training_rows
            )
        )
        accumulate = 1 if second_order else self.accumulation_steps
        batches_per_epoch = max(math.ceil(training_rows / batch_rows), 1)
        updates_per_epoch = max(math.ceil(batches_per_epoch / accumulate), 1)
        return Plan(
            batch_rows=batch_rows,
            second_order=second_order,
            accumulate=accumulate,
            updates_per_epoch=updates_per_epoch,
            total_steps=updates_per_epoch * self.epochs,
            warmup_steps=updates_per_epoch * self.warmup_epochs,
        )

    def prepare(
        self, features: Any, target: Any, sample_weight: Any
    ) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Converts the inputs to float32 matrix, encoded target and weights.

        Args:
            features: Numeric matrix.
            target: Labels or values.
            sample_weight: Optional per-row weights.

        Returns:
            The dense matrix, the encoded target and the row weights, which
            combine the estimator's own weights with `sample_weight`.
        """
        matrix = self.dense(features)
        encoded = self.vectorize(target)
        weights = self.weighting(encoded)
        if sample_weight is not None:
            weights = weights * numpy.asarray(
                sample_weight, dtype=numpy.float32
            )
        return matrix, encoded, weights

    def adjust(self, optimizer: Any, step: int, plan: Plan) -> None:
        """Applies the scheduled learning rate to every parameter group.

        Args:
            optimizer: The torch optimizer.
            step: Optimizer updates taken so far.
            plan: The run's update plan.
        """
        rate = scmle.models.deeplearning.training.anneal(
            step,
            plan.total_steps,
            self.learning_rate,
            plan.warmup_steps,
            self.schedule,
        )
        for group in optimizer.param_groups:
            group["lr"] = rate

    def train_epoch(
        self,
        torch: Any,
        network: Any,
        optimizer: Any,
        tensors: dict[str, Any],
        monitor: scmle.models.deeplearning.training.Monitor,
        plan: Plan,
        batches: tuple[Any, int],
    ) -> tuple[float, bool, int]:
        """Runs one pass over the shuffled training rows.

        Args:
            torch: The imported torch module.
            network: The network being trained.
            optimizer: The torch optimizer.
            tensors: Device tensors `x`, `y` and `w` (inputs, targets, weights).
            monitor: Records losses and gradient norms for the report.
            plan: The run's update plan.
            batches: The shuffled training row indices on the device, and the
                number of optimizer updates taken before this epoch.

        Returns:
            The summed update losses of the epoch, whether training is still
            healthy (False when the monitor saw a divergence) and the number
            of updates taken so far.
        """
        shuffled, step = batches
        epoch_loss, alive = 0.0, True
        pending, running = 0, 0.0
        for start in range(0, len(shuffled), plan.batch_rows):
            batch = shuffled[start : start + plan.batch_rows]
            if plan.second_order:
                value, norm = self.lbfgs_step(
                    torch, network, optimizer, tensors, batch
                )
                alive = monitor.step(value, norm, None)
                if not alive:
                    break
                epoch_loss += value
                step += 1
                continue
            if pending == 0:
                self.adjust(optimizer, step, plan)
                optimizer.zero_grad()
            loss = self.weighted_loss(torch, network, tensors, batch)
            (loss / plan.accumulate).backward()
            running += float(loss.detach())
            pending += 1
            last = start + plan.batch_rows >= len(shuffled)
            if pending < plan.accumulate and not last:
                continue
            norm = float(
                torch.nn.utils.clip_grad_norm_(
                    network.parameters(), self.max_grad_norm or math.inf
                )
            )
            value = running / pending
            alive = monitor.step(value, norm, self.max_grad_norm)
            if not alive:
                break
            optimizer.step()
            epoch_loss += value
            step += 1
            pending, running = 0, 0.0
        return epoch_loss, alive, step

    OPTIMIZERS = ("adam", "adamw", "sgd", "rmsprop", "lbfgs")
    CLASSIFIER_LOSSES = ("default", "focal")
    REGRESSOR_LOSSES = ("default", "huber")
    losses: tuple[str, ...] = ("default",)

    def vet(self) -> None:
        """Rejects unknown optimizers and losses before any work is done.

        Raises:
            ConfigError: For an unknown optimizer, a loss the task does not
                have, or a non-positive accumulation count.
        """
        if self.optimizer not in self.OPTIMIZERS:
            raise errors.ConfigError(
                f"unknown optimizer '{self.optimizer}'; use one of "
                f"{list(self.OPTIMIZERS)}"
            )
        if self.loss not in self.losses:
            raise errors.ConfigError(
                f"unknown loss '{self.loss}' for this task; use one of "
                f"{list(self.losses)}"
            )
        if self.accumulation_steps < 1:
            raise errors.ConfigError("accumulation_steps must be at least 1")

    def instantiate(self, torch: Any, network: Any) -> Any:
        """Creates the configured optimizer for the network's parameters.

        Args:
            torch: The imported torch module.
            network: The network whose parameters are optimised.

        Returns:
            The torch optimizer; Adam for any unrecognised name.
        """
        parameters = network.parameters()
        rate, decay = self.learning_rate, self.weight_decay
        if self.optimizer == "adamw":
            return torch.optim.AdamW(parameters, lr=rate, weight_decay=decay)
        if self.optimizer == "sgd":
            return torch.optim.SGD(
                parameters,
                lr=rate,
                momentum=self.momentum,
                nesterov=self.momentum > 0,
                weight_decay=decay,
            )
        if self.optimizer == "rmsprop":
            return torch.optim.RMSprop(parameters, lr=rate, weight_decay=decay)
        if self.optimizer == "lbfgs":
            return torch.optim.LBFGS(
                parameters, lr=1.0, max_iter=20, line_search_fn="strong_wolfe"
            )
        return torch.optim.Adam(parameters, lr=rate, weight_decay=decay)

    def lbfgs_step(
        self,
        torch: Any,
        network: Any,
        optimizer: Any,
        tensors: dict[str, Any],
        batch: Any,
    ) -> tuple[float, float]:
        """Runs one L-BFGS step; returns the starting loss and gradient norm.

        Args:
            torch: The imported torch module.
            network: The network being trained.
            optimizer: The L-BFGS optimizer.
            tensors: Device tensors `x`, `y` and `w` (inputs, targets, weights).
            batch: Indices of the rows to use (all training rows).

        Returns:
            The loss at the start of the step (the first closure evaluation)
            and the norm of the gradients left after it, without clipping.
        """

        def closure() -> Any:
            optimizer.zero_grad()
            loss = self.weighted_loss(torch, network, tensors, batch)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        norm = float(
            torch.nn.utils.clip_grad_norm_(network.parameters(), math.inf)
        )
        return float(loss.detach()), norm
