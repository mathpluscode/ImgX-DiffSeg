"""Module for optimization."""
import jax
import optax
from absl import logging
from omegaconf import DictConfig


def get_lr_schedule(config: DictConfig) -> optax.Schedule:
    """Get learning rate scheduler.

    Args:
        config: entire configuration.

    Returns:
        Scheduler
    """
    return optax.warmup_cosine_decay_schedule(**config.optimizer.lr_schedule)


def get_every_k_schedule(config: DictConfig) -> int:
    """Get k for gradient accumulations.

    Args:
        config: entire configuration.

    Returns:
        k, where gradients are accumulated every k step.
    """
    num_devices_per_replica = config.data.trainer.num_devices_per_replica
    batch_size_per_replica = config.data.trainer.batch_size_per_replica
    num_replicas = jax.local_device_count() // num_devices_per_replica
    batch_size_per_step = batch_size_per_replica * num_replicas
    if config.data.trainer.batch_size < batch_size_per_step:
        raise ValueError(
            f"Batch size {config.data.trainer.batch_size} is too small. "
            f"batch_size_per_replica * num_replicas = "
            f"{batch_size_per_replica} * {num_replicas} = "
            f"{batch_size_per_step}."
        )
    if config.data.trainer.batch_size % batch_size_per_step != 0:
        raise ValueError(
            "Batch size cannot be evenly divided by batch size per step."
        )
    every_k_schedule = config.data.trainer.batch_size // batch_size_per_step
    if every_k_schedule > 1:
        logging.info(
            f"Using gradient accumulation. "
            f"Each model duplicate is stored across {num_devices_per_replica} "
            f"shard{'s' if num_devices_per_replica > 1 else ''}. "
            f"Each step has {batch_size_per_step} samples. "
            f"Gradients are averaged every {every_k_schedule} steps. "
            f"Effective batch size is {config.data.trainer.batch_size}."
        )
    return every_k_schedule


def init_optimizer(
    config: DictConfig,
) -> tuple[optax.GradientTransformation, int]:
    """Initialize optimizer.

    Args:
        config: entire configuration.

    Returns:
        optimizer and every_k_schedule.
    """
    lr_schedule = get_lr_schedule(config)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.optimizer.grad_norm),
        getattr(optax, config.optimizer.name)(
            learning_rate=lr_schedule, **config.optimizer.kwargs
        ),
    )
    # accumulate gradient when needed
    every_k_schedule = get_every_k_schedule(config)
    if every_k_schedule == 1:
        # no need to accumulate gradient
        return optimizer, every_k_schedule
    optimizer = optax.MultiSteps(optimizer, every_k_schedule=every_k_schedule)
    return optimizer, every_k_schedule
