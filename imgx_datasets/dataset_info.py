"""Data set class for imgx_datasets."""
import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp


@dataclasses.dataclass
class DatasetInfo:
    """Data set class for imgx_datasets."""

    tfds_preprocessed_dir: Path
    image_spacing: tuple[float, ...]
    image_spatial_shape: tuple[int, ...]
    image_channels: int
    class_names: tuple[str, ...]
    classes_are_exclusive: bool

    @property
    def input_image_shape(self) -> tuple[int, ...]:
        """Input shape of image."""
        return (*self.image_spatial_shape, self.image_channels)

    @property
    def label_shape(self) -> tuple[int, ...]:
        """Shape of label."""
        return self.image_spatial_shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.image_spatial_shape)

    @property
    def num_classes(self) -> int:
        """Number of classes for segmentation."""
        raise NotImplementedError

    def logits_to_label(self, x: jnp.ndarray, axis: int) -> jnp.ndarray:
        """Transform logits to label with integers."""
        raise NotImplementedError

    def label_to_mask(
        self, x: jnp.ndarray, axis: int, dtype: jnp.dtype = jnp.float32
    ) -> jnp.ndarray:
        """Transform label to boolean mask."""
        raise NotImplementedError

    def logits_to_label_with_post_process(self, x: jnp.ndarray, axis: int) -> jnp.ndarray:
        """Transform logits to label with post-processing."""
        return self.post_process_label(self.logits_to_label(x, axis=axis))

    def post_process_label(self, label: jnp.ndarray) -> jnp.ndarray:
        """Label post-processing."""
        return label


class OneHotLabeledDatasetInfo(DatasetInfo):
    """Data set with mutual exclusive labels."""

    @property
    def num_classes(self) -> int:
        """Number of classes including background."""
        return len(self.class_names) + 1

    def logits_to_label(self, x: jnp.ndarray, axis: int) -> jnp.ndarray:
        """Transform logits to label with integers.

        Args:
            x: logits.
            axis: axis of num_classes.

        Returns:
            Label with integers.
        """
        return jnp.argmax(x, axis=axis)

    def label_to_mask(
        self, x: jnp.ndarray, axis: int, dtype: jnp.dtype = jnp.float32
    ) -> jnp.ndarray:
        """Transform label to boolean mask.

        Args:
            x: label.
            axis: axis of num_classes.
            dtype: dtype of output.

        Returns:
            One hot mask.
        """
        return jax.nn.one_hot(
            x=x,
            num_classes=self.num_classes,
            axis=axis,
            dtype=dtype,
        )
