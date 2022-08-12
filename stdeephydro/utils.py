import os
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


def setup_reproducibility(seed: int):
    os.environ['PYTHONHASHSEED'] = "0"
    tf.keras.utils.set_random_seed(seed)
    logger.info(f"Set random seed to {seed}.")
