import os
import tensorflow as tf
import logging
import random
import numpy as np

logger = logging.getLogger(__name__)


def setup_reproducibility(seed: int):
    os.environ['PYTHONHASHSEED'] = "0"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info(f"Set random seed to {seed}.")
