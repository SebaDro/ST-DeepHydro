import tensorflow as tf
import os
from stdeephydro import config


class TrainingMonitor:
    """
    Wrapper class for various monitoring strategies during training

    Parameters
    ----------
    out_dir: str
        Path to the output directory for monitoring callbacks
    save_checkpoints: bool
        Indicates whether to store checkpoints during training or not. If true, tf.keras.callbacks.ModelCheckpoint
        is used for training.
    log_tensorboard_events: bool
        Indicates whether to log events during training for Tensorboard analysis. If true,
        tf.keras.callbacks.TensorBoard is used for model training
    """

    def __init__(self, out_dir: str, save_checkpoints: bool, log_tensorboard_events: bool):
        self.__out_dir = out_dir
        self.__save_checkpoints = save_checkpoints
        self.__log_tensorboard_events = log_tensorboard_events

    def get_callbacks(self):
        callbacks = []
        if self.__save_checkpoints:
            callbacks.append(self.__create_checkpoint_callback())
        if self.__log_tensorboard_events:
            callbacks.append(self.__create_tensorboard_callback())
        return callbacks

    def __create_tensorboard_callback(self):
        fit_dir = os.path.join(self.__out_dir, "fit")
        return tf.keras.callbacks.TensorBoard(log_dir=fit_dir, histogram_freq=1)

    def __create_checkpoint_callback(self):
        checkpoint_dir = os.path.join(self.__out_dir, "checkpoints", "cp-{epoch:04d}.ckpt")
        return tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir, save_weights_only=True, verbose=1)


def training_monitor_factory(cfg: config.GeneralConfig, out_dir: str):
    """
    Creates a TrainingMonitor instance from config.GeneralConfig

    Parameters
    ----------
    cfg: config.GeneralConfig
        Runtime monitoring configuration parameters
    out_dir: str
        Monitoring output directory

    Returns
    -------
    TrainingMonitor
    """
    if out_dir is None:
        out_dir = os.path.join(cfg.output_dir, cfg.name)
    else:
        out_dir = out_dir
    return TrainingMonitor(out_dir, cfg.save_checkpoints, cfg.log_tensorboard_events)
