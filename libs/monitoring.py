import tensorflow as tf
import os
from libs import config


class TrainingMonitor:
    def __init__(self, out_dir: str, save_checkpoints: bool, save_model: bool):
        self.__out_dir = out_dir
        self.__save_checkpoints = save_checkpoints
        self.__save_model = save_model

    def get_callbacks(self):
        callbacks = []
        if self.__save_checkpoints:
            callbacks.append(self.__create_checkpoint_callback())
        if self.__save_model:
            callbacks.append(self.__create_tensorboard_callback())
        return callbacks

    def __create_tensorboard_callback(self):
        fit_dir = os.path.join(self.__out_dir, "fit")
        return tf.keras.callbacks.TensorBoard(log_dir=fit_dir, histogram_freq=1)

    def __create_checkpoint_callback(self):
        checkpoint_dir = os.path.join(self.__out_dir, "checkpoints", "cp-{epoch:04d}.ckpt")
        return tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir, save_weights_only=True, verbose=1)


def training_monitor_factory(cfg: config.GeneralConfig, out_dir: str):
    if out_dir is None:
        out_dir = os.path.join(cfg.output_dir, cfg.name)
    else:
        out_dir = out_dir
    return TrainingMonitor(out_dir, cfg.save_checkpoints, cfg.save_model)
