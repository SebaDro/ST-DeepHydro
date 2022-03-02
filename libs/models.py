import numpy as np
import os
import tensorflow as tf
import xarray as xr
from libs import config
from libs import dataset
from libs import monitoring
from libs import processing


class AbstractModel:

    def __init__(self, cfg: config.ModelConfig):
        self.__model = None
        self.__history = None
        self.__eval_results = None
        self._config = cfg

    def build(self, input_shape: tuple):
        self.__model = self._build_model(input_shape, self._config.params)

    def _build_model(self, input_shape: tuple, params: dict):
        raise NotImplementedError

    def _get_and_validate_params(self, params: dict):
        raise NotImplementedError

    def compile_and_fit(self, training_ds: dataset.AbstractDataset, validation_ds: dataset.AbstractDataset,
                        monitor: monitoring.TrainingMonitor = None) -> tf.keras.callbacks.History:
        """
        Compiles and fits a model using the given model training and validation datasets. For fitting the model
        a timeseries generator created batches of input time windows from the specified datasets.

        Parameters
        ----------
        training_ds: dataset.AbstractDataset
            Dataset that will be used for training
        validation_ds: dataset.AbstractDataset
            Dataset that will be used for validation
        monitor: monitoring.TrainingMonitor
            Encapsulates Tensorflow callback objects used for monitoring training progress
        Returns
        -------
        tf.keras.callbacks.History
            History object encapsulating the training progress

        """
        self.__model.compile(loss=self._config.loss,
                             optimizer=self._config.optimizer,
                             metrics=self._config.metrics)

        training_gen = self.__create_timeseries_generator(training_ds)
        validation_gen = self.__create_timeseries_generator(validation_ds)

        callbacks = monitor.get_callbacks() if monitor is not None else None

        self.__history = self.__model.fit(x=training_gen, validation_data=validation_gen, epochs=self._config.epochs,
                                          callbacks=callbacks)

        return self.__history

    def evaluate(self, test_ds: dataset.AbstractDataset):
        test_gen = self.__create_timeseries_generator(test_ds)
        return self.__model.evaluate(test_gen, return_dict=True)

    def predict(self, ds: dataset.AbstractDataset, basin, as_dataset: bool = True):
        gen = self.__create_timeseries_generator(ds)
        predictions = self.__model.predict(gen)
        if as_dataset:
            return self.to_dataset(ds, predictions, basin)
        else:
            return predictions

    def to_dataset(self, ds: dataset.AbstractDataset, predictions, basin):
        target_start_date = np.datetime64(ds.start_date) + np.timedelta64(self._config.timesteps, 'D')\
                            + np.timedelta64(self._config.offset,'D') - np.timedelta64(1, 'D')
        res_ds = ds.timeseries.sel(time=slice(target_start_date, np.datetime64(ds.end_date)))

        res_dict = {}
        for i, param in enumerate(ds.target_cols):
            non_nan_flags = np.invert(np.isnan(res_ds.sel(basin=basin)[param]))
            res_times = res_ds.time[non_nan_flags]
            res_dict[param] = xr.DataArray(predictions[:, i], coords=[res_times], dims=["time"])
        ds_prediction = xr.Dataset(res_dict)
        ds_prediction = ds_prediction.assign_coords({"basin": basin})
        ds_prediction = ds_prediction.expand_dims("basin")
        return ds_prediction

    def save_model(self, work_dir):
        storage_path = os.path.join(work_dir, "model")
        self.model.save(storage_path)

    def load_model(self, storage_path):
        return tf.keras.models.load_model(storage_path)

    def __create_timeseries_generator(self, ds: dataset.AbstractDataset):
        return processing.CustomTimeseriesGenerator(ds, self._config.batch_size, self._config.timesteps,
                                                    self._config.offset, ds.feature_cols, ds.target_cols, True)

    @property
    def model(self):
        return self.__model

    @property
    def history(self):
        return self.__history


class LstmModel(AbstractModel):

    def _build_model(self, input_shape: tuple, params: dict):
        hidden_layers, units, dropout = self._get_and_validate_params(params)

        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=input_shape))
        for i in range(0, hidden_layers - 1):
            model.add(tf.keras.layers.LSTM(units[i], return_sequences=True, dropout=dropout[i], use_bias=True))
        model.add(tf.keras.layers.LSTM(units[hidden_layers - 1], dropout=dropout[hidden_layers - 1], use_bias=True))
        model.add(tf.keras.layers.Dense(units=1))
        return model

    def _get_and_validate_params(self, params: dict):
        try:
            params = params["lstm"]
            hidden_layers = params["hiddenLayers"]
            units = params["units"]
            if len(units) != hidden_layers:
                raise config.ConfigError(f"Wrong number of layer unit definitions: {len(units)}. Expected: {hidden_layers}")
            dropout = params["dropout"]
            if len(dropout) != hidden_layers:
                raise config.ConfigError(f"Wrong number of dropout definitions: {len(dropout)}. Expected: {hidden_layers}")
            return hidden_layers, units, dropout
        except KeyError as ex:
            raise config.ConfigError(f"Required model parameter is missing: {ex}") from ex


def factory(cfg: config.ModelConfig) -> AbstractModel:
    if cfg.model_type == "lstm":
        return LstmModel(cfg)
    raise ValueError("No model for the given type '{}' available.".format(cfg.model_type))
