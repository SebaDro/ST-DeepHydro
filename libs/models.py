import numpy as np
import os
import tensorflow as tf
import xarray as xr
from libs import config
from libs import dataset
from libs import generator
from libs import monitoring


class AbstractModel:

    def __init__(self, cfg: config.ModelConfig):
        self.__model = None
        self.__history = None
        self.__eval_results = None
        self._config = cfg

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model

    @property
    def history(self):
        return self.__history

    def build(self, input_shape: tuple):
        """
        Builds the model architecture in accordance to the config.ModelConfig that has been passed for model
        instantiation and a given input_shape.

        Parameters
        ----------
        input_shape: tuple
            Shape of the model inputs. First axes contains the number of timesteps. Subsequent axis take into account
            optional spatial dimensions and variable dimension as last axes. Batch size is not relevant.

            One-dimensional: (timesteps, variables)
            Two-dimensional: (timesteps, x, y, variables)

        """
        param_tuple = self._get_and_validate_params(self._config.params)
        self.__model = self._build_model(input_shape, param_tuple)

    def _build_model(self, input_shape: tuple, params: tuple):
        raise NotImplementedError

    def _get_and_validate_params(self, params: dict):
        raise NotImplementedError

    def compile_and_fit(self, training_ds: dataset.HydroDataset, validation_ds: dataset.HydroDataset,
                        monitor: monitoring.TrainingMonitor = None) -> tf.keras.callbacks.History:
        """
        Compiles and fits a model using the given model training and validation datasets. For fitting the model
        a timeseries generator created batches of input time windows from the specified datasets.

        Parameters
        ----------
        training_ds: dataset.HydroDataset
            Dataset that will be used for training
        validation_ds: dataset.HydroDataset
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

    def evaluate(self, test_ds: dataset.HydroDataset, as_dataset: bool = False, basin: str = None):
        """
        Evaluates the trained model against the given dataset. The dataset will be wrapped by timeseries generator
        which aims as input for model evaluating. All metrics that have been specified as part of the model
        configuration will be calculated.

        Parameters
        ----------
        test_ds: dataset.HydroDataset
            Input dataset for model evaluation
        as_dataset: bool
            Indicates whether the calculated evaluation metrics should be returned as raw value or as xarray.Dataset
            indexed by the basin ID.
        basin: str
            ID of the basin to calculate the evaluation metrics for

        Returns
        -------
        Union[float, xr.Dataset]
            Evaluation metric either as dictionary or as basin indexed xarray.Dataset

        """
        test_gen = self.__create_timeseries_generator(test_ds)
        result = self.__model.evaluate(test_gen, return_dict=True)
        if as_dataset and basin is not None:
            res_dict = {}
            for key in result:
                res_dict[key] = (["basin"], [result[key]])
            return xr.Dataset(res_dict, coords=dict(basin=[basin]))
        else:
            return result

    def predict(self, ds: dataset.HydroDataset, basin: str, as_dataset: bool = True, remove_nan: bool = False):
        """
        Uses the trained model the calculate predictions for the given dataset.

        Parameters
        ----------
        ds: dataset.HydroDataset
            Input dataset for model predictions
        basin: str
            Basin ID
        as_dataset: bool
            Indicates if model predictions should be returned as raw numpy.ndarray or as xarray.Dataset
        remove_nan: bool
            Indicates if the timeseries generator should remove timesteps which contains NaN values for target
            variables. Default is false, since input targets does not matter for calculating predictions.

        Returns
        -------
            Model predictions

        """
        gen = self.__create_timeseries_generator(ds, remove_nan)
        predictions = self.__model.predict(gen)
        if as_dataset:
            return self.prediction_to_dataset(ds, predictions, basin, remove_nan)
        else:
            return predictions

    def prediction_to_dataset(self, ds: dataset.HydroDataset, predictions: np.ndarray, basin: str,
                              remove_nan: bool = False) -> xr.Dataset:
        """
        Creates a xarray.Dataset for raw model predictions. Therefore, the model outputs and the dataset that has been
        used as model input for calculating the predictions are aligned. The resulting xarrary.Dataset has the same
        coordinate dimensions as the input dataset. NaN values may be optionally removed.

        Parameters
        ----------
        ds: dataset.HydroDataset
            Source dataset that has been used as model input for generating predictions
        predictions: numpy.ndarray
            Raw model output
        basin: str
            Basin ID
        remove_nan: bool
            Indicates if timesteps which contain NaN values for the target variables in the input dataset should be
            preserved or not. If true, the resulting xarray.Dataset only contains those timesteps, which do not
            contain NaN values input dataset. Note, that this flag should be set in accordance to the flag that has been
            set for the model prediction method.

        Returns
        -------
        Model predictions as xarray.Dataset

        """
        target_start_date = np.datetime64(ds.start_date) + np.timedelta64(self._config.timesteps, 'D') + np.timedelta64(
            self._config.offset, 'D') - np.timedelta64(1, 'D')
        res_ds = ds.timeseries.sel(time=slice(target_start_date, np.datetime64(ds.end_date)))

        res_dict = {}
        for i, param in enumerate(ds.target_cols):
            if remove_nan:
                non_nan_flags = np.invert(np.isnan(res_ds.sel(basin=basin)[param]))
                res_times = res_ds.time[non_nan_flags]
            else:
                res_times = res_ds.time
            res_dict[param] = xr.DataArray(predictions[:, i], coords=[res_times], dims=["time"])
        ds_prediction = xr.Dataset(res_dict)
        ds_prediction = ds_prediction.assign_coords({"basin": basin})
        ds_prediction = ds_prediction.expand_dims("basin")
        return ds_prediction

    def save_model(self, storage_path: str):
        """
        Stores a trained model within the given directory.

        Parameters
        ----------
        storage_path: str
            Path to the storage directory.

        """
        storage_path = os.path.join(storage_path, "model")
        self.model.save(storage_path)

    def __create_timeseries_generator(self, ds: dataset.HydroDataset, remove_nan: bool = True):
        return generator.CustomTimeseriesGenerator(ds, self._config.batch_size, self._config.timesteps,
                                                   self._config.offset, ds.feature_cols, ds.target_cols, remove_nan)


class LstmModel(AbstractModel):

    def _build_model(self, input_shape: tuple, params: tuple):
        hidden_layers, units, dropout = params

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
                raise config.ConfigError(
                    f"Wrong number of layer unit definitions: {len(units)}. Expected: {hidden_layers}")
            dropout = params["dropout"]
            if len(dropout) != hidden_layers:
                raise config.ConfigError(
                    f"Wrong number of dropout definitions: {len(dropout)}. Expected: {hidden_layers}")
            return hidden_layers, units, dropout
        except KeyError as ex:
            raise config.ConfigError(f"Required model parameter is missing: {ex}") from ex


class CnnLstmModel(AbstractModel):

    def _build_model(self, input_shape: tuple, params: tuple):
        hidden_layers, units, dropout = params

        model = tf.keras.models.Sequential([
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(16, (3, 3), activation="relu")),
            tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation="relu")),
            tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
        ])
        for i in range(0, hidden_layers - 1):
            model.add(tf.keras.layers.LSTM(units[i], return_sequences=True, dropout=dropout[i], use_bias=True))
        model.add(tf.keras.layers.LSTM(units[hidden_layers - 1], use_bias=True))
        model.add(tf.keras.layers.Dense(units=1))
        return model

    def _get_and_validate_params(self, params: dict):
        try:
            lstm_params = params["lstm"]
            hidden_layers = lstm_params["hiddenLayers"]
            units = lstm_params["units"]
            if len(units) != hidden_layers:
                raise config.ConfigError(
                    f"Wrong number of layer unit definitions: {len(units)}. Expected: {hidden_layers}")
            dropout = lstm_params["dropout"]
            if len(dropout) != hidden_layers:
                raise config.ConfigError(
                    f"Wrong number of dropout definitions: {len(dropout)}. Expected: {hidden_layers}")
            return hidden_layers, units, dropout
        except KeyError as ex:
            raise config.ConfigError(f"Required model parameter is missing: {ex}") from ex


def factory(cfg: config.ModelConfig) -> AbstractModel:
    if cfg.model_type == "lstm":
        return LstmModel(cfg)
    if cfg.model_type == "cnn-lstm":
        return CnnLstmModel(cfg)
    raise ValueError("No model for the given type '{}' available.".format(cfg.model_type))


def load_model(storage_path: str, cfg: config.ModelConfig):
    """
    Loads a trained model from a given directory.

    Parameters
    ----------
    storage_path: str
        Path to the storage directory.
    cfg: str
        Model configuration

    Returns
    -------
    A trained model instance

    """
    model = factory(cfg)
    model.model = tf.keras.models.load_model(storage_path)
    return model
