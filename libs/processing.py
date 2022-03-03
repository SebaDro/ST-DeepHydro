import logging
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr

from libs import dataset


logger = logging.getLogger(__name__)


class AbstractProcessor:
    def __init__(self, scaling_params: tuple = None):
        self.__scaling_params = scaling_params

    @property
    def scaling_params(self):
        return self.__scaling_params

    @scaling_params.setter
    def scaling_params(self, value):
        self.__scaling_params = value

    def fit(self, ds: dataset.AbstractDataset):
        pass

    def process_and_fit(self, ds: dataset.AbstractDataset) -> dataset.AbstractDataset:
        pass

    def process(self, ds: dataset.AbstractDataset) -> dataset.AbstractDataset:
        pass

    def scale(self, ds: dataset.AbstractDataset):
        """
        Performs a min/max scaling on all variables of a datset. If processor has been
        fitted to a dataset, scaling will be done by using the minimum and maximum parameters from the fitting dataset.
        Else, minimum and maximum parameters will be calculated from the given dataset.

        Parameters
        ----------
        ds: dataset.AbstractDataset
            Dataset that holds xarray.Dataset timeseries data which will be scaled.
        """
        if self.scaling_params is None:
            min_params = ds.timeseries.min()
            max_params = ds.timeseries.max()
        else:
            min_params, max_params = self.scaling_params
        ds.timeseries = (ds.timeseries - min_params) / (max_params - min_params)

    def rescale(self, ds: dataset.AbstractDataset):
        min_params, max_params = self.scaling_params
        ds.timeseries = ds.timeseries * (max_params - min_params) + min_params

    def merge_input_and_prediction(self, ds_input: xr.Dataset, ds_prediction: xr.Dataset, pred_timeframe: bool = True):
        variables = list(ds_prediction.keys())

        if pred_timeframe:
            start_date = ds_prediction.time[0]
            end_date = ds_prediction.time[-1]
        else:
            start_date = ds_input.time[0]
            end_date = ds_input.timeseries.time[-1]

        ds_obs = ds_input.timeseries[variables].sel(time=slice(start_date, end_date))
        ds_obs = ds_obs.rename(dict((param, param + "_obs") for param in variables))
        ds_prediction = ds_prediction.rename(dict((param, param + "_pred") for param in variables))

        return xr.merge([ds_prediction, ds_obs], join="left") \
            if pred_timeframe \
            else xr.merge([ds_obs, ds_prediction], join="right")


class DefaultDatasetProcessor(AbstractProcessor):
    def __init__(self, scaling_params: tuple = None):
        """
        Initializes a DefaultDatasetProcessor instance that peforms several default processing steps on timeseries data
        wrapped by a dataset.AbstractDataset instance.

        Parameters
        ----------
        scaling_params: tuple
            Parameters that should be used for performing min-max-sacling on the timeseries data.
        """
        super().__init__(scaling_params)

    def fit(self, ds: dataset.AbstractDataset):
        """
        Fits the processor to a dataset which usually should be the training dataset. Fitting means, the processor will
        derive various parameters from the specified dataset which will be used for several subsequent processing steps.
        Usually, you will fit the processor on the training data to use the derived parameters for processing the
        validation and test datasets.

        Up to now, this method will derive the following parameters:
        - Minimum and maximum values for each variable, which will be used for performing a min-max-scalin.

        Parameters
        ----------
        ds: dataset.AbstractDataset
            Dataset that holds timeseries data as xarray.Dataset

        """
        self.__fit_scaling_params(ds)

    def process(self, ds: dataset.AbstractDataset):
        """
        Performs several processing steps on a dataset.LumpedDataset.

        Note, that it will use parameters that have been
        derived while fitting the processor to a dataset using the fit function. If this function have not been called
        before, it will automatically derive the same parameters form the specified dataset. This will lead to
        misleading results if you aim to process validation and test datsets by using processing parameters derived from
        a training dataset. Hence, it is strongly recommended to first call fit() on a dedicated dataset-

        Parameters
        ----------
        ds: dataset.AbstractDataset
            Dataset that will be processed

        Returns
        -------
            The resulting dataset.LumpedDataset after performing various processing steps on it

        """
        if self.scaling_params is None:
            logging.warning("Processor has not been fit to a dataset before. Thus, it will be fitted to the provided "
                            "dataset.")
            self.__fit_scaling_params(ds)
        ds.normalize(*self.scaling_params)
        return ds

    def __fit_scaling_params(self, ds: dataset.AbstractDataset):
        self.scaling_params = (ds.timeseries.min(), ds.timeseries.max())


class CustomTimeseriesGenerator(tf.keras.utils.Sequence):

    def __init__(self, ds: dataset.AbstractDataset, batch_size: int, timesteps: int, offset: int, feature_cols: list,
                 target_cols: list, drop_na: bool = False, input_shape: tuple = None):
        """
        A custom TimeseriesGenerator that creates batches of timeseries from a xarray.Dataset and optionally takes
        also into account NaN values.

        Parameters
        ----------
        ds: dataset.Dataset
            Dataset that holds forcings and streamflow timeseries data
        batch_size: int
            Size of the batches that will be created
        timesteps: int
            How many timesteps will be used for creating the input (forcings) timeseries
        offset: int
            Offset between inputs (forcings) and target (streamflow). An offset of 1 means that forcings for the last
            n-days will be taken as input and the the streamflow for n + 1 will be taken as target.
        feature_cols
        target_cols
        drop_na
        input_shape
        """
        self.ds = ds
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.offset = offset
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.drop_na = drop_na
        self.input_shape = input_shape
        self.idx_dict = self.__get_idx_df(drop_na)
        self.ds_inputs = ds.timeseries[feature_cols].to_array()
        self.ds_targets = ds.timeseries[target_cols].to_array()

    def __get_idx_df(self, drop_na):
        lag = self.timesteps + self.offset - 1

        date_idx_list = []
        basin_idx_list = []
        basins = self.ds.timeseries.basin.values
        for i, basin in enumerate(basins):
            if drop_na:
                # Only consider streamflow values which are not NaN
                non_nan_flags = np.invert(np.isnan(self.ds.timeseries.sel(basin=basin).streamflow))
                sel_indices = np.arange(0, len(self.ds.timeseries.sel(basin=basin).streamflow))[lag:][non_nan_flags.values[lag:]]
            else:
                sel_indices = np.arange(0, len(self.ds.timeseries.sel(basin=basin).streamflow))[lag:]
            basin_idx_list.extend([i] * len(sel_indices))
            date_idx_list.extend(sel_indices)
        return pd.DataFrame({"basin_idx": basin_idx_list, "time_idx": date_idx_list})

    def __len__(self):
        n_samples = len(self.idx_dict)
        return math.ceil(n_samples / self.batch_size)

    def __getitem__(self, idx):
        df_batch = self.idx_dict[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.input_shape:
            inputs = np.empty((0,) + self.input_shape)
        else:
            shape = self._get_input_shape()
            inputs = np.empty(shape)
        targets = np.empty((0, len(self.target_cols)))

        for index, row in df_batch.iterrows():
            start_date_idx = row.time_idx - self.timesteps
            end_date_idx = row.time_idx - self.offset + 1
            forcings_values = self.ds_inputs[:, row.basin_idx, start_date_idx:end_date_idx].values
            streamflow_values = self.ds_targets[:, row.basin_idx, row.time_idx].values
            forcings_values = np.moveaxis(forcings_values, 0, -1)
            inputs = np.vstack([inputs, np.expand_dims(forcings_values, axis=0)])
            targets = np.vstack([targets, np.expand_dims(streamflow_values, axis=0)])

        return inputs, targets

    def _get_input_shape(self):
        return (0, self.timesteps) + self.ds.get_input_shape()