import math
import numpy as np
import pandas as pd
import tensorflow as tf
from libs import dataset


class CustomTimeseriesGenerator(tf.keras.utils.Sequence):

    def __init__(self, ds: dataset.HydroDataset, batch_size: int, timesteps: int, offset: int, feature_cols: list,
                 target_cols: list, drop_na: bool = False, input_shape: tuple = None):
        """
        A custom TimeseriesGenerator that creates batches of timeseries from a xarray.Dataset and optionally takes
        also into account NaN values.

        Parameters
        ----------
        ds: dataset.HydroDataset
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
