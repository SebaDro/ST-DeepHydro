import math
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
import xarray as xr


class CustomTimeseriesGenerator(Sequence):

    def __init__(self, xds: xr.Dataset, batch_size: int, timesteps: int, offset: int, feature_vars: list,
                 target_var: str, drop_na: bool = True, joined_output: bool = False, input_shape: tuple = None):
        """
        A custom TimeseriesGenerator that creates batches of timeseries windows for input and target variables from a
        xarray.Dataset. The generator optionally takes into account ignoring NaN values.

        Parameters
        ----------
        xds: xarray.Dataset
            Dataset that holds forcings and streamflow timeseries data
        batch_size: int
            Size of the batches that will be created
        timesteps: int
            How many timesteps will be used for creating the input (forcings) timeseries
        offset: int
            Offset between inputs (forcings) and target (streamflow). An offset of 1 means that forcings for the last
            n-days will be taken as input and the the streamflow for n + 1 will be taken as target.
        feature_vars: list
            List of variables that should be used as input features
        target_vars: list
            List of variables that should be used as targets
        drop_na: bool
            Indicates whether NaN values for the target vars should be preserved for generating time win or not.
            Default: True
        input_shape: tuple
            Shape of the inputs to be used for generating time windows. If not specified, the input shape will be
            computed automatically from the given xarray.Dataset.

        """
        if all(i in xds.coords for i in ["basin", "time", "y", "x"]):
            self.xds = xds.transpose("basin", "time", "y", "x")
        elif all(i in xds.coords for i in ["basin", "time"]):
            self.xds = xds.transpose("basin", "time")
        else:
            raise ValueError(f"Coordinates should contain one of the sets: ['basin', 'time'],"
                             f"['basin', 'time', 'y', 'x']. Actual coordinates are: {list(xds.coords)}")

        self.batch_size = batch_size
        self.timesteps = timesteps
        self.offset = offset
        self.feature_vars = feature_vars
        self.target_var = target_var
        self.drop_na = drop_na
        self.joined_output = joined_output
        self.input_shape = input_shape
        self.idx_dict = self.__get_idx_df(drop_na, joined_output)
        self.ds_inputs = self.xds[feature_vars].to_array()
        if joined_output:
            self.ds_targets = self.xds[[target_var]].to_array()
        else:
            self.ds_targets = self.xds[[target_var]].to_array()

    def __get_idx_df(self, drop_na: bool, joined_output: bool):
        lag = self.timesteps + self.offset - 1

        date_idx_list = []
        basin_idx_list = []
        basins = self.xds.basin.values
        for i, basin in enumerate(basins):
            if drop_na:
                # Only consider target values which are not NaN
                non_nan_flags = np.invert(np.isnan(self.xds.sel(basin=basin)[self.target_var]))
                sel_indices = np.arange(0, len(self.xds.sel(basin=basin)[self.target_var]))[lag:][non_nan_flags.values[lag:]]
            else:
                sel_indices = np.arange(0, len(self.xds.sel(basin=basin)[self.target_var]))[lag:]
            basin_idx_list.extend([i] * len(sel_indices))
            date_idx_list.extend(sel_indices)
        df_idx = pd.DataFrame({"basin_idx": basin_idx_list, "time_idx": date_idx_list})
        if joined_output:
            df_idx = df_idx.groupby("time_idx")["basin_idx"].apply(list).reset_index(name="basin_idx")
            df_idx = df_idx[df_idx["basin_idx"].apply(len) == len(self.xds.basin.values)]
        return df_idx

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
        if self.joined_output:
            targets = np.empty((0, len(self.xds.basin.values)))
        else:
            targets = np.empty((0, 1))

        for index, row in df_batch.iterrows():
            start_date_idx = row.time_idx - self.timesteps
            end_date_idx = row.time_idx - self.offset + 1
            if self.joined_output:
                forcings_values = self.ds_inputs[:, start_date_idx:end_date_idx, ...].values
                forcings_values = np.moveaxis(forcings_values, 0, -1)
                inputs = np.vstack([inputs, np.expand_dims(forcings_values, axis=0)])
                streamflow_values = self.ds_targets[:, row.basin_idx, row.time_idx].values
                targets = np.vstack([targets, streamflow_values])
            else:
                forcings_values = self.ds_inputs[:, row.basin_idx, start_date_idx:end_date_idx, ...].values
                forcings_values = np.moveaxis(forcings_values, 0, -1)
                inputs = np.vstack([inputs, np.expand_dims(forcings_values, axis=0)])
                streamflow_values = self.ds_targets[:, row.basin_idx, row.time_idx].values
                targets = np.vstack([targets, np.expand_dims(streamflow_values, axis=0)])

        return inputs, targets

    def _get_input_shape(self):
        dim_indices = [dim for dim in self.xds[self.feature_vars].to_array().dims if
                       dim not in ["variable", "basin", "time"]]
        dim_size = tuple(self.xds[dim].size for dim in dim_indices)
        return (0, self.timesteps) + dim_size + (len(self.feature_vars),)
