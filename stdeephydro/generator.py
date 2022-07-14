import math
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
from typing import Union, List
import xarray as xr
import logging

logger = logging.getLogger(__name__)


class HydroMeteorologicalTimeseriesGenerator(Sequence):
    """
    A custom TimeseriesGenerator that creates batches of timeseries windows for input and target variables from a
    xarray.Dataset. The generator optionally takes into account ignoring NaN values.

    When passing more than one xarray.Dataset the generator creates a separate input batch for each one, which also can
    differ in the time window length. This can be useful when training models that accept multiple inputs of different
    number of timesteps. In this case, the first xarray.Dataset from the given list as well as the corresponding will be
    considered as a reference for all other datasets.

    Parameters
    ----------
    xds: xarray.Dataset or list of xarray.Dataset
        One or more Datasets that holds forcings and streamflow timeseries data.
    batch_size: int
        Size of the batches that will be created
    timesteps: int or list of int
        Timesteps that will be used for creating the input (forcings) timeseries. If you passed a list to the xds
        parameter, you also have to provide a list  with the same length to this parameter.
    offset: int
        Offset between inputs (forcings) and target (streamflow). An offset of 1 means that forcings for the last
        n-days will be taken as input and the streamflow for n + 1 will be taken as target.
    feature_vars: list
        List of variables that should be used as input features
    target_var: str
        List of variables that should be used as targets
    drop_na: bool
        Indicates whether NaN values for the target vars should be preserved for generating time win or not.
        Default: True
    joined_features: bool
        Indicates whether the timeseries batches should be prepared in a joined way, meaning input features are
        valid for all basins. If False, input feature timeseries are basin indexed. Default: False
    input_shape: tuple
        Shape of the inputs to be used for generating time windows. If not specified, the input shape will be
        computed automatically from the given xarray.Dataset.

    """

    def __init__(self, xds: Union[xr.Dataset, List[xr.Dataset]], batch_size: int, timesteps: Union[int, List[int]],
                 offset: int, feature_vars: list, target_var: str, drop_na: bool = True, joined_features: bool = False,
                 shuffle: bool = False, input_shape: tuple = None):
        self.xds_list = []
        self.ds_inputs_list = []
        if isinstance(xds, xr.Dataset):
            xds = [xds]
        self.ref_xds = xds[0]
        if isinstance(timesteps, int):
            timesteps = [timesteps]
        self.timesteps_list = timesteps
        self.ref_timesteps = timesteps[0]
        self.batch_size = batch_size
        self.offset = offset
        self.feature_vars = feature_vars
        self.target_var = target_var
        self.drop_na = drop_na
        self.joined_features = joined_features
        self.input_shape = input_shape
        self.shuffle = shuffle

        for x in xds:
            self.xds_list.append(self.__check_coords(x))
            self.ds_inputs_list.append(x[feature_vars].to_array().values)
        self.ds_targets = self.xds_list[0][[target_var]].to_array().values
        self.idx_dict = self.__get_idx_df(drop_na, joined_features)
        if self.shuffle:
            self.idx_dict = self.idx_dict.sample(frac=1).reset_index(drop=True)

    def __check_coords(self, xds: xr.Dataset):
        if all(i in xds.coords for i in ["basin", "time", "y", "x"]):
            xds = xds.transpose("basin", "time", "y", "x")
        elif all(i in xds.coords for i in ["basin", "time"]):
            xds = xds.transpose("basin", "time")
        else:
            raise ValueError(f"Coordinates should contain one of the sets: ['basin', 'time'],"
                             f"['basin', 'time', 'y', 'x']. Actual coordinates are: {list(xds.coords)}")
        return xds

    def __get_idx_df(self, drop_na: bool, joined_output: bool):
        lag = self.ref_timesteps + self.offset - 1

        date_idx_list = []
        basin_idx_list = []
        basins = self.ref_xds.basin.values
        for i, basin in enumerate(basins):
            if drop_na:
                # Only consider target values which are not NaN
                non_nan_flags = np.invert(np.isnan(self.ref_xds.sel(basin=basin)[self.target_var]))
                sel_indices = np.arange(0, len(self.ref_xds.sel(basin=basin)[self.target_var]))[lag:][
                    non_nan_flags.values[lag:]]
            else:
                sel_indices = np.arange(0, len(self.ref_xds.sel(basin=basin)[self.target_var]))[lag:]
            basin_idx_list.extend([i] * len(sel_indices))
            date_idx_list.extend(sel_indices)
        df_idx = pd.DataFrame({"basin_idx": basin_idx_list, "time_idx": date_idx_list})
        if joined_output:
            df_idx = df_idx.groupby("time_idx")["basin_idx"].apply(list).reset_index(name="basin_idx")
            df_idx = df_idx[df_idx["basin_idx"].apply(len) == len(self.ref_xds.basin.values)]
        return df_idx

    def __len__(self):
        n_samples = len(self.idx_dict)
        return math.ceil(n_samples / self.batch_size)

    def __getitem__(self, idx):
        df_batch = self.idx_dict[idx * self.batch_size:(idx + 1) * self.batch_size]
        input_lists = [[]] * len(self.xds_list)
        # for xds in self.xds_list:
        #     input_lists.append([])
        target_list = []

        for index, row in df_batch.iterrows():
            start_dates = [row.time_idx - t for t in self.timesteps_list]
            end_date_idx = row.time_idx - self.offset + 1
            if self.joined_features:
                for i, xds in enumerate(self.ds_inputs_list):
                    forcings_values = xds[:, start_dates[i]:end_date_idx, ...]
                    forcings_values = np.moveaxis(forcings_values, 0, -1)
                    input_lists[i].append(np.expand_dims(forcings_values, axis=0))

                streamflow_values = self.ds_targets[:, row.basin_idx, row.time_idx]
                target_list.append(streamflow_values)
            else:
                for i, xds in enumerate(self.ds_inputs_list):
                    forcings_values = xds[:, row.basin_idx, start_dates[i]:end_date_idx, ...]
                    forcings_values = np.moveaxis(forcings_values, 0, -1)
                    input_lists[i].append(np.expand_dims(forcings_values, axis=0))

                streamflow_values = self.ds_targets[:, row.basin_idx, row.time_idx]
                target_list.append(np.expand_dims(streamflow_values, axis=0))

        inputs = [np.vstack(i) for i in input_lists]
        targets = np.vstack(target_list)
        if len(inputs) == 1:
            inputs = inputs[0]
        return inputs, targets

    def _get_input_shapes(self):
        shapes = []
        for i, xds in enumerate(self.xds_list):
            dim_indices = [dim for dim in xds[self.feature_vars].to_array().dims if
                           dim not in ["variable", "basin", "time"]]
            dim_size = tuple(xds[dim].size for dim in dim_indices)
            shapes.append((0, self.timesteps_list[i]) + dim_size + (len(self.feature_vars),))
        return shapes

    def on_epoch_end(self):
        if self.shuffle:
            self.idx_dict = self.idx_dict.sample(frac=1).reset_index(drop=True)
