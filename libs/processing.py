import datetime
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


class LumpedDatasetProcessor(AbstractProcessor):
    def __init__(self, scaling_params: tuple = None):
        """
        Initializes a LumpedDatasetProcessor instance that peforms several default processing steps on timeseries data
        wrapped by a dataset.LumpedDataset.

        Parameters
        ----------
        scaling_params: tuple
            Parameters that should be used for performing min-max-sacling on the timeseries data.
        """
        super().__init__(scaling_params)

    def fit(self, ds: dataset.LumpedDataset):
        """
        Fits the processor to a dataset which usually should be the training dataset. Fitting means, the processor will
        derive various parameters from the specified dataset which will be used for several subsequent processing steps.
        Usually, you will fit the processor on the training data to use the derived parameters for processing the
        validation and test datasets.

        Up to now, this method will derive the following parameters:
        - Minimum and maximum values for each variable, which will be used for performing a min-max-scalin.

        Parameters
        ----------
        ds: dataset.LumpedDataset
            Dataset that holds timeseries data as xarray.Dataset

        """
        self.__fit_scaling_params(ds)

    def process(self, ds: dataset.LumpedDataset):
        """
        Performs several processing steps on a dataset.LumpedDataset.

        Note, that it will use parameters that have been
        derived while fitting the processor to a dataset using the fit function. If this function have not been called
        before, it will automatically derive the same parameters form the specified dataset. This will lead to
        misleading results if you aim to process validation and test datsets by using processing parameters derived from
        a training dataset. Hence, it is strongly recommended to first call fit() on a dedicated dataset-

        Parameters
        ----------
        ds: dataset.LumpedDataset
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

    def __fit_scaling_params(self, ds: dataset.LumpedDataset):
        self.scaling_params = (ds.timeseries.min(), ds.timeseries.max())


class CustomTimeseriesGenerator(tf.keras.utils.Sequence):

    def __init__(self, timeseries: xr.Dataset, batch_size, timesteps, offset, feature_cols, target_cols, drop_na=False):
        self.timeseries = timeseries
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.offset = offset
        self.drop_na = drop_na
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.idx_dict = self.__get_idx_df(drop_na)

    def __get_idx_df(self, drop_na):
        lag = self.timesteps + self.offset - 1

        date_list = []
        basin_list = []
        dates = self.timeseries.time[lag:].values
        basins = self.timeseries.basin.values
        for basin in basins:
            if drop_na:
                # Only consider streamflow values which are not NaN
                non_nan_flags = np.invert(np.isnan(self.timeseries.sel(basin=basin).streamflow))
                sel_dates = dates[non_nan_flags[lag:].values]
            else:
                sel_dates = dates
            basin_list.extend([basin] * len(sel_dates))
            date_list.extend(sel_dates)
        return pd.DataFrame({"basin": basin_list, "time": date_list})

    def __len__(self):
        n_samples = len(self.idx_dict)
        return math.ceil(n_samples / self.batch_size)

    def __getitem__(self, idx):
        df_batch = self.idx_dict[idx * self.batch_size:(idx + 1) * self.batch_size]

        inputs = np.empty((0, self.timesteps, len(self.feature_cols)))
        targets = np.empty((0, len(self.target_cols)))
        for index, row in df_batch.iterrows():
            start_date = row.time - datetime.timedelta(days=self.timesteps)
            end_date = row.time - datetime.timedelta(days=self.offset)
            forcings_values = self.timeseries.sel(basin=row.basin,
                                                  time=slice(start_date, end_date))[self.feature_cols].to_array().values
            streamflow_values = self.timeseries.sel(basin=row.basin, time=row.time)[self.target_cols].to_array().values
            inputs = np.vstack([inputs, np.expand_dims(forcings_values.transpose(), axis=0)])
            targets = np.vstack([targets, np.expand_dims(streamflow_values, axis=0)])
        return inputs, targets
