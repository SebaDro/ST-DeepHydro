import numpy as np
import pandas as pd
import xarray as xr
from libs import config
from libs import ioutils


class TimeseriesDataset:
    def __init__(self, input_timeseries, target_timeseries):
        self.__input_timeseries = input_timeseries
        self.__target_timeseries = target_timeseries
        self.__min_features = None
        self.__max_features = None
        self.__min_targets = None
        self.__max_targets = None

    @property
    def forcings_timeseries(self):
        return self.__input_timeseries

    @property
    def streamflow_timeseres(self):
        return self.__target_timeseries

    @property
    def min_features(self):
        return self.__min_features

    @property
    def max_features(self):
        return self.__max_features

    @property
    def min_targets(self):
        return self.__min_targets

    @property
    def max_targets(self):
        return self.__max_targets

    def normalize(self, min_features=None, max_features=None, min_targets=None, max_targets=None):
        axes = tuple(range(len(self.__input_timeseries.shape) - 1))
        self.__min_features = np.nanmin(self.__input_timeseries, axis=axes) if min_features is None else min_features
        self.__max_features = np.nanmax(self.__input_timeseries, axis=axes) if max_features is None else max_features
        self.__input_timeseries = (self.__input_timeseries - self.__min_features) / (self.__max_features - self.__min_features)

        self.__min_targets = np.nanmin(self.__target_timeseries, axis=0) if min_targets is None else min_targets
        self.__max_targets = np.nanmax(self.__target_timeseries, axis=0) if max_targets is None else max_targets
        self.__target_timeseries = (self.__target_timeseries - self.__min_targets) / (self.__max_targets - self.__min_targets)


class AbstractDataset:
    def __init__(self, timeseries: xr.Dataset):
        self.__timeseries = timeseries

    @property
    def timeseries(self):
        return self.__timeseries

    @timeseries.setter
    def timeseries(self, value):
        self.__timeseries = value

    def normalize(self, min_params: xr.Dataset, max_params: xr.Dataset):
        pass


class LumpedDataset(AbstractDataset):
    def __init__(self, timeseries: xr.Dataset, feature_variables: list, target_variables: list,
                 start_date: str = None, end_date: str = None):
        """
        Initializes a LumpedDataset instance, which wraps lumped forcings and streamflow timeseries for one or more
        basins as xarray.Dataset. Lumped means, the forcings and streamflow variables are only distributed in time but
        not in space. Forcings and streamflow values are store as variables.

        Parameters
        ----------
        timeseries: xarray.Dataset
            A xarray.Dataset that holds forcings and streamflow timeseries data. Tha variables are distribted in time
            but not in space. Dimension coordinates must be `time` for the temporal dimension and `basin` for separating
            the timeseries datasets for each basin.
        feature_variables: list
            List of names that represent the feature variables (e.g. 'temp', 'prec') within the xarray.Dataset
        target_variables
            List of names that represent the target variables (e.g. `streamflow`) within the xarray.Dataset.
        start_date:
            Start date of the timeseries
        end_date
            End date of the timeseries
        """
        self.__feature_cols = feature_variables
        self.__target_cols = target_variables
        self.__start_date = start_date
        self.__end_date = end_date
        super().__init__(timeseries)

    def normalize(self, min_params: xr.Dataset = None, max_params: xr.Dataset = None):
        """
        Performs a min/max scaling on all variables that are present within a xarray.Dataset.

        Parameters
        ----------
        min_params: xarray.Dataset
            Minimum parameters as reduced xarray.Dataset that will be used for scaling. If None, this parameter will be
            calculated from the current dataset.
        max_params: xarray.Dataset
            Maximum parameters as reduced xarray.Dataset that will be used for scaling. If None, this parameter will be
            calculated from the current dataset.
        """
        min_params = self.timeseries.min() if min_params is None else min_params
        max_params = self.timeseries.max() if max_params is None else max_params

        self.timeseries = (self.timeseries - min_params) / (max_params - min_params)


class AbstractDataLoader:

    def __init__(self, basins: list, forcings_dir: str, streamflow_dir: str, start_date: str, end_date: str,
                 forcings_vars: list, streamflow_vars: list):
        """
        Default initializer for DataLoader instances.

        Parameters
        ----------
        basins: list
            List of basins.
        forcings_dir: str
            Path to the directory that contains forcings timeseries data. The datasets may live in subdirectories of the
            speciified directory.
        streamflow_dir: str
            Path to the directory that contains streamflow timeseries data. The datasets may live in subdirectories of the
            speciified directory.
        start_date: str
            String that represents a date. It will be used as start date for subsetting the timeseries datasets.
        end_date: str
            String that represents a date. It will be used as end date for subsetting the timeseries datasets.
        forcings_vars: list
            List of forcings variable names. The variables will be used for subsetting the forcings dataset.
        streamflow_vars: list
            List of streamflow variable names. The variables will be used for subsetting the streamflow dataset.
        """
        self.__basins = basins
        self.__forcings_dir = forcings_dir
        self.__streamflow_dir = streamflow_dir
        self.__start_date = start_date
        self.__end_date = end_date
        self.__forcings_variables = forcings_vars
        self.__streamflow_variables = streamflow_vars

    @classmethod
    def from_config(cls, data_cfg: config.DataConfig, dataset_cfg: config.DatasetConfig):
        """
        Creates a DataLoader instance from configurations.

        Parameters
        ----------
        data_cfg: config.DataConfig
            DataConfig that holds data configuration parameters, such as basins, data directories and variables which
            will be used for loading and subsetting the forcings and streamflow datasets.
        dataset_cfg: config.DatasetConfig
            DatasetConfig that holds configuration parameters, such as start and end date for subsetting the
            forcings and streamflow datasets.


        Returns
        -------
        AbstractDataLoader
            A DataLoader instance

        """
        return cls(data_cfg.basins, data_cfg.forcings_cfg.data_dir, data_cfg.streamflow_cfg.data_dir,
                   dataset_cfg.start_date, dataset_cfg.end_date, data_cfg.forcings_cfg.variables,
                   data_cfg.streamflow_cfg.variables)

    @property
    def basins(self):
        return self.__basins
    
    @property
    def forcings_dir(self):
        return self.__forcings_dir
    
    @property
    def streamflow_dir(self):
        return self.__streamflow_dir
    
    @property
    def start_date(self):
        return self.__start_date
    
    @property
    def end_date(self):
        return self.__end_date
    
    @property
    def forcings_variables(self):
        return self.__forcings_variables
    
    @property
    def streamflow_variables(self):
        return self.__streamflow_variables

    def load_dataset(self):
        pass


class CamelsUSDataLoader(AbstractDataLoader):

    def load_dataset(self, basin: str = None) -> LumpedDataset:
        if basin is not None:
            return self.load_single_dataset(basin)
        else:
            return self.load_full_dataset()

    def load_single_dataset(self, basin: str) -> LumpedDataset:
        """
        Loads a single Camels-US dataset for the specified basin as LumpedDataset. The LumpedDataset wraps lumped
        forcings and timeseries timeseries for that basin as xarray.Dataset.

        Parameters
        ----------
        basin: str
            ID of the basin

        Returns
        -------
        LumpedDataset
            A LumpedDataset that holds lumped forcings and timeseries data

        """
        ds_timeseries = self.__load_xarray_dataset(basin)

        return LumpedDataset(ds_timeseries, self.forcings_variables, self.streamflow_variables,
                             self.start_date, self.end_date)

    def load_full_dataset(self) -> LumpedDataset:
        """
        Loads several Camels-US datasets as one single LumpedDataset. The LumpedDataset wraps lumped
        forcings and timeseries timeseries for all basins the loader has been initialized with as one single
        xarray.Dataset. Each basin ID will be used as dimension coordinate.

        Returns
        -------
        LumpedDataset
            A LumpedDataset that holds lumped forcings and timeseries data for several basins

        """
        ds_list = []
        for basin in self.basins:
            ds_list.append(self.__load_xarray_dataset(basin))
        ds_timeseries = xr.concat(ds_list, dim="basin")

        return LumpedDataset(ds_timeseries, self.forcings_variables, self.streamflow_variables,
                             self.start_date, self.end_date)

    def __load_xarray_dataset(self, basin):
        forcings_path = ioutils.discover_single_camels_us_forcings_file(self.forcings_dir, "daymet", basin)
        df_forcings = ioutils.load_forcings_camels_us(forcings_path)

        streamflow_path = ioutils.discover_single_camels_us_streamflow_file(self.streamflow_dir, basin)
        df_streamflow = ioutils.load_streamflow_camels_us(streamflow_path)

        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq="1D")
        df_forcings = df_forcings[self.start_date:self.end_date][self.forcings_variables].reindex(date_range)
        df_streamflow = df_streamflow[self.start_date:self.end_date][self.streamflow_variables].reindex(date_range)

        df_merged = df_forcings.join(df_streamflow, how="outer")

        ds_timeseries = xr.Dataset.from_dataframe(df_merged)
        ds_timeseries = ds_timeseries.rename({"index": "time"})
        ds_timeseries = ds_timeseries.assign_coords({"basin": basin})

        return ds_timeseries


