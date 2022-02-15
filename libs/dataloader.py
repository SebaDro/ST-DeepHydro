import pandas as pd
import xarray as xr

from libs import config
from libs import dataset
from libs import ioutils


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

    def load_dataset(self, basin: str = None) -> dataset.LumpedDataset:
        if basin is not None:
            return self.load_single_dataset(basin)
        else:
            return self.load_full_dataset()

    def load_single_dataset(self, basin: str) -> dataset.LumpedDataset:
        """
        Loads a single Camels-US dataset for the specified basin as LumpedDataset. The LumpedDataset wraps lumped
        forcings and timeseries data for that basin as xarray.Dataset.

        Parameters
        ----------
        basin: str
            ID of the basin

        Returns
        -------
        LumpedDataset
            A LumpedDataset that holds lumped forcings and timeseries data

        """
        ds_timeseries = self.load_xarray_dataset(basin, False)

        return dataset.LumpedDataset(ds_timeseries, self.forcings_variables, self.streamflow_variables,
                                     self.start_date, self.end_date)

    def load_full_dataset(self, as_dask: bool = False) -> dataset.LumpedDataset:
        """
        Loads several Camels-US datasets as one single LumpedDataset. The LumpedDataset wraps lumped
        forcings and timeseries data for all basins the loader has been initialized with as one single
        xarray.Dataset. Each basin ID will be used as dimension coordinate.

        Returns
        -------
        LumpedDataset
            A LumpedDataset that holds lumped forcings and timeseries data for several basins

        """
        ds_list = []
        for basin in self.basins:
            ds = self.load_xarray_dataset(basin, True)
            if as_dask:
                ds = ds.chunk()
            ds_list.append(ds)
        ds_timeseries = xr.concat(ds_list, dim="basin")

        if as_dask:
            ds_timeseries = ds_timeseries.chunk({"basin": len(ds_timeseries.basin)})

        return dataset.LumpedDataset(ds_timeseries, self.forcings_variables, self.streamflow_variables,
                                     self.start_date, self.end_date)

    def load_xarray_dataset(self, basin: str, normalize_streamflow: bool) -> xr.Dataset:
        """
        Loads CAMELS-US forcings and streamflow timeseries data for a certain basin as xarray.Dataset

        Parameters
        ----------
        basin: str
            Basin ID

        Returns
        -------
        xarray.Dataset
            A timeseries indexed xarray.Dataset that holds forcing and streamflow variables.

        """
        forcings_path = ioutils.discover_single_camels_us_forcings_file(self.forcings_dir, "daymet", basin)
        df_forcings = ioutils.load_forcings_camels_us(forcings_path)

        streamflow_path = ioutils.discover_single_camels_us_streamflow_file(self.streamflow_dir, basin)
        df_streamflow = ioutils.load_streamflow_camels_us(streamflow_path)

        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq="1D")
        df_forcings = df_forcings[self.start_date:self.end_date][self.forcings_variables].reindex(date_range)
        df_streamflow = df_streamflow[self.start_date:self.end_date][self.streamflow_variables].reindex(date_range)

        if normalize_streamflow:
            latitude, elevation, area = ioutils.load_forcings_gauge_metadata(forcings_path)
            df_streamflow[self.streamflow_variables] = convert_streamflow(df_streamflow[self.streamflow_variables], area)

        df_merged = df_forcings.join(df_streamflow, how="outer")

        ds_timeseries = xr.Dataset.from_dataframe(df_merged)
        ds_timeseries = ds_timeseries.rename({"index": "time"})
        ds_timeseries = ds_timeseries.assign_coords({"basin": basin})

        return ds_timeseries


class DistributedDataLoader(AbstractDataLoader):

    def load_dataset(self, basin: str = None) -> dataset.LumpedDataset:
        return self.load_single_dataset(basin)

    def load_single_dataset(self, basin: str) -> dataset.LumpedDataset:
        """
        Loads a single DistributedDataset for the specified basin from raster-based Daymet forcings and lumped Camels-US
        streamflow timeseries data files.

        Parameters
        ----------
        basin: str
            ID of the basin

        Returns
        -------
        LumpedDataset
            A DistributedDataset that holds forcings and timeseries data

        """
        ds_timeseries = self.load_xarray_dataset(basin)

        return dataset.LumpedDataset(ds_timeseries, self.forcings_variables, self.streamflow_variables,
                                     self.start_date, self.end_date)

    def load_xarray_dataset(self, basin: str) -> xr.Dataset:
        """
        Loads raster-based Daymet forcings and CAMELS-US streamflow timeseries data for a certain basin as
        xarray.Dataset. The xarray.Dataset is indexed by time and a basin ID for both forcings and streamflow variables.
        Forcings variables are also indexed by a spatial dimension which means 'x' and 'y' coordinates.

        Parameters
        ----------
        basin: str
            Basin ID

        Returns
        -------
        xarray.Dataset
            A xarray.Dataset that holds forcing and streamflow variables.

        """
        forcings_path = ioutils.discover_single_file_for_basin(self.forcings_dir, basin)
        ds_forcings = ioutils.load_forcings_daymet_2d(forcings_path)
        ds_forcings = ds_forcings.sel(time=slice(self.start_date, self.end_date))[self.forcings_variables]
        ds_forcings = ds_forcings.assign_coords({"basin": basin})
        ds_forcings = ds_forcings.expand_dims("basin")
        ds_forcings["time"] = ds_forcings.indexes["time"].normalize()

        streamflow_path = ioutils.discover_single_camels_us_streamflow_file(self.streamflow_dir, basin)
        df_streamflow = ioutils.load_streamflow_camels_us(streamflow_path)
        df_streamflow = df_streamflow[self.start_date:self.end_date][self.streamflow_variables]

        ds_streamflow = xr.Dataset.from_dataframe(df_streamflow)
        ds_streamflow = ds_streamflow.rename({"date": "time"})
        ds_streamflow = ds_streamflow.assign_coords({"basin": basin})
        ds_streamflow = ds_streamflow.expand_dims("basin")

        ds_timeseries = xr.merge([ds_forcings, ds_streamflow], join="left")

        return ds_timeseries


def convert_streamflow(streamflow, area: float):
    streamflow = streamflow * 0.028316846592  # [m³/s]
    streamflow = streamflow / area  # [m/s]
    streamflow = streamflow * 86400  # [m/d]
    return streamflow * 10 ** 3  # [mm/d]


def factory(data_cfg: config.DataConfig, dataset_cfg: config.DatasetConfig) -> AbstractDataLoader:
    if data_cfg.forcings_cfg.data_type == "camels-us" and data_cfg.streamflow_cfg.data_type == "camels-us":
        return CamelsUSDataLoader.from_config(data_cfg, dataset_cfg)
    raise ValueError("No data loader exists for the specified dataset types Dataset types '{}' and '{}'."
                     .format(data_cfg.forcings_cfg.data_type, data_cfg.streamflow_cfg.data_type))

