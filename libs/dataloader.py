import pandas as pd
import xarray as xr
import dask
from typing import Union

from libs import config
from libs import dataset
from libs import ioutils


class AbstractDatasetLoader:

    def __init__(self, basins: list, data_dir: str, variables: list):
        self.__basins = basins
        self.__data_dir = data_dir
        self.__variables = variables

    @classmethod
    def from_config(cls, data_cfg: config.DataTypeConfig, basins):
        return cls(basins, data_cfg.data_dir, data_cfg.variables)

    @property
    def basins(self):
        return self.__basins

    @property
    def data_dir(self):
        return self.__data_dir

    @property
    def variables(self):
        return self.__variables

    def load_full_dataset(self, start_date: str, end_date: str, as_dask: bool = False) -> xr.Dataset:
        pass

    def load_single_dataset(self, start_date: str, end_date: str, basin: str) -> xr.Dataset:
        pass


class CamelsUsStreamflowDataLoader(AbstractDatasetLoader):

    def __init__(self, basins: list, data_dir: str, variables: list, forcings_dir: str = None):
        super().__init__(basins, data_dir, variables)
        self.__forcings_dir = forcings_dir

    def load_single_dataset(self, start_date: str, end_date: str, basin: str) -> xr.Dataset:
        """
        Loads Camels-US streamflow timeseries data as xarray.Dataset for a single basin and the specified start and end
        date.

        Note, that streamflow will be converted from cubic feet per second to cubic meters per second.

        Parameters
        ----------
        start_date: str
            String that represents a date. It will be used as start date for subsetting the timeseries datasets.
        end_date: str
            String that represents a date. It will be used as end date for subsetting the timeseries datasets.
        basin: str
            ID of the basin to load streamflow data for.

        Returns
        -------
        xarray.Dataset:
            A time and basin indexed xarray.Dataset for streamflow timeseries data.

        """
        return self._load_xarray_dataset(start_date, end_date, basin)

    def load_full_dataset(self, start_date: str, end_date: str, as_dask: bool = False) -> xr.Dataset:
        """
        Loads Camels-US streamflow timeseries data as either as xarray.Dataset for the specified start and end date.
        The method will load streamflow data for all basins that have been specified for instantiating this
        data loader.

        Note, that streamflow will be normalized, which means it will be devided by the basin area and converted to
        meters per day. For loading basin area information you have to instantiate this data loader by additionally
        passing the forcings directory.

        Parameters
        ----------
        start_date: str
            String that represents a date. It will be used as start date for subsetting the timeseries datasets.
        end_date: str
            String that represents a date. It will be used as end date for subsetting the timeseries datasets.
        as_dask: bool:
            Indicates if the dataset should be load as Dask Array. Should be set as True if datasets does not fit into
            memory.

        Returns
        -------
        xarray.Dataset:
            A time and basin indexed xarray.Dataset for streamflow timeseries data.

        """
        ds_list = []
        for basin in self.basins:
            ds = self._load_xarray_dataset(start_date, end_date, basin, True)
            if as_dask:
                ds = ds.chunk()
            ds_list.append(ds)
        ds_timeseries = xr.concat(ds_list, dim="basin")

        if as_dask:
            ds_timeseries = ds_timeseries.chunk({"basin": len(ds_timeseries.basin)})

        return ds_timeseries

    def _load_xarray_dataset(self, start_date: str, end_date: str, basin: str, normalize: bool = False) -> xr.Dataset:
        streamflow_path = ioutils.discover_single_camels_us_streamflow_file(self.data_dir, basin)
        df_streamflow = ioutils.load_streamflow_camels_us(streamflow_path)

        date_range = pd.date_range(start=start_date, end=end_date, freq="1D")
        df_streamflow = df_streamflow[start_date:end_date][self.variables].reindex(date_range)

        if normalize:
            forcings_path = ioutils.discover_single_camels_us_forcings_file(self.__forcings_dir, "daymet", basin)
            latitude, elevation, area = ioutils.load_forcings_gauge_metadata(forcings_path)
            df_streamflow[self.variables] = normalize_streamflow(df_streamflow[self.variables], area)
        else:
            df_streamflow[self.variables] = streamflow_to_metric(df_streamflow[self.variables])

        ds_timeseries = xr.Dataset.from_dataframe(df_streamflow)
        ds_timeseries = ds_timeseries.rename({"index": "time"})
        ds_timeseries = ds_timeseries.assign_coords({"basin": basin})
        ds_timeseries = ds_timeseries.expand_dims("basin")

        return ds_timeseries


class CamelsUsForcingsDataLoader(AbstractDatasetLoader):

    def load_single_dataset(self, start_date: str, end_date: str, basin: str) -> xr.Dataset:
        """
        Loads Camels-US forcings timeseries data as xarray.Dataset for a single basin and the specified start and end
        date.

        Parameters
        ----------
        start_date: str
            String that represents a date. It will be used as start date for subsetting the timeseries datasets.
        end_date: str
            String that represents a date. It will be used as end date for subsetting the timeseries datasets.
        basin: str
            ID of the basin to load forcings data for.

        Returns
        -------
        xarray.Dataset:
            A time and basin indexed xarray.Dataset for forcings timeseries data.

        """
        ds_timeseries = self._load_xarray_dataset(start_date, end_date, basin)

        return ds_timeseries

    def load_full_dataset(self, start_date: str, end_date: str, as_dask: bool = False) -> xr.Dataset:
        """
        Loads Camels-US forcings timeseries data as xarray.Dataset for the specified start and end date. The method will
        load forcings data for all basins that have been specified for instantiating this data loader.

        Parameters
        ----------
        start_date: str
            String that represents a date. It will be used as start date for subsetting the timeseries datasets.
        end_date: str
            String that represents a date. It will be used as end date for subsetting the timeseries datasets.
        as_dask: bool:
            Indicates if the dataset should be load as Dask Array. Should be set as True if datasets does not fit into
            memory.

        Returns
        -------
        xarray.Dataset:
            A time and basin indexed xarray.Dataset for forcings timeseries data.

        """
        ds_list = []
        for basin in self.basins:
            ds = self._load_xarray_dataset(start_date, end_date, basin)
            if as_dask:
                ds = ds.chunk()
            ds_list.append(ds)
        ds_timeseries = xr.concat(ds_list, dim="basin")

        if as_dask:
            ds_timeseries = ds_timeseries.chunk({"basin": len(ds_timeseries.basin)})

        return ds_timeseries

    def _load_xarray_dataset(self, start_date: str, end_date: str, basin: str) -> xr.Dataset:
        forcings_path = ioutils.discover_single_camels_us_forcings_file(self.data_dir, "daymet", basin)
        df_forcings = ioutils.load_forcings_camels_us(forcings_path)

        date_range = pd.date_range(start=start_date, end=end_date, freq="1D")
        df_forcings = df_forcings[start_date:end_date][self.variables].reindex(date_range)

        ds_timeseries = xr.Dataset.from_dataframe(df_forcings)
        ds_timeseries = ds_timeseries.rename({"index": "time"})
        ds_timeseries = ds_timeseries.assign_coords({"basin": basin})
        ds_timeseries = ds_timeseries.expand_dims("basin")

        return ds_timeseries


class DaymetDataLoader(AbstractDatasetLoader):

    def load_dataset(self, start_date: str, end_date: str, as_dask: bool = False, basin: str = None) -> xr.Dataset:
        """
        Loads raster-based Daymet forcings as xarray.Dataset. The xarray.Dataset is indexed by time, basin ID and
        a spatial dimension by means of 'x' and 'y' coordinates.

        Parameters
        ----------
        start_date: str
            String that represents a date. It will be used as start date for subsetting the timeseries datasets.
        end_date: str
            String that represents a date. It will be used as end date for subsetting the timeseries datasets.
        as_dask: bool:
            Daymet data loader does not yet support loading raster datasets as Dask Array
        basin: str
            ID of the basin to load forcings data for.

        Returns
        -------
        xarray.Dataset
            A xarray.Dataset that holds forcing and streamflow variables.

        """
        return self.load_single_dataset(start_date, end_date, basin)

    def load_single_dataset(self, start_date: str, end_date: str, basin: str) -> xr.Dataset:
        """
        Loads raster-based Daymet forcings as xarray.Dataset for a single basin and the specified start and end date.
        The xarray.Dataset is indexed by time, basin ID and a spatial dimension by means of 'x' and 'y' coordinates.

        Parameters
        ----------
        start_date: str
            String that represents a date. It will be used as start date for subsetting the timeseries datasets.
        end_date: str
            String that represents a date. It will be used as end date for subsetting the timeseries datasets.
        basin: str
            Basin ID

        Returns
        -------
        xarray.Dataset
            A xarray.Dataset that holds forcing and streamflow variables.

        """
        ds_timeseries = self._load_xarray_dataset(start_date, end_date, basin)

        return ds_timeseries

    def _load_xarray_dataset(self, start_date: str, end_date: str, basin: str) -> xr.Dataset:
        forcings_path = ioutils.discover_single_file_for_basin(self.data_dir, basin)
        ds_forcings = ioutils.load_forcings_daymet_2d(forcings_path)
        ds_forcings = ds_forcings.sel(time=slice(start_date, end_date))[self.variables]
        ds_forcings = ds_forcings.assign_coords({"basin": basin})
        ds_forcings = ds_forcings.expand_dims("basin")
        ds_forcings["time"] = ds_forcings.indexes["time"].normalize()

        return ds_forcings


class HydroDataLoader:
    def __init__(self, forcings_data_loader: AbstractDatasetLoader, streamflow_data_loader: AbstractDatasetLoader,
                 forcings_vars: list, streamflow_vars: list):
        """
        Default initializer for a data loader instance which loads and combines forcings and streamflow timeseries data
        for basins.

        Parameters
        ----------
        forcings_data_loader: AbstractDatasetLoader
            Instance for loading forcings timeseries data as xarray.Dataset
        streamflow_data_loader: AbstractDatasetLoader
            Instance for loading streamflow timeseries data as xarray.Dataset
        forcings_vars: list
            List of forcings variable names. The variables will be used for subsetting the forcings dataset.
        streamflow_vars: list
            List of streamflow variable names. The variables will be used for subsetting the streamflow dataset.
        """
        self.__forcings_dataloader = forcings_data_loader
        self.__streamflow_dataloader = streamflow_data_loader
        self.__forcings_variables = forcings_vars
        self.__streamflow_variables = streamflow_vars

    @classmethod
    def from_config(cls, data_cfg: config.DataConfig):
        """
        Creates a BasinDataLoader instance from configurations.

        Parameters
        ----------
        data_cfg: config.DataConfig
            DataConfig that holds data configuration parameters, such as basins, data directories and variables which
            will be used for loading and subsetting the forcings and streamflow datasets.

        Returns
        -------
        HydroDataLoader
            A HydroDataLoader instance

        """
        forcings_dataloader = forcings_factory(data_cfg.forcings_cfg.data_type, data_cfg.basins,
                                               data_cfg.forcings_cfg.data_dir, data_cfg.forcings_cfg.variables)
        streamflow_dataloader = streamflow_factory(data_cfg.streamflow_cfg.data_type, data_cfg.basins,
                                                   data_cfg.streamflow_cfg.data_dir, data_cfg.streamflow_cfg.variables)
        return cls(forcings_dataloader, streamflow_dataloader,
                   data_cfg.forcings_cfg.variables, data_cfg.streamflow_cfg.variables)

    @property
    def forcings_variables(self):
        return self.__forcings_variables

    @property
    def streamflow_variables(self):
        return self.__streamflow_variables

    def load_single_dataset(self, start_date: str, end_date: str, basin: str):
        ds_forcings = self.__forcings_dataloader.load_single_dataset(start_date, end_date, basin)
        ds_streamflow = self.__streamflow_dataloader.load_single_dataset(start_date, end_date, basin)
        ds_timeseries = xr.merge([ds_forcings, ds_streamflow], join="left")

        return dataset.HydroDataset(ds_timeseries, self.forcings_variables, self.streamflow_variables,
                                    start_date, end_date)

    def load_full_dataset(self, start_date: str, end_date: str, as_dask: bool = False):
        ds_forcings = self.__forcings_dataloader.load_full_dataset(start_date, end_date, as_dask)
        ds_streamflow = self.__streamflow_dataloader.load_full_dataset(start_date, end_date, as_dask)
        ds_timeseries = xr.merge([ds_forcings, ds_streamflow], join="left")

        return dataset.HydroDataset(ds_timeseries, self.forcings_variables, self.streamflow_variables,
                                    start_date, end_date)


def streamflow_to_metric(streamflow):
    return streamflow * 0.028316846592  # [m³/s]


def normalize_streamflow(streamflow, area: float):
    streamflow = streamflow * 0.028316846592  # [m³/s]
    streamflow = streamflow / area  # [m/s]
    streamflow = streamflow * 86400  # [m/d]
    return streamflow * 10 ** 3  # [mm/d]


def forcings_factory(forcings_type: str, basins: list, forcings_dir: str, forcings_vars: list) -> AbstractDatasetLoader:
    if forcings_type == "camels-us":
        return CamelsUsForcingsDataLoader(basins, forcings_dir, forcings_vars)
    if forcings_type == "daymet":
        return DaymetDataLoader(basins, forcings_dir, forcings_vars)
    raise ValueError("No forcings data loader exists for the specified dataset type '{}.".format(forcings_type))


def streamflow_factory(streamflow_type: str, basins: list, streamflow_dir: str,
                       streamflow_vars: list) -> AbstractDatasetLoader:
    if streamflow_type == "camels-us":
        return CamelsUsStreamflowDataLoader(basins, streamflow_dir, streamflow_vars)
    raise ValueError("No streamflow data loader exists for the specified dataset type '{}.".format(streamflow_type))
