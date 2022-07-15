import logging
import pandas as pd
import xarray
import xarray as xr

from stdeephydro import config
from stdeephydro import dataset
from stdeephydro import ioutils

logger = logging.getLogger(__name__)


class AbstractDatasetLoader:
    """
    Abstract dataset loader base class, which should be used to implement subclasses for loading certain datasets.

    Parameters
    ----------
    data_dir: str
        Path to the root dir that contains the datasets
    variables: list of str
        List of names for variables that should be loaded as part of the dataset
    basins: list of str
        List of IDs that indicate for which basins the dataset should be loaded. If not set, datasets for all basins
        within the root data_dir will be loaded
    """

    def __init__(self, data_dir: str, variables: list, basins: list = None):
        self.__data_dir = data_dir
        self.__variables = variables
        self.__basins = basins

    @classmethod
    def from_config(cls, data_cfg: config.DataTypeConfig, basins):
        return cls(data_cfg.data_dir, data_cfg.variables, basins)

    @property
    def data_dir(self):
        return self.__data_dir

    @property
    def variables(self):
        return self.__variables

    @property
    def basins(self):
        return self.__basins

    def load_full_dataset(self, start_date: str, end_date: str) -> xr.Dataset:
        pass

    def load_single_dataset(self, start_date: str, end_date: str, basin: str) -> xr.Dataset:
        pass

    def load_joined_dataset(self, start_date: str, end_date: str) -> xr.Dataset:
        pass


class CamelsUsStreamflowDataLoader(AbstractDatasetLoader):
    """
    Data loader for streamflow timeseries data from the CAMELS-US dataset

    This class loads CAMELS-US streamflow timeseries data as xarray.Dataset. For this purpose, the data loader assumes
    the standard CAMELS-US folder structure when downloading the NCAR CAMELS-US dataset. This dataset contains
    streamflow timeseries as separate files for each basin within the directory
    'basin_dataset_public_v1p2/usgs_streamflow'.

    The resulting xarray.Dataset contains the streamflow variable with dimensions (basin, time).

    Parameters
    ----------
    data_dir: str
        Path to the 'usgs_streamflow' CAMELS-US directory
    variables: list of str
        List of CAMELS-US variables to load (i.e. only 'streamflow')
    basins:
        List of basin IDs
    forcings_dir:
        Path to a CAMELS-US forcings directory. If set, basin attributes will be read from the forcings files to
        normalize streamflow by using the basin area.
    as_dask:
        Indicates whether to load the dataset as Dask Array. Should be set to True if datasets does not fit into memory.
        Note, this is still an experimental feature, since timeseries window generation is not optimized for Dask
        Arrays, so far.
    """

    def __init__(self, data_dir: str, variables: list, basins: list = None, forcings_dir: str = None,
                 as_dask: bool = False):
        super().__init__(data_dir, variables, basins)
        self.__forcings_dir = forcings_dir
        self.__as_dask = as_dask

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

    def load_full_dataset(self, start_date: str, end_date: str) -> xr.Dataset:
        """
        Loads Camels-US streamflow timeseries data as either as xarray.Dataset for the specified start and end date.
        The method will load streamflow data for all basins that have been specified for instantiating this
        data loader.

        Note, that streamflow will be normalized if the data loader has been initialized with a given forcings directory.
        This means streamflow will be divided by the basin area and converted into meters per day. For loading basin
        area information you have to instantiate this data loader by additionally passing the forcings directory.

        Parameters
        ----------
        start_date: str
            String that represents a date. It will be used as start date for subsetting the timeseries datasets.
        end_date: str
            String that represents a date. It will be used as end date for subsetting the timeseries datasets.

        Returns
        -------
        xarray.Dataset:
            A time and basin indexed xarray.Dataset for streamflow timeseries data.

        """
        ds_list = []
        for basin in self.basins:
            if self.__forcings_dir is None:
                ds = self._load_xarray_dataset(start_date, end_date, basin, False)
            else:
                ds = self._load_xarray_dataset(start_date, end_date, basin, True)
            if self.__as_dask:
                ds = ds.chunk()
            ds_list.append(ds)
        ds_timeseries = xr.concat(ds_list, dim="basin")

        if self.__as_dask:
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
    """
    Data loader for forcings timeseries data from the CAMELS-US dataset

    This class loads CAMELS-US forcings timeseries data as xarray.Dataset. For this purpose, the data loader assumes
    the standard CAMELS-US folder structure when downloading the NCAR CAMELS-US dataset. This dataset contains
    forcings timeseries from three different sources (Daymet, Maurer, NLDAS) as separate files for each basin. Forcing
    files for each source are stored within separate subdirectories of 'basin_dataset_public_v1p2/basin_mean_forcing'.

    The resulting xarray.Dataset contains the forcings variables with dimensions (basin, time).

    Parameters
    ----------
    data_dir: str
        Path to the forcings CAMELS-US subdirectory (daymet, maurer or nldas)
    variables: list of str
        List of CAMELS-US forcing variables to load
    basins:
        List of basin IDs
    as_dask:
        Indicates whether to load the dataset as Dask Array. Should be set to True if datasets does not fit into memory.

    Notes
    -----
    Loading datasets as Dask Array is still an experimental feature. Some dataset processing routines, such as
    timeseries window generation, are not optimized for Dask Arrays. Therefore, the run_training command line script
    does not support this option.
    """

    def __init__(self, data_dir: str, variables: list, basins: list = None, as_dask: bool = False):
        super().__init__(data_dir, variables, basins)
        self.__as_dask = as_dask

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

    def load_full_dataset(self, start_date: str, end_date: str) -> xr.Dataset:
        """
        Loads Camels-US forcings timeseries data as xarray.Dataset for the specified start and end date. The method will
        load forcings data for all basins that have been specified for instantiating this data loader.

        Parameters
        ----------
        start_date: str
            String that represents a date. It will be used as start date for subsetting the timeseries datasets.
        end_date: str
            String that represents a date. It will be used as end date for subsetting the timeseries datasets.

        Returns
        -------
        xarray.Dataset:
            A time and basin indexed xarray.Dataset for forcings timeseries data.

        """
        ds_list = []
        for basin in self.basins:
            ds = self._load_xarray_dataset(start_date, end_date, basin)
            if self.__as_dask:
                ds = ds.chunk()
            ds_list.append(ds)
        ds_timeseries = xr.concat(ds_list, dim="basin")

        if self.__as_dask:
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
    """
    Data loader for raster-based and aggregated Daymet forcings stored in NetCDF file format.

    The data loader loads ORNL DAAC Daymet NetCDF data as xarray.Dataset. For this purpose the data loader expects that
    separate NetCDF files for each basin exists within a root data_dir. Each NetCDF file must contain a unique basin ID
    as part of the filename. Discovery of NetCDF files for specified basins will be performed using the pattern
    '{data_dir}/*{basin}*.nc', i.e. the basin ID has to be present in any file name within the directory.

    This data loader is able to read temporal distributed NetCDF data as well as spatio-temporally distributed data.
    Temporal distributed NetCDF files must contain forcing variables, which are only indexed by 'time' (e.g. basin
    aggregated forcings). Spatio-temporally NetCDF files must contain forcing variables which are indexed by 'time',
    'x' and 'y'.

    The resulting xarray.Dataset contains Daymet variables with either with dimensions (basin, time) for aggregated
    forcings or (basin, time, x, y) for raster-based forcings.

    Parameters
    ----------
    data_dir: str
        Path to the data directory that
    variables: list of str
        List of CAMELS-US forcing variables to load
    basins:
        List of basin IDs
    from_zarr: bool
        Indicates that datasets should be loaded from a Zarr store
    """

    def __init__(self, data_dir: str, variables: list, basins: list = None, from_zarr: bool = False):
        super().__init__(data_dir, variables, basins)
        self.__from_zarr = from_zarr

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
        ds_timeseries = self._discover_and_load_xarray_dataset(start_date, end_date, basin)

        return ds_timeseries

    def load_joined_dataset(self, start_date: str, end_date: str) -> xr.Dataset:
        """
        Loads raster-based Daymet forcings from the data_dir for the specified start and end date as a joined xarray.Dataset.
        If from_zarr is set to be true, it is assumed that data_dir points to a Zarr store and Daymet forcings will be
        loaded as Dask Array. Otherwise, all single NetCDF forcings files will be discovered from data_dir and loaded
        as Dask Array.

        The xarray.Dataset is indexed by time, and a spatial dimension by means of 'x' and 'y' coordinates.

        The dataset is meant to be valid for multiple basins e.g., if streamflow should be predicted for multiple
        basins, the joined Daymet forcings dataset covers all basins.

        Notes
        -----
        Loading joined data is still an experimental feature and may not be supported across the whole package.
        Therefore, it can't be used with the run_training command line script.

        Parameters
        ----------
        start_date: str
            String that represents a date. It will be used as start date for subsetting the timeseries datasets.
        end_date: str
            String that represents a date. It will be used as end date for subsetting the timeseries datasets.

        Returns
        -------
        xarray.Dataset
            A xarray.Dataset that holds forcing and streamflow variables.

        """
        if self.__from_zarr:
            ds_forcings = ioutils.load_forcings_daymet_2d_from_zarr(self.data_dir)
        else:
            daymet_files = ioutils.discover_daymet_files(self.data_dir, self.variables)
            ds_forcings = ioutils.load_multiple_forcings_daymet_2d(daymet_files)
        ds_forcings = ds_forcings.sel(time=slice(start_date, end_date))[self.variables]
        ds_forcings["time"] = ds_forcings.indexes["time"].normalize()
        return ds_forcings

    def _discover_and_load_xarray_dataset(self, start_date: str, end_date: str, basin: str) -> xr.Dataset:
        forcings_path = ioutils.discover_single_daymet_file_for_basin(self.data_dir, basin)
        return self._load_xarray_dataset(forcings_path, start_date, end_date, basin)

    def _load_xarray_dataset(self, path: str, start_date: str, end_date: str, basin: str = None) -> xr.Dataset:
        if self.__from_zarr:
            ds_forcings = ioutils.load_forcings_daymet_2d_from_zarr(path)
        else:
            ds_forcings = ioutils.load_forcings_daymet_2d(path)
        ds_forcings = ds_forcings.sel(time=slice(start_date, end_date))[self.variables]
        if basin is not None:
            ds_forcings = ds_forcings.assign_coords({"basin": basin})
            ds_forcings = ds_forcings.expand_dims("basin")
        ds_forcings["time"] = ds_forcings.indexes["time"].normalize()

        return ds_forcings


class HydroDataLoader:
    """
    Data loader class for loading forcings and streamflow data from arbitrary data sources as combined
    dataset.HydroDataset.

    This class holds certain forcings and streamflow data loader instances in order to load hydro-meteorological
    timeseries data as combined xarray.Datasets which comprises, basin indexed forcings and streamflow timeseries data.

    Parameters
    ----------
    forcings_data_loader: AbstractDatasetLoader
        Instance for loading forcings timeseries data as xarray.Dataset
    streamflow_data_loader: AbstractDatasetLoader
        Instance for loading streamflow timeseries data as xarray.Dataset
    forcings_vars: list
        List of forcings variable names. The variables will be used for subsetting the forcings dataset.
    streamflow_var: str
        Streamflow variable name. The variable will be used for subsetting the streamflow dataset.
    """
    def __init__(self, forcings_data_loader: AbstractDatasetLoader, streamflow_data_loader: AbstractDatasetLoader,
                 forcings_vars: list, streamflow_var: str):
        self.__forcings_dataloader = forcings_data_loader
        self.__streamflow_dataloader = streamflow_data_loader
        self.__forcings_variables = forcings_vars
        self.__streamflow_variable = streamflow_var

    @classmethod
    def from_config(cls, basins_file, forcings_cfg: config.DataTypeConfig, streamflow_cfg: config.DataTypeConfig):
        """
        Creates a HydroDataLoader instance from configurations.

        Note, that although the configuration file may contain a list of streamflow variables, the HydroDataLoader only
        support a single streamflow variable. Thus, only the first variable of the list will be considered.

        Parameters
        ----------
        basins_file
        forcings_cfg
        streamflow_cfg

        Returns
        -------
        HydroDataLoader
            A HydroDataLoader instance

        """
        with open(basins_file, 'r') as file:
            basins = [line.strip() for line in file.readlines()]
        forcings_dl = forcings_factory(forcings_cfg.data_type, basins, forcings_cfg.data_dir,
                                       forcings_cfg.variables)
        streamflow_dl = streamflow_factory(streamflow_cfg.data_type, basins, streamflow_cfg.data_dir,
                                           streamflow_cfg.variables)
        if len(streamflow_cfg.variables) > 0:
            logger.warning(f"Configuration contains {len(streamflow_cfg.variables)} streamflow variables,"
                           f" but only one is supported for training. Therefore, only the first variable"
                           f" '{streamflow_cfg.variables[0]}' will be considered.")
        return cls(forcings_dl, streamflow_dl, forcings_cfg.variables, streamflow_cfg.variables[0])

    @property
    def forcings_variables(self):
        return self.__forcings_variables

    @property
    def streamflow_variable(self):
        return self.__streamflow_variable

    def load_single_dataset(self, start_date: str, end_date: str, basin: str) -> dataset.HydroDataset:
        """
        Uses the forcings and streamflow DatasetsLoader to load a single timeseries dataset as xarray.Dataset which
        contains merged forcings and streamflow data.

        The resulting xarray.Dataset will be wrapped by a dataset.HydroDataset.

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
        dataset.HydroDataset
            Dataset which wraps forcings and streamflow timeseries data

        """
        ds_forcings = self.__forcings_dataloader.load_single_dataset(start_date, end_date, basin)
        ds_streamflow = self.__streamflow_dataloader.load_single_dataset(start_date, end_date, basin)
        ds_timeseries = xr.merge([ds_forcings, ds_streamflow], join="left")

        return dataset.HydroDataset(ds_timeseries, self.forcings_variables, self.streamflow_variable,
                                    start_date, end_date)

    def load_full_dataset(self, start_date: str, end_date: str):
        """
        Uses the forcings and streamflow DataLoaders to load forcings and streamflow for the specified start and
        end date as a joined xarray.Dataset. The method will load forcings and streamflow data for all basins that have
        been specified for instantiating the data loaders.

        The resulting xarray.Dataset will be wrapped by a dataset.HydroDataset.

        Parameters
        ----------
        start_date: str
            String that represents a date. It will be used as start date for subsetting the timeseries datasets.
        end_date: str
            String that represents a date. It will be used as end date for subsetting the timeseries datasets.

        Returns
        -------
        dataset.HydroDataset
            Dataset which wraps forcings and streamflow timeseries data

        """
        ds_forcings = self.__forcings_dataloader.load_full_dataset(start_date, end_date)
        ds_streamflow = self.__streamflow_dataloader.load_full_dataset(start_date, end_date)
        ds_timeseries = xr.merge([ds_forcings, ds_streamflow], join="left")

        return dataset.HydroDataset(ds_timeseries, self.forcings_variables, self.streamflow_variable,
                                    start_date, end_date)

    def load_joined_dataset(self, start_date: str, end_date: str):
        """
        Uses the forcings and streamflow DataLoaders to load forcings and streamflow for the specified
        start and end date as a joined xarray.Dataset. Forcings are loaded as a joined xarray.Datasets and streamflow as
        full xarray.Dataset and both merged.

        The forcings xarray.Dataset is indexed by time, and a spatial dimension by means of 'x' and 'y' coordinates.
        The streamflow xarray.Dataset is indexed by time, basin and a spatial dimension by means of 'x' and 'y'
        coordinates.

        The dataset is meant to be valid for multiple basins e.g., if streamflow should be predicted for multiple basins,
        the joined Daymet forcings dataset covers all basins.

        The resulting xarray.Dataset will be wrapped by a dataset.HydroDataset.

        Parameters
        ----------
        start_date: str
            String that represents a date. It will be used as start date for subsetting the timeseries datasets.
        end_date: str
            String that represents a date. It will be used as end date for subsetting the timeseries datasets.

        Returns
        -------
        dataset.HydroDataset
            Dataset which wraps forcings and streamflow timeseries data

        """
        ds_forcings = self.__forcings_dataloader.load_joined_dataset(start_date, end_date)
        ds_streamflow = self.__streamflow_dataloader.load_full_dataset(start_date, end_date)
        ds_timeseries = xr.merge([ds_forcings, ds_streamflow], join="left")

        return dataset.HydroDataset(ds_timeseries, self.forcings_variables, self.streamflow_variable,
                                    start_date, end_date)


def streamflow_to_metric(streamflow: xarray.DataArray) -> xarray.DataArray:
    """
    Converts streamflow from cubic feet per second to cubic meters per second

    Parameters
    ----------
    streamflow: xarray.DataArray
        Streamflow timeseries in cubic feet per second

    Returns
    -------
    xarray.DataArray
        Streamflow timeseries in cubic meters per second
    """
    return streamflow * 0.028316846592  # [m³/s]


def normalize_streamflow(streamflow: xarray.DataArray, area: float):
    """
    Normalizes streamflow, which has unit cubic feet per second [ft³/s], by dividing it by the basin area [m²] and
    converting into millimeters per day (mm/d).

    Parameters
    ----------
    streamflow: xarray.DataArray
        Streamflow timeseries in cubic feet per second [ft³/s]
    area: float
        Basin area in square meters [m²]

    Returns
    -------
    Streamflow timeseries in meters per day (mm/d)

    """
    streamflow = streamflow * 0.028316846592  # [m³/s]
    streamflow = streamflow / area  # [m/s]
    streamflow = streamflow * 86400  # [m/d]
    return streamflow * 10 ** 3  # [mm/d]


def forcings_factory(forcings_type: str, basins: list, forcings_dir: str, forcings_vars: list) -> AbstractDatasetLoader:
    """
    Creates a certain DataLoader instance for loading forcings timeseries for the specified forcings type.

    Parameters
    ----------
    forcings_type: str
        Forcings type. Supported: 'camels-us', 'daymet'
    basins: List of str
        List of basin IDs to load forcings for
    forcings_dir: str
        Path to the directory that contains forcings data files
    forcings_vars: List of str
        List of variable names to load forcings for

    Returns
    -------
    AbstractDatasetLoader
        A DatasetLoader instance that load forcings data

    """
    if forcings_type == "camels-us":
        return CamelsUsForcingsDataLoader(forcings_dir, forcings_vars, basins)
    if forcings_type == "daymet":
        return DaymetDataLoader(forcings_dir, forcings_vars, basins)
    raise ValueError("No forcings data loader exists for the specified dataset type '{}.".format(forcings_type))


def streamflow_factory(streamflow_type: str, basins: list, streamflow_dir: str, streamflow_vars: list) -> AbstractDatasetLoader:
    """
    Creates a certain DataLoader instance for loading streamflow timeseries for the specified streamflow type.

    Parameters
    ----------
    streamflow_type: str
        Streamflow type. Supported: 'camels-us'
    basins: List of str
        List of basin IDs to load streamflow for
    streamflow_dir: str
        Path to the directory that contains streamflow data files
    streamflow_vars: List of str
        List of variable names to load streamflow for

    Returns
    -------
    AbstractDatasetLoader
        A DatasetLoader instance that loads streamflow data

    """
    if streamflow_type == "camels-us":
        return CamelsUsStreamflowDataLoader(streamflow_dir, streamflow_vars, basins)
    raise ValueError("No streamflow data loader exists for the specified dataset type '{}.".format(streamflow_type))
