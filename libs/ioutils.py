from typing import Tuple

import pandas as pd
import xarray as xr
import glob
import logging
import os
import datetime as dt


logger = logging.getLogger(__name__)


def discover_single_file_for_basin(data_dir: str, basin: str) -> str:
    """
    Discovers a single dataset file for the specified basin. Discovery will be performed using the pattern
    '{data_dir}/*{basin}*', i.e. the basin ID has to be present in any file name within the directory. Note, that
    basin id '123' e.g. will match the following file names: 123_streamflow.txt, 123.nc, 00123456.nc4,
    streamflow_123.csv. Be sure, that your file names are unique, otherwise only the first occurence will be returned.

    Parameters
    ----------
    data_dir: str
        The data directory used for discovering a dataset file related to the specified basin
    basin: str
        ID of the basin

    Returns
    -------
    str
        Path of the file, which is related to the specified basin

    """
    # TODO Think about more sophisticated file discovery using regex, such as (?<![0-9])basin(?![0-9])
    files = glob.glob(f"{data_dir}/**/*{basin}*", recursive=True)
    if len(files) == 0:
        raise FileNotFoundError(f"Can't find file for basin {basin} within directory {data_dir}.")
    if len(files) > 1:
        logger.warning(f"Found multiple files for basin {basin} within directory {data_dir}. "
                       f"First one found will be returned.")
    return files[0]


def discover_files_for_basins(data_dir: str, basins: list) -> dict:
    """

    Parameters
    ----------
    data_dir: str
        The data directory used for discovering dataset files related to the specified basins
    basins: list
        List of basin IDs

    Returns
    -------
    dict
        Dict that holds the dataset path to each basin

    """
    file_dict = {}
    for basin in basins:
        file = discover_single_file_for_basin(data_dir, basin)
        file_dict[basin] = file
    return file_dict


def discover_single_camels_us_forcings_file(data_dir: str, forcings_type: str, basin: str):
    """
    Discovers a single CAMELS-US forcing file by using the pattern '{data_dir}/**/{basin}_lump_{forcings_type}_forcing_leap.txt'.

    Parameters
    ----------
    data_dir: str
        Path to the CAMELS-US data directory for forcings.
    forcings_type: str
        Type of the forcings timeseries, i.e. one of 'daymet', 'maurer', or 'nldas'
    basin: str
        ID of the basin, the forcings file will be discovered for.

    Returns
    -------
    str
        Path to the discovered forcings file

    """
    type_dict = {"daymet": "cida", "maurer": "maurer", "nldas": "nldas"}
    if forcings_type in type_dict:
        files = glob.glob(f"{data_dir}/**/{basin}_lump_{type_dict[forcings_type]}_forcing_leap.txt", recursive=True)
        if len(files) == 0:
            raise FileNotFoundError(f"Can't find file for basin {basin} within directory {data_dir}.")
        if len(files) > 1:
            logger.warning(f"Found multiple files for basin {basin} within directory {data_dir}. "
                           f"First one found will be returned.")
    else:
        raise ValueError(f"Invalid forcings type `{forcings_type}` specified.")
    return files[0]


def discover_single_camels_us_streamflow_file(data_dir: str, basin: str):
    """
    Discovers a single CAMELS-US streamflow file by using the pattern '{data_dir}/**/{basin}_streamflow_qc.txt'.

    Parameters
    ----------
    data_dir: str
        Path to the CAMELS-US data directory for streamflow.
    basin: str
        ID of the basin, the streamflow file will be discovered for.

    Returns
    -------
    str
        Path to the discovered streamflow file

    """
    files = glob.glob(f"{data_dir}/**/{basin}_streamflow_qc.txt", recursive=True)
    if len(files) == 0:
        raise FileNotFoundError(f"Can't find file for basin {basin} within directory {data_dir}.")
    if len(files) > 1:
        logger.warning(f"Found multiple files for basin {basin} within directory {data_dir}. "
                       f"First one found will be returned.")
    return files[0]


def discover_multiple_camels_us_forcings_files(data_dir: str, forcings_type: str, basins: list = None):
    """
    Discovers multiple CAMELS-US forcing files. All files will be considered that follow the pattern
    '{data_dir}/**/*_lump_{forcings_type}_forcing_leap.txt'.

    Parameters
    ----------
    data_dir: str
        Path to the CAMELS-US data directory for forcings.
    forcings_type: str
        Type of the forcing timeseries, i.e. one of 'daymet', 'maurer', or 'nldas'
    basins: list
        List of basins, the forcings files will be discovered for. If 'None', all present files will be considered

    Returns
    -------
    list
        List of forcing file paths for the specified basins.

    """
    type_dict = {"daymet": "cida", "maurer": "maurer", "nldas": "nldas"}
    if forcings_type in type_dict:
        files = glob.glob(f"{data_dir}/**/*_lump_{type_dict[forcings_type]}_forcing_leap.txt", recursive=True)
        if basins is not None:
            files = [f for f in files if (any(basin == os.path.basename(f)[0:8] for basin in basins))]
    else:
        raise ValueError(f"Invalid forcings type `{forcings_type}` specified.")
    return files


def discover_multiple_camels_us_streamflow_files(data_dir: str, basins: list = None):
    """
    Discovers multiple CAMELS-US streamflow files. All files will be considered that follow the pattern
    '{data_dir}/**/*_streamflow_qc.txt'.

    Parameters
    ----------
    data_dir: str
        Path to the CAMELS-US data directory for streamflow
    basins: list
        List of basins, the streamflow files will be discovered for. If 'None', all present files will be considered.

    Returns
    -------
    list
        List of streamflow file paths for the specified basins.

    """
    files = glob.glob(f"{data_dir}/**/*_streamflow_qc.txt")
    if basins is not None:
        files = [f for f in files if (any(basin == os.path.basename(f)[0:8] for basin in basins))]
    return files


def load_forcings(path: str, ds_type: str):
    """
    Load a dataset that contains forcing data

    Parameters
    ----------
    path: str
        Path to the forcings dataset
    ds_type: str
        Type of dataset. One of {camels-us, daymet-2d}

    Returns
    -------
    Dataset contating forcings timeseries data

    """
    if ds_type == "camels-us":
        return load_forcings_camels_us(path)
    if ds_type == "daymet-2d":
        return load_forcings_daymet_2d(path)
    raise ValueError("Unsupported forcings dataset type '{}'".format(ds_type))


def load_forcings_camels_us(path: str) -> pd.DataFrame:
    """
    Loads CAMELS forcing data from raw text files

    Parameters
    ----------
    path: str
        Path to the raw text file containing forcing data for a certain basin

    Returns
    -------
    pd.DataFrame
        DataFrame containing DateTime indexed forcing data for a basin

    """
    colnames = pd.read_csv(path, sep=' ', skiprows=3, nrows=1, header=None)
    df = pd.read_csv(path, sep='\t', skiprows=4, header=None, decimal='.',
                     names=colnames.iloc[0, 3:])
    dates = df.iloc[:, 0]
    df = df.drop(columns=df.columns[0])
    df["date"] = pd.to_datetime(dates.str.split(expand=True)
                                .drop([3], axis=1)
                                .rename(columns={0: "year", 1: "month", 2: "day"}))
    df = df.set_index("date")
    return df


def load_forcings_gauge_metadata(path: str) -> Tuple[float, float, float]:
    """
    Loads gauge metadata from the header of a CAMELS-USE forcings file.

    Parameters
    ----------
    path: str
        Path to the forcings file.

    Returns
    -------
    tuple
        (gauge latitude, gauge elevation, basin area [m²])

    """
    with open(path, 'r') as file:
        latitude = float(file.readline())
        elevation = float(file.readline())
        area = float(file.readline())
    return latitude, elevation, area


def load_forcings_daymet_2d(path: str) -> xr.Dataset:
    """

    Parameters
    ----------
    path: str
        Path to a Daymet NetCDF dataset

    Returns
    -------
    xarray.Dataset
        Dataset hat contains two dimensional Daymet forcings data

    """
    with xr.open_dataset(path) as ds:
        return ds


def load_streamflow(path: str, ds_type: str):
    """
    Load streamflow data

    Parameters
    ----------
    path: str
        Path to a streamflow dataset
    ds_type: str
        Type of the streamflow dataset. One of {camels-us}

    Returns
    -------
    Dataset containing streamflow timeseries data

    """
    if ds_type == "camels-us":
        return load_streamflow_camels_us(path)
    raise ValueError("Unsupported streamflow dataset type '{}'".format(ds_type))


def load_streamflow_camels_us(path: str) -> pd.DataFrame:
    """
    Loads CAMELS streamflow data from raw text files

    Parameters
    ----------
    path: str
        Path to the raw text file containing streamflow data for a certain basin

    Returns
    -------
    pd.DataFrame
        DataFrame containing DateTime indexed streamflow data for a basin

    """
    df = pd.read_csv(path, delim_whitespace=True, header=None, decimal='.', na_values=["-999.00"],
                     names=["gauge_id", "year", "month", "day", "streamflow", "qc_flag"], dtype={"gauge_id": str})
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df.drop(columns=["year", "month", "day"]).set_index("date")

    return df


def load_camels_us_basin_physical_characteristics(path: str) -> pd.DataFrame:
    """
    Loads physical characteristics for CAMEL-US basins

    Parameters
    ----------
    path: str
        Path to the metadata file

    Returns
    -------
    pd.DataFrame
        DataFrame containing physical characteristics for CAMEL-US basins

    """
    return pd.read_csv(path, delim_whitespace=True, decimal='.', dtype={"BASIN_ID": str})


def load_camels_us_gauge_information(path: str) -> pd.DataFrame:
    """
    Loads gauge information metadata for CAMEL-US basins

    Parameters
    ----------
    path: str
        Path to the metadata file

    Returns
    -------
    pd.DataFrame
        DataFrame containing physical characteristics for CAMEL-US basins

    """
    return pd.read_csv(path, delim_whitespace=True, decimal='.', dtype={"HUC_02": str, "GAGE_ID": str})


def create_out_dir(output: str, name: str) -> str:
    """
    Creates a directory in the given output folder for a given name and the current timestamp that can be used for
    storing outputs such as logs, monitoring metrics or saved models

    Parameters
    ----------
    output: str
        Output directory
    name: str
        Name of the current run

    Returns
    -------
    str
        Path of the created directory

    """
    timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S%z")
    out_dir = os.path.join(output, f"{timestamp}_{name}")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        logger.info(f"Created directory {out_dir} for storing outputs.")
    else:
        logger.warning(f"Directory {out_dir} already exists.")
    return out_dir
