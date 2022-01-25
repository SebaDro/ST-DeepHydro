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
                     names=["gauge_id", "year", "month", "day", "streamflow", "qc_flag"])
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
