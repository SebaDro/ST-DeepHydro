from libs import dataset
import unittest
import pandas as pd
import numpy as np
import xarray as xr


def create_streamflow_data():
    dates = pd.date_range("2021", periods=20)
    values = np.random.uniform(low=0, high=10, size=(20,))
    na_indices = [9, 10, 11, 18]
    values[na_indices] = np.NaN

    df = pd.DataFrame({"streamflow": values}, index=dates)
    return df


def create_streamflow_data_for_fitting():
    dates = pd.date_range("2021", periods=20)
    values = np.arange(5, 101, 5)

    df = pd.DataFrame({"streamflow": values}, index=dates)
    return df


def create_forcings_data():
    dates = pd.date_range("2021", periods=20)
    temp_values = np.random.uniform(low=0, high=10, size=(20,))
    prcp_values = np.random.uniform(low=0, high=10, size=(20,))

    df = pd.DataFrame({"temp": temp_values, "prcp": prcp_values}, index=dates)
    return df


def create_forcings_data_for_fitting():
    dates = pd.date_range("2021", periods=20)
    prcp_values = np.arange(3, 61, 3)
    temp_values = np.arange(4, 81, 4)

    df = pd.DataFrame({"temp": temp_values, "prcp": prcp_values}, index=dates)
    return df


class TestLumpedDataset(unittest.TestCase):
    def setUp(self):
        df_forcings = create_forcings_data()
        df_streamflow = create_streamflow_data()

        df_merged = df_forcings.join(df_streamflow, how="outer")

        ds_timeseries = xr.Dataset.from_dataframe(df_merged)
        ds_timeseries = ds_timeseries.rename({"index": "time"})
        ds_timeseries = ds_timeseries.assign_coords({"basin": "123"})

        self.__dataset = dataset.LumpedDataset(ds_timeseries, ["temp", "prcp"], ["streamflow"],
                                               "2021-01-01", "2021-01-20")

    def test_normalize(self):
        min_prcp_index = self.__dataset.timeseries.prcp.argmin(dim=["time"])
        max_prcp_index = self.__dataset.timeseries.prcp.argmax(dim=["time"])
        min_streamflow_index = self.__dataset.timeseries.streamflow.argmin(dim=["time"])
        max_streamflow_index = self.__dataset.timeseries.streamflow.argmax(dim=["time"])

        self.__dataset.normalize()

        self.assertEqual(self.__dataset.timeseries.prcp.min().values, 0)
        self.assertEqual(self.__dataset.timeseries.temp.min().values, 0)
        self.assertEqual(self.__dataset.timeseries.streamflow.min().values, 0)
        self.assertEqual(self.__dataset.timeseries.prcp.max().values, 1)
        self.assertEqual(self.__dataset.timeseries.temp.max().values, 1)
        self.assertEqual(self.__dataset.timeseries.streamflow.max().values, 1)

        self.assertEqual(self.__dataset.timeseries.streamflow.isel(min_streamflow_index), 0)
        self.assertEqual(self.__dataset.timeseries.streamflow.isel(max_streamflow_index), 1)
        self.assertEqual(self.__dataset.timeseries.prcp.isel(min_prcp_index), 0)
        self.assertEqual(self.__dataset.timeseries.prcp.isel(max_prcp_index), 1)

    def test_normalize_single(self):
        min_prcp = self.__dataset.timeseries.prcp.min()
        max_prcp = self.__dataset.timeseries.prcp.max()
        min_temp = self.__dataset.timeseries.temp.min()
        max_temp = self.__dataset.timeseries.temp.max()
        min_streamflow = self.__dataset.timeseries.streamflow.min()
        max_streamflow = self.__dataset.timeseries.streamflow.max()

        min_prcp_fit = 3
        max_prcp_fit = 60
        min_temp_fit = 4
        max_temp_fit = 80
        min_streamflow_fit = 5
        max_streamflow_fit = 100

        forcings_fit = create_forcings_data_for_fitting()
        streamflow_fit = create_streamflow_data_for_fitting()
        ds_timeseries_fit = xr.Dataset.from_dataframe(forcings_fit.join(streamflow_fit, how="outer"))

        self.__dataset.normalize(ds_timeseries_fit.min(), ds_timeseries_fit.max())

        prcp_min_exp = (min_prcp - min_prcp_fit) / (max_prcp_fit - min_prcp_fit)
        self.assertEqual(prcp_min_exp, self.__dataset.timeseries.prcp.min().values)

        prcp_max_exp = (max_prcp - min_prcp_fit) / (max_prcp_fit - min_prcp_fit)
        self.assertEqual(prcp_max_exp, self.__dataset.timeseries.prcp.max().values)

        temp_min_exp = (min_temp - min_temp_fit) / (max_temp_fit - min_temp_fit)
        self.assertEqual(temp_min_exp, self.__dataset.timeseries.temp.min().values)

        temp_max_exp = (max_temp - min_temp_fit) / (max_temp_fit - min_temp_fit)
        self.assertEqual(temp_max_exp, self.__dataset.timeseries.temp.max().values)

        streamflow_min_exp = (min_streamflow - min_streamflow_fit) / (max_streamflow_fit - min_streamflow_fit)
        self.assertEqual(streamflow_min_exp, self.__dataset.timeseries.streamflow.min().values)

        streamflow_max_exp = (max_streamflow - min_streamflow_fit) / (max_streamflow_fit - min_streamflow_fit)
        self.assertEqual(streamflow_max_exp, self.__dataset.timeseries.streamflow.max().values)
