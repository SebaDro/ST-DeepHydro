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


def create_forcings_data():
    dates = pd.date_range("2021", periods=20)
    temp_values = np.random.uniform(low=0, high=10, size=(20,))
    prcp_values = np.random.uniform(low=0, high=10, size=(20,))

    df = pd.DataFrame({"temp": temp_values, "prcp": prcp_values}, index=dates)
    return df


class LumpedDataset(unittest.TestCase):
    def setUp(self):
        df_forcings = create_forcings_data()
        df_streamflow = create_streamflow_data()

        df_merged = df_forcings.join(df_streamflow, how="outer")

        ds_timeseries = xr.Dataset.from_dataframe(df_merged)
        ds_timeseries = ds_timeseries.rename({"index": "time"})
        ds_timeseries = ds_timeseries.assign_coords({"basin": "123"})

        self.__dataset = dataset.LumpedDataset(ds_timeseries, ["temp", "prcp"], ["streamflow"],
                                               "2021-01-01", "2021-01-20")

    def test_normalize_single(self):
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

