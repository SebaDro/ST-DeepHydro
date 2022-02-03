import math
import numpy as np
import pandas as pd
import unittest
import xarray as xr

from libs import processing


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


class TestCustomTimeseriesGenerator(unittest.TestCase):
    def setUp(self):
        self.basin_1 = "123"
        df_forcings_1 = create_forcings_data()
        df_streamflow_1 = create_streamflow_data()
        df_merged_1 = df_forcings_1.join(df_streamflow_1, how="outer")

        ds_timeseries_1 = xr.Dataset.from_dataframe(df_merged_1)
        ds_timeseries_1 = ds_timeseries_1.rename({"index": "time"})
        ds_timeseries_1 = ds_timeseries_1.assign_coords({"basin": self.basin_1})

        self.basin_2 = "456"
        df_forcings_2 = create_forcings_data()
        df_streamflow_2 = create_streamflow_data()
        df_merged_2 = df_forcings_2.join(df_streamflow_2, how="outer")

        ds_timeseries_2 = xr.Dataset.from_dataframe(df_merged_2)
        ds_timeseries_2 = ds_timeseries_2.rename({"index": "time"})
        ds_timeseries_2 = ds_timeseries_2.assign_coords({"basin": self.basin_2})

        self.ds_timeseries = xr.concat([ds_timeseries_1, ds_timeseries_2], dim="basin")

    def test_timeseries_generation(self):
        batch_size = 6
        timesteps = 4
        offset = 1
        feature_cols = ["temp", "prcp"]
        target_cols = ["streamflow"]

        lag = timesteps + offset - 1

        gen = processing.CustomTimeseriesGenerator(self.ds_timeseries, batch_size, timesteps, offset, feature_cols,
                                                   target_cols, False)

        # First, check the number of batches
        # 20 timesteps with a time lag substracted for each of two basins
        exp_batches = math.ceil((20 - lag) * 2 / batch_size)
        self.assertEqual(exp_batches, len(gen))

        # Then check first, middle and last batches
        # First batch
        batch = 0
        inputs, targets = gen[batch]
        x1 = self.ds_timeseries.sel(basin=self.basin_1).to_array().values.transpose()[0:timesteps, 0:2]
        y1 = self.ds_timeseries.sel(basin=self.basin_1).to_array().values.transpose()[timesteps, 2]
        x2 = inputs[0]
        y2 = targets[0]
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

        # Middle batch for both basins
        batch = 2
        inputs, targets = gen[batch]
        # Last batch of basin 1
        x1 = self.ds_timeseries.sel(basin=self.basin_1).to_array().values.transpose()[-(timesteps+offset):-1, 0:2]
        y1 = self.ds_timeseries.sel(basin=self.basin_1).to_array().values.transpose()[-1, 2]
        x2 = inputs[3]
        y2 = targets[3]
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)
        # First batch of basin 2
        x1 = self.ds_timeseries.sel(basin=self.basin_2).to_array().values.transpose()[0:timesteps, 0:2]
        y1 = self.ds_timeseries.sel(basin=self.basin_2).to_array().values.transpose()[timesteps, 2]
        x2 = inputs[4]
        y2 = targets[4]
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

        # Last batch
        batch = 5
        inputs, targets = gen[batch]

        x1 = self.ds_timeseries.sel(basin=self.basin_2).to_array().values.transpose()[-(timesteps+offset):-1, 0:2]
        y1 = self.ds_timeseries.sel(basin=self.basin_2).to_array().values.transpose()[-1, 2]
        x2 = inputs[1]
        y2 = targets[1]
        # Last batch should contain only 2 elements
        self.assertEqual(2, len(inputs))
        np.testing.assert_array_equal(x1, x2)

    def test_timeseries_generation_with_nan(self):
        batch_size = 6
        timesteps = 4
        offset = 1
        feature_cols = ["temp", "prcp"]
        target_cols = ["streamflow"]

        lag = timesteps + offset - 1

        # Note that the generator does not consider create input/target pairs for targets with NaN values.
        # As a result the first two batches contains only inputs and targets for basin 1 and the other two batches
        # for basin 2
        gen = processing.CustomTimeseriesGenerator(self.ds_timeseries, batch_size, timesteps, offset, feature_cols,
                                                   target_cols, True)

        # First, check the number of batches
        # 20 timesteps with a time lag substracted for each of two basins
        exp_batches = math.ceil((20 - lag - 4) * 2 / batch_size)
        self.assertEqual(exp_batches, len(gen))

        # Then check first, middle and last batches
        # First batch
        batch = 0
        inputs, targets = gen[batch]
        x1 = self.ds_timeseries.sel(basin=self.basin_1).to_array().values.transpose()[0:timesteps, 0:2]
        y1 = self.ds_timeseries.sel(basin=self.basin_1).to_array().values.transpose()[timesteps, 2]
        x2 = inputs[0]
        y2 = targets[0]
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

        # Middle batch for both basins
        batch = 1
        inputs, targets = gen[batch]
        # Last batch of basin 1.
        x1 = self.ds_timeseries.sel(basin=self.basin_1).to_array().values.transpose()[-(timesteps+offset):-1, 0:2]
        y1 = self.ds_timeseries.sel(basin=self.basin_1).to_array().values.transpose()[-1, 2]
        x2 = inputs[5]
        y2 = targets[5]
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

        # First batch of basin 2
        batch = 2
        inputs, targets = gen[batch]
        x1 = self.ds_timeseries.sel(basin=self.basin_2).to_array().values.transpose()[0:timesteps, 0:2]
        y1 = self.ds_timeseries.sel(basin=self.basin_2).to_array().values.transpose()[timesteps, 2]
        x2 = inputs[0]
        y2 = targets[0]
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

        # Last batch
        batch = 3
        inputs, targets = gen[batch]

        x1 = self.ds_timeseries.sel(basin=self.basin_2).to_array().values.transpose()[-(timesteps+offset):-1, 0:2]
        y1 = self.ds_timeseries.sel(basin=self.basin_2).to_array().values.transpose()[-1, 2]
        x2 = inputs[5]
        y2 = targets[5]
        # Last batch should contain 6 elements
        self.assertEqual(6, len(inputs))
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)
