import math
import numpy as np
import pandas as pd
import unittest
import xarray as xr

from libs import dataset
from libs import generator


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


def create_one_dimensional_dataset():
    basin_1 = "123"
    df_forcings_1 = create_forcings_data()
    df_streamflow_1 = create_streamflow_data()
    df_merged_1 = df_forcings_1.join(df_streamflow_1, how="outer")

    ds_timeseries_1 = xr.Dataset.from_dataframe(df_merged_1)
    ds_timeseries_1 = ds_timeseries_1.rename({"index": "time"})
    ds_timeseries_1 = ds_timeseries_1.assign_coords({"basin": basin_1})

    basin_2 = "456"
    df_forcings_2 = create_forcings_data()
    df_streamflow_2 = create_streamflow_data()
    df_merged_2 = df_forcings_2.join(df_streamflow_2, how="outer")

    ds_timeseries_2 = xr.Dataset.from_dataframe(df_merged_2)
    ds_timeseries_2 = ds_timeseries_2.rename({"index": "time"})
    ds_timeseries_2 = ds_timeseries_2.assign_coords({"basin": basin_2})

    return dataset.HydroDataset(xr.concat([ds_timeseries_1, ds_timeseries_2], dim="basin"),
                                feature_variables=["temp", "prcp"], target_variable="streamflow")


def create_two_dimensional_dataset():
    dates = pd.date_range("2021", periods=20)
    x = np.arange(0, 14)
    y = np.arange(0, 12)
    basins = ["123"]

    temp_data = np.random.uniform(low=0, high=10, size=(1, 20, 12, 14))
    prcp_data = np.random.uniform(low=0, high=10, size=(1, 20, 12, 14))
    streamflow_data = np.random.uniform(low=0, high=10, size=(1, 20))
    na_indices = [9, 10, 11, 18]
    streamflow_data[0, na_indices] = np.NaN

    temp_xr = xr.DataArray(temp_data, coords=[basins, dates, y, x], dims=["basin", "time", "y", "x"])
    prcp_xr = xr.DataArray(prcp_data, coords=[basins, dates, y, x], dims=["basin", "time", "y", "x"])
    streamflow_xr = xr.DataArray(streamflow_data, coords=[basins, dates], dims=["basin", "time"])

    return dataset.HydroDataset(xr.Dataset(dict(temp=temp_xr, prcp=prcp_xr, streamflow=streamflow_xr)),
                                feature_variables=["temp", "prcp"], target_variable="streamflow")


def create_two_dimensional_joined_dataset():
    dates = pd.date_range("2021", periods=20)
    x = np.arange(0, 14)
    y = np.arange(0, 12)
    basins = ["123"]

    temp_data = np.random.uniform(low=0, high=10, size=(20, 12, 14))
    prcp_data = np.random.uniform(low=0, high=10, size=(20, 12, 14))
    streamflow_data = np.random.uniform(low=0, high=10, size=(1, 20))
    na_indices = [9, 10, 11, 18]
    streamflow_data[0, na_indices] = np.NaN

    temp_xr = xr.DataArray(temp_data, coords=[dates, y, x], dims=["time", "y", "x"])
    prcp_xr = xr.DataArray(prcp_data, coords=[dates, y, x], dims=["time", "y", "x"])
    streamflow_xr = xr.DataArray(streamflow_data, coords=[basins, dates], dims=["basin", "time"])

    return dataset.HydroDataset(xr.Dataset(dict(temp=temp_xr, prcp=prcp_xr, streamflow=streamflow_xr)),
                                feature_variables=["temp", "prcp"], target_variable=["streamflow"])


class TestCustomTimeseriesGenerator(unittest.TestCase):
    def setUp(self):
        self.basin_1 = "123"
        self.basin_2 = "456"
        self.ds = create_one_dimensional_dataset()
        self.ds_2d = create_two_dimensional_dataset()
        self.ds_2d_joined = create_two_dimensional_joined_dataset()

    def test_get_input_shape_1d(self):
        batch_size = 6
        timesteps = 8
        offset = 1
        feature_cols = ["temp", "prcp"]
        target_col = "streamflow"

        gen = generator.CustomTimeseriesGenerator(self.ds.timeseries, batch_size, timesteps, offset, feature_cols,
                                                  target_col, False)
        exp_shape = (0, 8, 2)
        self.assertEqual(exp_shape, gen._get_input_shape())

    def test_get_input_shape_2d(self):
        batch_size = 6
        timesteps = 6
        offset = 1
        feature_cols = ["temp", "prcp"]
        target_col = "streamflow"

        gen = generator.CustomTimeseriesGenerator(self.ds_2d.timeseries, batch_size, timesteps, offset, feature_cols,
                                                  target_col, False)
        exp_shape = (0, 6, 12, 14, 2)
        self.assertEqual(exp_shape, gen._get_input_shape())

    def test_timeseries_generation(self):
        batch_size = 6
        timesteps = 4
        offset = 1
        feature_cols = ["temp", "prcp"]
        target_col = "streamflow"

        lag = timesteps + offset - 1

        gen = generator.CustomTimeseriesGenerator(self.ds.timeseries, batch_size, timesteps, offset, feature_cols,
                                                  target_col, False)

        # First, check the number of batches
        # Substract time lag values from 20 timesteps for each of two basins
        exp_batches = math.ceil((20 - lag) * 2 / batch_size)
        self.assertEqual(exp_batches, len(gen))

        # Then check first, middle and last batches
        # First batch
        batch = 0
        inputs, targets = gen[batch]
        x1 = self.ds.timeseries.sel(basin=self.basin_1).to_array().values.transpose()[0:timesteps, 0:2]
        y1 = self.ds.timeseries.sel(basin=self.basin_1).to_array().values.transpose()[timesteps, 2]
        x2 = inputs[0]
        y2 = targets[0]
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

        # Middle batch for both basins
        batch = 2
        inputs, targets = gen[batch]
        # Last batch of basin 1
        x1 = self.ds.timeseries.sel(basin=self.basin_1).to_array().values.transpose()[-(timesteps + offset):-1, 0:2]
        y1 = self.ds.timeseries.sel(basin=self.basin_1).to_array().values.transpose()[-1, 2]
        x2 = inputs[3]
        y2 = targets[3]
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)
        # First batch of basin 2
        x1 = self.ds.timeseries.sel(basin=self.basin_2).to_array().values.transpose()[0:timesteps, 0:2]
        y1 = self.ds.timeseries.sel(basin=self.basin_2).to_array().values.transpose()[timesteps, 2]
        x2 = inputs[4]
        y2 = targets[4]
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

        # Last batch
        batch = 5
        inputs, targets = gen[batch]

        x1 = self.ds.timeseries.sel(basin=self.basin_2).to_array().values.transpose()[-(timesteps + offset):-1, 0:2]
        y1 = self.ds.timeseries.sel(basin=self.basin_2).to_array().values.transpose()[-1, 2]
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
        target_col = "streamflow"

        lag = timesteps + offset - 1

        # Note that the generator does not consider create input/target pairs for targets with NaN values.
        # As a result the first two batches contains only inputs and targets for basin 1 and the other two batches
        # for basin 2
        gen = generator.CustomTimeseriesGenerator(self.ds.timeseries, batch_size, timesteps, offset, feature_cols,
                                                  target_col, True)

        # First, check the number of batches
        # Substract time lag and 4 NaN values from 20 timesteps for each of two basins
        nr_nan = 4
        exp_batches = math.ceil((20 - lag - nr_nan) * 2 / batch_size)
        self.assertEqual(exp_batches, len(gen))

        # Then check first, middle and last batches
        # First batch
        batch = 0
        inputs, targets = gen[batch]
        x1 = self.ds.timeseries.sel(basin=self.basin_1).to_array().values.transpose()[0:timesteps, 0:2]
        y1 = self.ds.timeseries.sel(basin=self.basin_1).to_array().values.transpose()[timesteps, 2]
        x2 = inputs[0]
        y2 = targets[0]
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

        # Middle batch for both basins
        batch = 1
        inputs, targets = gen[batch]
        # Last batch of basin 1.
        x1 = self.ds.timeseries.sel(basin=self.basin_1).to_array().values.transpose()[-(timesteps + offset):-1, 0:2]
        y1 = self.ds.timeseries.sel(basin=self.basin_1).to_array().values.transpose()[-1, 2]
        x2 = inputs[5]
        y2 = targets[5]
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

        # First batch of basin 2
        batch = 2
        inputs, targets = gen[batch]
        x1 = self.ds.timeseries.sel(basin=self.basin_2).to_array().values.transpose()[0:timesteps, 0:2]
        y1 = self.ds.timeseries.sel(basin=self.basin_2).to_array().values.transpose()[timesteps, 2]
        x2 = inputs[0]
        y2 = targets[0]
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

        # Last batch
        batch = 3
        inputs, targets = gen[batch]

        x1 = self.ds.timeseries.sel(basin=self.basin_2).to_array().values.transpose()[-(timesteps + offset):-1, 0:2]
        y1 = self.ds.timeseries.sel(basin=self.basin_2).to_array().values.transpose()[-1, 2]
        x2 = inputs[5]
        y2 = targets[5]
        # Last batch should contain 6 elements
        self.assertEqual(6, len(inputs))
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

    def test_timeseries_generation_2d(self):
        batch_size = 6
        timesteps = 4
        offset = 1
        feature_cols = ["temp", "prcp"]
        target_col = "streamflow"

        lag = timesteps + offset - 1
        shape = (
        timesteps, len(self.ds_2d.timeseries.indexes["y"]), len(self.ds_2d.timeseries.indexes["x"]), len(feature_cols))

        gen = generator.CustomTimeseriesGenerator(self.ds_2d.timeseries, batch_size, timesteps, offset, feature_cols,
                                                  target_col, False, False, shape)

        # First, check the number of batches
        # Substract time lag from 20 timesteps for only one basin
        exp_batches = math.ceil((20 - lag) / batch_size)
        self.assertEqual(exp_batches, len(gen))

        # Then check first, middle and last batches
        # First batch
        batch = 0
        inputs, targets = gen[batch]
        x1 = np.moveaxis(self.ds_2d.timeseries.sel(basin=self.basin_1)[feature_cols].to_array().values, 0, -1)[
             batch * batch_size:batch * batch_size + timesteps]
        y1 = np.moveaxis(self.ds_2d.timeseries.sel(basin=self.basin_1)[[target_col]].to_array().values, 0, -1)[
            batch * batch_size + timesteps]
        x2 = inputs[0]
        y2 = targets[0]
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

        # Middle batch
        batch = 1
        i = 2
        inputs, targets = gen[batch]
        x1 = np.moveaxis(self.ds_2d.timeseries.sel(basin=self.basin_1)[feature_cols].to_array().values, 0, -1)[
             i + batch * batch_size:i + batch * batch_size + timesteps]
        y1 = np.moveaxis(self.ds_2d.timeseries.sel(basin=self.basin_1)[[target_col]].to_array().values, 0, -1)[
            i + batch_size + timesteps]
        x2 = inputs[i]
        y2 = targets[i]
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

        # Last batch
        batch = 2
        i = 3
        inputs, targets = gen[batch]
        x1 = np.moveaxis(self.ds_2d.timeseries.sel(basin=self.basin_1)[feature_cols].to_array().values, 0, -1)[
             -(timesteps + offset):-1]
        y1 = np.moveaxis(self.ds_2d.timeseries.sel(basin=self.basin_1)[[target_col]].to_array().values, 0, -1)[-1]
        x2 = inputs[i]
        y2 = targets[i]
        # Last batch should contain only 4 elements
        self.assertEqual(4, len(inputs))
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

    def test_timeseries_generation_2d_without_nan(self):
        batch_size = 6
        timesteps = 4
        offset = 1
        feature_cols = ["temp", "prcp"]
        target_col = "streamflow"

        lag = timesteps + offset - 1
        shape = (timesteps, len(self.ds_2d.timeseries.indexes["y"]),
                 len(self.ds_2d.timeseries.indexes["x"]), len(feature_cols))

        # Note that the generator does not consider create input/target pairs for targets with NaN values.
        # As a result the number of batches is lower
        gen = generator.CustomTimeseriesGenerator(self.ds_2d.timeseries, batch_size, timesteps, offset, feature_cols,
                                                  target_col, True, False, shape)

        # First, check the number of batches
        # Substract time lag and 4 NaN values from 20 timesteps
        nr_nan = 4
        exp_batches = math.ceil((20 - lag - nr_nan) / batch_size)
        self.assertEqual(exp_batches, len(gen))

        # Then check first, middle and last batches
        # First batch
        batch = 0
        inputs, targets = gen[batch]
        x1 = np.moveaxis(self.ds_2d.timeseries.sel(basin=self.basin_1)[feature_cols].to_array().values, 0, -1)[
             batch * batch_size:batch * batch_size + timesteps]
        y1 = np.moveaxis(self.ds_2d.timeseries.sel(basin=self.basin_1)[[target_col]].to_array().values, 0, -1)[
            batch * batch_size + timesteps]
        x2 = inputs[0]
        y2 = targets[0]
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

        # Last batch (due to removal of NaN, the second batch is also the last batch)
        batch = 1
        inputs, targets = gen[batch]
        # Due to NaN values, now the target at index position 13 is first target value of the second batch
        i_target = 13
        x1 = np.moveaxis(self.ds_2d.timeseries.sel(basin=self.basin_1)[feature_cols].to_array().values, 0, -1)[
             i_target - timesteps:i_target]
        y1 = np.moveaxis(self.ds_2d.timeseries.sel(basin=self.basin_1)[[target_col]].to_array().values, 0, -1)[i_target]
        x2 = inputs[0]
        y2 = targets[0]
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

        # Check also last values of last batch
        x1 = np.moveaxis(self.ds_2d.timeseries.sel(basin=self.basin_1)[feature_cols].to_array().values, 0, -1)[
             -(timesteps + offset):-1]
        y1 = np.moveaxis(self.ds_2d.timeseries.sel(basin=self.basin_1)[[target_col]].to_array().values, 0, -1)[-1]
        x2 = inputs[5]
        y2 = targets[5]
        # Last batch should contain 6 elements
        self.assertEqual(6, len(inputs))
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)


    def test_timeseries_generation_2d_with_joined_output(self):
        batch_size = 6
        timesteps = 4
        offset = 1
        feature_cols = ["temp", "prcp"]
        target_col = "streamflow"

        lag = timesteps + offset - 1
        shape = (timesteps, len(self.ds_2d_joined.timeseries.indexes["y"]),
                 len(self.ds_2d_joined.timeseries.indexes["x"]), len(feature_cols))

        # Note that the generator does not consider create input/target pairs for targets with NaN values.
        # As a result the number of batches is lower
        gen = generator.CustomTimeseriesGenerator(self.ds_2d_joined.timeseries, batch_size, timesteps, offset,
                                                  feature_cols, target_col, True, True, shape)

        # First, check the number of batches
        # Substract time lag and 4 NaN values from 20 timesteps
        nr_nan = 4
        exp_batches = math.ceil((20 - lag - nr_nan) / batch_size)
        self.assertEqual(exp_batches, len(gen))

        # Then check first, middle and last batches
        # First batch
        batch = 0
        inputs, targets = gen[batch]
        x1 = np.moveaxis(self.ds_2d_joined.timeseries.sel(basin=self.basin_1)[feature_cols].to_array().values, 0, -1)[
             batch * batch_size:batch * batch_size + timesteps]
        y1 = np.moveaxis(self.ds_2d_joined.timeseries.sel(basin=self.basin_1)[[target_col]].to_array().values, 0, -1)[
            batch * batch_size + timesteps]
        x2 = inputs[0]
        y2 = targets[0]
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

        # Last batch (due to removal of NaN, the second batch is also the last batch)
        batch = 1
        inputs, targets = gen[batch]
        # Due to NaN values, now the target at index position 13 is first target value of the second batch
        i_target = 13
        x1 = np.moveaxis(self.ds_2d_joined.timeseries.sel(basin=self.basin_1)[feature_cols].to_array().values, 0, -1)[
             i_target - timesteps:i_target]
        y1 = np.moveaxis(self.ds_2d_joined.timeseries.sel(basin=self.basin_1)[[target_col]].to_array().values, 0, -1)[i_target]
        x2 = inputs[0]
        y2 = targets[0]
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

        # Check also last values of last batch
        x1 = np.moveaxis(self.ds_2d_joined.timeseries.sel(basin=self.basin_1)[feature_cols].to_array().values, 0, -1)[
             -(timesteps + offset):-1]
        y1 = np.moveaxis(self.ds_2d_joined.timeseries.sel(basin=self.basin_1)[[target_col]].to_array().values, 0, -1)[-1]
        x2 = inputs[5]
        y2 = targets[5]
        # Last batch should contain 6 elements
        self.assertEqual(6, len(inputs))
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)
