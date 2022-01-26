from libs import dataset
import unittest
import pandas as pd
import numpy as np


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


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.__streamflow_data = create_streamflow_data()
        self.__forcings_data = create_forcings_data()

    def test_factory(self):
        data = dataset.factory(self.__forcings_data, self.__streamflow_data, "camels-us", ["temp"],  ["streamflow"])

        self.assertIsInstance(data, dataset.CamelUSDataset)
        self.assertEqual(list(data.forcings), ["temp"])
        self.assertEqual(list(data.streamflow), ["streamflow"])
        self.assertEqual(len(data.forcings), 20)
        self.assertEqual(len(data.streamflow), 20)
        self.assertRaises(ValueError, dataset.factory,  self.__forcings_data, self.__streamflow_data, "non-valid-type",
                          ["temp"], ["streamflow"])

        data = dataset.factory(self.__forcings_data, self.__streamflow_data, "camels-us", ["temp"], ["streamflow"],
                               "2021-01-03", "2021-01-06")
        self.assertEqual(len(data.forcings), 4)
        self.assertEqual(len(data.streamflow), 4)


class TestTimeseriesDataset(unittest.TestCase):
    def setUp(self):
        self.__streamflow_data = create_streamflow_data()
        self.__forcings_data = create_forcings_data()

    def test_normalize(self):
        data = dataset.TimeseriesDataset(self.__forcings_data.to_numpy(), self.__streamflow_data.to_numpy())
        min_input_indices = np.argmin(data.forcings_timeseries, axis=0)
        max_input_indices = np.argmax(data.forcings_timeseries, axis=0)
        min_target_indices = np.nanargmin(data.streamflow_timeseres, axis=0)
        max_target_indices = np.nanargmax(data.streamflow_timeseres, axis=0)
        data.normalize()

        np.testing.assert_array_equal(np.min(data.forcings_timeseries, axis=0), np.array([0, 0]))
        np.testing.assert_array_equal(np.max(data.forcings_timeseries, axis=0), np.array([1, 1]))
        self.assertEqual(data.forcings_timeseries[min_input_indices[0], 0], 0)
        self.assertEqual(data.forcings_timeseries[min_input_indices[1], 1], 0)
        self.assertEqual(data.forcings_timeseries[max_input_indices[0], 0], 1)
        self.assertEqual(data.forcings_timeseries[max_input_indices[1], 1], 1)

        self.assertEqual(data.streamflow_timeseres[min_target_indices[0], 0], 0)
        self.assertEqual(data.streamflow_timeseres[max_target_indices[0], 0], 1)
