import numpy as np
import pandas as pd


class TimeseriesDataset:
    def __init__(self, input_timeseries, target_timeseries):
        self.__input_timeseries = input_timeseries
        self.__target_timeseries = target_timeseries
        self.__min_features = None
        self.__max_features = None
        self.__min_targets = None
        self.__max_targets = None

    @property
    def forcings_timeseries(self):
        return self.__input_timeseries

    @property
    def streamflow_timeseres(self):
        return self.__target_timeseries

    @property
    def min_features(self):
        return self.__min_features

    @property
    def max_features(self):
        return self.__max_features

    @property
    def min_targets(self):
        return self.__min_targets

    @property
    def max_targets(self):
        return self.__max_targets

    def normalize(self, min_features=None, max_features=None, min_targets=None, max_targets=None):
        axes = tuple(range(len(self.__input_timeseries.shape) - 1))
        self.__min_features = np.nanmin(self.__input_timeseries, axis=axes) if min_features is None else min_features
        self.__max_features = np.nanmax(self.__input_timeseries, axis=axes) if max_features is None else max_features
        self.__input_timeseries = (self.__input_timeseries - self.__min_features) / (self.__max_features - self.__min_features)

        self.__min_targets = np.nanmin(self.__target_timeseries, axis=0) if min_targets is None else min_targets
        self.__max_targets = np.nanmax(self.__target_timeseries, axis=0) if max_targets is None else max_targets
        self.__target_timeseries = (self.__target_timeseries - self.__min_targets) / (self.__max_targets - self.__min_targets)


class AbstractDataset:
    def __init__(self, forcings, streamflow):
        self.__forcings = forcings
        self.__streamflow = streamflow

    @property
    def forcings(self):
        return self.__forcings

    @property
    def streamflow(self):
        return self.__streamflow


class CamelUSDataset(AbstractDataset):
    def __init__(self, forcings: pd.DataFrame, streamflow: pd.DataFrame, feature_cols: list, target_cols: list,
                 start_time: str = None, end_time: str = None):
        super().__init__(forcings[start_time:end_time][feature_cols],
                         streamflow[start_time:end_time][target_cols])


def factory(forcings, streamflow, dataset_type: str, feature_cols: list, target_cols: list,
            start_time: str = None, end_time: str = None) -> AbstractDataset:
    if dataset_type == "camels-us":
        return CamelUSDataset(forcings, streamflow, feature_cols, target_cols, start_time, end_time)
    raise ValueError("Dataset type '{}' is not supported.".format(dataset_type))
