from libs import dataset
import pandas as pd
import numpy as np


class AbstractProcessingPipeline:
    def __init__(self, scaling_params: tuple = None):
        self.__scaling_params = scaling_params

    @property
    def scaling_params(self):
        return self.__scaling_params

    @scaling_params.setter
    def scaling_params(self, value):
        self.__scaling_params = value

    def fit(self, ds: dataset.AbstractDataset):
        pass

    def process_and_fit(self, ds: dataset.AbstractDataset) -> dataset.TimeseriesDataset:
        pass

    def process(self, ds: dataset.AbstractDataset) -> dataset.TimeseriesDataset:
        pass


class CamelsUsProcessingPipeline(AbstractProcessingPipeline):

    def fit(self, ds: dataset.CamelUSDataset):
        inputs, targets = align_datasets(ds.forcings, ds.streamflow)
        self.__fit_scaling_params(inputs, targets)

    def process_and_fit(self, ds: dataset.CamelUSDataset):
        inputs, targets = align_datasets(ds.forcings, ds.streamflow)
        timeseries_data = dataset.TimeseriesDataset(inputs.to_numpy(), targets.to_numpy())
        timeseries_data.normalize()
        self.scaling_params = (timeseries_data.min_features, timeseries_data.max_features,
                               timeseries_data.min_features, timeseries_data.max_targets)

        return timeseries_data

    def process(self, ds: dataset.CamelUSDataset):
        inputs, targets = align_datasets(ds.forcings, ds.streamflow)
        timeseries_data = dataset.TimeseriesDataset(inputs.to_numpy(), targets.to_numpy())
        timeseries_data.normalize(*self.scaling_params)

        return timeseries_data

    def __select_data(self):
        inputs = self.forcings[self.feature_cols]
        targets = self.streamflow[self.target_cols]
        return inputs, targets

    def __fit_scaling_params(self, inputs, targets):
        inputs, targets = align_datasets(inputs, targets)
        axes = tuple(range(len(inputs.shape) - 1))
        inputs = inputs.to_numpy()
        targets = targets.to_numpy()
        min_features = np.nanmin(inputs, axis=axes)
        max_features = np.nanmax(inputs, axis=axes)
        min_targets = np.nanmin(targets, axis=0)
        max_targets = np.nanmax(targets, axis=0)
        self.scaling_params = (min_features, max_features, min_targets, max_targets)


def align_datasets(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    Ensures that two time indexed datasets cover the same timespan

    Parameters
    ----------
    df1
        First pandas.DataFrame
    df2
        Second pandas.DataFrame

    Returns
    -------
    Two pandas.Dataframes that cover the same timespan

    """
    common_start_date = max(df1.index.min(), df2.index.min())
    common_end_date = min(df1.index.max(), df2.index.max())

    return df1[common_start_date:common_end_date], df2[common_start_date:common_end_date]


def factory(ds: dataset.AbstractDataset) -> AbstractProcessingPipeline:
    if isinstance(ds, dataset.CamelUSDataset):
        return CamelsUsProcessingPipeline(ds)
    raise ValueError("Dataset type '{}' is not supported.".format(type(ds)))
