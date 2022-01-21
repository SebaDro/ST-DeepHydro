class TimeseriesDataset:
    def __init__(self, forcings_timeseries, streamflow_timeseries):
        self.__forcings_timeseries = forcings_timeseries
        self.__streamflow_timeseries = streamflow_timeseries

    @property
    def forcings_timeseries(self):
        return self.__forcings_timeseries

    @property
    def streamflow_timeseres(self):
        return self.__streamflow_timeseries
