import sys
import pandas as pd

from defs import StationCol


station_cols = StationCol._all()


def stations_seen(trips: pd.DataFrame) -> pd.DataFrame:

    station_lists = []

    for dir in ["start", "end"]:
        trip_columns = [f"{dir}_{col}" for col in station_cols]
        station_lists.append(trips[trip_columns].rename(columns=dict(zip(trip_columns, station_cols))))

    return pd.concat(station_lists, ignore_index=True)


def to_stations(trip_paths) -> pd.DataFrame:
    """
    Extracts a unique list of stations given a set of trip files
    :param trip_paths:
    :return: A DataFrame containing the stations and their position. Indexed by station_id
    """
    unique_stations = pd.DataFrame(columns=station_cols)
    for trip_path in trip_paths:
        trips = pd.read_csv(trip_path)
        # Not all trips specify the exact same coordinate for the same stations..
        unique_stations = pd.concat([unique_stations, stations_seen(trips)], ignore_index=True).drop_duplicates(
            subset="station_id")
    unique_stations.set_index("station_id", drop=True, verify_integrity=True, inplace=True)

    return unique_stations


def with_elevation(stations: pd.DataFrame) -> pd.DataFrame:
    """
    Augments a stations dataframe with elevation column (`ele`)
    :param stations:
    :return:
    """
    import ele

    coordinates = stations[[StationCol.lon, StationCol.lat]].values
    elevation = [ele.find_ele(*lonlat) for lonlat in coordinates]
    stations = stations.copy()
    stations[StationCol.ele] = elevation
    return stations


def pipe(initial_arg, *fns):
    import functools

    return functools.reduce(lambda arg, fn: fn(arg), fns, initial_arg)


if __name__ == '__main__':
    stations = pipe(sys.argv[1:], to_stations, with_elevation)

    stations.to_csv(sys.stdout, index=True)

