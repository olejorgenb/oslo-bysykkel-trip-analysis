import pandas as pd
import matplotlib.pyplot as plt

from defs import StationCol, TripCol

import pyproj
wgs = pyproj.Geod(ellps='WGS84')


def with_unique_multi_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.is_unique():
        return df

    return df.set_index(df.groupby(level=df.index.names).cumcount(), append=True)


def read_stations() -> pd.DataFrame:
    return pd.read_csv("stations.csv", index_col=0)


def read_trips() -> pd.DataFrame:
    trips = pd.read_csv("data/all.csv")
    return trips[TripCol._all()].set_index([TripCol.station_from, TripCol.station_to])


def calc_dist(lonlat_from, lonlat_to):
    lat_from, lon_from = lonlat_from
    lat_to,   lon_to   = lonlat_to
    return wgs.inv(lon_from, lat_from, lon_to, lat_to)[2]


def coord_of(stations, id):
    # return tuple(stations.loc[id, [StationCol.lon, StationCol.lat]])
    return stations.loc[id, StationCol.lon], stations.loc[id, StationCol.lat]


def route_distance(trips: pd.DataFrame, stations: pd.DataFrame) -> pd.Series:
    u = trips.index.unique()
    d = u.map(lambda uv: calc_dist(coord_of(stations, uv[0]), coord_of(stations, uv[1])))
    return pd.Series(d, index=u.to_flat_index(), name="d")


def route_ele_gain(trips: pd.DataFrame, stations: pd.DataFrame) -> pd.Series:
    u = trips.index.unique()
    g = u.map(lambda uv: stations.loc[uv[1], StationCol.ele] - stations.loc[uv[0], StationCol.ele])
    return pd.Series(g, index=u.to_flat_index(), name="g")


if __name__ == '__main__':
    V = read_stations()
    E = read_trips()

    d = route_distance(E, V)
    g = route_ele_gain(E, V)

    route_count = E.index.value_counts()

    route_traveled_dist = route_count * d
    route_ele_gained = route_count * g