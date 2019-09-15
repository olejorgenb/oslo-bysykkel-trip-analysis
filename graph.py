import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
import aggdraw
import math

from defs import StationCol, TripCol

import pyproj


# Route: (station_from_id, station_to_id)
# Trip: an route "instance"


def with_unique_multi_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.is_unique():
        return df

    return df.set_index(df.groupby(level=df.index.names).cumcount(), append=True)


def read_stations() -> pd.DataFrame:
    return pd.read_csv("stations.csv", index_col=0)


def read_trips() -> pd.DataFrame:
    trips = pd.read_csv("data/all.csv")
    # trips[TripCol.started_at] = trips[TripCol.started_at].map(lambda date: date.time())
    return trips[TripCol._all()].set_index([TripCol.station_from, TripCol.station_to])


def with_utm32_coords(stations: pd.DataFrame) -> pd.DataFrame:
    proj = pyproj.Proj(proj='utm', zone=32, ellps='WGS84')
    degrees = stations[[StationCol.lon, StationCol.lat]].values
    xy = np.apply_along_axis(lambda lonlat: proj(*lonlat), 1, degrees)
    stations = stations.copy()
    stations["x"] = xy[:, 0]
    stations["y"] = xy[:, 1]
    return stations
    # return [list(proj(*lonlat)) for lonlat in degrees]


def calc_dist(lonlat_from, lonlat_to, wgs=pyproj.Geod(ellps='WGS84')):
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


def mk_mapper(index):
    index_map = dict(zip(index, range(len(index))))
    return np.vectorize(index_map.__getitem__)


def colored_matrix_plot(c, sort_by):
    def to_numpy(df: pd.DataFrame, reindex) -> np.ndarray:
        index = reindex(np.array(list(zip(*df.index))))
        return index.T

    def to_mat(series, mapper):
        n = len(series)
        coords = to_numpy(series, mapper)
        row_idxs = coords[:,0]
        col_idxs = coords[:,1]
        mat = np.zeros((np.max(row_idxs)+1, np.max(col_idxs)+1))
        mat[row_idxs, col_idxs] = series.values
        return mat

    by_ele = V.sort_values(sort_by)

    mat = to_mat(c, mk_mapper(by_ele.index))

    plt.colorbar(plt.matshow(mat))

    return mat


def tuples_to_numpy(tuples):
    return np.array(list(zip(*tuples)))


def to_df(series: pd.Series, index: pd.Index) -> pd.DataFrame:
    n = len(V)
    mat = np.zeros((n, n), dtype=series.dtype)

    # omg.. pandas chokes on memory when doing this in `.loc` space..
    coords = tuples_to_numpy(series.index)
    loc_iloc_mapper = mk_mapper(index)
    icoords = loc_iloc_mapper(coords)

    mat[icoords[0], icoords[1]] = series

    df = pd.DataFrame(index=index, columns=index, data=mat)

    return df


def draw_map(pos: pd.DataFrame,
             routes: pd.Series,
             v_color: pd.Series=None,
             res=2000
) -> Image:
    xnorm = pos.x - pos.x.min()
    ynorm = pos.y - pos.y.min()

    m_per_px = max(xnorm.max(), ynorm.max()) / res

    xnorm = xnorm / m_per_px
    ynorm = ynorm / m_per_px

    img = Image.new('RGBA', (math.ceil(xnorm.max()), math.ceil(ynorm.max())))
    drawer = aggdraw.Draw(img)

    for (a,b), color in routes.iteritems():
        x1, y1 = xnorm[a], ynorm[a]
        x2, y2 = xnorm[b], ynorm[b]

        color = tuple(color[:4])

        p = aggdraw.Pen(color, 3)

        drawer.line((x1, y1, x2, y2), p)

    for id, x, y in zip(xnorm.index, xnorm, ynorm):
        r = 6
        if v_color is not None:
            c = tuple(v_color.loc[id])
        else:
            c = (200, 60, 20)
        b = aggdraw.Brush(c)
        drawer.ellipse((x - r, y - r, x + r, y + r), b)

    drawer.flush()
    return img


def normalized(series: pd.Series) -> pd.Series:
    a, b = series.min(), series.max()
    return (series - a) / (b - a)


def apply_cmap(cmap, series: pd.Series) -> pd.Series:
    c = cmap(normalized(series))
    c = (c * 255).astype(np.int)

    return pd.Series(data=c.tolist(), index=series.index)


def opacity_cmap(series: pd.Series, color=(0, 0, 0)) -> pd.Series:
    c = np.repeat(np.array([*color, 0]).reshape(1, -1), len(series), axis=0)
    c[:, 3] = np.round(series.values * 255)
    return pd.Series(data=c.tolist(), index=series.index)


if __name__ == '__main__':
    V = read_stations()
    E = read_trips()

    d = route_distance(E, V)
    g = route_ele_gain(E, V)

    route_count = E.index.value_counts()

    df = to_df(route_count, V.index)

    d = df.sum(axis=0) - df.sum(axis=1)
    d.name = "bike_surplus"


    # route_traveled_dist = (route_count * d).astype(np.int64)
    # route_ele_gained = (route_count * g).astype(np.int64)

    # rev_route_count = route_count[route_count.index.map(lambda v: tuple(reversed(v)))]
    # rev_route_count = rev_route_count.fillna(0).astype(np.int64)
    # route_count_df = pd.DataFrame(index=route_count.index,
    #                               data={"forward": route_count.values, "backward": rev_route_count.values})
    # route_count_df["diff"] = route_count_df["forward"] - route_count_df["backward"]


    hoff = 566
    org = 571
    skøyen = 627

    pos = with_utm32_coords(V)
    cmap = plt.get_cmap("viridis")

    sc = pd.Series(index=d.index)
    omg1 = [(255,0,0)]*(len(d))
    omg2 = [(0,255,0)]*(len(d))
    sc[d >= 0] = pd.Series(index=d.index, data=omg1)
    sc[d < 0] = pd.Series(index=d.index, data=omg2)

    img = draw_map(pos, opacity_cmap(normalized(route_count)), sc)

    plt.imshow(img)
    plt.tight_layout()

    plt.show()