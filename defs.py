

class Col:
    # yuck
    @classmethod
    def _all(cls):
        return [getattr(cls, attr) for attr in dir(cls) if not attr.startswith("_")]


class StationCol(Col):
    id = "station_id"
    lat = "station_latitude"
    lon = "station_longitude"
    ele = "station_elevation"


class TripCol(Col):
    station_from = "start_station_id"
    station_to = "end_station_id"
    started_at = "started_at"
    ended_at = "ended_at"
    duration = "duration"
