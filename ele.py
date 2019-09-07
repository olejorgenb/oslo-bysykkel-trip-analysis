"""
Created 01 September 2019
Ole Jørgen Brønner, olejorgen@yahoo.no

Lookup elevation in Norway

Dependencies:
- owslib
- rasterio
"""

import owslib.wcs as wcs
import rasterio as rio
import io

# See https://kartkatalog.geonorge.no/search?Facets%5B0%5D.name=theme&Facets%5B0%5D.value=H%C3%B8ydedata&Facets%5B1%5D.name=dataaccess&Facets%5B1%5D.value=%C3%85pne%20data&Facets%5B2%5D.name=DistributionProtocols&Facets%5B2%5D.value=WCS-tjeneste
# for other datasets.

# DOM/DSM (Digital Overflate Modell): Includes buildings, trees, etc.
# DTM (Digital Terreng Modell)  : Attempts to model the "ground" by excluding buildings, etc.
# NB: owslib doesn't support 1.1.2 which is the native WCS version of geonorge atm. Request an older version explicitly as a workaround
wcs_dtm1_32 = "https://wms.geonorge.no/skwms1/wcs.hoyde-dtm1_32?version=1.0.1"
wcs_dom1_32 = "https://wms.geonorge.no/skwms1/wcs.hoyde-dom1_32?version=1.0.1"

wcs_source = wcs_dtm1_32


client = wcs.WebCoverageService(wcs_source)


def bbox_around(long, lat, d=1e-4*3):
    # IMPROVEMENT: calculate a square box (square in lat/lons is not square in meters)
    return [long-d, lat-d, long+d, lat+d]


def find_ele(lon, lat):
    """
    Find elevation in meters at given WGS84 coordinate
    :param lon:
    :param lat:
    :return: elevation in meters
    """
    # https://mapserver.org/ogc/wcs_server.html#test-with-a-getcoverage-request
    res = client.getCoverage(
        '1',
        bbox=bbox_around(lon, lat),
        crs='EPSG:4326',  # Note: same as WGS84
        width=100, height=100,  # IMPROVEMENT: estimate resolution in a better way
        format='geotiff'
    )

    bs = res.read()
    with rio.open(io.BytesIO(bs)) as ds:
        ele_map = ds.read()

    i, j = ds.index(lon, lat)
    return ele_map[0, i, j]
