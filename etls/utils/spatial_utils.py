"""
Author: Anastassios Dardas, PhD - Lead Geospatial Computing Engineer.

Date Created: 2023.

Date Modified: Dec. 2024.

About: Spatial operations that include the following:
    - Vector:
        - convert dataframe to geodataframe
        - exporting to geoparquet
        - Morton Z-order index (multidimensional to 1D for data locality)
        - H3 indexing processes for points, lines, and polygons
        - Auto-acquiring UTM projections
        - Reprojections
        - Auto-acquiring geographic units
        - Intersection over Union

    - Raster:
        - Read TIF file
        - Raster TIF file characteristics including spatial
        - Construct GCPs
        - Construct AffineTransformer
        - Coordinates to pixel space conversion
        - Pixel space to coordinate conversion
        - Random selection of random GCPs along pixel space of TIF file
"""

# Custom data-engineering packages
from .data_utils import DataEng
from .regex_utils import ReMethods

# Data Engineering packages
from pandas import DataFrame
import numpy as np

# Spatial (data) engineering packages - Vector
from geopandas import GeoDataFrame
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, MultiPoint, MultiLineString
from shapely.ops import transform as shp_transform
from shapely import geometry
from pyproj import CRS, Proj, Transformer
import h3
import utm

# Spatial (data) engineering packages - Raster
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.transform import from_gcps, AffineTransformer
import cv2

# Miscellaneous packages
from operator import add, sub
from typing import Union, List
import math


class SpatialOps:

    def __init__(self):
        self.map_scale_inch2meter = 39.37

        # List of geographic units anticipated in projected coordinate systems and their conversion relative to metre.
        # Derived from cs2cs -lu from OSGeo4w Shell
        # Source: https://gis.stackexchange.com/questions/200075/proj4-fails-to-create-projection-with-units-degrees
        self.cs2cs = {
            "km": ["kilometer", 1000],
            "m": ["metre", 1],
            "dm": ["decimeter", 1 / 10],
            "cm": ["centimeter", 1 / 100],
            "mm": ["millimeter", 1 / 1000],
            "kmi": ["International Nautical Mile", 1852],
            "in": ["International Inch", 0.0254],
            "ft": ["International Foot", 0.3048],
            "yd": ["International Yard", 0.9144],
            "mi": ["International Statute Mile", 1609.344],
            "fath": ["International Fathom", 1.8288],
            "ch": ["International Chain", 20.1168],
            "link": ["International Link", 0.201168],
            "us-in": ["U.S. Surveyor's Inch", 1 / 39.37],
            "us-ft": ["U.S. Surveyor's Foot", 0.304800609601219],
            "us-yd": ["U.S. Surveyor's Yard", 0.914401828803658],
            "us-ch": ["U.S. Surveyor's Chain", 20.11684023368047],
            "us-mi": ["U.S. Surveyor's Statute Mile", 1609.347218694437],
            "ind-yd": ["Indian Yard", 0.91439523],
            "ind-ft": ["Indian Foot", 0.30479841],
            "ind-ch": ["Indian Chain", 20.11669506]
        }

        self.cs2cs_items = list(self.cs2cs.values())
        self.cs2cs_unit_name = [i[0] for i in self.cs2cs_items]

        # Construct list of Units that will match to Proj4 string style.
        self.unit_list = [f"+units={u}" for u in self.cs2cs.keys()]

        # UTM letter relative to hemisphere position
        # Source: https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system#/media/File:Universal_Transverse_Mercator_zones.svg
        self.utm_letter_hemisphere = {
            "A": "South",
            "B": "South",
            "C": "South",
            "D": "South",
            "E": "South",
            "F": "South",
            "G": "South",
            "H": "South",
            "J": "South",
            "K": "South",
            "L": "South",
            "M": "South",
            "N": "North",
            "P": "North",
            "Q": "North",
            "R": "North",
            "S": "North",
            "T": "North",
            "U": "North",
            "V": "North",
            "W": "North",
            "X": "North",
            "Y": "North",
            "Z": "North"
        }

        # Operation - add and subtract
        self.ops = (add, sub)

    def pipe2gdf(self, df: DataFrame, geom_field: str, epsg: str) -> GeoDataFrame:
        """
        Convert DataFrame to GeoDataFrame.

        :param df: DataFrame with geometry field.
        :param geom_field: Name of the geometry field in the DataFrame.
        :param epsg: Spatial Coordinate Reference System (e.g., "epsg:4326").

        :return: GeoDataFrame.
        """
        gdf = GeoDataFrame(df, crs=epsg, geometry=geom_field)
        return gdf

    def geoparquet(self, gdf: GeoDataFrame, output_path: str, index=False, compression='zstd'):
        """
        Outputs the GeoDataFrame to GeoParquet.

        :param gdf: GeoDataFrame.
        :param output_path: Output path including naming the file and .geoparquet extension.
        :param index: Index field to include - currently set to False as default.
        :param compression: Compression type for the geoparquet. Currently default to 'zstd'.

        :return: Indirectly - saves the GeoDataFrame as a geoparquet file.
        """
        gdf.to_parquet(path = output_path, index = index, compression = compression)

    def coord2pixel_raster(self, raster_transform: rasterio.Affine, xs: Union[float, List], ys: Union[float, List]):
        """
        Convert coordinates to pixel space.

        :param raster_transform: Affine Transformer.
        :param xs: Longitude (x) coordinate.
        :param ys: Latitude (y) coordinate.

        :return: Pixel space.
        """
        return rasterio.transform.rowcol(transform=raster_transform, xs=xs, ys=ys)

    def pixel2coord_raster(self, raster_transform: rasterio.Affine, row_y_pix, col_x_pix) -> List:
        """
        Acquire geographic coordinate from pixel coordinates.

        :param raster_transform: Rasterio Affine transformation.
        :param row_y_pix: Y pixel. Refer to height.
        :param col_x_pix: X pixel. Refer to width.

        :return: List containing geographic coordinates.
        """
        x, y = rasterio.transform.xy(transform = raster_transform,
                                     rows      = row_y_pix,
                                     cols      = col_x_pix)

        return [x,y]

    def morton_z(self, x, y) -> np.int64:
        """
        Create Z2 index (multi-dimensional to 1D) via bit interleaving approach.

        :param x: Normalized longitude value (or pixel X).
        :param y: Normalized latitude value (or pixel Y).

        :return: Integer64.
        """
        inter_list = [x, y]
        tmp_value = []
        for i in inter_list:
            i &= 0x7fffffff
            i = (i ^ (i << 32)) & 0x00000000ffffffff
            i = (i ^ (i << 16)) & 0x0000ffff0000ffff
            i = (i ^ (i << 8)) & 0x00ff00ff00ff00ff
            i = (i ^ (i << 4)) & 0x0f0f0f0f0f0f0f0f
            i = (i ^ (i << 2)) & 0x3333333333333333
            i = (i ^ (i << 1)) & 0x5555555555555555

            tmp_value.append(i)

        zIndex = tmp_value[0] | tmp_value[1] << 1

        return zIndex

    def read_tif_file(self, tif_file: str, method="r+"):
        """
        Read raster TIF file.

        :param tif_file: Name of the raster TIF file.
        :param method: Default to r+ (access mode when working with the TIF).

        :return: OpenDataset
        """
        try:
            return rasterio.open(tif_file, method)

        except rasterio._err.CPLE_AppDefinedError:
            return rasterio.open(tif_file, method, IGNORE_COG_LAYOUT_BREAK="YES")

    def raster_spatial_info(self, tif_file: str) -> List:
        """
        Acquire raster spatial information.

        :param tif_file: Name of the TIF file.

        :return: List => Source as OpenDataset, GCPS EPSG, Affine Transformer, EPSG, width, and height of pixel space.
        """

        with self.read_tif_file(tif_file = tif_file) as src:
            gcps_info   = src.gcps
            gcps_coords = gcps_info[0]
            gcps_epsg   = gcps_info[1]
            transform   = from_gcps(gcps_coords)
            src.crs     = gcps_epsg
            get_width   = src.width
            get_height  = src.height

        return [src, gcps_epsg, transform, src.crs, get_width, get_height]

    def raster_characteristics(self, tif_file: str) -> List:
        """
        Acquire raster characteristics - mainly spatial.

        :param tif_file: Raster TIF file.

        :return: List of Spatial characteristics (OpenDataset, CRS, transform, width & height).
        """
        # Read TIF and transform the map if it has GCPs
        try:
            raster_inf = self.raster_spatial_info(tif_file=tif_file)
            crs        = raster_inf[3]
            transform  = raster_inf[2]
            get_width  = raster_inf[4]
            get_height = raster_inf[5]

        # Otherwise, read the TIF and get the transform directly
        except rasterio.errors.CRSError:
            raster_inf = self.read_tif_file(tif_file=tif_file)
            crs        = raster_inf.crs
            transform  = raster_inf.transform
            get_width  = raster_inf.width
            get_height = raster_inf.height

        return [raster_inf, crs, transform, get_width, get_height]

    def get_pixel_coordinates(self, polygon: Polygon, img_dim: tuple):
        """
        Acquire pixel coordinates within a provided polygon.

        :param polygon: Shapely Polygon geometry. May need to convert geographic coordinates to pixels first.
        :param img_dim: Tuple representing (height, width) of the image.

        :return: List of pixel coordinates (x,y) within the polygon.
        """

        mask = np.zeros(img_dim, dtype=np.uint8)
        #x,y  = polygon.exterior.xy
        #pts  = np.array([[int(x[i]), int(y[i])]
        #                 for i in range(len(x))], dtype=np.int32).reshape((-1, 1, 2))
        #cv2.fillPoly(mask, [pts], 255)

        pnts = np.array(polygon.exterior.coords).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pnts], (255, 0, 0))
        pixel_coords = np.where(mask == 255)

        return np.array(list(zip(pixel_coords[1], pixel_coords[0])))

    def get_pixel_coordinates_by_line(self, line: LineString, img_dim: tuple):
        mask       = np.zeros(img_dim, dtype=np.uint8)
        line_coord = line.coords.xy
        x1,y1 = int(line_coord[0][0]), int(line_coord[0][1])
        x2,y2 = int(line_coord[1][0]), int(line_coord[1][1])
        cv2.line(mask, (x1, y1), (x2, y2), 255)
        pixel_coords = np.where(mask == 255)

        return np.array(list(zip(pixel_coords[1], pixel_coords[0])))

    def construct_gcp(self, row, col, x, y) -> GroundControlPoint:
        """
        Construct ground control point directly.

        :param row: Row value (e.g., rows_from_top).
        :param col: Column value (e.g., columns_from_left).
        :param x: Longitude coordinate.
        :param y: Latitude coordinate.

        :return: GroundControlPoint.
        """
        return GroundControlPoint(row = row, col = col, x = x, y = y)

    def affine_transformer_from_gcps(self, gcps_list: List) -> AffineTransformer:
        """
        Construct AffineTransformer from list of GCPs.

        :param gcps_list: List of GCPs via list of GroundControlPoints.

        :return: AffineTransformer
        """
        return AffineTransformer(from_gcps(gcps_list))

    def densifyLne(self, line: LineString, step) -> LineString:
        """
        Identifies the density of the LineString. Mainly used in "acqh3_fromlne" function.

        :param line: Shapely LineString geometry.
        :param step: The size (usually in math.degrees) of the step.

        :return: LineString.
        """
        if line.length < step:
            return line

        length           = line.length
        current_distance = step
        new_points       = []

        new_points.append(line.interpolate(0.0, normalized=True))

        while current_distance < length:
            new_points.append(line.interpolate(current_distance))
            current_distance += step

        new_points.append(line.interpolate(1.0, normalized=True))

        return LineString(new_points)

    def acqH3_frompnt(self, lat, lon, res: int):
        """
        Acquire H3 spatial index from a point.

        :param lat: Latitude degree coordinate.
        :param lon: Longitude degree coordinate.
        :param res: The H3 resolution set - higher, more precise, and more indices.

        :return: H3 spatial index value.
        """
        return h3.geo_to_h3(lat=lat, lng=lon, resolution=res)

    def acqH3_fromlne(self, line: LineString, res: int) -> List:
        """
        Acquire H3 spatial indices from a LineString.

        :param line: Shapely LineString geometry.
        :param res: The H3 resolution set - higher, more precise, and more indices.

        :return: List of H3 spatial indices.
        """
        result_set   = set()
        vertex_hexes = [h3.geo_to_h3(t[1], t[0], res) for t in list(line.coords)]
        result_set.update(vertex_hexes)

        endpoint_hex_edges = (
            DataEng.flatten([h3.get_h3_unidirectional_edges_from_hexagon(h) for h in [vertex_hexes[0], vertex_hexes[1]]])
        )

        step = math.degrees(min([h3.exact_edge_length(e, unit='rads') for e in endpoint_hex_edges]))

        densified_line = self.densifyLne(line = line, step = step)
        line_hexes     = [h3.geo_to_h3(t[1], t[0], res) for t in list(densified_line.coords)]
        result_set.update(line_hexes)

        neighboring_hexes              = set(DataEng.flatten([h3.k_ring(h,1) for h in result_set])) - result_set
        intersecting_neighboring_hexes = filter(lambda h: Polygon(h3.h3_set_to_multi_polygon([h], True)[0][0]).distance(line) == 0,
                                                neighboring_hexes)

        result_set.update(intersecting_neighboring_hexes)

        return list(result_set)

    def acqH3_poly(self, poly: Polygon, res: int, geo_json_conformant=True) -> List:
        """
        Acquire H3 spatial indices from a Polygon.

        :param poly: Shapely Polygon geometry.
        :param res: H3 resolution set.
        :param geo_json_conformant: True or False - depending on how the latitude and longitude structured in the polygon.
                                    Usually it is True, but if it seems off - then set to False.

        :return: List of H3 indices.
        """
        result_set = set()
        vertex_hexes = [h3.geo_to_h3(t[1], t[0], res) for t in list(poly.exterior.coords)]
        for i in range(len(vertex_hexes)-1):
            result_set.update(h3.h3_line(vertex_hexes[i], vertex_hexes[i+1]))

        result_set.update(list(h3.polyfill(geometry.mapping(poly), res, geo_json_conformant=geo_json_conformant)))

        return list(result_set)

    def acq_minH3_frompoly_iter(self, poly: Polygon, res: int, geo_json_conformant=True) -> List:
        """
        Acquire a list of H3 spatial indices from a polygon. However, if the set is empty increase the resolution
        by 1 iteratively until it finds a set. From there, the set will then be used to find its parent H3 spatial index
        (i.e., by using the updated set and the original resolution).

        :param poly: Shapely polygon.
        :param res: Original H3 resolution number set.
        :param geo_json_conformant: True or False - depending on how the latitude and longitude structured in the polygon.
                                    Usually it is True, but if it seems off - then set to False.

        :return: List containing the new resolution level, and spatial index value(s) at its original resolution.
        """
        tmp_res     = res
        min_h3      = []
        while len(min_h3) == 0:
            min_h3 = self.acqH3_poly(poly=poly, res=tmp_res, geo_json_conformant=geo_json_conformant)

            if min_h3 == "Limit":
                break
            else:
                tmp_res += 1

        if min_h3 != "Limit":
            min_h3 = list(set([h3.h3_to_parent(h = m, res = res) for m in min_h3]))
            return [tmp_res, min_h3]

        else:
            # Polygon might be too small
            centroid_pnt = poly.centroid
            min_h3       = [self.acqH3_frompnt(lat = centroid_pnt.y, lon = centroid_pnt.x, res = res)]
            return [res, min_h3]

    def break_multi_geom(self, geom: Union[Polygon, Point, LineString, MultiLineString, MultiPolygon, MultiPoint]):
        """
        Deconstruct multi-geometries into smaller geometries.

        :param geom: Shapely multigeometries.

        :return: Either list of geometries or a Shapely geometry.
        """
        if isinstance(geom, Union[MultiPoint, MultiPolygon, MultiLineString]):
            return list(geom.geoms)
        else:
            return geom

    def auto_acqH3_fromgeom(self, geom: Union[Polygon, Point, LineString], min_res: int, max_res: int) -> List:
        """
        Auto acquire minimum and maximum level of H3 indices based on the Shapely geometry.

        :param geom: Shapely geometry either as Point, Polygon, or LineString.
        :param min_res: Minimum resolution set for H3.
        :param max_res: Maximum resolution set for H3.

        :return: List -> list of minimum H3 indices, list of maximum H3 indices, min res, max res, and feature type.
        """
        if isinstance(geom, Polygon):
            # Remove the Z
            if geom.has_z:
                lines = [xy[:2] for xy in list(geom.exterior.coords)]
                geom  = Polygon(lines)

            h3_inf      = self.acq_minH3_frompoly_iter(poly=geom, res=min_res)
            min_res     = h3_inf[0]
            min_h3      = h3_inf[1]
            max_h3      = self.acqH3_poly(poly=geom, res=max_res)

            if max_h3 is None:
                centroid_pnt = geom.centroid
                max_h3 = self.acqH3_frompnt(lat = centroid_pnt.y, lon = centroid_pnt.x, res = max_res)

            return [min_h3, max_h3, min_res, max_res, "polygon"]

        elif isinstance(geom, LineString):
            min_h3 = self.acqH3_fromlne(line=geom, res=min_res)
            max_h3 = self.acqH3_fromlne(line=geom, res=max_res)
            return [min_h3, max_h3, min_res, max_res, "line"]

        elif isinstance(geom, Point):
            min_h3 = [self.acqH3_frompnt(lat=geom.y, lon=geom.x, res=min_res)]
            max_h3 = [self.acqH3_frompnt(lat=geom.y, lon=geom.x, res=max_res)]
            return [min_h3, max_h3, min_res, max_res, "point"]

    def _dict_index(self, grp_df: DataFrame, merge_info: DataFrame, unique_value, output_path) -> dict:
        """
        Construct spatial index dictionary.

        :param grp_df: Grouped DataFrame by the minimum H3 resolution level.
        :param merge_info: DataFrame containing merged information such as index.
        :param unique_value: Unique value (i.e., identifier in the dataset) to recall.
        :param output_path: Where the spatial index and geo-partitioned file of that observation is stored.

        :return: Spatial Index dictionary as string-like. To reload from compressed file, use ast.literal_eval function.
        """

        min_h3    = grp_df['min_h3'].iloc[0]
        data_dict = {}
        for g in range(len(grp_df)):
            index   = grp_df['index'].iloc[g]
            sub_df  = merge_info.query('min_h3 == @min_h3 and index == @index')
            max_res = sub_df['max_res'].iloc[0]
            for s in sub_df['max_h3']:
                tmp_dict = {s : [f"{output_path}:{unique_value}:{index}"]}
                if data_dict.get(max_res):
                    if data_dict[max_res].get(s):
                        data_dict[max_res][s].append(f"{output_path}:{unique_value}:{index}")
                    else:
                        data_dict[max_res].update(tmp_dict)
                else:
                    data_dict[max_res] = tmp_dict

        return data_dict

    def build_h3_index_table(self, tmp_gdf: GeoDataFrame, unique_value, output_path) -> DataFrame:
        """
        Construct spatial index table of the geo-partitioned file. It will later on be in use to build a master spatial
        index lookup and to facilitate searching data that may not have been successful with label matching.

        :param tmp_gdf: Partitioned GeoDataFrame.
        :param unique_value: Unique value (i.e., identifier in the dataset) to recall.
        :param output_path: Output path to where the spatial index table will be stored.

        :return: DataFrame containing the following schema:
            - min_h3 => minimum H3 resolution (i.e., spatial index).
            - table  => hierarchical dictionary of the spatial index from min_h3. The format is as followed:
                {"res_level" (int) : {"H3_spatial_index" (str) : ["output_path:unique_value:index]}}
        """
        min_h3 = tmp_gdf.explode('min_h3')
        max_h3 = tmp_gdf.explode('max_h3')

        grp_min_h3 = (
            min_h3
            .groupby('min_h3', as_index=False)
            .agg({'max_h3': lambda x: x.index.tolist()})
            .explode('max_h3')
            .rename(columns={'max_h3': 'index'})
        )

        grp_max_h3 = (
            max_h3
            .groupby('max_h3', as_index=False)
            .agg({'min_h3' : lambda x: x.index.tolist(),
                  'max_res': lambda x: x.tolist()})
            .explode(['min_h3', 'max_res'])
            .rename(columns={'min_h3': 'index'})
        )

        merge_info = grp_min_h3.merge(grp_max_h3, on=['index'])

        spatial_table = (
            grp_min_h3
            .groupby('min_h3', as_index=False)
            .apply(lambda a: DataFrame({'min_h3': [a['min_h3'].iloc[0]],
                                        'table' : [self._dict_index(grp_df       = a,
                                                                    merge_info   = merge_info,
                                                                    unique_value = unique_value,
                                                                    output_path  = output_path)]}))
        )

        return spatial_table

    def IoU_calc(self, geom1, geom2):
        """
        Intersection over Union calculation.

        :param geom1: 1st Shapely geometry - ideally should be polygon.
        :param geom2: 2nd Shapely geometry - ideally should be polygon.

        :return: Intersection Over Union rounded to the thousandths decimal place.
        """
        intersect_area = geom1.intersection(geom2).area
        return round(intersect_area / (geom2.area + geom1.area - intersect_area), 5) * 100

    def _match_geog_unit_func(self, value, items):
        """
        Filter process as part of the acq_unit_converter_by_other_unit function.

        :param value: Geographic unit value as string.

        :param items: Dictionary of the cs2cs items.

        :return: Filtered cs2cs item.
        """
        return next(filter(lambda v: v[0] == value, items))[1]

    def acq_unit_converter_by_other_unit(self, value: str) -> List:
        """
        Find close match of geographic unit to the cs2cs geographic units.

        :param value: String value of the geographic unit.

        :return: List -> matched geographic unit, original geographic unit, and matched ratio.
        """
        try:
            get_value = self._match_geog_unit_func(value = value, items = self.cs2cs_items)
            return [get_value, value, 1.0]

        except StopIteration:
            tmp_value = ReMethods().max_sequence_matcher(str1 = value, str2_list = self.cs2cs_unit_name)
            get_value = self._match_geog_unit_func(value = tmp_value[0], items = self.cs2cs_items)
            return [get_value, tmp_value[0], tmp_value[1]]

    def preproj_pnt_utm(self, lat, lon):
        """
        Acquire the UTM (i.e., projected coordinate system) based on the latitude and longitude degree coordinates.
        UTM is more precise to compute distance, buffer, and other GIS processes than standard degree coordinates.

        :param lat: Latitude degree value.
        :param lon: Longitude degree value.

        :return: UTM projected coordinate system.
        """
        return utm.from_latlon(latitude=lat, longitude=lon)

    def acquireUTMproj(self, proj_type: str, zone: int, south=False) -> str:
        """
        Acquires the EPSG (or CRS) based on the provided UTM. For more help in determining what UTM zone,

        :param proj_type:
        :param zone:
        :param south:
        :return:
        """
        return ":".join(CRS.from_dict({'proj'  : proj_type,
                                       'zone'  : zone,
                                       'south' : south}).to_authority())

    def latlon_to_utm_epsg_proj(self, lat, lon):
        """
        Acquire projected coordinate system dynamically from decimal degree lat and lon coordinates.

        :param lat: Latitude (y) coordinate.
        :param lon: Longitude (x) coordinate.

        :return: Projected coordinate system.
        """
        pre_proj   = self.preproj_pnt_utm(lat = lat, lon = lon)
        utm_zone   = pre_proj[2]
        utm_letter = pre_proj[3]
        zone_pos   = self.utm_letter_hemisphere[utm_letter]
        south      = True if zone_pos == "South" else False
        new_epsg   = self.acquireUTMproj(proj_type = "utm",
                                         zone      = utm_zone,
                                         south     = south).lower()

        return new_epsg

    def projGeomTransform(self, orig_epsg: str, new_epsg: str, geom):
        """
        Reproject geometry to projected coordinate system.

        :param orig_epsg: Original EPSG.
        :param new_epsg: EPSG to be projected to.
        :param geom: Shapely geometry.

        :return: Re-projected shapely geometry.
        """
        project = Transformer.from_proj(Proj(orig_epsg), Proj(new_epsg), always_xy=True)
        return shp_transform(project.transform, geom)

    def get_crs_from_epsg(self, epsg: int):
        """
        Acquire CRS from EPSG.

        :param epsg: EPSG.

        :return: CRS.
        """
        return CRS.from_epsg(code=epsg)

    def acq_crs_info_from_raster(self, raster_crs: rasterio.crs.CRS) -> List:
        """
        Acquire specific CRS information from the Raster TIF's CRS.

        :param raster_crs: TIF's CRS.

        :return: List -> original EPSG, geographic unit of that EPSG, and determining whether it needs to be reprojected.
        """
        # In try & except: Acquire EPSG whether directly from the CRS or WKT that is then converted
        # to proj4 (a method to transform between coordinate systems).
        try:
            orig_epsg = raster_crs.data['init'].lower()
            epsg_code = orig_epsg.split(":")[-1]

            # Identify if the EPSG code is 4326 - important as this will need to be converted to meters
            if epsg_code == "4326":
                orig_geog_unit    = "degrees"
                need_to_reproject = True

            # Otherwise, identify the units from the EPSG code
            else:
                orig_geog_unit    = [u.unit_name for u in self.get_crs_from_epsg(int(epsg_code)).axis_info][0]
                need_to_reproject = False

        except KeyError:
            orig_epsg         = rasterio.crs.CRS.from_wkt(raster_crs.wkt).to_proj4()
            prj_list          = orig_epsg.split(" ")
            pre_geog_unit     = [u for p in prj_list for u in self.unit_list if p in u][0].split("=")[-1]
            orig_geog_unit    = self.cs2cs[pre_geog_unit][0]
            need_to_reproject = True

        return [orig_epsg, orig_geog_unit, need_to_reproject]

    def select_random_gcps_from_tif(self, id_map: Union[str, int], tif_file: str, num_gcps: int = 10, unique_samples: bool = False) -> DataFrame:
        """
        Randomly select GCPs from TIF for georeferencing evaluation.

        :param id_map: Unique identifier of the map.
        :param tif_file: Name of the TIF file corresponding to the unique identifier of the map.
        :param num_gcps: Number of random GCPs on the map - default is 10.
        :param unique_samples: NumPy parameter to replace unique random samples - default to False.

        :return: DataFrame with the following schema:
            - id                    => Unique identifier of the map.
            - tif_file              => Name of the TIF file.
            - pix_height            => Height of the TIF file in pixel space.
            - pix_width             => Width of the TIF file in pixel space.
            - random_pix_height     => Random selected height location of the pixel.
            - random_pix_width      => Random selected width location of the pixel.
            - raw_orig_epsgs        => Original EPSG
            - raw_orig_geog_units   => Original geographic units from the original EPSG.
            - raw_orig_pnts         => Converted random selected pixel space to point geometry in original EPSG.
            - raw_deci_pnts         => Converted (if applicable) of random selected pixel space to decimal degrees.
            - raw_proj_epsgs        => Re-projected EPSG (if applicable).
            - raw_proj_geog_units   => Re-projected geographic unit of the re-projected EPSG.
            - raw_proj_pnts         => Converted random selected pixel space to reprojected point geometry.
            - same_epsgs            => True or False --> if the reprojected EPSG is the same as the original one.
        """
        tif_inf     = self.raster_characteristics(tif_file = tif_file)
        crs         = tif_inf[1]
        transform   = tif_inf[2]
        get_width   = tif_inf[3]
        get_height  = tif_inf[4]

        # Acquire height and width of the TIF in pixel
        height_pixel = np.arange(1, get_height+1)
        width_pixel  = np.arange(1, get_width+1)

        # Generate n number of random pixel spaces
        random_height = np.random.choice(height_pixel, num_gcps, replace=unique_samples)
        random_width  = np.random.choice(width_pixel, num_gcps, replace=unique_samples)

        # In try & except: Acquire EPSG whether directly from the CRS or WKT that is then converted
        # to proj4 (a method to transform between coordinate systems).
        geog_info         = self.acq_crs_info_from_raster(raster_crs = crs)
        orig_epsg         = geog_info[0]
        orig_geog_unit    = geog_info[1]
        need_to_reproject = geog_info[2]

        # Acquire geographic coordinate from the pixel space, convert to Shapely point, and acquire decimal degree version of it.
        orig_pnts, deci_pnts, proj_pnts, proj_epsgs, proj_geog_units, same_epsgs = ([] for i in range(6))
        for h, w in zip(random_height, random_width):
            pixel_coord = self.pixel2coord_raster(raster_transform = transform,
                                                  row_y_pix        = h,
                                                  col_x_pix        = w)

            # Convert Shapely point to decimal degrees - not directly important but can be for later use.
            pixel_pnt = Point(pixel_coord)
            deci_pnt  = self.projGeomTransform(orig_epsg = orig_epsg,
                                               new_epsg  = "epsg:4326",
                                               geom      = pixel_pnt)

            # If the TIF file contains a spatial reference that is either 4326 or a localized projected coordinate system
            # then reproject automatically based on the decimal degree point in its UTM zone.
            if need_to_reproject:

                # Acquire reprojected epsg code
                new_epsg = self.latlon_to_utm_epsg_proj(lat = deci_pnt.coords[0][1],
                                                        lon = deci_pnt.coords[0][0])

                # Re-project decimal degree point
                new_code   = int(new_epsg.split(":")[-1])
                reproj_pnt = self.projGeomTransform(orig_epsg = "epsg:4326",
                                                    new_epsg  = new_epsg,
                                                    geom      = deci_pnt)

                # Acquire projected geographic unit
                proj_geog_unit = [u.unit_name for u in self.get_crs_from_epsg(new_code).axis_info][0]
                same_epsg      = False

            else:
                reproj_pnt     = pixel_pnt
                proj_geog_unit = orig_geog_unit
                new_epsg       = orig_epsg
                same_epsg      = True

            orig_pnts.append(pixel_pnt)
            deci_pnts.append(deci_pnt)
            proj_pnts.append(reproj_pnt)
            proj_geog_units.append(proj_geog_unit)
            proj_epsgs.append(new_epsg)
            same_epsgs.append(same_epsg)

        # Construct the DataFrame with the necessary details.
        tmp_df = DataFrame({"id"                    : [id_map] * num_gcps,
                            "tif_file"              : [tif_file] * num_gcps,
                            "pix_height"            : [get_height] * num_gcps,
                            "pix_width"             : [get_width] * num_gcps,
                            "random_pix_height"     : random_height,
                            "random_pix_width"      : random_width,
                            "raw_orig_epsgs"        : [orig_epsg] * num_gcps,
                            "raw_orig_geog_units"   : [orig_geog_unit] * num_gcps,
                            "raw_orig_pnts"         : orig_pnts,
                            "raw_deci_pnts"         : deci_pnts,
                            "raw_proj_epsgs"        : proj_epsgs,
                            "raw_proj_geog_units"   : proj_geog_units,
                            "raw_proj_pnts"         : proj_pnts,
                            "same_epsgs"            : same_epsgs})

        return tmp_df