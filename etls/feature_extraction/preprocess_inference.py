"""
Author: Anastassios Dardas, PhD - Lead Geospatial Computing Engineer.

Date Created: Oct. 2024

Last Updated: Nov. 2024

About:


"""
import pyproj.exceptions

"""
Test packages
"""
from pandas import read_parquet


from tqdm import tqdm
from typing import List, Union
from geopandas import GeoDataFrame
from pandas import DataFrame, read_csv, concat
from shapely.geometry import Point, Polygon, LineString, box
from shapely.wkt import loads
from numpy import array_split
from functools import partial
from multiprocessing import Manager, cpu_count

# Import custom-made packages outside of relative path
import os
import sys
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Custom packages to import
from ..cdr import CDR2Vec, FeatExtractFromCDR, FeatExtractCDR, LegendItemsCDR, LegendAnnotatedExtract
from ..utils import DataEng, SpatialOps, ParallelPool, ReMethods


class PreprocessInf:

    def __init__(self,
                 cdr_token: str,
                 inhouse_feat: Union[DataFrame, str],
                 eval_df: Union[DataFrame, str, dict],
                 feat_tif_file: Union[DataFrame, str],
                 binary_df: Union[DataFrame, str],
                 crude_grp_binary: Union[DataFrame, str, None],
                 output_dir: str,
                 legend_output_dir: str,
                 eval_id_field: str = "ID",
                 eval_cog_id_field: str = "COG ID",
                 local_schema_process: Union[List, None] = None,
                 cdr_systems: Union[dict, None] = None,
                 match_threshold: float = 0.9,
                 min_res: int = 6,
                 max_res: int = 9,
                 georeference_data: bool = True,
                 legend_data: bool = True):

        """
        :param cdr_token:
        :param inhouse_feat:
        :param eval_df:
        :param feat_tif_file:
        :param binary_df:
        :param output_dir:
        :param legend_output_dir:
        :param eval_id_field:
        :param eval_cog_id_field:
        :param local_schema_process:
        :param cdr_systems:
        :param match_threshold:
        :param min_res:
        :param max_res:
        :param georeference_data:
        :param legend_data:
        """

        # Evaluation set - identify COG IDs based on the (NGMDB) ID
        id_cog_link     = self._link_cog_ids(inhouse_feat      = inhouse_feat,
                                             eval_df           = eval_df,
                                             eval_id_field     = eval_id_field,
                                             eval_cog_id_field = eval_cog_id_field)

        self.cog_ids              = id_cog_link['cog_id'].tolist()
        self.output_dir           = output_dir
        self.legend_output_dir    = legend_output_dir
        self.min_res              = min_res
        self.max_res              = max_res
        self.georeference_data    = georeference_data
        self.legend_data          = legend_data
        self.cdr_systems          = cdr_systems
        self.local_schema_process = local_schema_process
        self.match_threshold      = match_threshold
        self.binary_df            = self._read_files(file=binary_df)

        if crude_grp_binary is not None:
            self.crude_grp_binary = self._read_files(file=crude_grp_binary)
        else:
            self.crude_grp_binary = None

        # Candidate TIF files from the binary rasters - reduce the size and each one per ID will be selected.
        # It is to facilitate with the pixel-space to geographic coordinate conversion.
        feat_tif_file      = self._read_files(file=feat_tif_file)
        id_set             = id_cog_link['id'].unique()
        self.feat_tif_file = (
            feat_tif_file
            .query('id in @id_set and has_file == True')
            .sort_values(['id', 'size_hrf'], ascending=True)
            .drop_duplicates('id', keep='first')
            [['id', 'path']]
        )

        # Build configuration for feature extraction download from the CDR.
        feat_extract_config = {
            "token"      : cdr_token,
            "cog_ids"    : self.cog_ids,
            "output_dir" : self.output_dir
        }

        # Build configuration for legend annotation download from the CDR.
        legend_anno_config = {
            "token"      : cdr_token,
            "cog_ids"    : self.cog_ids,
            "output_dir" : self.legend_output_dir,
            "validated"  : "true"
        }

        """
        Conduct Legend Item extraction and then apply that during feature extraction to see if there is alignment 
        If there is alignment, can proceed to evaluation. If there isn't, then use spatial indexing instead which will 
        still have high compute complexity to iterate each GeoParquet. To prevent that, would need to construct hierarchical 
        lookup table.
        """
        self.legend_info = LegendItemsCDR(config = legend_anno_config).concat_L1
        self.legend_info.to_parquet(f"{legend_output_dir}/legend_master_file.parquet")
        #self.legend_info = read_parquet('data/ground_truth/annotated_legend/legend_master_file.parquet')

        self.id_cog_list = id_cog_link[['id', 'cog_id']]

        # Multithreading download performer results from the CDR
        if self.cdr_systems is not None and self.local_schema_process is None:
            download_from_cdr       = self._feat_cdr_pipeline(config=feat_extract_config)
            self.polygon_infer_file = download_from_cdr[0][1]
            self.line_infer_file    = download_from_cdr[1][1]
            self.point_infer_file   = download_from_cdr[2][1]

            self.polygon_infer_file.to_parquet(f"{output_dir}/polygon_pre_master.parquet")
            self.line_infer_file.to_parquet(f"{output_dir}/line_pre_master.parquet")
            self.point_infer_file.to_parquet(f"{output_dir}/point_pre_master.parquet")

            #self.polygon_infer_file = read_parquet('data/inferenced_cdr/Feature Extraction/polygon_pre_master.parquet')
            #self.line_infer_file    = read_parquet('data/inferenced_cdr/Feature Extraction/line_pre_master.parquet')
            #self.point_infer_file   = read_parquet('data/inferenced_cdr/Feature Extraction/point_pre_master.parquet')

            # Parallelization
            self.data_extract = self._feature_extraction_conversion()

    def _read_files(self, file):
        if isinstance(file, str):
            get_ext = os.path.splitext(file)[-1]
            if get_ext == ".csv":
                df = read_csv(file)
            elif get_ext == ".parquet":
                df = read_parquet(file)

        else:
            df = file

        return df

    def _prep_and_execute_process(self, df: DataFrame, feat_type):
        L1 = Manager().list()
        partial_func = partial(self._convert_cdr, L1=L1)
        sub_poly     = df.query('output_file.notna()', engine='python')
        split_dfs    = array_split(sub_poly, cpu_count())
        ParallelPool(start_method='spawn', partial_func=partial_func, main_list=split_dfs, num_cores=cpu_count())
        df_extr    = concat(L1).reset_index().drop(columns=['index'])
        df_extr.to_parquet(f"{self.output_dir}/{feat_type}_performers_master.parquet")

        return df_extr

    def _feature_extraction_conversion(self):

        feature_dict = {}
        if len(self.polygon_infer_file) > 0:
            poly_extr = self._prep_and_execute_process(df=self.polygon_infer_file, feat_type="polygon")

            if len(poly_extr) > 0:
                feature_dict['polygon'] = poly_extr
                poly_extr.to_parquet(f"{self.output_dir}/polygon_schema.parquet")

        if len(self.line_infer_file) > 0:
            line_extr = self._prep_and_execute_process(df=self.line_infer_file, feat_type='line')

            if len(line_extr) > 0:
                feature_dict['line'] = line_extr
                line_extr.to_parquet(f"{self.output_dir}/line_schema.parquet")

        if len(self.point_infer_file) > 0:
            point_extr = self._prep_and_execute_process(df=self.point_infer_file, feat_type='point')

            if len(point_extr) > 0:
                feature_dict['point'] = point_extr
                point_extr.to_parquet(f"{self.output_dir}/point_schema.parquet")

        return feature_dict

    def _eval_df(self, eval_df: Union[str, DataFrame, dict], eval_id_field: str = "ID", eval_cog_id_field: str = "COG ID") -> dict:
        """
        Part of _link_cog_ids function - if applicable str or DataFrame, to read and convert to dictionary.

        :param eval_df: Dictionary that points (NGMDB) ID to the COG ID, or str that points to CSV file, or DataFrame
                        that needs to be converted to a dictionary.
        :param eval_id_field: Name of the ID field in the CSV or DataFrame. Default is None assuming that eval_df is a
                              dictionary.
        :param eval_cog_id_field: Name of the COG ID field in the CSV or DataFrame. Default is None assuming that eval_df
                                  is a dictionary.

        :return: Dictionary containing ID to COG ID.
        """

        # Construct dictionary
        if isinstance(eval_df, str) or isinstance(eval_df, DataFrame):
            if isinstance(eval_df, str):
                get_ext = os.path.splitext(eval_df)[-1]

                if get_ext == ".csv":
                    eval_df = read_csv(eval_df)
                    eval_df = eval_df[[eval_id_field, eval_cog_id_field]].drop_duplicates([eval_id_field, eval_cog_id_field])
                    eval_df = {i: c for i, c in zip(eval_df[eval_id_field], eval_df[eval_cog_id_field])}

                elif get_ext == ".json":
                    eval_df = DataEng.read_json(config_file = eval_df)

        return eval_df

    def _link_cog_ids(self, inhouse_feat: Union[DataFrame, str], eval_df: Union[DataFrame, str, dict],
                      eval_id_field: str, eval_cog_id_field: str) -> DataFrame:
        """
        Link the COG IDs to the (NGMDB) IDs.

        :param inhouse_feat: DataFrame or str (as a csv file) containing feature extraction data that are ready to be evaluated.
        :param eval_df: Dictionary that points (NGMDB) ID to the COG ID, or str that points to CSV file, or DataFrame
                        that needs to be converted to a dictionary.
        :param eval_id_field: Name of the ID field in the CSV or DataFrame. Default is None assuming that eval_df is a
                              dictionary.
        :param eval_cog_id_field: Name of the COG ID field in the CSV or DataFrame. Default is None assuming that eval_df
                                  is a dictionary.

        :return: Updated DataFrame containing the COG ID information.
        """
        eval_df = self._eval_df(eval_df           = eval_df,
                                eval_id_field     = eval_id_field,
                                eval_cog_id_field = eval_cog_id_field)

        eval_lambda = lambda x: eval_df[x]

        # If the georeference data is a string (aka csv file), then read it first
        if isinstance(inhouse_feat, str):
            get_ext = os.path.splitext(inhouse_feat)[-1]
            if get_ext == ".csv":
                inhouse_feat = read_csv(inhouse_feat)
            elif get_ext == ".parquet":
                inhouse_feat = read_parquet(inhouse_feat)

        inhouse_df  = inhouse_feat.assign(cog_id=lambda a: list(map(eval_lambda, a['id'])))

        return inhouse_df

    def _feat_cdr_pipeline(self, config: dict) -> List:
        """
        Main function to download data from the CDR per COG ID.

        :param config: Config dictionary.

        :return: List containing valuable information. Each inferenced download contains the following items:
            - when selecting the first --> list of COG URLs.
            - when selecting the second (which is what is used) --> Concatenated DataFrame in the following schema:
                - 'cog_id'          => COG ID
                - 'output_file'     => Path to the file that was exported
                - 'response_code'   => Response Code when requesting to download. Only applicable if there is a failed
                                       download for the first time.
                - 'message'         => Direct message that indicates whether the download has been success or not.
                                        "success-done" means downloaded all contents.
                                        "failed" - if there is an output file indicates that it downloaded stuff, but
                                                   not all contents successfully.
                                                 - if there is no output file indicates that it failed in the CDR either
                                                   no access, the COG ID does not exist, or the CDR server is not available.
                - 'feature_type'    => Feature type (polygon, line, or point).
                - 'system'          => Specific system (i.e., performer) downloading contents from.
                - 'sys_version'     => Specific version of that system for version control.
        """

        # Pull result data from the CDR instead & then do conversions required.
        feat_cdr         = FeatExtractCDR(config=config, cdr_systems=self.cdr_systems)

        # Polygon extraction rest-endpoint
        infer_polygon_df = feat_cdr.polygon_results(georeference_data=self.georeference_data,
                                                    legend_data=self.legend_data)

        # Line extraction rest-endpoint
        infer_lines_df   = feat_cdr.line_results(georeference_data=self.georeference_data,
                                                 legend_data=self.legend_data)

        # Point extraction rest-endpoint
        infer_points_df  = feat_cdr.point_results(georeference_data=self.georeference_data,
                                                  legend_data=self.legend_data)

        return [infer_polygon_df, infer_lines_df, infer_points_df]

    def _coord_poly_line_convert(self, tmp_df: DataFrame, crs,  cog_id, tif_file, transform,
                                 feat_type: str = Union['polygon', 'line']) -> Union[DataFrame, GeoDataFrame, None]:
        """
        Converting CDR JSON data containing pixel space to geographic coordinates or projected features,
        acquire spatial indexing, and convert to GeoDataFrame. For polygons and lines.

        :param tmp_df: DataFrame being converted.
        :param crs: EPSG system acquired from the TIF file.
        :param transform: Affine Transformation acquired from the TIF file.
        :param feat_type: str ==> either polygon or line to determine what Shapely geometry will we convert.

        :return: DataFrame with converted and supplemented information.
        """

        geom    = lambda x: Polygon(x) if feat_type == "polygon" else LineString(x)
        h3_info = lambda x: SpatialOps().auto_acqH3_fromgeom(geom=x, min_res=self.min_res, max_res=self.max_res)

        # For pixels-based only
        if self.georeference_data is False:

            # Lambda functions:
            # pre_coord => pixel space to geographic coordinate
            # geom      => create Polygon or LineString
            # h3_info   => Automatic H3 spatial indexing based on min and max resolution.
            pre_coord = lambda x: SpatialOps().pixel2coord_raster(raster_transform=transform, row_y_pix=x[0], col_x_pix=x[1])
            #self.feat_type = feat_type

            try:
                # Constructing GeoDataFrame process with necessary information.
                exp_df = (
                    tmp_df
                    # since the coordinates are nested, explode.
                    .explode('coordinates')
                    # Reset the index to trace which coordinates belong to which feature
                    .reset_index()
                    # Groupby the index and convert the individual pixel space coordinate pair to geographic coordinates.
                    .groupby('index', as_index=False)
                    .apply(lambda a: a.assign(pre_coord = lambda d: list(map(pre_coord, d['coordinates']))))
                    # Restack the converted geographic coordinates back to its index (i.e., reduce DataFrame size)
                    .groupby('index', as_index=False)
                    .agg({'pre_coord'   : lambda x: x.tolist(),
                          'coordinates' : lambda x: x.tolist()})
                    # Assign geometries
                    .assign(geom = lambda a: list(map(geom, a['pre_coord'])))
                    # Convert to GeoDataFrame
                    .pipe(lambda a: SpatialOps().pipe2gdf(df=a, geom_field='geom', epsg=crs.lower()))
                    # Re-project to EPSG:4326 to obtain H3 spatial indexing
                    .to_crs('epsg:4326')
                    # Concatenate with the GeoDataFrame to the DataFrame containing H3 spatial indexing information
                    .assign(inf = lambda d: list(map(h3_info, d['geom'])))
                    .pipe(lambda a: concat([a, DataFrame(list(a['inf']), columns=['min_h3', 'max_h3', 'min_res', 'max_res', 'geom_type'])], axis=1))
                    .drop(columns=['inf', 'geom_type', 'pre_coord'])
                    # Re-project back to the original spatial reference system
                    .to_crs(crs.lower())
                )

                return exp_df

            except pyproj.exceptions.CRSError:
                print(f"Invalid CRS projection, check {cog_id} and used {tif_file}")
                return None

        # For georeferenced / projected features
        else:

            try:
                exp_df = (
                    tmp_df
                    .reset_index()
                    .assign(geom = lambda a: list(map(geom, a['coordinates'])))
                    .pipe(lambda a: SpatialOps().pipe2gdf(df=a, geom_field='geom', epsg='epsg:4326')) # assumes that it is epsg:4326
                    .assign(inf = lambda d: list(map(h3_info, d['geom'])))
                    .pipe(lambda a: concat([a, DataFrame(list(a['inf']), columns=['min_h3', 'max_h3', 'min_res', 'max_res', 'geom_type'])],axis=1))
                    .drop(columns=['inf', 'geom_type'])
                    # Re-project back to the original spatial reference system
                    .to_crs(crs.lower())
                )

                return exp_df

            except pyproj.exceptions.CRSError:
                print(f"Invalid CRS projection, check {cog_id} and used {tif_file}")
                return None

    def _coord_point_convert(self, tmp_df: DataFrame, crs, cog_id, tif_file, transform) -> Union[DataFrame, GeoDataFrame, None]:
        """
        Converting CDR JSON data containing pixel space to geographic coordinates or projected features,
        acquire spatial indexing, and convert to GeoDataFrame. For points only.

        :param tmp_df: DataFrame being converted.
        :param crs: EPSG system acquired from the TIF file.
        :param transform: Affine Transformation acquired from the TIF file.

        :return: DataFrame with converted and supplemented information.
        """
        # Auto-generate H3 spatial indexing at min. and max. resolution.
        h3_info = lambda x: SpatialOps().auto_acqH3_fromgeom(geom=x,
                                                             min_res=self.min_res,
                                                             max_res=self.max_res)

        # For pixels-based only
        if self.georeference_data is False:

            # Create Point geometry during pixel space to geographic coordinate conversion
            point_func = lambda x: Point(SpatialOps().pixel2coord_raster(raster_transform = transform,
                                                                         row_y_pix        = x[1],
                                                                         col_x_pix        = x[0]))
            try:
                # Construct GeoDataFrame
                geom_extract = (
                    tmp_df
                    .reset_index()
                    # Assign geometry via Point function
                    .assign(geom=lambda a: list(map(point_func, a['coordinates'])))
                    # Convert DataFrame to GeoDataFrame
                    .pipe(lambda a: SpatialOps().pipe2gdf(df=a, geom_field='geom', epsg=crs.lower()))
                    # Temporarily re-project to epsg:4326 to conduct H3 spatial indexing
                    .to_crs('epsg:4326')
                    # H3 spatial indexing process and concat back to the original GeoDataFrame
                    .assign(inf=lambda d: list(map(h3_info, d['geom'])))
                    .pipe(lambda a: concat([a, DataFrame(list(a['inf']), columns=['min_h3', 'max_h3', 'min_res', 'max_res',
                                                                                  'geom_type'])], axis=1))
                    .drop(columns=['inf', 'geom_type'])
                    # Reproject back to the original spatial reference system
                    .to_crs(crs.lower())
                )

                return geom_extract

            except pyproj.exceptions.CRSError:
                print(f"Invalid CRS projection, check {cog_id} and used {tif_file}")
                return None

        # For georeferenced / projected features
        else:
            point_func = lambda x: Point(x)
            try:
                geom_extract = (
                    tmp_df
                    .reset_index()
                    .assign(geom=lambda a: list(map(point_func, a['coordinates'])))
                    .pipe(lambda a: SpatialOps().pipe2gdf(df=a, geom_field='geom', epsg='epsg:4326'))
                    # H3 spatial indexing process and concat back to the original GeoDataFrame
                    .assign(inf=lambda d: list(map(h3_info, d['geom'])))
                    .pipe(lambda a: concat([a, DataFrame(list(a['inf']), columns=['min_h3', 'max_h3', 'min_res', 'max_res',
                                                                                  'geom_type'])], axis=1))
                    .drop(columns=['inf', 'geom_type'])
                    # Reproject back to the original spatial reference system
                    .to_crs(crs.lower())
                )

                return geom_extract

            except pyproj.exceptions.CRSError:
                print(f"Invalid CRS projection, check {cog_id} and used {tif_file}")

                return None

    def _convert_cdr(self, data_info: DataFrame, L1):
        """
        During parallel processing - Convert CDR feature extracted data into GeoDataFrame and export as GeoParquet.
        This is also the development of the master file for inferenced files, which will facilitate the orchestration
        during evaluation metrics.

        :param data_info: Temporary orchestrated master set to read CDR files for extraction.
        :param cdr_systems: CDR System information as a dictionary.
        :param L1: List manager to append extraction information.
        """

        tmp_info = []
        for f in tqdm(data_info.itertuples(index=True), total=data_info.shape[0]):
            tmp_sys   = f.system
            tmp_sys_v = f.sys_version
            feat_type = f.feature_type
            cog_id    = f.cog_id
            message   = f.message
            tmp_file  = f.output_file
            get_id    = self.id_cog_list.query('cog_id == @cog_id')['id'].iloc[0]

            # Successfully downloaded and complete - proceed conversion process
            if message == "success-done" and tmp_file is not None:
                #print(tmp_file, feat_type)
                # Convert JSON inferenced CDR to DataFrame
                extract_df = FeatExtractFromCDR(cdr_file           = tmp_file,
                                                cdr_systems        = self.cdr_systems,
                                                feat_type          = feat_type,
                                                georeferenced_data = self.georeference_data,
                                                legend_data        = self.legend_data).extract_df

                # Need to double-check
                extract_df = extract_df.query('coordinates.notna() and p_num_geom > 0', engine='python')

                # Augmentation process - i.e., spatial conversions
                if extract_df is not None:

                    extract_df.to_parquet(f"{self.output_dir}/{cog_id}/inferenced/{feat_type}/{tmp_sys}__{tmp_sys_v}__{cog_id}__{feat_type}_extract.parquet")

                    # Reset index for data-structure safety
                    clean_extract = extract_df.reset_index().drop(columns=['index'])

                    # Two separate DataFrames - for organizational purposes and will be concatenated back
                    sys_extract  = clean_extract[['system', 'system_version', 'cog_id', 'p_num_geom', 'px_bbox',
                                                  'abbreviation', 'color', 'pattern']]
                    geom_extract = clean_extract[['type', 'coordinates']]

                    # Main output process
                    output_loc  = f"{self.output_dir}/{cog_id}/inferenced/{feat_type}"
                    DataEng.checkdir(dir_name=output_loc)

                    # Read TIF file that belongs to the COG ID via NGMDB ID - need this to acquire pixel-space (if applicable)
                    # to geographic coordinates
                    tif_file    = self.feat_tif_file.query('id == @get_id')['path'].iloc[0]
                    raster_char = SpatialOps().raster_characteristics(tif_file=tif_file)
                    crs         = raster_char[1]
                    transform   = raster_char[2]

                    del raster_char

                    if feat_type == "polygon" or feat_type == "line":

                        # Convert Legend Item bounding box to geographic coordinate Polygon - will be used for IoU
                        # during feature extraction metric eval pipeline.
                        if feat_type == "polygon":
                            px_bbox = list(map(lambda x: self._convert_bbox(x=x, transform=transform), clean_extract['px_bbox']))

                            # Assuming the legend exists (i.e., downloaded and filled with contents) and there is px_bbox;
                            # otherwise, will crash and cascade effects kicks in. Indicating importance.
                            output_anno  = "Spatial Index as alternative"
                            query_legend = self.legend_info.query('cog_id == @cog_id and response_code == 200 and message == "success-done"')
                            if len(query_legend) > 0:
                                legend_file    = query_legend['output_file'].iloc[0]
                                extract_legend = (
                                    LegendAnnotatedExtract(cdr_file=legend_file)
                                    .legend_df
                                    .assign(px_bbox = lambda a: list(map(lambda x: self._convert_bbox(x=x, transform=transform), a['px_bbox'])))
                                )

                                output_anno = f"{self.legend_output_dir}/{cog_id}/annotated/legend/{cog_id}_extracted.parquet"
                                extract_legend.to_parquet(output_anno)

                            # If there is no legend annotation - set to None (pre-determine that spatial indexing will get involved)
                            else:
                                extract_legend = None
                                output_anno    = "Use Spatial Index"

                        # For Lines
                        elif feat_type == "line":
                            px_bbox        = "Use Abbreviation"
                            output_anno    = "Use Abbreviation"
                            extract_legend = None

                        del extract_df, clean_extract

                        geom_df = self._coord_poly_line_convert(tmp_df    = geom_extract,
                                                                feat_type = feat_type,
                                                                crs       = str(crs),
                                                                cog_id    = cog_id,
                                                                tif_file  = tif_file,
                                                                transform = transform)

                    # For points
                    elif feat_type == "point":
                        px_bbox        = "Use Abbreviation"
                        output_anno    = "Use Abbreviation"
                        extract_legend = None
                        del extract_df, clean_extract

                        geom_df     = self._coord_point_convert(tmp_df    = geom_extract,
                                                                crs       = str(crs),
                                                                cog_id    = cog_id,
                                                                tif_file  = tif_file,
                                                                transform = transform)

                    if geom_df is not None:
                        # Concatenate system-extract DataFrame with the GeoDataFrame and assign legend bounding box
                        concat_geom = concat([sys_extract, geom_df], axis=1).assign(px_bbox=px_bbox)

                        # Legend abbreviation assign
                        if output_anno is not None:
                            abbrev_values = self._legend_iou(performer_df = concat_geom,
                                                             legend_df    = extract_legend,
                                                             get_id       = get_id)

                            fin_geom = concat([concat_geom, abbrev_values], axis=1)

                            #print(get_id, fin_geom.columns)
                            abbrev_val_grp = (
                                fin_geom
                                .assign(count = 1)
                                .groupby(['abbreviation', 'geo_value'], as_index=False)
                                .agg({'count'         : 'sum',
                                      #'geo_value'     : 'unique',
                                      'matched_ratio' : 'unique',
                                      'match_binary'  : 'unique'})
                                .pipe(lambda a: a.assign(perc = lambda d: d['count'] / sum(d['count'])))
                                .explode(['geo_value', 'matched_ratio', 'match_binary'])
                            )

                            #print(fin_geom.columns)

                        else:
                            concat_geom['geo_value']     = None
                            concat_geom['matched_ratio'] = None
                            concat_geom['legend_iou']    = None
                            abbrev_val_grp               = None
                            fin_geom = concat_geom.copy(deep=True)

                            del concat_geom

                        # Delete to free-up memory
                        del geom_df, sys_extract, geom_extract

                        # Convert the concatenated DataFrame to GeoDataFrame and export as a geoparquet file
                        output_file = f"{output_loc}/{tmp_sys}__{tmp_sys_v}__{cog_id}__{feat_type}.geoparquet"
                        geom_gdf    = SpatialOps().pipe2gdf(df=fin_geom, geom_field='geom', epsg=str(crs).lower())

                        #print(geom_gdf.columns, get_id)
                        SpatialOps().geoparquet(gdf=geom_gdf, output_path=output_file)

                        if abbrev_val_grp is not None:
                            output_set = f"{output_loc}/{tmp_sys}__{tmp_sys_v}__{cog_id}__{feat_type}__synopsis.parquet"
                            abbrev_val_grp.to_parquet(output_set)
                        else:
                            output_set = None

                        # Append the information that will be constructed as a master file for inferenced.
                        tmp_info.append([get_id, cog_id, tmp_sys, tmp_sys_v, feat_type, message, 'extracted data',
                                         str(crs).lower(), output_file, output_anno, output_set, self.georeference_data, self.legend_data])

                    else:
                        tmp_info.append([get_id, cog_id, tmp_sys, tmp_sys_v, feat_type, message, f"CRS failure - {str(crs).lower()}",
                                         None, None, None, None, self.georeference_data, self.legend_data])

                # Append information that will be constructed as a master file for inferenced - except no data was extracted
                else:
                    tmp_info.append([get_id, cog_id, tmp_sys, tmp_sys_v, feat_type, message, 'no-data extracted/missing features',
                                     None, None, None, None, self.georeference_data, self.legend_data])

            else:
                tmp_info.append([get_id, cog_id, tmp_sys, tmp_sys_v, feat_type, message, 'no-data',
                                 None, None, None, None, self.georeference_data, self.legend_data])

        L1.append(DataFrame(tmp_info, columns=['id', 'cog_id', 'system', 'system_version', 'feature_type',
                                               'message', 'data', 'crs', 'output_file', 'legend_extracted',
                                               'output_synopsis', 'georeferenced_data', 'legend_data_added']))

    def _legend_iou(self, performer_df, legend_df, get_id):
        iou_data     = []
        for p,a in zip(performer_df['px_bbox'], performer_df['abbreviation']):
            if (p != "Use Spatial Index") and (p != "Use Abbreviation") and (p is not None):
                try:
                    tmp_data = []
                    p        = loads(p)
                    for d in legend_df['px_bbox']:
                        d = loads(d)
                        tmp_data.append(SpatialOps().IoU_calc(geom1=p, geom2=d))
                    max_iou     = max(tmp_data)
                    idx_lgnd    = tmp_data.index(max_iou)
                    lgnd_abbrev = re.sub(r'\W+', '', legend_df['abbreviation'].iloc[idx_lgnd])
                    lgnd_label  = re.sub(r'\W+', '', legend_df['label'].iloc[idx_lgnd])

                    if lgnd_abbrev is None:
                        iou_data.append([lgnd_label, None, max_iou, None])

                    else:
                        iou_data.append([lgnd_abbrev, None, max_iou, None])

                except TypeError:
                    iou_data.append(["Use Spatial Index", None, None, None])

            else:
                if p == "Use Spatial Index" or p is None:
                    iou_data.append(["Use Spatial Index", None, None, False])

                elif p == "Use Abbreviation":
                    a = re.sub(r'\W+', ' ', a).lower()

                    sub_bin = (
                        self.binary_df
                        .query('id == @get_id')
                        .explode(['geo_value_vector'])
                    )

                    if len(sub_bin) > 0:
                        pre_values  = [[d.lower(), d] for d in sub_bin['geo_value_vector'].unique()]
                        geo_values  = [p[0] for p in pre_values]
                        uniq_values = [p[1] for p in pre_values]
                        match       = self._sequence_matcher(str1=a, str2_list=geo_values, threshold=self.match_threshold)

                        if len(geo_values) > 0:
                            if match[0] != "Use Spatial Index":
                                match_value = geo_values.index(match[0])
                                get_value   = uniq_values[match_value]
                                iou_data.append([get_value, match[1], None, True])

                            else:
                                if self.crude_grp_binary is not None:
                                    sub_bin = (
                                        self.crude_grp_binary
                                        .query('id == @get_id')
                                        .explode(['unmatched_binary'])
                                    )

                                    pre_values  = [[d.lower(), d] for d in sub_bin['unmatched_binary'].unique()]
                                    geo_values  = [p[0] for p in pre_values]
                                    uniq_values = [p[1] for p in pre_values]

                                    if len(geo_values) > 0:
                                        match = self._sequence_matcher(str1=a,
                                                                       str2_list=geo_values,
                                                                       threshold=self.match_threshold)

                                        if match[0] != "Use Spatial Index":
                                            match_value = geo_values.index(match[0])
                                            get_value   = uniq_values[match_value]
                                            iou_data.append([get_value, match[1], None, False])
                                            #print('success', get_id)
                                        else:
                                            iou_data.append([match[0], match[1], None, False])

                                    else:
                                        iou_data.append(["Use Spatial Index", None, None, False])

                                else:
                                    iou_data.append(["Use Spatial Index", None, None, False])

                        else:
                            iou_data.append([match[0], match[1], None, False])

                    elif self.crude_grp_binary is not None:
                        sub_bin = (
                            self.crude_grp_binary
                            .query('id == @get_id')
                            .explode(['unmatched_binary'])
                        )

                        pre_values  = [[d.lower(), d] for d in sub_bin['unmatched_binary'].unique()]
                        geo_values  = [p[0] for p in pre_values]
                        uniq_values = [p[1] for p in pre_values]

                        if len(geo_values) > 0:

                            match = self._sequence_matcher(str1=a, str2_list=geo_values, threshold=self.match_threshold)

                            if match[0] != "Use Spatial Index":
                                match_value = geo_values.index(match[0])
                                get_value   = uniq_values[match_value]
                                iou_data.append([get_value, match[1], None, False])

                            else:
                                iou_data.append([match[0], match[1], None, False])

                        else:
                            iou_data.append(["Use Spatial Index", None, None, False])


        """
        match_binary interpretation (applies for polygons only) -- ensure that feature type is used 
            - valid geo_value, None    ==> Identified geologic value from performer based on IoU from their pixel bounding box
            - Use Spatial Index, False ==> Forcefully use spatial index, no matches in the binary 
            
            - Under "Use Abbreviation" (applies for points and lines)
                - valid geo_value, True      ==> Identified geologic value from performer based on matching from matched binary set. 
                - valid geo_value, False     ==> Identified geologic value from performer based on matching from the crude binary set.
                - "Use Spatial Index", False ==> Forcefully use spatial index, no matches in the binary neither crude set; 
                                                 resort to the crude set files based on feature type (similar geometry types).  
        """

        return DataFrame(iou_data, columns=['geo_value', 'matched_ratio', 'legend_iou', 'match_binary'])

    def _convert_bbox(self, x, transform):
        try:
            # Might need to change the ordering back
            min_points = SpatialOps().pixel2coord_raster(raster_transform=transform, row_y_pix=x[1], col_x_pix=x[0])
            max_points = SpatialOps().pixel2coord_raster(raster_transform=transform, row_y_pix=x[3], col_x_pix=x[2])

            maxx = max_points[0]
            minx = min_points[0]
            maxy = min_points[1]
            miny = max_points[1]

            bbox = box(minx=minx, miny=miny, maxx=maxx, maxy=maxy).wkt

            return bbox

        except TypeError:
            return "Use Spatial Index"

    def _sequence_matcher(self, str1, str2_list, threshold):
        seq_match = ReMethods().max_sequence_matcher(str1      = str1,
                                                     str2_list = str2_list)

        try:
            if seq_match[1] > threshold:
                return [seq_match[0], seq_match[1]]

            else:
                return ["Use Spatial Index", seq_match[1]]

        except TypeError:
            return ["Use Spatial Index", None]

    def _local_cdr_conversion(self, local_schema_process: List):

        for cdr_file in tqdm(local_schema_process):
            data   = CDR2Vec(cdr_file = cdr_file)
            system = data.json_data['system']
            sys_v  = data.json_data['system_version']
            cog_id = data.json_data['cog_id']
            # Feature extract
            conv_d = data.extracted

            # Need to confirm with UIUC, Uncharted, and UMN with their tools that are locally saved
            # how to retrieve COG ID.
            #return [None, None, None]
