"""
Author: Anastassios Dardas, PhD - Lead Geospatial Computing Engineer

Date Created: Oct. 2023

Date Modified: Nov. 2023

About:
"""

from etls.feature_extraction import PreprocessInf

if __name__ == "__main__":
    """
    3rd Pre-Eval Pipeline (Performers & supplemental ground-truth)
        - Download annotated legend items from the CDR
            - Convert annotated legend items and export as GeoParquet
        - Download performer results from the CDR 
            - Convert downloaded results from CDR via JSON to DataFrame
            - If applicable (not georeferenced), convert pixel space coordinates to geographic coordinates (projected)
            - Export data as GeoParquet and create master file for inferenced data

    Process time: 16 CPU EC2
        Multithreading download from CDR 
            -->  0 min.  1 sec. (Legend Annotation download)
            --> 15 min. 43 sec. (Polygon performers download from CDR)
            -->  1 min.  5 sec. (Line performers download from CDR)
            -->  0 min. 14 sec. (Point performers download from CDR)

        Total: 17 min. 3 sec. 

        Conversion, Indexing, Identifying geologic values per geometry
            --> 17 min. 17 sec. (Polygons)
            -->  7 min. 39 sec. (Lines) 
            -->  1 min. 09 sec. (Points) 

        Total: 26 min. 05 sec. 

    Total: 
    """

    # Parameters where performers / users may need to make a change
    cdr_token   = "---Add your token here---"  # Performers change this
    cdr_systems = {
        "performers": [{"system": "uiuc-icy-resin", "system_version": "0.4.6"},  # Performers change this
                       {"system": "uncharted-points", "system_version": "0.0.5"},
                       {"system": "umn-usc-inferlink", "system_version": "0.0.5"}]
    }

    output_inf_dir = "data/inferenced_cdr/Feature Extraction"  # Performers can change this to their desired output

    # Performers/ users to leave these parameters as is - these are from previous processes (i.e., Ground-truth)
    legend_output_dir = "data/ground_truth/annotated_legend"  # Ground-truth pre-process (do not change)
    inhouse_feat      = "data/ground_truth/ft_inhouse_inventory.parquet"  # Ground-truth pre-process (do not change)
    eval_dict         = "data/ground_truth/cog_id_info.json"  # Ground-truth pre-process (do not change)
    match_binary      = "data/ground_truth/ft_match_binary.parquet"  # Ground-truth pre-process (do not change)
    crude_grp         = "data/ground_truth/Feature Extraction/ft_crude_grp.parquet"  # Ground-truth pre-process (do not change)
    feat_tif_file     = "data/ground_truth/ft_tif_files.parquet"  # Ground-truth pre-process (do not change)

    performer_cdr = PreprocessInf(cdr_token         = cdr_token,
                                  cdr_systems       = cdr_systems,
                                  inhouse_feat      = inhouse_feat,
                                  eval_df           = eval_dict,
                                  binary_df         = match_binary,
                                  crude_grp_binary  = crude_grp,
                                  feat_tif_file     = feat_tif_file,
                                  output_dir        = output_inf_dir,
                                  legend_output_dir = legend_output_dir)
