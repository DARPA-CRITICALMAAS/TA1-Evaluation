"""
Author: Anastassios Dardas, PhD - Lead Geospatial Computing Engineer

Date Created: Sept. 2023

Date Modified: Oct. 2023

About:
"""

from etls.eval_prep import PrepEval
from etls.georeferencing import GeorefInspect, PreEvalGeoref, GeorefEval

if __name__ == "__main__":
    """
    Georeferencing and Feature Extraction inventory of the ground-truth set. 
    
    16 CPU machine => 22 seconds. 
    """
    data = PrepEval(kwargs='configs/inventory.json', output_path='data/ground_truth')

    georef_path    = data.inventory_data.georef_path
    inhouse_georef = data.in_house_georef_invent
    missing_georef = data.missing_georef_invent
    georef_validate = GeorefInspect(georef_path       = georef_path,
                                    inhouse_inventory = inhouse_georef)

    tif_files  = georef_validate.tif_files
    tif_grp_id = georef_validate.tif_grp
    tot_miss_georef = georef_validate.total_missing

    """
    16 CPU machine => 22 seconds. 
    """
    georef_eval_gcps = PreEvalGeoref(tif_files = tif_files,
                                     eval_df   = data.eval_dict)

    # Change here
    cdr_token   = "---Add your token here ---"
    output_dir  = "data/inferenced_cdr/Georeferencing"
    cdr_systems = {
        "annotated"  : [{"system" : "ngmdb", "system_version": "2.0"},
                        {"system" : "polymer", "system_version" : "0.0.1"}],
        "performers" : [{"system" : "uncharted-georeference", "system_version": "0.0.6"},
                        {"system" : "umn-usc-inferlink", "system_version": "0.0.7"}]
    }

    """
    16 CPU machine => 11 seconds. 
    """
    georef_metrics = GeorefEval(cdr_token      = cdr_token,
                                georef_dataset = georef_eval_gcps.concat_df,
                                output_dir     = output_dir,
                                cdr_systems    = cdr_systems)

    #failed_cogs_download = georef_metrics.failed_download
    rmse_data            = georef_metrics.concat_df
    #rmse_data.to_parquet(f"{output_dir}/rmse_metrics_georeferencing.parquet")

    augment_data = (
        data
        .inventory_data
        .eval_df
        .merge(rmse_data, left_on=['COG ID'], right_on=['cog_id'])
    )

    #augment_data = rmse_data.merge(data.inventory_data.eval_df, left_on=['cog_id'], right_on=['COG ID'])
    #augment_data.to_csv("data/final_metrics/georeferencing_rmse_12_month.csv", index=False)
    #rmse_data_missing = rmse_data.query('log == "annotated-missing"')
