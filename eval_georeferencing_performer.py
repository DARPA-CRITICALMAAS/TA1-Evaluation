from etls.georeferencing import PreEvalGeoref, GeorefEval
from pandas import read_csv

if __name__ == "__main__":
    """
    16 CPU machine => 22 seconds. 
    """

    # Do not change this - the output path of these files were defined in an earlier process (pre_eval_pipeline_georeference.py)
    tif_files = "data/ground_truth/georef_tif_files.parquet"
    eval_dict = "data/ground_truth/cog_id_info.json"

    georef_eval_gcps = PreEvalGeoref(tif_files=tif_files,
                                     eval_df=eval_dict)

    # Change here
    cdr_token = "---Add your token here ---"
    output_dir = "data/inferenced_cdr/Georeferencing"
    cdr_systems = {
        "annotated": [{"system": "ngmdb", "system_version": "2.0"},
                      {"system": "polymer", "system_version": "0.0.1"}],
        "performers": [{"system": "uncharted-georeference", "system_version": "0.0.6"},
                       {"system": "umn-usc-inferlink", "system_version": "0.0.7"}]
    }

    """
    16 CPU machine => 11 seconds. 
    """
    georef_metrics = GeorefEval(cdr_token=cdr_token,
                                georef_dataset=georef_eval_gcps.concat_df,
                                output_dir=output_dir,
                                cdr_systems=cdr_systems)

    # failed_cogs_download = georef_metrics.failed_download
    rmse_data = georef_metrics.concat_df
    rmse_data.to_parquet(f"{output_dir}/rmse_metrics_georeferencing.parquet")

    # Do not change this.
    eval_df      = read_csv("data/eval_final_set.csv")
    augment_data = (
        eval_df
        .merge(rmse_data, left_on=['COG ID'], right_on=['cog_id'])
    )

    # You may change the output_dir to another folder
    augment_data.to_csv(f"{output_dir}/georeferencing_rmse_12_month.csv", index=False)
    # rmse_data_missing = rmse_data.query('log == "annotated-missing"')
