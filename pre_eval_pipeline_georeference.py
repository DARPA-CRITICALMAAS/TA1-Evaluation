"""
Author: Anastassios Dardas, PhD - Lead Geospatial Computing Engineer

Date Created: Sept. 2023

Date Modified: Oct. 2023

About:

Note: To be run only once unless completely new ground-truth data or on a clean slate of the original ground-truth data
      that has no pre-process files added.
"""

from etls.eval_prep import PrepEval
from etls.georeferencing import GeorefInspect

if __name__ == "__main__":
    """
    Georeferencing and Feature Extraction inventory of the ground-truth set. 
    
    16 CPU machine => 22 seconds. 
    """

    output_path = 'data/ground_truth'
    data        = PrepEval(kwargs='configs/inventory.json', output_path=output_path)

    georef_path     = data.inventory_data.georef_path
    inhouse_georef  = data.in_house_georef_invent
    missing_georef  = data.missing_georef_invent
    georef_validate = GeorefInspect(georef_path       = georef_path,
                                    inhouse_inventory = inhouse_georef)

    tif_files       = georef_validate.tif_files
    tif_grp_id      = georef_validate.tif_grp
    tot_miss_georef = georef_validate.total_missing

    tif_files.to_parquet(f"{output_path}/georef_tif_files.parquet")
    eval_dict = f"{output_path}/cog_id_info.json"
