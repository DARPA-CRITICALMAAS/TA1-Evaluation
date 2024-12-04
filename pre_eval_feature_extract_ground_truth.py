"""
Author: Anastassios Dardas, PhD - Lead Geospatial Computing Engineer

Date Created: Sept. 2023

Date Modified: Nov. 2023

About: First two of the three pre-eval pipeline processes. The pre-eval pipeline of the ground-truth does the following:

    PrepEval class
    -------------
    - Check inventory --> any missing ID from master ground-truth set of IDs.
                      --> any missing TIF and/or shapefiles

    - Conducts geologic value matches --> using binary raster filename to retrieve geologic values and find matches in
                                          the shapefiles to identify which observations in the ground-truth set can be
                                          evaluated. If there are no matches, then a more "brute-force" approach would
                                          be implemented based on same geometry types.

    PreprocessVector
    ----------------
    - Matches --> partition out the ground-truth data directly as geoparquet file. The partition process is in the form
                  of "ground_truth/Feature Extraction/{ngmdb_id}/partition/matched/{name_of_shapefile_len_of_matched_values}/{geologic_value}/sub_geog.geoparquet

    - crude_match --> not much of a partition process since we don't know exactly where to look for in the ground-truth
                      data. Instead convert shapefiles that have been matched by geometry type to the unmatched geologic
                      values (e.g., a point geologic value) to geoparquet.

    Both processes have hierarchical spatial indexing.

Note: To be run only once unless completely new ground-truth data or on a clean slate of the original ground-truth data.
"""

from etls.eval_prep import PrepEval
from etls.feature_extraction import PreprocessVector

if __name__ == "__main__":
    """
    1st Pre-Eval Pipeline (Ground-truth)
        - Check inventory of ground-truth data and prepare for sub-sequent pre-eval pipelines 
          in preparation for the eval pipeline.  
          
    WARNING: Make a copy of the Feature Extraction folder before running this and especially the next pre-eval pipeline. 
             In case need to debug or re-run there cannot be post-process files in there; otherwise, it will crash.
             
    Process time: 16 CPU 
        --> 0 min. 21 sec. (main inventory & scanning shapefiles)
        --> 0 min. 13 sec. (crude-match process - scanning shapefiles)
    
    Total: 0 min. 34 sec.               
    """

    ground_data  = PrepEval(kwargs='configs/inventory.json', output_path='data/ground_truth')

    # Important outputs - can be viewed in the output_path
    feat_path      = ground_data.inventory_data.feature_path
    inhouse_feat   = ground_data.in_house_ft_invent
    match_binary   = ground_data.match_binary_vector
    crude_match    = ground_data.crude_match
    unmatch_binary = ground_data.unmatch_binary_vector_final
    feat_tif_file  = ground_data.inventory_data.prep_invent_ftif[0]
    #missing_feat   = ground_data.missing_ft_invent
    eval_dict      = ground_data.eval_dict

    """
    2nd Pre-Eval Pipeline (Ground-truth)
        - Partition out the ground-truth data (pre-orchestration step)
        - Dependent on the 1st Pre-Eval Pipeline 
    
    Process time: 16 CPU 
        --> 1 min. 37 sec. (match-binary partition & indexing process)
        --> 0 min. 57 sec. (crude-match partition & indexing process)
        
    Total time: 2 min. 34 sec. 
    """
    preprocess_grnd = PreprocessVector(match_binary       = match_binary,
                                       crude_match_binary = crude_match,
                                       output_dir         = feat_path)

    # Important outputs - can be viewed in the output_dir (i.e., 'data/ground_truth/Feature Extraction')
    crude_grp    = preprocess_grnd.crude_grp
    match_schema = preprocess_grnd.match_binary_grnd_schema
    crude_schema = preprocess_grnd.crude_match_grnd_schema
