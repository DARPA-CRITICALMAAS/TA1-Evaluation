"""
Author: Anastassios Dardas, PhD - Lead Geospatial Computing Engineer

Date Created: Nov. 2024

Date Modified: Dec. 2024

About: Feature Extraction Evaluation & Aggregate the data

Notes: Evaluating a geometry to geometry basis between two sets (one as inferenced and the other as ground-truth) can
       easily reach to billions of combinations if done naively. Not only is this not computationally efficient, but
       it is not sustainable and can take forever to complete. Fortunately, this has been avoided by implementing four
       orders of filtering processes and a bit reduction in data skewness per CPU during parallelization. The four orders
       of filtering are:
            - Performer set to Ground-truth set - to filter
                - By NGMDB ID / COG ID
                - Partitioned geologic value (if applicable) otherwise use the crude match instead (i.e., geometry type).
                - Filter by matched spatial index at the minimum level.

            - For each candidate feature from the performer
                - Convert the feature to pixel space in preparation for F1 as final step
                - Filter ground-truth set that matches to the last level spatial index
                - Spatial overlap
                - Perform IoU and then keep only the ground-truth feature that had the highest IoU.
                - Convert ground-truth geometry to pixel space in prep. for F1.
                - Perform F1.

       This approach makes it vastly faster; however, it does take time to complete the process.
        - IoU + F1 => Polygons (23 hrs. 55 min); Points (6 hrs. 6 min); Lines (4 hrs.)
        - IoU only (comment out F1) => Polygons (5 min. 11 sec.); Points (7 min. 10 sec.); Lines (2 min. 57 sec.)

        To make the F1 process substantially faster, the geometries of the ground-truth data will likely need to be
        pre-computed into pixel space prior during the pre-eval ground-truth pipeline as a modification.
"""

from etls.feature_extraction import FeatureEval
from etls.feature_extraction import AggMetrics


# Ground-truth compiled information - do not change
grnd_match_schema = 'data/ground_truth/Feature Extraction/ft_match_binary_grnd_schema.parquet'
grnd_crude_schema = 'data/ground_truth/Feature Extraction/ft_crude_match_grnd_schema.parquet'
feat_tif_file     = 'data/ground_truth/ft_tif_files.parquet'

# Performer compiled information - change only the directory path if needed (i.e., data/inferenced_cdr/Feature Extraction/)
# Refer to pre_eval_feature_extract_performer.py the output_inf_dir parameter you've set it to.
inf_polygon_schema = 'data/inferenced_cdr/Feature Extraction/polygon_schema.parquet'
inf_line_schema    = 'data/inferenced_cdr/Feature Extraction/line_schema.parquet'
inf_point_schema   = 'data/inferenced_cdr/Feature Extraction/point_schema.parquet'

# Output directory for where the evaluation results are to be stored during evaluation.
output_dir = "data/final_metrics"

if __name__ == "__main__":
    feat_eval = FeatureEval(grnd_match_schema  = grnd_match_schema,
                            grnd_crude_schema  = grnd_crude_schema,
                            inf_polygon_schema = inf_polygon_schema,
                            inf_line_schema    = inf_line_schema,
                            inf_point_schema   = inf_point_schema,
                            feat_tif_file      = feat_tif_file,
                            output_dir         = output_dir,
                            eval_set           = "data/eval_final_set.csv",
                            evaluate_feature   = "all", # performers can change this to 'point', 'line', 'polygon', or 'all' for what feature type to evaluate
                            dynamic_buffer     = 0.05)

    data = AggMetrics(polygon_results_path = f'{output_dir}/polygon',
                      line_results_path    = f'{output_dir}/line',
                      point_results_path   = f'{output_dir}/point',
                      eval_set             = 'data/eval_final_set.csv')