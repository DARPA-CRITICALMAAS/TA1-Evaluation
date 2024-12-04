# Classes to access & download data from the CDR
from .cdr_bulk_cog_pull import BulkCOG
from .cdr_inferenced_outputs import CDR2Vec, FeatExtractFromCDR, LegendAnnotatedExtract
from .cdr_bulk_legenditems_pull import LegendItemsCDR
from .cdr_bulk_gcps import GCPS
from .cdr_cog_ngmdb_check import COG_NGMDB_ID
from .cdr_feature_extract_pull import FeatExtractCDR
from .gen_cdr import GeorefConvert, CDR2Data
