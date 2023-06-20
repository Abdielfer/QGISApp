# import dc_extract
import dc_extract.describe.describe as d
from dc_extract.extract_cog import extract_cog as exc
name = r'C:/Users/abfernan/CrossCanFloodMapping/CatchmentsPlyg/Test.gpkg'
out_dir_local = r'C:/Users/abfernan/CrossCanFloodMapping/CatchmentsPlyg/testFullBasin' # Optional defaults to dc_extract/describe/data/{collections_or_search}/dce.gpkg
 # Optional defaults to datacube prod
# collections and asset descriptions only
# d.collections(name)

# Once user has supplied optional filter parameters
bbox = '-123.3250,51.9500,-120.1473,53.7507' # Optional defaults to None, no bbox filter applied
# datetime = None # Optional defaults to None, no bbox filter applied
collections = 'cdem' #example : 'landcover'
collection_asset = 'cdem:dem'
d.search(name,bbox=bbox,cols=collections)
exc.extract_cog(bbox=bbox,bbox_crs='EPSG:4326',out_dir= out_dir_local,collections=collection_asset)#out_dir=name,geom_file='path to gpkg'

# 52.92827,-122.53823
# 53.02376,-122.46681
# crs: 'EPSG:4326'
