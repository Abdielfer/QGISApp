# import dc_extract
import dc_extract.describe.describe as d
from dc_extract.extract_cog import extract_cog as exc
import hydra 
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import util as U
import logging

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
exc.extract_cog(bbox=bbox,bbox_crs='EPSG:4326',out_dir= out_dir_local,collections=collection_asset)

def dc_extraction(cfg: DictConfig):
    args = OmegaConf.create(cfg.parameters['dc_extraction'])
    exc.extract_cog(bbox=bbox,bbox_crs='EPSG:4326',out_dir= out_dir_local,collections=collection_asset)
    return True

def dc_serach(cfg: DictConfig):
    args = OmegaConf.create(cfg.parameters['dc_searsh'])
    d.search(name=args['name'],bbox=args['bbox'],cols=args['collection'])
    return True

    

def logger(cfg: DictConfig, nameByTime):
    '''
    You can log all you want here!
    '''
    logging.info(f"Excecution number: {nameByTime}")
    logging.info(f"Output directory :{cfg['output_dir']}")
    logging.info(f"dc_extract inputs: {cfg.parameters.dc_extraction}")

@hydra.main(version_base=None, config_path=f"config", config_name="mainConfigPC.yaml")
def main(cfg: DictConfig):
    nameByTime = U.makeNameByTime()
    logger(cfg,nameByTime)
    
if __name__ == "__main__":
    with U.timeit():
        main()  
