# import dc_extract
import dc_extract.describe.describe as d
from dc_extract.extract_cog import extract_cog as exc
import hydra 
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import util as U
import logging

def dc_describe(cfg: DictConfig):
    '''
    Configurate the call of the d.describe() with hydra parameters.
    '''
    args = OmegaConf.create(cfg.parameters['dc_description'])
    d.collections(name = args['name'])
    return True

def dc_serach(cfg: DictConfig):
    '''
    Configurate the call of the d.search()  with hydra parameters.
    '''
    args = OmegaConf.create(cfg.parameters['dc_search'])
    print(args) 
    d.search(name= r'C:/Users/abfernan/CrossCanFloodMapping/GISAutomation/AutomationOutput/example/testSearch.gpkg',
             bbox='-123.3250,51.9500,-120.1473,53.7507',
             cols= 'cdem',
             )
    return True

def dc_extraction(cfg: DictConfig):
    '''
    Configurate the call of the exc.extract_cog() with hydra parameters.
    '''
    args = OmegaConf.create(cfg.parameters['dc_extrac_cog'])
    print(args) 
    # exc.extract_cog(bbox=args['bbox'],
    #                 bbox_crs= args['bbox_crs'],
    #                 out_dir= cfg['output_dir'],
    #                 collections=args['collection'],)
    return True

def logger(cfg: DictConfig, nameByTime):
    '''
    You can log all you want here!
    '''
    logging.info(f"Excecution number: {nameByTime}")
    logging.info(f"Output directory :{cfg['output_dir']}")
    logging.info(f"dc_search inputs: {cfg.parameters.dc_search}")
    logging.info(f"dc_description inputs: {cfg.parameters.dc_description}")
    logging.info(f"dc_extract inputs: {cfg.parameters.dc_extrac_cog}")

@hydra.main(version_base=None, config_path=f"config", config_name="mainConfigPC")
def main(cfg: DictConfig):
    nameByTime = U.makeNameByTime()
    logger(cfg,nameByTime)
    # d = dc_describe(cfg)
    # logging.info(f"Description output path: {d}")
    # print(f"Describe -->{d}")
    se = dc_serach(cfg)
    logging.info(f"Search output path: {se}")
    print(f"Search-->{se}")
    # ex = dc_extraction(cfg)
    # logging.info(f"Extraction output path: {ex}")
    # print(f"Extraction -->{ex}")

if __name__ == "__main__":
    with U.timeit():
        main()  
