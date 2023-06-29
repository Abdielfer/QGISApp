# import dc_extract
import dc_extract.describe.describe as d
from dc_extract.extract_cog import extract_cog as exc
import hydra 
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import util as U
import logging 
# from wbw_test import checkIn as chIn   ### IMPORTANT ###: DO NOT USE. If two instance the license are created it can kill my license. Thank you!!


def dc_describe(cfg: DictConfig):
    '''
    Configurate the call of d.describe() with hydra parameters.
    '''
    instantiate(OmegaConf.create(cfg.parameters['dc_describeCollections']))
    return True

def dc_serach(cfg: DictConfig):
    '''
    Configurate the call of d.search()  with hydra parameters.
    '''
    out = instantiate(OmegaConf.create(cfg.parameters['dc_search']))
    return out

def dc_extraction(cfg: DictConfig):
    '''
    Configurate the call of extract_cog() with hydra parameters.
    '''
    out = instantiate(OmegaConf.create(cfg.parameters['dc_extrac_cog']))
    return out

def logger(cfg: DictConfig, nameByTime):
    '''
    You can log all you want here!
    '''
    logging.info(f"Excecution number: {nameByTime}")
    logging.info(f"Output directory :{cfg['output_dir']}")
    # logging.info(f"dc_search inputs: {cfg.parameters.dc_search}")
    # logging.info(f"dc_description inputs: {cfg.parameters.dc_describeCollections}")
    logging.info(f"dc_extract inputs: {cfg.parameters.dc_extrac_cog}")

@hydra.main(version_base=None, config_path=f"config", config_name="mainConfigPC")
def main(cfg: DictConfig):
    # nameByTime = U.makeNameByTime()
    # logger(cfg,nameByTime)
    # # dc_describe(cfg)
    # # dc_serach(cfg)
    # ex = dc_extraction(cfg)
    # logging.info(f"Extraction output path: {ex}")
    instantiate(OmegaConf.create(cfg.transformation['clipRasterGdal']))
    # chIn   # To check in the wbtools license

if __name__ == "__main__":
    with U.timeit():
        main()  
