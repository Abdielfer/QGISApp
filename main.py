# import dc_extract
import os
import dc_extract.describe.describe as d
from dc_extract.extract_cog import extract_cog as exc
import hydra 
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import util as U
import logging 
# from wbw_test import checkIn as chIn   ### IMPORTANT ###: DO NOT USE. If two instance the license are created it can kill my license. Thank you!!

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
    # shpFile = cfg.transformation['mask']
    # bbox = U.get_Shpfile_bbox(shpFile)
    # print(f"The bbox is: {bbox}")
    # U.dc_describe(cfg)
    # U.dc_serach(cfg)
    # U.dc_extraction(cfg)
    # logging.info(f"Extraction output path: {ex}")
    # instantiate(OmegaConf.create(cfg.transformation['clipRasterGdal']))
    # chIn   # To check in the wbtools license

    # cliped = instantiate(OmegaConf.create(cfg.transformation['clipRasterGdal']))
    # cliped = U.clipRasterGdal(ras_in,mask,ras_out)
    # U.reproject_tif(cliped,'EPSG:4326')
    config = OmegaConf.create(cfg.transformation['crop_tif'])
    tif = config['tif_file']
    # rastOut = U.RasterGDAL(tif)
    # rastOut.printRaster()
    config['tif_file'] = U.reproject_tif(tif,'EPSG:4326')
    cropTif = instantiate(config)
    print(f"cropTif: {cropTif}")
    

if __name__ == "__main__":
    with U.timeit():
        main()  
