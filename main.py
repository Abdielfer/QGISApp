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
    nameByTime = U.makeNameByTime()
    logger(cfg,nameByTime)
    # shpFile = cfg.transformation['mask']
    # bbox = U.get_Shpfile_bbox(shpFile)
    # print(f"The bbox is: {bbox}")
    # U.dc_describe(cfg)
    # U.dc_serach(cfg)
    U.dc_extraction(cfg)
    # print(ex)
    # chIn   # To check in the wbtools license
    
    ### Take file from the dc_output folder and create a new folder to write the trasformations ###
    tifFile = U.listFreeFilesInDirByExt_fullPath('C:/Users/abfernan/CrossCanFloodMapping/GISAutomation/dc_output', ext='.tif')
    print(f"The tif file is: {tifFile}")    
    config = OmegaConf.create(cfg.transformation['clip_raster_to_polygon'])
    config['inputRaster'] = U.reproject_tif(tifFile[0],'EPSG:4326')
    shpPathFormCropTif = config['maskVector']
    _,name,_=U.get_parenPath_name_ext(shpPathFormCropTif)
    paret = r'C:\Users\abfernan\CrossCanFloodMapping\GISAutomation\data'
    newDir = U.ensureDirectory(os.path.join(paret,name))
    config['outPath'] = str(os.path.join(newDir,str(name+'_crop.tif')))  
    print(f"The config is: {config}")
    
    # # Crop the tif
    cropTif = instantiate(config)
    print(f"Croped Tif path: {cropTif}")
    # Compute HAND

    handOutputPath = U.addSubstringToName(cropTif,'_HAND')
    handOutputPath = U.replaceExtention(handOutputPath,'.map')
    U.computeHAND(cropTif,handOutputPath)

    # Clean dc_output folder
    # U.clearTransitFolderContent('C:/Users/abfernan/CrossCanFloodMapping/GISAutomation/dc_output')

if __name__ == "__main__":
    with U.timeit():
        main()  
