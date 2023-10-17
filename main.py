# import dc_extract
import os
from typing import Tuple
import hydra 
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import util as U
import logging 
import numpy as np
import pcraster as pcr
from pcraster import *
# from wbw_test import checkIn as chIn   ### IMPORTANT ###: DO NOT USE. If two instance of the license are created, it can kill my license. Thank you!!

from osgeo import gdal
from osgeo.gdalconst import *

def settingsForClipDEMAndHandComputing(maskVectorPath:os.path)-> Tuple:
    '''
    To be run after dc_extract. Make sure the dc_output directory contains only one file (The right one..)
    
    '''
    transitForlder = r'C:/Users/abfernan/CrossCanFloodMapping/GISAutomation/dc_output' 
    ### Take DEM from dc_extraction output folder (*.tif) and create a new folder to write Outputs ###
    tifFile = U.listFreeFilesInDirByExt_fullPath(transitForlder, ext='.tif')
        ##  Create Output paths for the DEM's products (Clip.tif & HAND.map) at the maskVector directory. 
    path,communName,_ = U.get_parenPath_name_ext(maskVectorPath)
    clipPath =os.path.join(path,str(communName+'_Clip.tif'))
    HandPathMap = os.path.join(path,str(communName+'_HAND.map'))
            # Clip the DTM
    U.clipRasterByMask(tifFile[0],maskVectorPath,clipPath)
     # # Clean dtransitForlder
    U.clearTransitFolderContent(transitForlder)
    return clipPath, HandPathMap

def logger(cfg: DictConfig, nameByTime):
    '''
    You can log all you want here!
    '''
    logging.info(f"Excecution number: {nameByTime}")
    logging.info(f"Output directory :{cfg['output_dir']}")
    # logging.info(f"dc_search inputs: {cfg.dc_Extract_params.dc_search}")
    # logging.info(f"dc_description inputs: {cfg.dc_Extract_params.dc_describeCollections}")
    # logging.info(f"dc_extract inputs: {cfg.dc_Extract_params.dc_extrac_cog}")

def createShpList(parentDir)-> os.path:
    listOfPath = U.listALLFilesInDirByExt_fullPath(parentDir,ext='.shp')
    OutCSVPath = os.path.join(parentDir,'listOfShpFiles.csv')
    U.createCSVFromList(OutCSVPath,listOfPath)
    return OutCSVPath 

def computeHANDfromBasinPolygon(cfg: DictConfig, csvListOfBasins:os.path):
    '''
    #########################################################
    ##### Compute HAN in a serie of basins. #################
    ##### Bisins are provided as input path in a csv file. ##
    #########################################################
    '''
    listOfPath = U.createListFromCSV(csvListOfBasins)
    for f in listOfPath:
        bbox = U.get_Shpfile_bbox_str(f)
        d ={'bbox':bbox}
        print(f"Extrtaxted BBox : {bbox}")
        U.dc_extraction(cfg,args = d)
        clipPath, HANDPath = settingsForClipDEMAndHandComputing(f)
        print(f"ClipPath: {clipPath}")
        print(f"HANDPath: {HANDPath}")
        U.computeHAND(clipPath,HANDPath)

def computeProximityFromDEM(DEMPath)->os.path:
    '''
    Starting from a DEM, compute the main river network with PCRaster tools and then compte proximity raster with computeProximity() from GDAL. 
    @demPath: Path to the DEM file in *.tif format
    @return: Path to the computed proximity map. 
    '''
    # Set output path.
    path,communName,_ = U.get_parenPath_name_ext(DEMPath)
    lddOutPath = os.path.join(path,str(communName+'_ldd.map'))
    mainRiverMapPath = os.path.join(path,str(communName+'_mainRiver.map'))
    mainRiverTiffPath = os.path.join(path,str(communName+'_mainRiver.tif'))
    
    # Extract input crs.\
    input_crs = U.extractProjection(DEMPath)
   
    # # Convert DEM to *.map
    # DEMMap = U.saveTiffAsPCRaster(DEMPath)
    # pcr.setclone(DEMMap)
    DEM = pcr.readmap(DEMPath)
    pcr.setclone(DEMPath)

    # # Compute flow direction with D8. 
    with U.timeit(): 
         print('#####......Computing LDD .......######')
         FlowDir = lddcreate(DEM,1e31,1e31,1e31,1e31)
         pcr.report(FlowDir,lddOutPath)
         print('#####......LDD Ready .......######')

    #Compute river network
    print('#####......Computing Strahler order.......######')
    strahlerOrder = streamorder(lddOutPath)
    MainRiver = ifthen(strahlerOrder >= 9,strahlerOrder)
    pcr.report(MainRiver,mainRiverMapPath)
    # Verify main river projection
    mainRiver_crs = U.extractProjection(mainRiverMapPath)
    print(f"mainRiverMap_crs : {mainRiver_crs}")
    
    if mainRiver_crs is None:
        print('Yeaa is None')
        U.reproject_tif(mainRiverMapPath,input_crs)

    # Save mainRiver as Tif.
    saved = U.translateRaster(mainRiverMapPath,mainRiverTiffPath)
    print(f"Main River saved  --> {saved}")
    
    # Compute proximity
    proximityPath = U.computeProximity(mainRiverTiffPath,value=[9,10,11,12,13,14])
    return proximityPath


@hydra.main(version_base=None, config_path=f"config", config_name="mainConfigPC")
def main(cfg: DictConfig):
    # chIn   # To check in the wbtools license
    # nameByTime = U.makeNameByTime()
    # logger(cfg,nameByTime)
    # U.dc_extraction(cfg)
    mapFile = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\AL_Lethbridge_ok\AL_Lethbridge_FullBasin_Clip.map'
    # computeProximityFromDEM(mapFile)
    ToReproject = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\QC_Bromont_OK\QC_Bromont_EffectiveBasin_Clip.map'
    U.reproject_PCRaster(ToReproject,'EPSG:3979')
    

if __name__ == "__main__":
    with U.timeit():
        main()  


#####   Block to write into main.py to performe automated tasks 

    ### Download, merging and resampling HRDTM tiles ###
    # WbWDir = cfg['output_dir']
    # WbTransf = U.generalRasterTools(WbWDir)
    # csvTilesList = [r'C:\Users\abfernan\CrossCanFloodMapping\SResDEM\Data\NF_Shearstown\NF_Shearstown_HRDEM.csv',
    #                 r'C:\Users\abfernan\CrossCanFloodMapping\SResDEM\Data\NF_StJohns\StJohns_HRDEM.csv',
    #                 r'C:\Users\abfernan\CrossCanFloodMapping\SResDEM\Data\QC_Quebec\QC_HRDEM\QC_HRDEM_Quebec_600015_34.csv',
    #                 r'C:\Users\abfernan\CrossCanFloodMapping\SResDEM\Data\QC_Quebec\QC_HRDEM\QC_HRDEM_Levis.csv']
    
    # for csv in csvTilesList:
    #     WbTransf.mosaikAndResamplingFromCSV(csv,8,'Ftp_dtm')
