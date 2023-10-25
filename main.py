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

def removeFilesBySubstring(parentPath,subStr:str=''):
    list = U.listALLFilesInDirBySubstring_fullPath(parentPath,subStr)
    for i in list:
        U.removeFile(i)

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
    ##### Compute HAND in a serie of basins. #################
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
        U.computeHANDPCRaster(clipPath,HANDPath)

def computeProximityFromDEMList(csvListOfDEMs)->os.path:
    '''
    Starting from a DEM" list, compute for each raster in the list:
        -  the main river network with PCRaster tools.
        -  compte proximity raster with computeProximity() from GDAL. 
    
    @demPath: Path to the DEM file in *.tif format
    @return: Path to the computed proximity map. 
    '''
    listOfPath = U.createListFromCSV(csvListOfDEMs)
    for DEMPath in listOfPath:
        with U.timeit():
            print(f"Start processing --> {DEMPath}")   
            mainRiverTiff = U.extractHydroFeatures(DEMPath)
            ##### Compute proximity
            proximity = U.computeProximity(mainRiverTiff,value=[5,6,7,8,9,10,11,12,13,14])
            print(f"Proximity created at --> {proximity}")
            print('########_____________________#########')

def DEMFeaturingForMLP_WbT(DEM):
    '''
    The goal of this function is to compute all necesary(or desired) maps for MLP classification inputs, starting from a DEM. The function use WhiteboxTools library. All output adress are managed into the class WbT_DEM_FeatureExtraction(). Also the WbT_working directory is setted at the same parent dir of the input DEM. 
    The steps are (See function description formmore details.):
    1- DEM correction <fixNoDataAndfillDTM()>
    2- Slope <computeSlope()>
    3- Compute flow direction <D8_pointe()>
    4- Compute Flow accumulation <DInfFlowAcc()>
    5- Extract stream.
    6- Compute stream order with Strahler Order.
    7- Compute HAND.
    8- Compute HAND_euclidean
    9- Compute distance to stream.
    10- Smapling(TODO)    
    '''
    DEM_Features = U.WbT_DEM_FeatureExtraction(DEM)
    DEM_Features.wbT_geomorphons()
    # DEM_Features.fixNoDataAndfillDTM()
    # U.replace_no_data_value(DEM_Features.FilledDEM)
    # DEM_Features.computeSlope()
    # D8Pointer = DEM_Features.d8_Pointer()  
    # FAcc = DEM_Features.d8_flow_accumulation()
    # stream = DEM_Features.extract_stream(FAcc)
    # strahlerOrder = DEM_Features.computeStrahlerOrder(D8Pointer,stream)
    # mainRiver = DEM_Features.thresholdingStrahlerOrders(strahlerOrder, maxStrahOrder=3)
    # DEM_Features.WbT_HAND(mainRiver)
    # DEM_Features.WbT_HAND_euclidean(mainRiver)
    # DEM_Features.wbT_geomorphons()
    # U.computeProximity(mainRiver)
    return True  # True - If no error is encountered in the process, otherwhise, WbT error will apears. 

def runFunctionInLoop(csvList, function):
    '''
    Given a list <csvList>, excecute the <function> in loop, with one element from the csv as argument at the time.  
    '''
    listOfPath = U.createListFromCSV(csvList)
    for path in listOfPath:
        if os.path.exists(path):
            with U.timeit():
                function(path)
        else:
            print(f"Path not found -> {path}")

def cropTifListWithMaskList(cfg: DictConfig, maskList:os.path):
    wdir = cfg['output_dir']
    maskList = U.createListFromCSV(maskList)
    tifList = U.listFreeFilesInDirByExt_fullPath(wdir,'.tif')
    for i in tifList:
        _,tifName,_ = U.get_parenPath_name_ext(i)
        for j in maskList:
            _,maskName,_ =U.get_parenPath_name_ext(j)
            if maskName in tifName:
                outPath = os.path.join(wdir,maskName+'_clip.tif')
                print('-----------------------Croping --------------------')
                out = U.crop_tif(i,j,outPath)
                print(f'{outPath}')
                print('-----------------------Croped --------------------  \n')
    

    
    
    return True

@hydra.main(version_base=None, config_path=f"config", config_name="mainConfigPC")
def main(cfg: DictConfig):
    # chIn   # To check in the wbtools license
    # nameByTime = U.makeNameByTime()
    # logger(cfg,nameByTime)
    # U.dc_extraction(cfg)
    csvList = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\ListOfBasinsPolygonsShp.csv'
    cropTifListWithMaskList(cfg,csvList)
    
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

#### Compute operation on alist of path from a csv
# listOfPath = U.createListFromCSV(csvList)
#     for DEMPath in listOfPath:
#         if os.path.exists(DEMPath):
#             with U.timeit():
#                 DEMFeaturingForMLP_WbT(DEMPath)
#         else:
#             print(f"Is not dir-> {DEMPath}")