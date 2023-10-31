# import dc_extract
import os
import pandas as pd
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

def logger(cfg: DictConfig, nameByTime):
    '''
    You can log all you want here!
    '''
    logging.info(f"Excecution number: {nameByTime}")
    logging.info(f"Output directory :{cfg['output_dir']}")
    # logging.info(f"dc_search inputs: {cfg.dc_Extract_params.dc_search}")
    # logging.info(f"dc_description inputs: {cfg.dc_Extract_params.dc_describeCollections}")
    # logging.info(f"dc_extract inputs: {cfg.dc_Extract_params.dc_extrac_cog}")

def runFunctionInLoop(csvList, function):
    '''
    Given a list <csvList>, excecute the <function> in loop, with one element from the csv as argument, at the time.  
    '''
    listOfPath = U.createListFromCSV(csvList)
    for path in listOfPath:
        if os.path.exists(path):
            with U.timeit():
                function(path)
        else:
            print(f"Path not found -> {path}")

def createShpList(parentDir)-> os.path:
    listOfPath = U.listALLFilesInDirByExt_fullPath(parentDir,ext='.shp')
    OutCSVPath = os.path.join(parentDir,'listOfShpFiles.csv')
    U.createCSVFromList(OutCSVPath,listOfPath)
    return OutCSVPath 

def DEMFeaturingForMLP_WbT(DEM)-> list:
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
    8- Compute distance to stream.

    @DEM: Deigital elevation mode raster.
    @Return: A list of some produced maps. The maps to be added to a multiband *.tif.
    
    NOTE: Also save a list of produced features in a csv file for further automation process. 
    '''
    outList = [DEM]
    DEM_Features = U.WbT_DEM_FeatureExtraction(DEM)
    geomorph = DEM_Features.wbT_geomorphons()
    U.replace_no_data_value(geomorph)
    outList.append(geomorph)
    FloodOrd = DEM_Features.FloodOrder()
    U.replace_no_data_value(FloodOrd)
    outList.append(FloodOrd)
    DEM_Features.fixNoDataAndfillDTM()
    slope = DEM_Features.computeSlope()
    outList.append(slope)
    D8Pointer = DEM_Features.d8_Pointer()  
    FAcc = DEM_Features.d8_flow_accumulation()
    U.replace_no_data_value(FAcc)
    outList.append(FAcc)
    stream = DEM_Features.extract_stream(FAcc)
    strahlerOrder = DEM_Features.computeStrahlerOrder(D8Pointer,stream)
    mainRiver = DEM_Features.thresholdingStrahlerOrders(strahlerOrder, maxStrahOrder=3)
    HAND = DEM_Features.WbT_HAND(mainRiver)
    outList.append(HAND)
    proximity = U.computeProximity(mainRiver)
    outList.append(proximity)
    ## Catchment extraction
    DEM_Features.watershedHillslopes(D8Pointer,mainRiver)
    ### Save the list of features path as csv. 
    csvPath = U.replaceExtention(DEM,'_FeaturesList.csv')
    U.createCSVFromList(outList,csvPath)
    return outList  

def fromDEMtoDataFrame(DEM,labels,target:str='percentage',mask:os.path=None)->pd.DataFrame:
    '''
    Testing the process of sampling automatically a DEM of multiples bands and a polygon, to produce a DataSet. 
    '''
    ## Create features for dataset
    bandsList = DEMFeaturingForMLP_WbT(DEM)
    ## Extract the band name from the full path of features
    colList = U.extractNamesListFromFullPathList(bandsList)
    ## Build a multiband raster to ensure spatial correlation between features.
    rasterMultiband = U.addSubstringToName(DEM,'_dataset')
    U.stackBandsInMultibandRaster(bandsList)
    ## Crop the multiband raster if needed.
    if mask:
        cropped = U.addSubstringToName(rasterMultiband,'_crop')
        raster = U.crop_tif(rasterMultiband,mask,cropped)
    else:
        raster = rasterMultiband
    ## Random sampling the raster with a density defined by the ratio. This is the more expensive opperation..by patient. 
    samplesArr = U.randomSamplingMultiBandRaster(raster,ratio=0.0001)
    ## Build a dataframe with the samples
    df = pd.DataFrame(samplesArr,columns=colList)
    ## Extract coordinates columns to sample from label. 
    xyList = samplesArr[:,0:2]
    labesColumn = []
    for i in range(0, xyList.shape[0]):
        labesColumn.append(U.getFieldValueFromPolygon(labels,target,xyList[i,0],xyList[i,1]))
    ## Add labels to the DataFrame
    df['target'] = labesColumn
    ## Saving DataFrame as csv
    scv_output = U.replaceExtention(rasterMultiband,'_DFrame.csv')
    df.to_csv(scv_output)
    return df

@hydra.main(version_base=None, config_path=f"config", config_name="mainConfigPC")
def main(cfg: DictConfig):
    # chIn   # To check-in the wbtools license
    # nameByTime = U.makeNameByTime()
    # logger(cfg,nameByTime)
    # U.dc_extraction(cfg)
    # # runFunctionInLoop(csvList,DEMFeaturingForMLP_WbT)
    DEM = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\BC_Quesnel_ok\BC_Quesnel_FullBasin_clip.tif' 
    DEmFeaturs = U.WbT_DEM_FeatureExtraction(DEM)
    DEmFeaturs.FloodOrder()
    # WbTransf = U.generalRasterTools(WbWDir)

    # raster= r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\BC_Quesnel_ok\BC_Quesnel_dataset.tif'
    # bandList_csv = r'C:/Users/abfernan/CrossCanFloodMapping/FloodMappingProjData/HRDTMByAOI/BC_Quesnel_ok/BC_Quesnel_bands.csv'
    # samplingArea = r'C:/Users/abfernan/CrossCanFloodMapping/FloodMappingProjData/HRDTMByAOI/BC_Quesnel_ok/SamplingArea.shp'
    # labels = r'C:/Users/abfernan/CrossCanFloodMapping/FloodMappingProjData/HRDTMByAOI/BC_Quesnel_ok/BC_Quesnel_floodLabel.shp'
    # bandList =U.createListFromCSV(bandList_csv)
    # colList = U.extractNamesListFromFullPathList(bandList,['x_coord','y_coord'])
    # df = fromDEMtoDataFrame(raster,colList,labels,mask=samplingArea)
    # print(df.head)

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
