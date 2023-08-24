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
    return [clipPath, HandPathMap]

def logger(cfg: DictConfig, nameByTime):
    '''
    You can log all you want here!
    '''
    logging.info(f"Excecution number: {nameByTime}")
    logging.info(f"Output directory :{cfg['output_dir']}")
    # logging.info(f"dc_search inputs: {cfg.dc_Extract_params.dc_search}")
    # logging.info(f"dc_description inputs: {cfg.dc_Extract_params.dc_describeCollections}")
    logging.info(f"dc_extract inputs: {cfg.dc_Extract_params.dc_extrac_cog}")

def createShpList(parentDir):
    listOfPath = U.listALLFilesInDirByExt_fullPath(parentDir,ext='.shp')
    OutCSVPath = os.path.join(parentDir,'listOfShpFiles.csv')
    U.createCSVFromList(OutCSVPath,listOfPath)


@hydra.main(version_base=None, config_path=f"config", config_name="mainConfigPC")
def main(cfg: DictConfig):
    # chIn   # To check in the wbtools license
    # nameByTime = U.makeNameByTime()
    # logger(cfg,nameByTime)
    
    cdem = r'C:\Users\abfernan\CrossCanFloodMapping\SResDEM\Data\cdsm16m\QC_Quebec.tif'
    cdemFill =  "'QC_Quebec_filled_WangLiu.tif'"
    paretPath,name,ext = U.get_parenPath_name_ext(cdem)
    cdemName = str(name+ext)
    print(f"cdem Name = {cdemName}")
    DEMTranf = U.WbT_dtmTransformer(workingDir=paretPath) 
    # cdemFill = DEMTranf.fixNoDataAndfillDTM(cdem)
    difRaster = U.addSubstringToName(cdem,'_CdemFillMenusCDEM_tets')
    statement = str(cdemFill+' - '+"'"+cdemName+"'" +" > 0.05") # Remouve some noice because of aproximation with -0.05
    print(f"Statement : {statement}")
    DEMTranf.rasterCalculator(difRaster, statement)
    
    # cdemFill2 = np.round(cdemFill,2).astype('float32')
    # cdemMin,cdemMax, rcdemMean,cdemMode, cdemSTD,_ = U.computeRaterStats(cdemFill)
    # cdemSlope = DEMTranf.computeSlope(cdemFill)
    # cdem_SlpMin,cdem_SlpMax, rcdem_SlpMean,cdem_SlpMode, cdem_SlpSTD,_ = U.computeRaterStats(cdemSlope)
    # cdemFAcc = DEMTranf.DInfFlowCalculation(cdemSlope)
    # cdem_FAccMin,cdem_FAccMax, rcdem_FAccMean,cdem_FAccMode, cdem_FAccSTD,_ = U.computeRaterStats(cdemFAcc) 
    # print(f"Elevation stat: min={cdemMin}, max={cdemMax},mean= {rcdemMean}, mode={cdemMode}, std={cdemSTD} \n ")
    # print(f"Slope stat: min={cdem_SlpMin}, max={cdem_SlpMax},mean= {rcdem_SlpMean}, mode={cdem_SlpMode}, std={cdem_SlpSTD} \n")
    # print(f"FAcc stat: min={cdem_FAccMin}, max={cdem_FAccMax},mean= {rcdem_FAccMean}, mode={cdem_FAccMode}, std={cdem_FAccSTD} \n")
    # DEMTranf.computeRasterHistogram(cdemFill)

    
    ###################################
    # csv = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\ListOfBasins.csv'
    # listOfPath = U.createListFromCSV(csv)
    # for f in listOfPath:
    #     bbox = U.get_Shpfile_bbox_str(f)
    #     d ={'bbox':bbox}
    #     print(f"Extrtaxted BBox : {bbox}")
    #     U.dc_extraction(cfg,args = d)
    #     clipPath, HANDPath = settingsForClipDEMAndHandComputing(f)
    #     print(f"ClipPath: {clipPath}")
    #     print(f"HANDPath: {HANDPath}")
    #     U.computeHAND(clipPath,HANDPath)
    ###################################


if __name__ == "__main__":
    with U.timeit():
        main()  
