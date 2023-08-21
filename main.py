# import dc_extract
import os
from typing import Tuple
import hydra 
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import util as U
import logging 

import pcraster as pcr
from pcraster import *
# from wbw_test import checkIn as chIn   ### IMPORTANT ###: DO NOT USE. If two instance of the license are created, it can kill my license. Thank you!!

def test(DEMPath,LDD) -> os.path:
    '''
    NOTE: Important to ensure the input DEM has a well defined NoData Value ex. -9999. 

    1- *.tif in (DEMPath) is converted to PCRaster *.map format -> U.saveTiffAsPCRaster(DEM)
    2- pcr.setClone(DEMMap) : Ensure extention, CRS and other characteristics for creating new *.map files.
    3- Read DEM in PCRasterformat
    4- Compute flow direction with d8 algorithm -> lddcreate()
    5- Compute strahler Order-> streamorder(FlowDir)
    
    @DEMPath : Input path to the DEM in *.tif format.
    @HANDPath : Output path for the HAND.map result.
    
    '''
    path,communName,_ = U.get_parenPath_name_ext(DEMPath)
    subCatch =os.path.join(path,str(communName+'_subCatch.map'))
    areaMinPath = os.path.join(path,str(communName+'_areaMin.map'))
    areaMaxPath = os.path.join(path,str(communName+'_areaMax.map'))
    outletsPath = os.path.join(path,str(communName+'_Outlets.map'))
    FAccByCatchment = os.path.join(path,str(communName+'_MaxFAcc.map'))
    pcr.setclone(DEMPath)
    DEM = pcr.readmap(DEMPath)
    FlowDir = pcr.readmap(LDD)
    threshold = 8
    print('#####......Computing Strahler order.......######')
    strahlerOrder = streamorder(FlowDir)
    strahlerRiver = ifthen(strahlerOrder>=threshold,strahlerOrder)
    print('#####......Finding outlets.......######')
    junctions = ifthen(downstream(FlowDir,strahlerOrder) != strahlerRiver, boolean(1))
    outlets = ordinal(cover(uniqueid(junctions),0))
    pcr.report(outlets,outletsPath)
    print('#####......Calculating subcatchment.......######')
    subCatchments = catchment(FlowDir,outlets)
    pcr.report(subCatchments,subCatch)
    print('#####......Computing HAND.......######')
    massMap = pcr.spatial(pcr.scalar(1.0))
    Resultflux = accuflux(ldd, massMap)
    FAccByCatchment = areamaximum(Resultflux,Resultflux)
    areaMin = areaminimum(DEM,subCatchments)
    areaMax = areamaximum(DEM,subCatchments)
    pcr.report(areaMin,areaMinPath)
    pcr.report(areaMax,areaMaxPath)
    pcr.report(FAccByCatchment,areaMinPath)

    aguila(areaMin)
    aguila(areaMax)
    pass


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
    
    DEMMap = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\BC_Quesnel_ok\BC_Quesnel_FullBasin_Clip.map'
    LDDPath = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\BC_Quesnel_ok\BC_Quesne_LDD.map'
    test(DEMMap,LDDPath)

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
