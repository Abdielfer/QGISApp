# import dc_extract
import os
import shutil
import hydra 
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import util as U
import logging 
import pandas as pd

# from wbw_test import checkIn as chIn   ### IMPORTANT ###: DO NOT USE. If two instance of the license are created, it can kill my license. Thank you!!
KMP_DUPLICATE_LIB_OK=True

def logger(cfg: DictConfig, nameByTime):
    '''
    You can log all you want here!
    '''
    logging.info(f"Excecution number: {nameByTime}")
    logging.info(f"Output directory :{cfg['output_dir']}")
    # logging.info(f"dc_search inputs: {cfg.dc_Extract_params.dc_search}")
    # logging.info(f"dc_description inputs: {cfg.dc_Extract_params.dc_describeCollections}")
    # logging.info(f"dc_extract inputs: {cfg.dc_Extract_params.dc_extrac_cog}")
       
def customFunction(path):
    Datum = r'C:\Users\abfernan\CrossCanFloodMapping\SResDEM\Data\Analyses1\DATUM\CGG2013an83_CAN.tif'
    pathOutput = U.addSubstringToName(path,'_Ellip')
    U.DatumCorrection_DifferentResolution(path,Datum,pathOutput)
    pass

def runFunctionInLoop(csvList, function = customFunction):
    '''
    Given a list <csvList>, excecute the <function> in loop, with one element from the csv as argument, at the time.  
    '''
    listOfPath = U.createListFromCSV(csvList)
    for path in listOfPath:
        print(path)
        function(path)

def intFucntion(x):
    return int(x)    

@hydra.main(version_base=None, config_path=f"config", config_name="mainConfigPC")
def main(cfg: DictConfig):
    
    # nameByTime = U.makeNameByTime()
    # logger(cfg,nameByTime)
    # U.dc_extraction(cfg)
    # U.multiple_dc_extract_ByPolygonList(cfg)
   
    csv = r'C:\Users\abfernan\CrossCanFloodMapping\SResDEM\Data\ExploringError_CDSM_HRDEM\SourceErrorStudy\hrdtm_List.csv'
    runFunctionInLoop(csv)



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

    #####  Plot histogrames 
    # csvTifList = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\ListOfBasinsDEM16mTif.csv'
    # DEM = U.createListFromCSV(csvTifList)
    # U.plotRasterPDFComparison(DEM,ax_x_units='m',save=False,show=True,title='Full Dataset Histogram of relative elevation')
    
    #### Overwrite Projection
    # U.overwriteShapefileProjection(out_shp)
    # configFile = r'C:\Users\abfernan\CrossCanFloodMapping\GISAutomation\config\mainConfigPC.yaml'
    
    #### Overwrite Hydra config file
    # newParams = {'normalizers': '[100,22,53]'}
    # U.overWriteHydraConfig(configFile,newParams)

    ### Regional statictics with polygons
    # tif = r'c:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\AL_Lethbridge_ok\AL_Lethbridge_cdem_fill_hillslope.tif'
    # watersheds = r'c:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\AL_Lethbridge_ok\AL_Lethbridge_watershed.shp'
    # U.raster_max_by_polygons(tif,watersheds)

    ### Building shapefile from csv.
    # csvDataFrame = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\AL_Lethbridge_ok\AL_Lethbridge_cdem_features_DSet_withClasses.csv'
    # U.buildShapefilePointFromCsvDataframe(csvDataFrame, EPGS=3979) 
    
     ####  Parallelizing. 
    # allFloodList = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\cdem_label_mask.csv'
    # pathList = U.createListFromCSV(allFloodList, delim=';')
    # maxParalelizer(customFunction,pathList)

    #### Create list of files by substring and save it to csv
    # subString = 'DSet.csv'
    # wdr = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI'
    # listToCSV = U.listALLFilesInDirBySubstring_fullPath(wdr,subString)
    # csvToSave = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\DatasetNoLabelsList.csv'
    # U.createCSVFromList(csvToSave,listToCSV)

    ### Rasterize Polygon Multi steps
    # print(f'-->> {shpFile}')
    # U.transformShp_Value(shpFile,targetField='percentage', baseField= 'percentage', funct=intFucntion)
    # U.rasterizePolygonMultiSteps(shpFile,attribute='percentage',burnValue=1) 


     #### Custom Function to add a new column to dataset      
    # def customFunction(pathList):
    #     print('_____________________New Dataset __________________')
    #     csvPath = pathList[0]
    #     print(csvPath)
    #     dataset = pd.read_csv(csvPath,index_col=False)
    #     colName = dataset.columns
    #     print(colName)
    #     if 'Unnamed: 0' in colName:
    #         dataset.drop('Unnamed: 0', axis=1,inplace=True)
    #     outputDatasetPath = U.addSubstringToName(csvPath,'_RelElev')
    #     dem = pathList[1]
    #     dataFrameWithRelElev = U.addCollFromRasterToDataFrame(dataset,dem)
    #     new_order = ['x_coord', 'y_coord', 'Cilp', 'RelElev','GMorph', 'FloodOrd',
    #     'Slope', 'd8fllowAcc', 'HAND', 'proximity', 'Labels']
    #     outDataFrame = U.reorder_dataframe_columns(dataFrameWithRelElev,new_order)
    #     print(outDataFrame.columns)
    #     outDataFrame.to_csv(outputDatasetPath,index=None)

    ####   Standardize Dataset columns from reference dataset mean and std by columns.
    # source = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\A_DatasetsForMLP\StratifiedSampling\class1_Full.csv'
    # objective = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\A_DatasetsForMLP\StratifiedSampling\class1_Full_Validation.csv'
    # colNames = ['RelElev','GMorph','FloodOrd','Slope','d8fllowAcc','HAND','proximity']
    # newDataset = U.standardizeDatasetByColName(source,objective,colNames)
    # output = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\A_DatasetsForMLP\StratifiedSampling\class1_Full_Standardized_Validation.csv'
    # newDataset.to_csv(output, index=None)