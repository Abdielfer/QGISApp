# import dc_extract
import os
import shutil
import hydra 
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import util as U
import logging 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
       
def customFunction(pathList):
    U.removeFile(pathList)
    

def runFunctionInLoop(csvList, function = customFunction):
    '''
    Given a list <csvList>, excecute the <function> in loop, with one element from the csv as argument, at the time.  
    '''
    listOfPath = U.createListFromCSV_multiplePathPerRow(csvList)
   
        
def intFucntion(x):
    return int(x)

def compare_bboxes(bbox1, bbox2):
    # Unpack the bounding boxes
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2
    # Compare the bounding boxes
    if xmin1 == xmin2 and ymin1 == ymin2 and xmax1 == xmax2 and ymax1 == ymax2:
        return True
    else:
        return False

@hydra.main(version_base=None, config_path=f"config", config_name="mainConfigPC")
def main(cfg: DictConfig):
    
    # nameByTime = U.makeNameByTime()
    # logger(cfg,nameByTime)
    # U.dc_extraction(cfg)
    # U.multiple_dc_extract_ByPolygonList(cfg)
    vectorInput = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\A_DatasetsForMLP\RegionalModelingApplication\BestModelsApplication\ValidationDataset_RastMode_Normalized_class1_RMA_2401241404.shp'

    # rasterOutput = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\A_DatasetsForMLP\RegionalModelingApplication\RMA_outputs\ValidationSet_RastComb_Normalized_class1_RMA_2401181901.tif'
    rasterOutput = U.replaceExtention(vectorInput, '.tif')

    U.rasterizePointsVector(vectorInput,rasterOutput,atribute='y_hat',pixel_size=16)
    U.reproject_tif(rasterOutput)

      
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

    ### Regional statistics with polygons
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
    #     new_order = ['x_coord', 'y_coord', 'RelElev','GMorph', 'FloodOrd',
    #     'Slope', 'd8fllowAcc', 'HAND', 'proximity', 'Labels']
    #     outDataFrame = U.reorder_dataframe_columns(dataFrameWithRelElev,new_order)
    #     print(outDataFrame.columns)
    #     outDataFrame.to_csv(outputDatasetPath,index=None)

    #### Build balanced and Stratified Dataset by classes.
    # csv = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\Dataset_LabelsMode_List.csv'
    # wdir = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\A_DatasetsForMLP\StratifiedSampling_RasterizedMode'
    # datasetList = U.createListFromCSV(csv)
    # U.buildBalanceStratifiedDatasetByClasses1And5(datasetList,wdir,targetCol='LabelsMode')


     ##### Concat Datasets
    # class1_Trin = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\A_DatasetsForMLP\StratifiedSampling_RasterizedMode\class5_Full_Training.csv'
    # class1_Val = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\A_DatasetsForMLP\StratifiedSampling_RasterizedMode\class5_Full_Validation.csv'
    # class1Full = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\A_DatasetsForMLP\StratifiedSampling_RasterizedMode\class5_Full_RaterizedMode.csv'
    # class1TrainDF = pd.read_csv(class1_Trin,index_col=False)
    # class1ValDF = pd.read_csv(class1_Val,index_col=False)
    # full = pd.concat([class1TrainDF,class1ValDF])
    # full.to_csv(class1Full,index=None)

    ####   Min-Max normalization by dataset columns from reference dataset mean, std, min, max.
    # source = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\A_DatasetsForMLP\StratifiedSampling\class1_Full.csv'
    # objective = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\A_DatasetsForMLP\StratifiedSampling\class1_Full_Validation.csv'
    # colNames = ['RelElev','GMorph','FloodOrd','Slope','d8fllowAcc','HAND','proximity']
    # newDataset = U.normalizeDatasetByColName(source,objective,colNames)
    # output = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\A_DatasetsForMLP\StratifiedSampling\class1_Full_Standardized_Validation.csv'
    # newDataset.to_csv(output, index=None)
        

    ##### Create Dataset of error from a set of pairs of tiles. 
    # hrTiles_csv =  r'C:\Users\abfernan\CrossCanFloodMapping\SResDEM\Data\ExploringError_CDSM_HRDEM\SourceErrorStudy\hrdsm_16m.csv'
    # hrTiles_csv_list = U.createListFromCSV(hrTiles_csv)
    # cdem_csv = r'C:\Users\abfernan\CrossCanFloodMapping\SResDEM\Data\ExploringError_CDSM_HRDEM\SourceErrorStudy\sr_cdsm.csv'
    # cdem_csv_list = U.createListFromCSV(cdem_csv)
    # datsetOutName = r'C:\Users\abfernan\CrossCanFloodMapping\SResDEM\Data\ExploringError_CDSM_HRDEM\SourceErrorStudy\error_hrdsm_srcdsm_16m.csv'
    
    # col_names = ['x_coord', 'y_coord', 'hrdsm', 'cdsm', 'error']
    # arrayDataSet = np.zeros((1,5))
    # for hr in hrTiles_csv_list:
    #     _,HResName,_ = U.get_parenPath_name_ext(hr)
    #     for lowRes in cdem_csv_list:
    #         _,lowResName,_ = U.get_parenPath_name_ext(lowRes)
    #         if HResName in lowResName:
    #             print('Sampling ->> ',HResName)
    #             samples = U.twoRaster_ErrorAnalyse(hr,lowRes,10000)
    #             arrayDataSet = np.concatenate([arrayDataSet,samples])
    #             cdem_csv_list.remove(lowRes)
   
    # dataSet = pd.DataFrame(arrayDataSet,columns=col_names)
    # print(dataSet.describe())
    # dataSet.to_csv(datsetOutName,index=None)
        
    #### Sampling full raster to DataSet
    # raster = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\AL_Lethbridge_ok\AL_Lethbridge_FullBasin_Cilp_FullDataset_AOI.tif'
    # array = U.sampling_Full_raster_GDAL(raster)
    # dataSet = pd.DataFrame(array)
    # print(dataSet.head())
    # print(dataSet.describe())
    # datsetOutName = r'C:\Users\abfernan\CrossCanFloodMapping\GISAutomation\dc_output\testingFullSampling.csv'
    # dataSet.to_csv(datsetOutName,index=None)  