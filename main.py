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
from collections import Counter

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
    
def runFunctionInLoop(csvList, function:None):
    '''
    Given a list <csvList>, excecute the <function> in loop, with one element from the csv as argument, at the time.  
    '''
    listOfPath = U.createListFromCSV_multiplePathPerRow(csvList)
    outListOfDatasets = U.addSubstringToName(csvList, "_DatasetsList")
    finalListofDatasetsPaths = []
    for i in listOfPath:
        outDataSetPath = U.replaceExtention(i[1],'.csv')
        dataset = fromDEMTo_TrainingDataset_percentRasterSampling(i[1],i[0],i[2],outDatasetPath=outDataSetPath)
        finalListofDatasetsPaths.append(dataset)
        
    U.createCSVFromList(outListOfDatasets,finalListofDatasetsPaths)
        
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

                #############################
                #####   Wrap Functions  #####
                #############################

#####  DataFrame preprocesing wraps functions  ####
def addColumnFromRasterToDataFrame(pathList,outPath:str=None,prefix=''):
    listOfPath = U.createListFromCSV_multiplePathPerRow(pathList)
    outListOfDatasets = U.addSubstringToName(pathList,"_Dataset_ColAdded_List")
    finalListofDatasetsPaths = []
    for i in listOfPath:
        ## Import the datset 
        csvPath = i[1]
        dataset = pd.read_csv(csvPath,index_col=False)
        colName = dataset.columns
        print(colName)
        if 'Unnamed: 0' in colName:
            dataset.drop('Unnamed: 0', axis=1,inplace=True)
        ## Add the column to the existent dataset with the in <colName>. NOTE: If colneme Exist, it'll be replaced 
        demPath = i[0]  ## The new column can be a mask or a feature. 
        dataFrame = U.addCollFromRasterToDataFrame(dataset,demPath, colName='Labels')
        # print(dataFrame.columns)
        print(dataFrame.describe())
        
        ## Save the new Dataset
        if outPath:
            outputDatasetPath = outPath
        else:
            outputDatasetPath = U.addSubstringToName(csvPath,prefix)
        dataFrame.to_csv(outputDatasetPath,index=None)
        
        ## Keep track in the created dataset for further opperations. 
        finalListofDatasetsPaths.append(outputDatasetPath)
        
    U.createCSVFromList(outListOfDatasets,finalListofDatasetsPaths)
    return outListOfDatasets

def buildBalanceStratifiedDatasetByClasses1And5_FromDatasetList(iouList:os.path, wdir:os.path, prefix:str=''):
    datasetList = U.createListFromCSV(iouList)
    U.createFullBalanceStratifiedDatasetsByClasses1And5(datasetList,wdir,targetCol='Labels',prefix=prefix)

def buildBalanceStratifiedDatasets_SingleClass(iouList:os.path, wdir:os.path, classVal:int = 1, prefix:str=''):
    targetCol='Labels'
    return U.createBalanceStratifiedDatasets_SingleClass(iouList,wdir,targetCol,classValue=classVal,saveIndividuals=True,prefix = prefix)

def concatDatasets(datsetListPath:os.path, outDatasetPath:os.path)->os.path:
    datasetList = U.createListFromCSV(datsetListPath)
    globalDataset = pd.DataFrame()
    for i in datasetList:
        dataset_From_i = pd.read_csv(i,index_col=None)
        globalDataset = pd.concat([globalDataset,dataset_From_i])

    print(globalDataset.describe())
    globalDataset.to_csv(outDatasetPath, index=None)
    return outDatasetPath

def dataFrameReorderAndCleaning(datasetSourcePath:os.path, cleanByColName:str = None, colNewOrder:list= None)->os.path:
    '''
    @datasetSourcePath:os.path: Path to the dataset to be transformed.
    @cleanByColName:str (default = None): Colname objective to be use as refferences in the cleaning. (ex. cleanDataframe = datasetSourcePath[datasetSourcePath[cleanByColName]>0] ) 
    @colNewOrder:list (default = None): List of new order of columns to find in the output dataset. This also allows to reduce the column's number in the dataset as desired. (ex. colNeworder =['RelElev','GMorph','Slope','d8fllowAcc','dInfFAcc','HAND','proximity','Labels']).
    @return: os.path: Path to the reordered and cleaned dataset.csv.
    '''
    ## Cleaning
    if cleanByColName is None:
        cleanDataframe = pd.read_csv(datasetSourcePath, index_col= None)
    else:
        # Apply your custom dataframeCleaningFunction() just bellow!!
        source = pd.read_csv(datasetSourcePath, index_col= None)
        cleanDataframe = dataframeCleaningFunction(source,cleanByColName)
    
    ## Reordering if called
    if colNewOrder is None:
        reorderedDataframe = cleanDataframe
    else:
        reorderedDataframe = U.reorder_dataframe_columns(cleanDataframe,colNewOrder)
    
    NewFile = U.addSubstringToName(datasetSourcePath,"_Clean")
    print(reorderedDataframe.describe())
    reorderedDataframe.to_csv(NewFile, index=None)
    return NewFile

def dataframeCleaningFunction(source:pd.DataFrame, col:str)->pd.DataFrame:
    '''
    Fuction to be customized for Dataframe cleaning opperation. 
    Ex. Remove all negatives values from column 'Elevation'
        cleanDataframe = source[source['Elevation']>0]
    '''
    cleanDataframe = source[source[col]>=0]
    return cleanDataframe

def proximityBuffer(datasetPath:os.path, Buffer:int = 3000)->pd.DataFrame:
    dataset = pd.read_csv(datasetPath, index_col= None)
    proximityBuffered = dataset[dataset['proximity']<Buffer]
    return proximityBuffered

def minmax_normalization(source:os.path, objective:os.path = None,output:os.path = None, colList:list=None):
    '''
    This wrap function group the steps to perform min-max normalization if a dataset. The normalization values are taken from the @source, and the normalization process is performed on the @objetive.
    If @objective is None, the normalization is performed in the @source. 
    If @
    @source: os.path: path to *.csv containing the dataset from which the Min-Max values per column in @colList are extracted.
    @objective: os.path: path to *.csv containing the dataset to be normalized.
    @colList: list: Default(None): List of columns to be normalized. If default, normalization is performed through all columns.      
    '''
    if objective is None:
        objective = source
    newDataset = U.normalizeDatasetByColName(source,objective,colList)
    print(newDataset.describe())
    if output is None:
        output = U.addSubstringToName(objective, "_Normal")
    
    newDataset.to_csv(output, index=None)

#### Dataset extraction from DEM wraps functions    
def dataset_Extraction_Process(csvList,trainingOrInference:str='tr',prefix:str="")->os.path:  # 
    '''
    Given a list <csvList>, excecute operations in loop, with one element from the csv as argument, at the time.  
    @csvList: path: Path to the csv containing the path to dem,labels raster, sampling areas vectors,FeaturesPathlist, multibandAOI,multibandFullBasin. 
    @trainingOrValidation:str(Defaylt='tr'). Options: 'tr'== call <fromDEMTo_TrainingDataset_percentRasterSampling> to sample a percent of the ROI; 'Inf'== call <fromDEMTo_ValidationDataset_FullRasterSampling> to sample the entire region for inference.
    '''
    listOfPath = U.createListFromCSV_multiplePathPerRow(csvList)
    outListOfDatasets = U.addSubstringToName(csvList, "_ValidatingProcess")
    finalListofDatasetsPaths = []
    for i in listOfPath:
        if trainingOrInference =='tr':
            outDataSetPath = U.replaceExtention(i[1],'.csv')
            if prefix:
                outDataSetPath = U.addSubstringToName(outDataSetPath,prefix)
            fromDEMTo_TrainingDataset_percentRasterSampling(i[1],i[0],i[2],outDatasetPath=outDataSetPath)
            finalListofDatasetsPaths.append(outDataSetPath)
            U.createCSVFromList(outListOfDatasets,finalListofDatasetsPaths)
        if trainingOrInference =='Inf':
            outDataSetPath = U.replaceExtention(i[0],'.csv')
            if prefix:
                outDataSetPath = U.addSubstringToName(outDataSetPath,prefix)
            if len(i)>2:
                fromDEMTo_ValidationDataset_FullRasterSampling(i[0],i[0],outDatasetPath=outDataSetPath,featurePathList=i[2],multibandRaster=i[1])
            else:
                fromDEMTo_ValidationDataset_FullRasterSampling(i[1],i[0],outDatasetPath=outDataSetPath)
            finalListofDatasetsPaths.append(outDataSetPath)
            U.createCSVFromList(outListOfDatasets,finalListofDatasetsPaths)
 
    return outListOfDatasets
 
def fromDEMTo_ValidationDataset_FullRasterSampling(dem:os.path=None ,labelRaster:os.path=None, labelsColName:str = "Labels", outDatasetPath:os.path = None,featurePathList:os.path=None, multibandRaster:os.path = None)->pd.DataFrame:
    '''
    This function wrap the process of build features from the DEM, sampling the Full Raster Multiband  in a dataframe, and adding the labels column from a labels raster.
    Conditional behaiviour: If provided path exist, the current multiband will be samples (and cropped if mast not None), otherwhise, the multiband will be created from the list of raster bands.
    @dem:os.path: Path to the DEM form which the features will be created.
    @labelRaster::os.path: ath the the raster of labels.    
    @labelsColName:str (Default = "Labels"): Labels name to be assigned to the output dataframe. 
    @outDatasetPath:os.path (Required)(Default = None): Path to the dataset <*.csv> will be created.
    @featurePathList:os.path(Optional)(Default = None) If features exist,this list will contain the path to the raster features.It MUST be provided in combination to the <multibandRaster path>
    @multibandRaster: (Optional) Path to the existent multiband raster to sample from.
    '''
        
    if outDatasetPath:
        if featurePathList is not None:
            list = U.createListFromCSV(featurePathList)
            ## We asume, the DEM is a muliband
            csvFilePath = U.from_MultibandRaster_toDataFrame_ForRegModelingApplication(list,rasterMultiband=multibandRaster)
        else:
            list = U.DEMFeaturingForMLP_WbT(dem)
            multibandPath = U.addSubstringToName(dem,"_featureStack")
            csvFilePath = U.from_TifList_toDataFrame_ForRegModelingApplication(list,outRasterMultibandPath=multibandPath)
        
        featureStackDataFrame = pd.read_csv(csvFilePath)

        ### Add labels to each sampled point from the <labelRaster>
        dataFrame = U.addCollFromRasterToDataFrame(featureStackDataFrame,labelRaster,labelsColName) 
        print(dataFrame.describe())
        dataFrame.to_csv(outDatasetPath, index=None)
        print(f"Dataset saved at : {outDatasetPath}")
        return outDatasetPath
    else:
        print("Process NOT started : Add an output dataset path")
        return None

def fromDEMTo_TrainingDataset_percentRasterSampling(dem,labelRaster,mask:os.path = None,labelsColName:str = "Labels", outDatasetPath:os.path = None)-> os.path:
    '''
    This function wrap the process of:
     - build features from the DEM, 
     - sampling a portion of a Raster Multiband of features to a dataframe with: <U.from_TifList_toDataFrame()> 
     - add the labels column from a labels raster.
    '''
    if outDatasetPath is not None:
        bandList = U.DEMFeaturingForMLP_WbT(dem)
        dataframe = U.from_TifList_toDataFrame(bandList,labelRaster,mask=mask,labelColName=labelsColName)
        dataframe.to_csv(outDatasetPath, index=None)
        print(f"Dataset saved to : {outDatasetPath}")
        return outDatasetPath
    else:
        print("Process NOT started : Add an output dataset path")
        return None

def inferenceDatasetPreprocessing(colNeworder:list =[], prefix:str=""):
    sourceList = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\Inference_4DInfFlowDir_DatasetsList.csv'
    #sourceList = dataset_Extraction_Process(featureLabelsMascList,trainingOrInference='Inf')
    
    listDatasetToNormaliza = U.createListFromCSV(sourceList)
    sourceC1 = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\A_DatasetsForMLP\DinfFlowDatasets\Class1\DInfFlowDir_Class1_TrainingDatasetsList_StratUndersamp.csv'
    sourceC5 = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\A_DatasetsForMLP\DinfFlowDatasets\Class5\DInfFlowDir_Class5_TrainingDatasetsList_StratUndersamp.csv'

    for i in listDatasetToNormaliza:
        print(f"\n In progress ->> {i}")
        reorderPath = dataFrameReorderAndCleaning(i,colNewOrder=colNeworder)
        reorderBoofered = proximityBuffer(reorderPath)
        outPathC1 = U.addSubstringToName(i, prefix)
        minmax_normalization(sourceC1,reorderBoofered,output=outPathC1,colList=['RelElev','GMorph','Slope','dInfFAcc','HAND','proximity'])
        outPathC5 = U.addSubstringToName(i, "_4Class5")
        minmax_normalization(sourceC5,reorderBoofered,output=outPathC5,colList=['RelElev','GMorph','Slope','dInfFAcc','HAND','proximity'])


@hydra.main(version_base=None, config_path=f"config", config_name="mainConfigPC")
def main(cfg: DictConfig):
    # nameByTime = U.makeNameByTime()
    # logger(cfg,nameByTime)
    # DEM = r'C:\Users\abfernan\CrossCanFloodMapping\SResDEM\Data\ExploringError_CDSM_HRDEM\BC_Sunshine_Coast2020\dtm\dtm_16m.tif'
    # bbox = U.get_raster_bbox(DEM)
    # print(bbox)
    # args =  {"bbox":bbox,"suffix":"NB_StJohn_Mosaic_dem"}
    # U.dc_extraction(cfg)
    # U.multiple_dc_extract_ByPolygonList(cfg)

    DatasetIn= r'c:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\NF_StJohns_2020_ok\NF_StJohns_cdem\NF_StJohns16m_Elevation_tester.tif'
    labelsRaster = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\NF_StJohns_2020_ok\NF_StJohns_cdem\NF_StJohns16m_C1_CombinedSampling.tif'
    outDataset = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\NF_StJohns_2020_ok\NF_StJohns_cdem\NF_StJohns16m_TestingOneRelaceNoData.csv' 

    fromDEMTo_ValidationDataset_FullRasterSampling(dem=DatasetIn, labelRaster=labelsRaster, outDatasetPath=outDataset)
    # cols = ['x_coord','y_coord','GMorph', 'RelElev', 'Slope', 'dInfFAcc', 'HAND','proximity','Labels']

    # listOfPath = U.createListFromCSV(outListPath)
    # outListOfDatasets = U.addSubstringToName(outListPath, "_DatasetCleanedsList")
    # finalListofDatasetsPaths = []
    # for i in listOfPath:
    #     out = dataFrameReorderAndCleaning(i,colNewOrder=cols)
    #     finalListofDatasetsPaths.append(out)
    
    # U.createCSVFromList(outListOfDatasets,finalListofDatasetsPaths)


if __name__ == "__main__":
    with U.timeit():
        main()  

#####   Block to write into main.py to perform automated tasks 

    ### Download, merging and resampling HRDTM tiles ###
    # WbWDir = cfg['output_dir']
    # WbTransf = U.generalRasterTools(WbWDir)
    # csvTilesList = [r'C:\Users\abfernan\CrossCanFloodMapping\SResDEM\Data\NF_Shearstown\NF_Shearstown_HRDEM.csv',
    #                 r'C:\Users\abfernan\CrossCanFloodMapping\SResDEM\Data\NF_StJohns\StJohns_HRDEM.csv',
    #                 r'C:\Users\abfernan\CrossCanFloodMapping\SResDEM\Data\QC_Quebec\QC_HRDEM\QC_HRDEM_Quebec_600015_34.csv',
    #                 r'C:\Users\abfernan\CrossCanFloodMapping\SResDEM\Data\QC_Quebec\QC_HRDEM\QC_HRDEM_Levis.csv']
    # for csv in csvTilesList:
    #     WbTransf.mosaikAndResamplingFromCSV(csv,8,'Ftp_dtm')

    #####  Plot histograms 
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
  
    ######  Rasterize point vectors
    # vectorInput = r'C:\Users\...\*.shp'
    # rasterOutput = U.replaceExtention(vectorInput, '.tif')
    # U.rasterizePointsVector(vectorInput,rasterOutput,atribute='y_hat',pixel_size=16)
    # U.reproject_tif(rasterOutput)


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
        
    