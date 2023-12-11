# import dc_extract
import os
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
       
def customFunction(DatasetList):
    '''
     
    '''
    wDri = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\A_DatasetsForMLP\StratifiedSampling'
    
    ####   Empty Dataset creation 
    class1_Full_Training = pd.DataFrame()
    class1_Full_Validation = pd.DataFrame()
    class5_Full_Training = pd.DataFrame()
    class5_Full_Validation = pd.DataFrame()
    print(len(DatasetList))

    for csvPath in DatasetList:
        print('_____________________New Dataset __________________')
        print(f"---- {csvPath}")
        ### Split in Class1 And Class5 and Save It.
        Class1,Class5 = U.extractFloodClassForMLP(csvPath)
        if Class1 is not None:
            ##### Stratified Sampling per Class
                        ###  Class 1
            X_train, y_train, X_test, y_test = U.stratifiedSplit_WithRandomBalanceUndersampling(Class1,'Labels')
                # Create balanced Dataset and Save it
            trainSetClass1 = U.addSubstringToName(Class1,'_balanceTrain')
            class1_X_Train = X_train
            class1_X_Train['Labels'] = y_train
            class1_X_Train.to_csv(trainSetClass1)
            class1_Full_Training = pd.concat([class1_Full_Training,class1_X_Train], ignore_index=True)

            ValSetClass1 = U.addSubstringToName(Class1,'_balanceValid')
            class1_X_Val = X_test
            class1_X_Val['Labels'] = y_test
            class1_X_Val.to_csv(ValSetClass1)
            class1_Full_Validation = pd.concat([class1_Full_Validation,class1_X_Val], ignore_index=True)
            Class1 = None
        
        if Class5 is not None:   
            ###  Class 5
            X_train, y_train, X_test, y_test = U.stratifiedSplit_WithRandomBalanceUndersampling(Class5,'Labels')
                # Create balanced Dataset and Save it
            trainSetClass5 = U.addSubstringToName(Class5,'_balanceTrain')
            class5_X_Train = X_train
            class5_X_Train['Labels'] = y_train
            class5_X_Train.to_csv(trainSetClass5)
            class5_Full_Training = pd.concat([class5_Full_Training,class5_X_Train], ignore_index=True)

            ValSetClass5 = U.addSubstringToName(Class5,'_balanceValid')
            class5_X_Val = X_test
            class5_X_Val['Labels'] = y_test
            class5_X_Val.to_csv(ValSetClass5)
            class5_Full_Validation = pd.concat([class5_Full_Validation,class5_X_Val], ignore_index=True)
            Class5 = None


    C1_Full_Train = os.path.join(wDri,'class1_Full_Training.csv')
    class1_Full_Training.to_csv(C1_Full_Train,index=None)

    C1_Full_Validation = os.path.join(wDri,'class1_Full_Validation.csv')
    class1_Full_Validation.to_csv(C1_Full_Validation,index=None)
    
    C2_Full_Train = os.path.join(wDri,'class5_Full_Training.csv')
    class5_Full_Training.to_csv(C2_Full_Train,index=None)

    C5_Full_Validation = os.path.join(wDri,'class5_Full_Validation.csv')
    class5_Full_Validation.to_csv(C5_Full_Validation,index=None)

    return True

    
def runFunctionInLoop(csvList, function = customFunction):
    '''
    Given a list <csvList>, excecute the <function> in loop, with one element from the csv as argument, at the time.  
    '''
    listOfPath = U.createListFromCSV_multiplePathPerRow(csvList)
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
    
    # subString = '_RelElev.csv'
    # wdr = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI'
    # listToCSV = U.listALLFilesInDirBySubstring_fullPath(wdr,subString)
    csvToSave = r'C:\Users\abfernan\CrossCanFloodMapping\FloodMappingProjData\HRDTMByAOI\AllDataset_RelElev.csv'
    # U.createCSVFromList(csvToSave,listToCSV)
    listToCSV = U.createListFromCSV(csvToSave)
    U.buildBalanceStratifiedDatasetByClasses(listToCSV) 



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
