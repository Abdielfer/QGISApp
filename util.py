import os, ntpath
import glob
import pathlib
import shutil
from time import strftime
from typing import Tuple, List, overload
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import rasterio as rio
from rasterio.plot import show_hist
from rasterio.enums import Resampling
from datetime import datetime
from whitebox.whitebox_tools import WhiteboxTools, default_callback
import whitebox_workflows as wbw   
from torchgeo.datasets.utils import download_url
from osgeo import gdal,ogr, osr
from osgeo import gdal_array
from osgeo.gdalconst import *
gdal.UseExceptions()

import pcraster as pcr
from pcraster import *

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

### General applications ##
class timeit(): 
    '''
    to compute execution time do:
    with timeit():
         your code, e.g., 
    '''
    def __enter__(self):
        self.tic = datetime.now()
    def __exit__(self, *args, **kwargs):
        print('runtime: {}'.format(datetime.now() - self.tic))

def makeNameByTime():
    name = strftime("%y%m%d%H%M")
    return name

### Configurations And file management
def importConfig():
    with open('./config.txt') as f:
        content = f.readlines()
    # print(content)    
    return content

def getLocalPath():
    return os.getcwd()

def makePath(str1,str2):
    return os.path.join(str1,str2)

def ensureDirectory(pathToCheck:os.path)->os.path:
    if not os.path.isdir(pathToCheck): 
        os.mkdir(pathToCheck)
        print(f"Confirmed directory at: {pathToCheck} ")
    return pathToCheck

def relocateFile(inputFilePath, outputFilePath):
    '''
    NOTE: @outputFilePath must contain the complete filename
    Sintax:
     @shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
    '''
    shutil.move(inputFilePath, outputFilePath)
    return True

def makeFileCopy(inputFilePath, outputFilePath):
    try:
        shutil.copy(inputFilePath, outputFilePath)
        return outputFilePath
    except shutil.SameFileError:
        print("Source and destination represents the same file.")
    except PermissionError:
        print("Permission denied.")
    except:
        print("Error occurred while copying file.")

def removeFile(filePath):
    try:
        os.remove(filePath)
        return True
    except OSError as error:
        print(error)
        print("File path can not be removed")
        return False

def createTransitFolder(parent_dir_path, folderName:str = 'TransitDir'):
    path = os.path.join(parent_dir_path, folderName)
    ensureDirectory(path)
    return path

def clearTransitFolderContent(path:str, filetype = '/*'):
    '''
    NOTE: This well clear dir without removing the parent dir itself. 
    We can replace '*' for an specific condition ei. '.tif' for specific fileType deletion if needed. 
    @Arguments:
    @path: Parent directory path
    @filetype: file type to delete. @default ='/*' delete all files. 
    '''
    files = glob.glob(path + filetype)
    for f in files:
        os.remove(f)
    return True

def listFreeFilesInDirByExt(cwd:str, ext = '.tif'):
    '''
    @ext = *.tif by default.
    NOTE:  THIS function list only files that are directly into <cwd> path. 
    '''
    cwd = os.path.abspath(cwd)
    print(f"Current working directory: {cwd}")
    file_list = []
    for (root, dirs, file) in os.walk(cwd):
        for f in file:
            print(f"File: {f}")
            _,_,extent = get_parenPath_name_ext(f)
            if extent == ext:
                file_list.append(f)
    return file_list

def listFreeFilesInDirByExt_fullPath(cwd:str, ext = '.csv') -> list:
    '''
    @ext = *.csv by default.
    NOTE:  THIS function list only files that are directly into <cwd> path. 
    '''
    cwd = os.path.abspath(cwd)
    # print(f"Current working directory: {cwd}")
    file_list = []
    for (root,_, file) in os.walk(cwd, followlinks=True):
        for f in file:
            # print(f"Current f: {f}")
            _,extent = splitFilenameAndExtention(f)
            # print(f"Current extent: {extent}")
            if ext == extent:
                file_list.append(os.path.join(root,f))
    return file_list

def listALLFilesInDirByExt(cwd, ext = '.csv'):
    '''
    @ext = *.csv by default.
    NOTE:  THIS function list ALL files that are directly into <cwd> path and children folders. 
    '''
    fullList: list = []
    for (root, _, _) in os.walk(cwd):
         fullList.extend(listFreeFilesInDirByExt(root, ext)) 
    return fullList

def listALLFilesInDirByExt_fullPath(cwd, ext = '.csv'):
    '''
    @ext: NOTE <ext> must contain the "." ex: '.csv'; '.tif'; etc...
    NOTE:  THIS function list ALL files that are directly into <cwd> path and children folders. 
    '''
    fullList = []
    for (root, _, _) in os.walk(cwd):
        # print(f"Roots {root}")
        localList = listFreeFilesInDirByExt_fullPath(root, ext)
        # print(f"Local List len :-->> {len(localList)}")
        fullList.extend(localList) 
    return fullList

def isSubstringInText(subStr,text)->bool:
    if subStr in text:
        return True
    return False

def createListFromCSV(csv_file_location, delim:str =','):  
    '''
    @return: list from a <csv_file_location>.
    Argument:
    @csv_file_location: full path file location and name.
    '''       
    df = pd.read_csv(csv_file_location, index_col= None, delimiter = delim)
    out = []
    for i in range(0,df.shape[0]):
        out.append(df.iloc[i,:].tolist()[0])    
    return out

def createListFromCSVColumn(csv_file_location, col_idx, delim:str =','):  
    '''
    @return: list from <col_id> in <csv_file_location>.
    Argument:
    @csv_file_location: full path file location and name.
    @col_idx : number or str(name)of the desired collumn to extrac info from (Consider index 0 <default> for the first column, if no names are assigned in csv header.)
    @delim: Delimiter to pass to pd.read_csv() function. Default = ','.
    '''       
    x=[]
    df = pd.read_csv(csv_file_location,index_col=None, delimiter = delim)
    if isinstance(col_idx,str):  
        colIndex = df.columns.get_loc(col_idx)
    elif isinstance(col_idx,int): 
        colIndex = col_idx
    fin = df.shape[0] ## rows count.
    for i in range(0,fin): 
        x.append(df.iloc[i,colIndex])
    return x
 
def createListFromExelColumn(excell_file_location,Sheet_id:str, col_idx:str):  
    '''
    @return: list from <col_id> in <excell_file_location>.
    Argument:
    @excell_file_location: full path file location and name.
    @col_id : number of the desired collumn to extrac info from (Consider index 0 for the first column)
    '''       
    x=[]
    df = pd.ExcelFile(excell_file_location).parse(Sheet_id)
    for i in df[col_idx]:
        x.append(i)
    return x

def splitFilenameAndExtention(file_path):
    '''
    pathlib.Path Options: 
    '''
    fpath = pathlib.Path(file_path)
    extention = fpath.suffix
    name = fpath.stem
    return name, extention 

def replaceExtention(inPath,newExt: str)->os.path :
    '''
    Just remember to add the poin to the new ext -> '.map'
    '''
    dir,fileName = ntpath.split(inPath)
    _,actualExt = ntpath.splitext(fileName)
    return os.path.join(dir,ntpath.basename(inPath).replace(actualExt,newExt))

def get_parenPath_name_ext(filePath):
    '''
    Ex: user/folther/file.txt
    parentPath = pathlib.PurePath('/src/goo/scripts/main.py').parent 
    parentPath => '/src/goo/scripts/'
    parentPath: can be instantiated.
         ex: parentPath[0] => '/src/goo/scripts/'; parentPath[1] => '/src/goo/', etc...
    '''
    parentPath = pathlib.PurePath(filePath).parent
    name,ext = splitFilenameAndExtention(filePath)
    return parentPath, name, ext
  
def addSubstringToName(path, subStr: str, destinyPath = None) -> os.path:
    '''
    @path: Path to the raster to read. 
    @subStr:  String o add at the end of the origial name
    @destinyPath (default = None)
    '''
    parentPath,name,ext= get_parenPath_name_ext(path)
    if destinyPath != None: 
        return os.path.join(destinyPath,(name+subStr+ext))
    else: 
        return os.path.join(parentPath,(name+subStr+ ext))

def replaceName_KeepPathAndExt(path, newName: str) -> os.path:
    '''
    @path: Path to the raster to read. 
    @subStr:  String o add at the end of the origial name
    @destinyPath (default = None)
    '''
    parentPath,_,ext= get_parenPath_name_ext(path)
    return os.path.join(parentPath,(newName+ext))
    
def createCSVFromList(pathToSave: os.path, listData:list):
    '''
    This function create a *.csv file with one line per <lstData> element. 
    @pathToSave: path of *.csv file to be writed with name and extention.
    @listData: list to be writed. 
    '''
    parentPath,name,_ = get_parenPath_name_ext(pathToSave)
    textPath = makePath(parentPath,(name+'.txt'))
    with open(textPath, 'w') as output:
        for line in listData:
            output.write(str(line) + '\n')
    read_file = pd.read_csv (textPath)
    print(f'Creating CSV at {pathToSave}')
    read_file.to_csv (pathToSave, index=None)
    removeFile(textPath)
    return True

def updateDict(dic:dict, args:dict)->dict:
    outDic = dic
    for k in args.keys():
        if k in dic.keys():
            outDic[k]= args[k]
    return outDic

###################            
### General GIS ###
###################

def plotImageAndMask(img, mask,imgName:str='Image', mskName: str= 'Mask'):
    # colList = ['Image','Mask']
    image = img.detach().numpy() if torch.is_tensor(img) else img.numpy().squeeze()
    mask_squeezed = mask.detach().numpy() if torch.is_tensor(mask) else mask.numpy().squeeze()
    fig, axs = plt.subplots(1,2, figsize=(10,5), sharey=True)
    axs[0].imshow(image, cmap='Greys_r')
    axs[0].set(xlabel= imgName)
    axs[1].imshow(mask_squeezed, cmap='Greys_r')
    axs[1].set(xlabel= mskName)
    plt.rcParams['font.size'] = '15'
    fig.tight_layout()
 
def imageToTensor(img,DTYPE:str = 'float32'):
    imagTensor = img.astype(DTYPE)
    # imagTensor = np.transpose(imagTensor, (2, 0, 1)).astype(DTYPE)
    imagTensor = torch.tensor(imagTensor)
    return imagTensor

def reshape_as_image(arr):
    '''
    From GDL
    Parameters
    ----------
    arr : arr as image in raster order (bands, rows, columns)
    return: array-like in the image form of (rows, columns, bands)
    '''       
    return np.ma.transpose(arr, [1, 2, 0]).astype('float32')

def reshape_as_raster(arr):
    '''  
    From GDL
        swap the axes order from (rows, columns, bands) to (bands, rows, columns)
    Parameters
    ----------
    arr : array-like in the image form of (rows, columns, bands)
    return: arr as image in raster order (bands, rows, columns)
    '''
    return np.transpose(arr, [2, 0, 1])

def plotHistComparison(DEM1,DEM2, bins:int = 50):
    # Reding raster 1:
    data_DEM1,_= readRasterWithRasterio(DEM1)  # Return an Array
    data_DEM1 = np.resize(data_DEM1,(1))
    # Reding raster 2:
    data_DEM2,_= readRasterWithRasterio(DEM2)  # Return an Array
    # data_DEM2 = np.resize(data_DEM2,(1))
    # Setting plot
    n_bins = bins
    fig, ax = plt.subplots(1,sharey=True, tight_layout=True)
    # x=np.array((data_DEM1[0],data_DEM2[0]))
    
    ax.hist(data_DEM1, n_bins, density=True, histtype='step', label=['cdem'],stacked=True, fill=False)
    # ax.hist(data_DEM2[0], n_bins, density=True, histtype='step', label=colors,stacked=True, fill=False)
    ax.legend(prop={'size': 10})
    ax.set_title('cdem_16m vs srdem_8m') 
    
    fig.tight_layout()
    plt.show()
    pass

def getNeighboursValues(raster)-> np.array:
    '''
    Inspect the 8 neighbours of each pixel to list their values. If the pixel being NoData, the neighbour list is empty. 
    @raster: os.path to the raster to inst=pect.
    @return: array of lists. 
    '''
    # Convert raster to NumPy array
    arr,profil = readRasterWithRasterio(raster)
    NOData = profil['nodata']
    # Get dimensions of array
    rows, cols = arr.shape

    # Create empty array to store neighbours
    neighbours = np.empty((rows, cols), dtype=object)

    # Iterate over each cell in array
    for i in range(rows):
        for j in range(cols):
            # Get value of current cell
            val = arr[i, j]

            # Check if value is NaN or NoData
            if np.isnan(val) or val == NOData:
                neighbours[i, j] = []
            else:
                # Get indices of neighbouring cells
                indices = [(i-1, j-1), (i-1, j), (i-1, j+1),
                           (i, j-1),             (i, j+1),
                           (i+1, j-1), (i+1, j), (i+1, j+1)]

                # Get values of neighbouring cells
                vals = [arr[x, y] for x, y in indices if 0 <= x < rows and 0 <= y < cols]
                # Add values to neighbours array
                neighbours[i, j] = vals

    return neighbours


#######################
### Rasterio Tools  ###
#######################

def readRasterWithRasterio(rasterPath:os.path) -> tuple[np.array, dict]:
    '''
    Read a raster with Rasterio.
    return:
     Raster data as np.array
     Raster.profile: dictionary with all rater information
    '''
    inRaster = rio.open(rasterPath, mode="r")
    profile = inRaster.profile
    rasterData = inRaster.read()
    print(f"raster data shape in ReadRaster : {rasterData.shape}")
    return rasterData, profile

def read_tiff_advanced(file_path: str) -> Tuple[np.ndarray, str, str]:
    try:
        with rio.open(file_path) as dataset:
            image_data = dataset.read()
            file_extension = dataset.profile['driver']
            crs = dataset.crs.to_string()
        return image_data, file_extension, crs
    except Exception as e:
        print(f"The TIFF in the path {file_path} is corrupted.")
        return None, None, None

def createRaster(savePath:os.path, data:np.array, profile, noData:int = None):
    '''
    parameter: 
    @savePath: Most contain the file name ex.: *name.tif.
    @data: np.array with shape (bands,H,W)
    '''
    B,H,W = data.shape[-3],data.shape[-2],data.shape[-1] 
    # print(f"C : {B}, H : {H} , W : {W} ")
    profile.update(dtype = rio.uint16, nodata = noData, blockysize = profile['blockysize'])
    with rio.open(
        savePath,
        mode="w",
        #out_shape=(B, H ,W),
        **profile
        ) as new_dataset:
            # print(f"New Dataset.Profile: ->> {new_dataset.profile}")
            new_dataset.write(data)
            print("Created new raster>>>")
    return savePath

def write_multilayer_tiff(file_path: str, data: List[np.ndarray], file_extension: str, crs: str) -> None:
    profile = {
        'driver': file_extension,
        'dtype': 'float32',
        'count': len(data),
        'width': data[0].shape[1],
        'height': data[0].shape[0],
        'crs': crs,
        'transform': rio.transform.from_origin(0, 0, 1, 1)
    }
    
    with rio.open(file_path, 'w', **profile) as dst:
        for i, layer in enumerate(data):
            dst.write(layer.astype('float32'), i + 1)

def plotHistogram(raster, CustomTitle:str = None, bins: int=50, bandNumber: int = 1):
    if CustomTitle is not None:
        title = CustomTitle
    else:
        title = f"Histogram of band : {bandNumber}"    
    data,_ = readRasterWithRasterio(raster)
    
    show_hist(source=data, bins=bins, title= title, 
          histtype='stepfilled', alpha=0.5)
    return True

def replaceRastNoDataWithNan(rasterPath:os.path,extraNoDataVal: float = None)-> np.array:
    rasterData,profil = readRasterWithRasterio(rasterPath)
    NOData = profil['nodata']
    rasterDataNan = np.where(((rasterData == NOData)|(rasterData == extraNoDataVal)), np.nan, rasterData) 
    return rasterDataNan

def computeRaterStats(rasterPath:os.path):
    '''
    Read a reaste and return: 
    @Return
    @rasMin: Raster min.
    @rasMax: Raster max.
    @rasMean: Rater mean.
    @rasMode: Raster mode.
    @rasSTD: Raster standard deviation.
    @rasNoNaNCont: Raster count of all NOT NoData pixels
    '''
    rasDataNan = replaceRastNoDataWithNan(rasterPath)
    rasMin = np.nanmin(rasDataNan)
    rasMax = np.nanmax(rasDataNan)
    rasMean = np.nanmean(rasDataNan)
    rasSTD = np.nanstd(rasDataNan)
    rasNoNaNCont = np.count_nonzero(rasDataNan != np.nan)
    # Compute mode
    vals,counts = np.unique(rasDataNan, return_counts=True)
    index = np.argmax(counts)
    rasMode = vals[index]
    return rasMin, rasMax, rasMean,rasMode, rasSTD, rasNoNaNCont

def computeRasterValuePercent(rasterPath, value:int=1)-> float:
    '''
    Compute the percent of pixels of value <value: default =1> in a raster. 
    @rasterPath: Path to the raster to be analyzed.
    @value: Value to verify percent in raster. Default = 1. 
    @return: The computed percent of <value> within the nonNoData values in the input raster.  
    '''
    rasDataNan = replaceRastNoDataWithNan(rasterPath)
    rasNoNaNCont = np.count_nonzero(rasDataNan != np.nan)
    valuCont = np.count_nonzero(rasDataNan == value)
    return (valuCont/rasNoNaNCont)*100

###########################
####   PCRaster Tools  ####
###########################

def computeHAND(DEMPath,HANDPath,saveDDL:bool=True,saveStrahOrder:bool=True,saveSubCath:bool = False) -> os.path:
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
    ### Prepare output file path. 
    path,communName,_ = get_parenPath_name_ext(DEMPath)
    lddPath =os.path.join(path,str(communName+'_ldd.map'))
    strahleOrdPath =os.path.join(path,str(communName+'_StrOrder.map'))
    subCatch =os.path.join(path,str(communName+'_subCatch.map'))
    DEMMap = translateToPCRaster(DEMPath)
    pcr.setclone(DEMMap)
    DEM = pcr.readmap(DEMMap)
    aguila(DEM)
    ## Flow Direcction (Use to take long...)
    print("#####......Computing D8 flow dir.......######")
    threshold = 8
    FlowDir = lddcreate(DEM,1e31,1e31,1e31,1e31)
    if saveDDL: 
        pcr.report(FlowDir,lddPath)

    # Compute river network
    print('#####......Computing Strahler order.......######')
    strahlerOrder = streamorder(FlowDir)
    strahlerRiver = ifthen(strahlerOrder>=threshold,strahlerOrder)
    if saveStrahOrder:
        pcr.report(strahlerRiver, strahleOrdPath)
    
    print('#####......Finding outlets.......######')
    junctions = ifthen(downstream(FlowDir,strahlerOrder) != strahlerRiver, boolean(1))
    outlets = ordinal(cover(uniqueid(junctions),0))
    
    print('#####......Calculating subcatchment.......######')
    subCatchments = catchment(FlowDir,outlets)
    if saveSubCath:
        pcr.report(subCatchments,subCatch)
    
    # Compute HAND
    print('#####......Computing HAND.......######')
    areaMin = areaminimum(DEM,subCatchments)
    HAND = DEM - areaMin
    # Save HAND as *.map
    pcr.report(HAND,HANDPath)
    
    ##### Print maps
            ## Uncomment to show images of computed values. 
    # print('#####......Ready to print.......######')
    # aguila(HAND)
    # aguila(subCatchments)
    # aguila(areaMin)
    
    #Save HAND as *.tif
    handTifOut = replaceExtention(HANDPath,'.tif')
    translateToTiff(HANDPath,handTifOut)
    return handTifOut

def extractHydroFeatures(DEMPath) -> bool:
    '''
    NOTE: Important to ensure the input DEM has a well defined NoData Value ex. -9999. 
    1- *.tif in (DEMPath) is converted to PCRaster *.map format -> U.saveTiffAsPCRaster(DEM)
    2- pcr.setClone(DEMMap) : Ensure extention, CRS and other characteristics for creating new *.map files.
    3- Read DEM in PCRasterformat

    4- Compute strahler Order-> streamorder(FlowDir)
    5- Compute main river -> streamorder >= 10
    6- Extract outlets: Defines as outlet all the intersections in the river network.
    7- Compute subcatchment extention corresponding to each outlet.
    8- Compute flow accumulation (in cell numbers)

    Some other measurements are available as options. Uncomment the lines to compute and/or save. 
    
    @DEMPath : Input path to the DEM in *.tif format.
    @LDDPath : Input path to a precalculated LDD map through (pcr.lddcreation())
    @Return: <True> if the process is complete without error. Otherwhise, you'll receive an error from the PCRaster algorithm. 
    '''
    path,communName,_ = get_parenPath_name_ext(DEMPath)
    # Create output names
    slopePath = os.path.join(path,str(communName+'_slp.map'))
    lddOutPath = os.path.join(path,str(communName+'_ldd.map'))
    subCatchPath =os.path.join(path,str(communName+'_subCatch.map'))
    areaMinPath = os.path.join(path,str(communName+'_areaMin.map'))
    areaMaxPath = os.path.join(path,str(communName+'_areaMax.map'))
    outletsPath = os.path.join(path,str(communName+'_Outlets.map'))
    flowAccumulationPath = os.path.join(path,str(communName+'_FlowAcc.map'))
    maxFAccByCatchmentPath = os.path.join(path,str(communName+'_MaxFAccByCatch.map'))
    strahlerOrderPath = os.path.join(path,str(communName+'_StrahOrder.map'))
    mainRiverPath = os.path.join(path,str(communName+'_mainRiverPath.map'))
    
    pcr.setclone(DEMPath)
    DEM = pcr.readmap(DEMPath)
    print('#####......Computing Slope.......######')
    slp = slope(DEM)
    saveIt(slp,slopePath)
    
    print('#####......Computing LDD .......######')
    with timeit(): 
        FlowDir = lddcreate(DEM,1e31,1e31,1e31,1e31)
    saveIt(FlowDir,lddOutPath)
    print('#####......LDD Ready .......######')

    print('#####......Computing Strahler order and Main River.......######')
    threshold = 5
    strahlerOrder = streamorder(FlowDir)
    strahlerRiver = ifthen(strahlerOrder >=threshold,strahlerOrder)
    #####  Get Max strahler order
    array = pcraster.pcr2numpy(strahlerOrder, np.nan)
    # Get the maximum value
    max_value = np.nanmax(array)
    limit = (max_value-3)
    print(f'Max Satrahler order = {max_value}. For main river considered {limit} to {max_value}')
    
    ## Extract Main river with the 3 las strahler orders
    MainRiver = ifthen(strahlerOrder >= limit,strahlerOrder)
    saveIt(strahlerRiver,strahlerOrderPath)
    _,mainRiverTif = saveIt(MainRiver,mainRiverPath)
    print(f"Main_River.tif saved  --> {mainRiverTif} \n")
    
    print('#####......Finding outlets.......######')
    junctions = ifthen(downstream(FlowDir,strahlerOrder) != strahlerRiver, boolean(1))
    outlets = ordinal(cover(uniqueid(junctions),0))
    # saveIt(outlets,outletsPath)

    print('#####......Calculating subcatchment.......######')
    subCatchments = catchment(FlowDir,outlets)
    # saveIt(subCatchments,subCatchPath)
    
    print('#####......Computing subcatchment measures.......######')
    massMap = pcr.spatial(pcr.scalar(1.0))
    flowAccumulation = accuflux(FlowDir, massMap)
    MaxFAccByCatchment = areamaximum(flowAccumulation,subCatchments)
    # areaMin = areaminimum(DEM,subCatchments)    # Optional
    areaMax = areamaximum(DEM,subCatchments)    # Optional
    
    ## Saving subcatchment measures
    saveIt(flowAccumulation,flowAccumulationPath)
    # saveIt(areaMin,areaMinPath)    # Optional
    saveIt(areaMax,areaMaxPath)    # Optional
    saveIt(MaxFAccByCatchment,maxFAccByCatchmentPath)
    del DEM,areaMax,massMap,flowAccumulation,MaxFAccByCatchment,subCatchments,outlets,junctions,MainRiver,strahlerRiver,strahlerOrder,FlowDir
    return mainRiverTif

def saveIt(dataset, path):
        pcr.report(dataset,path)
        path_Reproj = assigneProjection(path)
        translatedTIff = translateToTiff(path_Reproj)
        if path_Reproj and (path != path_Reproj):
            removeFile(path)
            return path_Reproj,translatedTIff
        else:
            return path,translatedTIff
######################
####   GDAL Tools  ###
######################

class RasterGDAL():
    '''
    Some info about GDAL deo Transform
    adfGeoTransform[0] /* top left x */
    adfGeoTransform[1] /* w-e pixel resolution */
    adfGeoTransform[2] /* rotation, 0 if image is "north up" */
    adfGeoTransform[3] /* top left y */
    adfGeoTransform[4] /* rotation, 0 if image is "north up" */
    adfGeoTransform[5] /* n-s pixel resolution */
    
    '''
    def __init__(self, rasterPath) -> None:
        gdal.AllRegister() # register all of the drivers
        gdal.DontUseExceptions()
        self.inputPath = rasterPath
        self.ds = gdal.Open(rasterPath)
        if self.ds is None:
            print('Could not open image')
            sys.exit(1)   
        # get image size
        self.rows = self.ds.RasterYSize
        self.cols = self.ds.RasterXSize
        self.NumOfBands = self.ds.RasterCount
        # get georeference info
        transform = self.ds.GetGeoTransform()
        self.xOrigin = transform[0]
        self.yOrigin = transform[3]
        self.pixelWidth = transform[1]
        self.pixelHeight = transform[5]
        self.projection = self.ds.GetProjection()
        self.MetaData = self.ds.GetMetadata()
        self.band1 = self.ds.GetRasterBand(1)
        self.NoData = self.band1.GetNoDataValue()

    def setDirGDAL(self, path ):
        os.chdir()
    
    def getRasterDataset(self):
        return self.ds 
   
    def getRasterNpArray(self, maskNoData:bool = True)-> np.array:
        arr = self.ds.ReadAsArray()
        if maskNoData:
            arr = np.ma.masked_equal(arr, self.NoData)
        return arr
    
    def computePixelOffset(self,x,y):
        # compute pixel offset
        xOffset = int((x - self.xOrigin) / self.pixelWidth)
        yOffset = int((y - self.yOrigin) / self.pixelHeight)
        return xOffset, yOffset

    def closeRaster(self):
        self.ds = None

    def translateRaster(self, outpPath, format:str = "GeoTiff"):
        """
        Ref: https://gdal.org/api/python/osgeo.gdal.html#osgeo.gdal.Translate
        """
        gdal.Translate(outpPath,self.ds,format=format)
        return True

    def saveTiffAsPCRaster(self):
        outpPath = ntpath.basename(self.inputPath).replace('.tif','.map') 
        gdal.Translate(outpPath,self.ds,format='PCRaster')
        return outpPath

    def printRaster(self):
        print("---- Image size ----")
        print(f"Row : {self.rows}")
        print(f"Cols : {self.cols}")
        print(f"xOrigin : {self.xOrigin}")
        print(f"yOrigin : {self.yOrigin}") 
        print(f"NumOfBands : {self.NumOfBands}")
        print(f"pixelWidth : {self.pixelWidth}")
        print(f"pixelHeight : {self.pixelHeight}")
        print(f"projection : {self.projection}")
        print(f"MetaData : {self.MetaData}")

###  GDAL independent functions
def replace_no_data_value(dataset_path, new_value:float = -9999):
    dataset = gdal.Open(dataset_path, gdal.GA_Update)
    band = dataset.GetRasterBand(1)
    old_value = band.GetNoDataValue()
    for i in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(i)
        band_array = band.ReadAsArray()
        band_array[band_array == old_value] = new_value
        band.WriteArray(band_array)
        band.SetNoDataValue(new_value)
    dataset.FlushCache()

def clipRasterByMask(DEMPath:os.path, vectorMask, outPath)-> gdal.Dataset:
    mask_bbox = get_Shpfile_bbox(vectorMask)
    gdal.Warp(outPath, DEMPath,outputBounds=mask_bbox,cutlineDSName=vectorMask, cropToCutline=True)
    print(f"Successfully clipped at : {outPath}")
    return outPath

def translateToTiff(inPath) -> bool:
    """
    Write a *tif raster from an appropriate(GDAL accepted formats) raster input. The function return a raster with the same characteristic,
      with NoDataValue seted to -9999.
    @inPath: os.path: Path to the input raster
    @Return: os.path: Path to the *.tif raster. 
    
    """
     # Open the input raster
    input_raster = gdal.Open(inPath)
   
    # Transform the dataset NoData value into -9999
    input_band = input_raster.GetRasterBand(1)
    input_band.SetNoDataValue(-9999)

    # Create an empty raster file with the same CRS as the input raster
    output_path = os.path.splitext(inPath)[0] + '.tif'
    driver = gdal.GetDriverByName('GTiff')
    output_raster = driver.CreateCopy(output_path, input_raster, 0)

    # Write the dataset with the transformed NoData into the empty raster
    output_band = output_raster.GetRasterBand(1)
    output_band.WriteArray(input_band.ReadAsArray())

    # Close the rasters
    input_raster = None
    output_raster = None

    return output_path

def translateToPCRaster(inputPath) -> str:
    outpPath = replaceExtention(inputPath,'.map')
    options = gdal.TranslateOptions(format='PCRaster',noData=-9999) # creationOptions=["COMPRESS=LZW"]
    gdal.Translate(outpPath,inputPath,options=options) #outputType=gdal.GDT_Float32
    return outpPath

def readRasterAsArry(rasterPath):
   return gdal_array.LoadFile(rasterPath)

def extractProjection(inPath):
    '''
    Extract the projection of the dataset with GDAL.GetProjection()
    @inPath: path to the input file. Must be in one of the GDAL format. ex. 'GTiff'
    @Return: projection file
    '''
    # Open the input TIFF file
    dataset = gdal.Open(inPath)
    # Get the input CRS
    crs = dataset.GetProjection()
    print(crs)
    return crs

def reproject_PCRaster(tif_file,output_crs:str='EPSG:3979') -> str:
    """
    Reprojects a PCraster file to the specified coordinate reference system (CRS).
    Args:
        map_file(str): Path to the input *.map file.
        output_crs (str): Output coordinate reference system (CRS) in the format 'EPSG:<code>'.

    Returns:
        str: Path to the reprojected *.map file.
    NOTE: NOData value do not works. 
    """
    # get input name and extention
    parent,inputNeme,ext = get_parenPath_name_ext(tif_file)
    # Open the input TIFF file
    dataset = gdal.Open(tif_file)
    
    # Create a spatial reference object for the output CRS
    output_srs = osr.SpatialReference()
    output_srs.ImportFromEPSG(int(output_crs.split(':')[1]))
    output_file_path = os.path.join(parent,inputNeme + '_' + ext)
    
    # Create the output dataset
    '''
    NO definas input dataset crs. 
       srcSRS=input_srs, 
    '''
    gdal.Warp(output_file_path, dataset, dstSRS=output_srs, outputType=gdal.GDT_Float32, dstNodata=-9999, creationOptions=['PCRASTER_VALUESCALE=VS_SCALAR'])
    # gdal.Warp(output_file_path, dataset, dstSRS=output_srs, resampleAlg=gdal.GRA_Bilinear, dstNodata=-9999,outputType=gdal.GDT_Float32)
    # Close the datasets
    del dataset
    return output_file_path

def reproject_tif(tif_file, output_crs:str='EPSG:3979') -> str:
    """
    Reprojects a TIFF file to the specified coordinate reference system (CRS).
    Args:
        tif_file (str): Path to the input TIFF file.
        output_crs (str): Output coordinate reference system (CRS) in the format 'EPSG:<code>'. Default <'EPSG:3979'>

    Returns:
        str: Path to the reprojected TIFF file.
    """
    # get input name and extention
    parent,inputNeme,ext = get_parenPath_name_ext(tif_file)
    # Open the input TIFF file
    replace_no_data_value(tif_file)
    dataset = gdal.Open(tif_file,gdal.GA_Update)
    # Create a spatial reference object for the output CRS
    output_srs = osr.SpatialReference()
    output_srs.ImportFromEPSG(int(output_crs.split(':')[1]))
    output_file_path = os.path.join(parent,inputNeme + '_' + ext)
    
    # Create the output dataset
    '''
    Do not define input dataset crs. 
       srcSRS=input_srs, 
    It works better. Do not know way..!!
    '''
    gdal.Warp(output_file_path, dataset, dstSRS=output_srs, resampleAlg=gdal.GRA_Bilinear, dstNodata=-9999)#outputType=gdal.GDT_Float32
    # Close the datasets
    del dataset
    return output_file_path

def assigneProjection(raster_file, output_crs:str='EPSG:3979') -> str:
    '''
    Assigne prejection <outpt_crs> to files in *.tif and *.map format IF they have no projection defined.
    parameters:
     @raster_file: os.path: Path to the raster to be reprojected.
     @output_crs: <EPSG:####> projection.
     @return: The path to the reprojected file.
    '''
    _,communName,ext = get_parenPath_name_ext(raster_file)
    input_crs = extractProjection(raster_file)
    if not input_crs:
        print(f'Reprojecting..... {communName}{ext}')
        if "map" in ext:
            return reproject_PCRaster(raster_file,output_crs)
        if 'tif' in ext:
            return reproject_tif(raster_file,output_crs) 


def crop_tif(inputRaster:os.path, maskVector:os.path, outPath:os.path)->os.path:
    """
    Crops a TIFF file using a shapefile as a mask.
    NOTE: It is important to FILL THE NEW DATASET WITH np.nan to avoid ending with big extentions of valid values, instead of NoData. 

    Args:
        inputRaster (str): Path to the input TIFF file.
        maskVector (str): Path to the input shapefile.
        outPath (str): Path to the output TIFF file.
    Returns:
        str: Path to the output TIFF file.
    """
    print(f'Into crop_tif, tif_file: {inputRaster}')
    # Open the input TIFF file
    dataset = gdal.Open(inputRaster)
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    count = dataset.RasterCount
    datatype = gdal.GetDataTypeName(dataset.GetRasterBand(1).DataType)
    print(f'cols,rows: {cols}:--{rows}')
    print(f'datatype: {datatype}')
    # Open the shapefile
    shapefile_ds = ogr.Open(maskVector)
    layer = shapefile_ds.GetLayer()
    # Get the extent of the shapefile
    extent = layer.GetExtent()
    print(f'extent: {extent}')
    # Set the output file format
    driver = gdal.GetDriverByName('GTiff')
    # Create the output dataset
    output_dataset = driver.Create(outPath, cols,rows,count, gdal.GDT_Float32)
    output_dataset.GetRasterBand(1).Fill(-99999)  # Important step to ensure DO NOT FILL the whole extention with valid values. 
    # Set the geotransform and projection
    output_dataset.SetGeoTransform(dataset.GetGeoTransform())
    output_dataset.SetProjection(dataset.GetProjection())
    # Perform the cropping
    print(f'output_dataset: {output_dataset}')
    gdal.Warp(output_dataset, dataset, outputBounds=extent, cutlineDSName=maskVector, cropToCutline=True)
    # ,cutlineLayer = 'bc_quesnel'
    # Close the datasets
    dataset = output_dataset= shapefile_ds = None
    return outPath

def get_Shpfile_bbox(file_path) -> Tuple[float, float, float, float]:
    driver = ogr.GetDriverByName('ESRI Shapefile')
    data_source = driver.Open(file_path, 0)
    layer = data_source.GetLayer()
    extent = layer.GetExtent()
    min_x, max_x, min_y, max_y = extent
    return min_x, min_y, max_x, max_y

def get_Shpfile_bbox_str(file_path) -> str:
    driver = ogr.GetDriverByName('ESRI Shapefile')
    data_source = driver.Open(file_path, 0)
    layer = data_source.GetLayer()
    extent = layer.GetExtent()
    min_x, max_x, min_y, max_y = extent
    bboxStr = str(round(min_x, 2))+','+str(round(min_y,2))+','+str(round(max_x,2))+','+str(round(max_y,2))
    return bboxStr

def computeProximity(inRaster, value:int= 1, outPath:os.path = None) -> os.path:
    '''
    Compute the horizontal distance to features in the input raster.
    @inRaster: A raster with features to mesure proximity from. A 0-1 valued raster,where the 1s are cells of the river network. 
    @outPath: Path to save the output raster. If None,the output is create in the same folder as the input with prefix: <_proximity.tif>.
    @values: list of values to be considered as terget in the inRaster. Default [1].  
    '''
    if outPath is None:  
        path,communName,_ = get_parenPath_name_ext(inRaster)
        # Create output name
        outPath =os.path.join(path,str(communName+'_proximity.tif'))
    ds = gdal.Open(inRaster,GA_ReadOnly)
    band = ds.GetRasterBand(1)
    gt = ds.GetGeoTransform()
    sr = ds.GetProjection()
    cols = ds.RasterXSize
    rows = ds.RasterYSize

    # create empty proximity raster
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(outPath, cols, rows, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(sr)
    out_band = out_ds.GetRasterBand(1)

    # compute proximity
    gdal.ComputeProximity(band, out_band, [f'VALUES= {value}', 'DISTUNITS=GEO'])
    # delete input and output rasters
    del ds, out_ds
    return outPath

def get_neighbors_GDAL(raster_file, shape_file):
    """
    This function takes a raster and a shapefile as input and returns an array containing
    the values of the 8 neighbors of each pixel in the raster. The inputs must be in the
    same reference system. The output array will have the same size of the input raster
    and must contain at each cell a list of shape [1:8], with the values of the 8 neighbors
    of the questioned pixel, starting from the left up corner. The function considers only
    the pixels that are not np.nan or noData value in any of the two input layers. If the
    questioned pixel is noData in any of the inputs or np.nan, the list of neighbors values
    must be empty.
                               Neighbourhood order
                                #############
                                ## 1  2  3 ##
                                ## 4  X  5 ##
                                ## 6  7  8 ##
                                #############

    @raster_file: path to raster file.
    @shape_file: path to shapefile.
    @return: array containing values of 8 neighbors for each pixel in raster
    """
    # Open raster file and get band, width, height, and transform information
    dataset = gdal.Open(raster_file)
    band = dataset.GetRasterBand(1)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    transform = dataset.GetGeoTransform()

    # Open shapefile and get layer information
    shape = gdal.OpenEx(shape_file)
    layer = shape.GetLayer()

    # Create output array with same size as input raster
    output_array = np.empty((height, width), dtype=object)

    # Loop through each pixel in raster and get its neighbors' values
    for y in range(height):
        for x in range(width):
            # Get value at current pixel location
            value = band.ReadAsArray(x, y, 1, 1)[0][0]

            # Check if current pixel is not np.nan or noData value in either input layer
            if not np.isnan(value) and value != band.GetNoDataValue():
                # Get coordinates for current pixel location
                x_coord = transform[0] + (x * transform[1]) + (y * transform[2])
                y_coord = transform[3] + (x * transform[4]) + (y * transform[5])

                ## Create polygon for current pixel location
                # This could be slow, but is less error-prone than rasterize polygons. 
                ring = ogr.Geometry(ogr.wkbPolygon)
                ring.AddPoint(x_coord, y_coord)
                ring.AddPoint(x_coord + transform[1], y_coord)
                ring.AddPoint(x_coord + transform[1], y_coord + transform[5])
                ring.AddPoint(x_coord, y_coord + transform[5])
                ring.AddPoint(x_coord, y_coord)
                poly = ogr.Geometry(ogr.wkbPolygon)
                poly.AddGeometry(ring)

                # Loop through each feature in shapefile and check if it intersects with current pixel location polygon
                neighbors = []
                for feature in layer:
                    if feature.GetGeometryRef().Intersects(poly):
                        neighbor_value = feature.GetField("value")
                        if not np.isnan(neighbor_value) and neighbor_value != band.GetNoDataValue():
                             # Get indices of neighbouring cells
                            indices = [(x_coord-transform[0], y_coord-transform[3]),        #1
                                    (x_coord-transform[0], y_coord),                        #2
                                    (x_coord-transform[0], y_coord+transform[3]),           #3
                                    (x_coord, y_coord-transform[3]),                        #4
                                    (x_coord, y_coord+transform[3]),                        #5
                                    (x_coord+transform[0], y_coord-transform[3]),           #6
                                    (x_coord+transform[0], y_coord),                        #7
                                    (x_coord+transform[0], y_coord+transform[3])]           #8

                            # Get values of neighbouring cells that are not NaN or NoData in either input layer
                            vals = [band[x, y] for x, y in indices]
                            # Add values to neighbours array
                            neighbors[x_coord, y_coord] = vals
                # Add list of neighbor values to output array at current pixel location
                output_array[y,x] = neighbors

            else:
                # Add empty list to output array at current pixel location since current pixel is noData or np.nan
                output_array[y,x] = []

    return output_array

############################
#### Datacube_ Extract  ####
############################

def dc_describe(cfg: DictConfig)-> bool:
    '''
    Configurate the call of d.describe() with hydra parameters.
    '''
    instantiate(OmegaConf.create(cfg.dc_Extract_params['dc_describeCollections']))
    return True

def dc_search(cfg: DictConfig)-> str :
    '''
    Configurate the call of d.search()  with hydra parameters.
    return the output path of the search result.
    '''
    out = instantiate(OmegaConf.create(cfg.dc_Extract_params['dc_search']))
    return out

def dc_extraction(cfg: DictConfig, args:dict=None)-> str:
    '''
    Configurate the call of extract_cog() with hydra parameters.
    return the output path of the extracted file.
    '''
    dict_DcExtract = OmegaConf.create(cfg.dc_Extract_params['dc_extrac_cog'])
    if args is not None:
        dict_DcExtract = updateDict(dict_DcExtract,args)
    print(f"New dcExtract Dict:  {dict_DcExtract}")
   
    ##  procede to extraction
    out = instantiate(dict_DcExtract)
    return out

    
######################################
####   WhiteBoxTools and  Rasterio ###
######################################

## LocalPaths and global variables: to be adapted to your needs ##
currentDirectory = os.getcwd()
wbt = WhiteboxTools()
wbt.set_working_dir(currentDirectory)
wbt.set_verbose_mode(True)
wbt.set_compress_rasters(True) # compress the rasters map. Just ones in the code is needed

    ## Pretraitment #
class WbT_dtmTransformer():
    '''
     This class contain some functions to generate geomorphological and hydrological features from DEM.
    Functions are based on WhiteBoxTools and Rasterio libraries. For optimal functionality DTM’s most be high resolution, ideally Lidar derived  1m or < 2m. 
    '''
    def __init__(self, workingDir:str = None) -> None:
        if (workingDir is not None and os.path.isdir(workingDir)): # Creates output dir if it does not already exist 
            self.workingDir = workingDir
            wbt.set_working_dir(workingDir)
        print(f"Working dir at: {self.workingDir}")    
        
    def fixNoDataAndfillDTM(self, inDTMName, eraseIntermediateRasters = False)-> os.path:
        '''
        Ref:   https://www.whiteboxgeo.com/manual/wbt_book/available_tools/hydrological_analysis.html#filldepressions
        To ensure the quality of this process, this method execute several steep in sequence, following the Whitebox’s authors recommendation (For mor info see the above reference).
        Steps:
        1-	Correct no data values to be accepted for all operation. 
        2-	Fill gaps of no data.
        3-	Fill depressions.
        4-	Remove intermediary results to save storage space (Optionally you can keep it. See @Arguments).  
        @Argument: 
        -inDTMName: Input DTM name
        -eraseIntermediateRasters(default = True): Erase intermediate results to save storage space. 
        @Return: True if all process happened successfully, ERROR messages otherwise. 
        @OUTPUT: DTM <filled_ inDTMName> Corrected DTM with wang_and_liu method. 
        '''
        dtmNoDataValueSetted = addSubstringToName(inDTMName,'_NoDataOK')
        
        wbt.set_nodata_value(
            inDTMName, 
            dtmNoDataValueSetted, 
            back_value=0.0, 
            callback=default_callback
            )
        dtmMissingDataFilled = addSubstringToName(inDTMName,'_MissingDataFilled')
        wbt.fill_missing_data(
            dtmNoDataValueSetted, 
            dtmMissingDataFilled, 
            filter=11, 
            weight=2.0, 
            no_edges=True, 
            callback=default_callback
            )
        output = addSubstringToName(inDTMName,"_filled_WangLiu")
        wbt.fill_depressions_wang_and_liu(
            dtmMissingDataFilled, 
            output, 
            fix_flats=True, 
            flat_increment=None, 
            callback=default_callback
            )
        if eraseIntermediateRasters:
            try:
                os.remove(os.path.join(wbt.work_dir,dtmNoDataValueSetted))
                os.remove(os.path.join(wbt.work_dir,dtmMissingDataFilled))
            except OSError as error:
                print("There was an error removing intermediate results : \n {error}")
        return output

    def d8FPointerRasterCalculation(self, inFilledDTMName):
        '''
        @argument:
         @inFilledDTMName: DTM without spurious point ar depression.  
        @UOTPUT: D8_pioter: Raster tu use as input for flow direction and flow accumulation calculations. 
        '''
        output = addSubstringToName(inFilledDTMName,"_d8Pointer")
        wbt.d8_pointer(
            inFilledDTMName, 
            output, 
            esri_pntr=False, 
            callback=default_callback
            )
    
    def d8_flow_accumulation(self, inFilledDTMName):
        d8FAccOutputName = addSubstringToName(inFilledDTMName,"_d8fllowAcc" ) 
        wbt.d8_flow_accumulation(
            inFilledDTMName, 
            d8FAccOutputName, 
            out_type="cells", 
            log=False, 
            clip=False, 
            pntr=False, 
            esri_pntr=False, 
            callback=default_callback
            ) 
        return d8FAccOutputName
    
    def jensePourPoint(self,inOutlest,d8FAccOutputName):
        jensenOutput = "correctedSnapPoints.shp"
        wbt.jenson_snap_pour_points(
            inOutlest, 
            d8FAccOutputName, 
            jensenOutput, 
            snap_dist = 15.0, 
            callback=default_callback
            )
        print("jensePourPoint Done")

    def watershedConputing(self,d8Pointer, jensenOutput):  
        output = addSubstringToName(d8Pointer, "_watersheds")
        wbt.watershed(
            d8Pointer, 
            jensenOutput, 
            output, 
            esri_pntr=False, 
            callback=default_callback
        )
        print("watershedConputing Done")

    def DInfFlowCalculation(self, inD8Pointer, log = False):
        ''' 
        Compute DInfinity flow accumulation algorithm.
        Ref: https://www.whiteboxgeo.com/manual/wbt_book/available_tools/hydrological_analysis.html#dinfflowaccumulation  
        We keep the DEFAULT SETTING  from source, which compute "Specific Contributing Area". 
        See ref for the description of more output’s options. 
        @Argument: 
            @inD8Pointer: D8-Pointer raster
            @log (Boolean): Apply Log-transformation on the output raster
        @Output: 
            DInfFlowAcculation map. 
        '''
        output = addSubstringToName(inD8Pointer,"_dInfFAcc")
        wbt.d_inf_flow_accumulation(
            inD8Pointer, 
            output, 
            out_type="Specific Contributing Area", 
            threshold=None, 
            log=log, 
            clip=False, 
            pntr=True, 
            callback=default_callback
            )
        return output

    def computeSlope(self,inDTMName):
        outSlope = addSubstringToName(inDTMName,'_Slope')
        wbt.slope(inDTMName,
                outSlope, 
                zfactor=None, 
                units="degrees", 
                callback=default_callback
                )
        return outSlope
    
    def computeAspect(self,inDTMName):
        outAspect = addSubstringToName(inDTMName,'_aspect')
        wbt.aspect(inDTMName, 
                outAspect, 
                zfactor=None, 
                callback=default_callback
                )
        return outAspect

    def computeRasterHistogram(self,inRaster):
        '''
        For details see Whitebox Tools references at:
        https://www.whiteboxgeo.com/manual/wbt_book/available_tools/mathand_stats_tools.html?#RasterHistogram
        @return: An *.html file with the computed histogram. The file is autoloaded. 
        '''
        output = addSubstringToName(inRaster,'_histogram')
        output = replaceExtention(output, '.html')
        wbt.raster_histogram(
            inRaster, 
            output, 
            callback=default_callback
            )   
        return output

    def rasterCalculator(self, output, statement:str)-> os.path:
        '''
        For details see Whitebox Tools references at:
        https://www.whiteboxgeo.com/manual/wbt_book/available_tools/mathand_stats_tools.html#RasterCalculator
        
        @statement : string of desired opperation. Raster must be cuoted inside the statement str. ex "'raster1.tif' - 'rater2.tif'"
        '''
        wbt.raster_calculator(
            output, 
            statement, 
            callback=default_callback
            )
        return output

    def get_WorkingDir(self):
        return str(self.workingDir)
    
    def set_WorkingDir(self,NewWDir):
        wbt.set_working_dir(NewWDir)

class generalRasterTools():
    def __init__(self, workingDir):
        if os.path.isdir(workingDir): # Creates output dir, if it does not already exist. 
            self.workingDir = workingDir
            wbt.set_working_dir(workingDir)
        else:
            self.workingDir = input('Enter working directory')
            ensureDirectory(self.workingDir)
            wbt.set_working_dir(self.workingDir)
        # print('Current working directory : ', self.workingDir)
    
    def computeMosaic(self, outpouFileName:str):
        '''
        Compute wbt.mosaic across all .tif files into the workingDir.  
        @return: Return True if mosaic succeed, False otherwise. Result is saved to wbt.work_dir. 
        Argument
        @verifiedOutpouFileName: The output file name. IMPORTANT: include the "*.tif" extention.
        '''
        verifiedOutpouFileName = checkTifExtention(outpouFileName)
        outFilePathAndName = os.path.join(wbt.work_dir,verifiedOutpouFileName)
        if wbt.mosaic(
            output=outFilePathAndName, 
            method = "nn"  # Calls mosaic tool with nearest neighbour as the resampling method ("nn")
            ) != 0:
            print('ERROR running mosaic')  # Non-zero returns indicate an error.
            return False 
        return True

    def rasterResampler(sefl,inputRaster, outputRaster, outputCellSize:int,resampleMethod = 'bilinear'):
        '''
        wbt.Resampler ref: https://www.whiteboxgeo.com/manual/wbt_book/available_tools/image_processing_tools.html#Resample
        NOTE: It performes Mosaic if several inputs are provided, in addition to resampling. See refference for details. 
        @arguments: inputRaster, resampledRaster, outputCellSize:int, resampleMethod:str
        Resampling method; options include 'nn' (nearest neighbour), 'bilinear', and 'cc' (cubic convolution)
        '''
        verifiedOutpouFileName = checkTifExtention(outputRaster)
        outputFilePathAndName = os.path.join(wbt.work_dir,verifiedOutpouFileName)
        if isinstance(inputRaster, list):
            inputs = sefl.prepareInputForResampler(inputRaster)
        else: 
            inputs = inputRaster        
        wbt.resample(
            inputs, 
            outputFilePathAndName, 
            cell_size=outputCellSize, 
            base=None, 
            method= resampleMethod, 
            callback=default_callback
            )
        return outputFilePathAndName
     
    def mosaikAndResamplingFromCSV(self,csvName, outputResolution:int, csvColumn:str, clearTransitDir = True):
        '''
        Just to make things easier, this function download from *csv with list of dtm_url,
         do mosaik and resampling at once. 
        NOTE: If only one DTM is provided, mosaik is not applyed. 
        Steps:
        1- create TransitFolder
        2- For *.csv in the nameList:
             - create destination Folder with csv name. 
             - import DTM into TransitFolder
             - mosaik DTM in TransitFoldes if more than one is downloaded.
             - resample mosaik to <outputResolution> argument
             - clear TransitFolder
        '''
        ## Preparing for download
        transitFolderPath = createTransitFolder(self.workingDir)
        sourcePath_dtm_ftp = os.path.join(self.workingDir, csvName) 
        name,ext = splitFilenameAndExtention(csvName)
        print('filename :', name, ' ext: ',ext)
        destinationFolder = makePath(self.workingDir,name)
        ensureDirectory(destinationFolder)
        dtmFtpList = createListFromCSVColumn(sourcePath_dtm_ftp,csvColumn)
        
        ## Download tails to transit folder
        downloadTailsToLocalDir(dtmFtpList,transitFolderPath)
        savedWDir = self.workingDir
        resamplerOutput = makePath(destinationFolder,(name +'_'+str(outputResolution)+'m.tif'))
        resamplerOutput_CRS_OK = makePath(destinationFolder,(name +'_'+str(outputResolution)+'m.tif'))
        setWBTWorkingDir(transitFolderPath)
        
        ## recover all downloaded *.tif files path
        dtmTail = listFreeFilesInDirByExt(transitFolderPath, ext = '.tif')
        crs,_ = self.get_CRSAndTranslation_GTIFF(dtmFtpList[0])
        
        ## Merging tiles and resampling
        self.rasterResampler(dtmTail,resamplerOutput,outputResolution)
        self.set_CRS_GTIF(resamplerOutput, resamplerOutput_CRS_OK, crs)
        setWBTWorkingDir(savedWDir)
        
        ## Celaning transit folder. 
        if clearTransitDir: 
            clearTransitFolderContent(transitFolderPath)

    def rasterToVectorLine(sefl, inputRaster, outputVector):
        wbt.raster_to_vector_lines(
            inputRaster, 
            outputVector, 
            callback=default_callback
            )

    def rasterVisibility_index(sefl, inputDTM, outputVisIdx, resFator = 2.0):
        '''
        Both, input and output are raster. 
        '''
        wbt.visibility_index(
                inputDTM, 
                outputVisIdx, 
                height=2.0, 
                res_factor=resFator, 
                callback=default_callback
                )           

    def gaussianFilter(sefl, input, output, sigma = 0.75):
        '''
        input@: kernelSize = integer or tupel(x,y). If integer, kernel is square, othewise, is a (with=x,hight=y) rectagle. 
        '''
        wbt.gaussian_filter(
        input, 
        output, 
        sigma = sigma, 
        callback=default_callback
        )
    
    def prepareInputForResampler(self,nameList):
        inputStr = ''   
        if len(nameList)>1:
            for i in range(len(nameList)-1):
                inputStr += nameList[i]+';'
            inputStr += nameList[-1]
            return inputStr
        return str(*nameList)

    def get_CRSAndTranslation_GTIFF(self,input_gtif):
        '''
         @input_gtif = "path/to/input.tif"
         NOTE: Accept URL as input. 
        '''
        with rio.open(input_gtif) as src:
        # Extract spatial metadata
            input_crs = src.crs
            input_gtif  = src.transform
            src.close()
            return input_crs, input_gtif  

    def set_CRS_GTIF(self,input_gtif, output_tif, in_crs):
        arr, kwds = self.separate_array_profile(input_gtif)
        kwds.update(crs=in_crs)
        with rio.open(output_tif,'w', **kwds) as output:
            output.write(arr)
        return output_tif

    def set_Tanslation_GTIF(self, input_gtif, output_tif, in_gt):
        arr, kwds = self.separate_array_profile(input_gtif)
        kwds.update(transform=in_gt)
        with rio.open(output_tif,'w', **kwds) as output:
            output.write(arr)
        return output_tif

    def separate_array_profile(self, input_gtif):
        with rio.open(input_gtif) as src: 
            profile = src.profile
            print('This is a profile :', profile)
            arr = src.read()
            src.close() 
        return arr, profile

    def ensureTranslationResolution(self, rioTransf:rio.Affine, desiredResolution: int):
        '''
        NOTE: For now it works for square pixels ONLY!!
        Compare the translation values for X and Y transformation with @desiredResolution. 
        If different, the values are replaced by the desired one. 
        return:
         @rioAfine:rio.profiles with the new resolution
        '''
        if rioTransf[0] != desiredResolution:
            newTrans = rio.Affine(desiredResolution,
                                rioTransf[1],
                                rioTransf[2],
                                rioTransf[3],
                                -1*desiredResolution,
                                rioTransf[5])
        return newTrans

    def get_rasterResolution(self, inRaster):
        with rio.open(inRaster) as src:
            profile = src.profile
            transformation = profile['transform']
            res = int(transformation[0])
        return res
    
    def get_WorkingDir(self):
        return str(self.workingDir)

#WbW. TO SLOW: To be checked.  
def clip_raster_to_polygon(inputRaster, maskVector, outPath, maintainDim:bool = False )->os.path:
    wbt.clip_raster_to_polygon(
        inputRaster, 
        maskVector, 
        outPath, 
        maintainDim, 
        callback=default_callback
        )
    return outPath

# Helpers
def setWBTWorkingDir(workingDir):
    wbt.set_working_dir(workingDir)

def checkTifExtention(fileName):
    if ".tif" not in fileName:
        newFileName = input("enter a valid file name with the '.tif' extention")
        return newFileName
    else:
        return fileName

def downloadTailsToLocalDir(tail_URL_NamesList, localPath):
    '''
    Import the tails in the url <tail_URL_NamesList>, 
        to the local ydirectory defined in <localPath>.
    '''
    confirmedLocalPath = ensureDirectory(localPath)
    for url in tail_URL_NamesList:
        download_url(url, confirmedLocalPath)
    print(f"Tails downloaded to: {confirmedLocalPath}")

