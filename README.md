# GISAutomation

### Settings:
 - Clone repository: 
   # Adding dc_extract to the repo. Two options:
        1 - Clone the cd_extract repo OUTSIDE your GISAutomation folder. Make a copy of "dc_extract" into your GISAutomation folder. If necesary, "remouve remote" from "dc_extract" into your repo, to avoid conflicts with Git operations. 
        2- OR: Add dc_extract as submodule into your GISAutomation local repo. 
 - Create Conda environment:
    1.Create a conda env from the datacue-extract.yml requirements, located into the dc_extract folder. Choose your OS. 
    2.Add these packages to the conda env: qgis, pcraster and whiteboxtool: 
            conda install -c conda-forge qgis
            conda install -c conda-forge pcraster
            conda install -c conda-forge whitebox_tools


### Data flow Chart:
1- Extraction of a subset from a data datacube with dc_extraction tools. Subset defined by bounding box or *.gpkg file.   (NOTE: From experience, works better with bbox.) 
- Input: collection/asset; bbox=(xMin,yMin, xMax,yMax). 
- Output: collection/asset subset (*.tif, *.shp, *.gpkg  or other specific format from the collection/asset)

2- Clip the extracted subset with a mask if needed.(Ex. catchment extraction from a DEM using a *.shp file with a basin polygon as mask. NOTE: util.clipRasterGdal() is tested and works OK) 

3- Performe the desired operation on the extracted subset.(Ex. From a DEM computes slope, flow accumulation, river network extraction and HAND coefficient.  -->>> util.computeHAND(DEMPath,HANDPath)) 

### Configuration files (hydra):
    Main configuration (config\): 
        Must match the Operative system requirements (ex. mainConfigPC). In this file you'll set the paths in the right format for each OS. 
    
    Extraction parameters(config\parameters\dc_extractParamsCanvas.yaml):
        A file containing the necessary information for dc_extrac fuctions(See:https://gccode.ssc-spc.gc.ca/datacube/dc_extract/-/blob/main/docs_md/dc_extract_tools.md ).
   
    Transformation (config\transformation\yourFile.ymal):
        Add as much configurations as desired to call "ydra.utils import instantiate" in the main.py. 
        ex: config\transformation\wbwTools.yaml
