# GISAutomation

### Settings:
 - Clone repository: 
   # Adding dc_extract to the repo. Two options:
        1 - Clone the cd_extract repo OUTSIDE your GISAutomation file. Make a copy of dc_extract into yor GISAutomation file. I necesary, "remouve remote" to the dc_extract into your repo, to avoid conflicts with Git operations. 
        2- OR: Add dc_extract as submodule into your GISAutomation local repo. 
 - Create Conda environment:
    1.Create a conda env from the datacue-extract.yml located into the dc_extract folder. Choose your OS. 
    2.Add packages: qgis, pcraster and whiteboxtool to your env: 
            conda install -c conda-forge qgis
            conda install -c conda-forge pcraster
            conda install -c conda-forge whitebox_tools


### Data flow Chart:
1- Extraction of a subset from a data collection/asset with dc_extraction tools. Subset defined by bounding box or *.gpkg file.   (NOTE: From experinece, works better with bbox.) 
- Input: bbox=(xMin,yMin, xMax,yMax), 
- Output: collection/asset subset (*.tif, *.shp, *.gpkg  or other specific format from the collection/asset)

2- Clip the extracted subset with a mask if needed.(Ex. catchment extraction from a DEM using a *.shp file with a basin polygon as mask. NOTE: util.clipRasterGdal() is tested.) 

3- Performe the desired operation on the extracted subset.(Ex. From a DEM computes slope, flow accumulation, river network extraction, HAND coefficient, etc.  -->>> 
        ex. util.computeHAND(DEMPath,HANDPath)) 

### Configuration:
    Main configuration (config\): 
        that must match the Operative system requirements (ex. mainConfigPC). In this file you'll set the paths in the right format for each OS. 
    
    Extraction parameters(config\parameters\dc_extractParamsCanvas.yaml):
        A file containing the necessary information for dc_extrac fuctions(See:https://gccode.ssc-spc.gc.ca/datacube/dc_extract/-/blob/main/docs_md/dc_extract_tools.md ).
   
    Transformation (config\transformation\yourFile.ymal):
        Add as much configurations as desired to call "ydra.utils import instantiate" in the main.py. 
        ex: config\transformation\wbwTools.yaml
