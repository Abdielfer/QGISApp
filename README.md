# GISAutomation
Data flow Chart:
1- Extraction of a subset from a data collection/asset with dc_extraction tools. Subset defined by bounding box or *.gpkg file.  
- Input: 
- Output: collection/asset subset (*.tif, *.shp, *.gpkg  or other specific format from the collection/asset)

2- Clip the extracted subset with a mask if needed.(Ex. catchment extraction from a DEM using a *.shp file with a basin polygon as mask).

3- Performe the desired operation on the extracted subset.(Ex. From a DEM computes slope, flow accumulation, river network extraction, HAND coefficient, etc.) This operation can be performed with WhiteBoxTools toolbox for instance. 

Configuration:
Configuration is composed of two files. First: The main configuration that must match the Operative system requirements (configPC,configLinux, configMac). In this file you'll set the path in the right format for each OS. Second: a file containing the necessary information for subset extraction (See:https://gccode.ssc-spc.gc.ca/datacube/dc_extract/-/blob/main/docs_md/dc_extract_tools.md ), the list of required operations after the extraction and several useful setting variables leading the additional operations behaviour.  You can find a canvas of configuration files in config/example/configCanvas.yml.