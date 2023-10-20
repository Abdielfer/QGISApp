# Convert the following code from R to Python


# library(raster)
# library(sf)
# library(sp)
# library(rgdal)
# library(whitebox)
# library(sf)


# #user input: 
# ui_command <- commandArgs(trailingOnly = TRUE)
# inshp <- ui_command[1]
# indem <- ui_command[2]
# sttype <-  ui_command[3]
# stval <-  ui_command[4]

# #sttype <- 'Strahler'
# #stval <- 5

# ###########################################################################
# #folder: ## FOR HPC:
# whatsubdir <- 'HAND'
# maindir <- '/space/partner/nrcan/geobase/work/data/hem'
# analysisdir <- paste(maindir, '/Python_Tst/elevation/', whatsubdir, '/', sep='')
# dem_touse <- paste('/gpfs/fs5/nrcan/nrcan_geobase/work/data/hem/mcdtm/' , indem, sep='')


# print('set working directory:')
# abc <- unlist(strsplit(inshp, "\\."))
# abc1 <- abc[1]
# goworkdir <- paste(analysisdir, abc1, sep='')
# de <- dir.exists(goworkdir)

# if (de == TRUE){
#   print(goworkdir)
#   setwd(goworkdir)
  
# }else{
#   dir.create(goworkdir)
#   setwd(goworkdir)
#   print(goworkdir)
# }

# mastershape <- paste(analysisdir, 'Shapefile/', inshp , sep='')
# print('location of shapefile is: ')
# print(mastershape)

# print('read shapefile')
# AllWS <- st_read(mastershape)

# #first feature get wsid number:
# startcount <- AllWS[1,]
# sc <- startcount$WSID

# outppts <- paste('pourpts_',  sttype, stval,'.shp', sep='')
# outws <- paste('watershedbound_', sttype, stval, '.shp', sep='')
# allstreamshp <- 'raster_streams.shp'
# streamshp <- paste("stream_", stval, ".shp", sep="")
# wstif <- paste('watershed_', sttype, stval, '.tif', sep = '')
# streamrastertif <- paste('ST_', sttype, stval, '.tif', sep='')
# outjenson_pts <- paste('pp_',  sttype, stval, '.shp', sep='')
# outflowpathTif <- paste('flowlength_', sttype, stval, '.tif', sep='')
# sr <- paste("streams", sttype,"_", stval, ".tif", sep="")
# sob <- paste('SOrdBsn_', sttype, stval, '.tif', sep = '')


# make_streampoly <- function(dem, newdem) {
  
#   wbt_raise_walls(input = 'shapebuffer.shp',
#             dem = dem,
#             output = 'raisewalls.tif',
#             height = 0.1)
  
#   #wbt_fill_missing_data(
#   # fill singles
#   wbt_fill_single_cell_pits(
#     dem = 'raisewalls.tif', 
#     output = 'fscp.tif'
#   )
#    #fill pits:
#   wbt_breach_depressions_least_cost(dem = 'fscp.tif', 
#              output = 'bdlc.tif',
#              dist = 300,
#              fill= FALSE)
                       
#   wbt_fill_depressions_wang_and_liu(
#     dem = "bdlc.tif",
#     output = newdem)
  
  
#   wbt_d8_flow_accumulation(input = newdem,
#                            output = "D8FA.tif",
#                            out_type = 'specific contributing area',
#                            clip=TRUE, 
#                            log=TRUE)
#   #quantiles from flow accum: 
#   q <- quantile(raster("D8FA.tif"), probs = c(0.1, 0.954))
#   q
  
#   wbt_d8_pointer(dem = newdem,
#                  output = "D8pointer.tif")
  
#   wbt_extract_streams(flow_accum = "D8FA.tif",
#                       output = "raster_streams.tif",
#                       threshold = q[2])
  
#   streams_r <- raster('raster_streams.tif')
#   plot(streams_r , col='blue' )
  
#   wbt_raster_streams_to_vector(streams= "raster_streams.tif",
#                                d8_pntr = "D8pointer.tif",
#                                output= allstreamshp
#   )
  
# }


# stream_select <- function(streamrastertif, sttype, stval){
  
#   if (sttype == "Hack"){
#     i_stval <- as.integer(stval)
#     up <- i_stval + 1
#     reclass_df <- c(0, i_stval, 1, up , 100, NA)
#     reclass_m <- matrix(reclass_df,
#                         ncol = 3,
#                         byrow = TRUE)
    
    
#     wbt_hack_stream_order(
#       d8_pntr="D8pointer.tif",
#       streams ='raster_streams.tif',
#       output = streamrastertif
#     )
    
#   }
  
#   if (sttype == 'Strahler'){
#     up <- stval+1
#     reclass_df <- c(0, stval, NA, up, 100, 1)
#     reclass_m <- matrix(reclass_df,
#                         ncol = 3,
#                         byrow = TRUE)
#     wbt_strahler_stream_order(
#       d8_pntr="D8pointer.tif",
#       streams = 'raster_streams.tif',
#       output = streamrastertif)
    
#   }
  
  
#   r <- raster(streamrastertif)
#   print("open stream order raster to filter")
#   print(streamrastertif)
  
  
#   y <- reclassify(r, reclass_m, right=FALSE) 
  
#   #plot(y)
#   print("reclassified")
#   writeRaster(y,sr, overwrite=TRUE)
  
#   wbt_raster_streams_to_vector(streams= sr,
#                                d8_pntr = "D8pointer.tif",
#                                output= streamshp
#   )
#   #copy_rename_prj(streamshp)
#   print(streamshp)
  
#   print("finished: stream_select function")
# }


# getdangles <- function(streamshp, outppts){
#   #this reads the stream file and gets point features from the dangles, used to 
#   # then rebuild a stream network based on these 'pour points'
#   sfls <- st_read(streamshp)
  
#   plot(sfls)
#   # break it to points
#   sfpts <- st_geometry(sfls) %>% 
#     lapply(., function(x) {
#       st_sfc(x) %>% 
#         st_cast(., 'POINT')})
  
 
#   sfls_ends <- sapply(sfpts, function(p) {
#     p[c(length(p))]
#     p[1]
    
#   }) %>% 
#     st_sfc() %>%
#     st_sf('geom' = .)
  
#   # check with
#   #plot(sfls$geometry)
#   #plot(sfls_ends, add = TRUE, pch = 19, col = 'red')
#   st_write(sfls_ends, outppts, delete_dsn = TRUE, append = TRUE)
  
#   print("finished:getdangles function ")
  
#   sstpts <- st_read(outppts)
#   #copy_rename_prj(outppts)
  
#   plot(sstpts)
  
# }

# ################### Watersheds function ###############
# make_watersheds <- function(str_raster,pp_shp,  outws) {

#   wbt_jenson_snap_pour_points(pour_pts = pp_shp,
#                               streams = str_raster,
#                               output = outjenson_pts,
#                               snap_dist = 120) #careful with this! Know the units of your data
  
  
#   streams <- raster(str_raster)
  
#   wbt_watershed(d8_pntr = "D8pointer.tif",
#                 pour_pts = outjenson_pts,
#                 output = wstif)
  
#   wbt_strahler_order_basins(
#     d8_pntr = "D8pointer.tif",
#     streams = str_raster,
#     output = sob
#   )
  
#   wbt_raster_to_vector_polygons(wstif, outws)
#   print("finished make_watersheds function")
# }

# #HAND:
# makehand <- function(outdem, outhand){
#   wbt_elevation_above_stream(
#     dem = outdem, 
#     streams = sr, 
#     output = outhand
#   )
# }

# #crop raster:

# croppoly <- function(demi, shp_path, outdem){
#   print(demi)
#   poly_extent <- st_read(shp_path)
#   print('read in poly')
#   poly_pro <- st_transform(poly_extent, crs='EPSG:3979')
#   pp_buffer <- st_buffer(poly_pro, dist=1200)
#   #plot(pp_buffer) 
#   dtm <- raster(demi)
#   dtm_crop <-crop(dtm, pp_buffer)
#   plot(dtm_crop, main= 'Cropped DEM')
#   dtm_mask <- mask(dtm_crop, pp_buffer)
#   plot(dtm_mask, main= 'Mask DEM')
#   plot(poly_pro, add=TRUE, border="red", col = scales::alpha("lightblue", 0.9))
#   print('cropped')
#   writeRaster(dtm_mask, outdem, overwrite=TRUE)
#   print('raster written')
  

#   st_write(pp_buffer, 'shapebuffer.shp', driver = "ESRI Shapefile")
  
  
# }


# #for loop to iterate through rows of shapefile: 
# for (i in 1:(nrow(AllWS))){
  
#   print(i)
#   print(sc)
#   setwd(goworkdir)
#   print('working directory is: ')
#   getwd()
  
#   print('create new dir:')
#   newdir <- paste('WS', sc, sep='')
#   #dir.create(newdir)
  
#   if (!dir.exists(newdir)){
#     dir.create(newdir)
#   }else{
#     print("dir exists")
#   }
  
  
#   print('change to new dir:')
#   nwd2 <- paste(goworkdir,'/',  newdir, sep='')
#   print(nwd2)
#   setwd(nwd2)
  
 
#   #select one WS at a time: 
#   crop_dem <- paste('WS_DEM_', sc, '.tif', sep='')
#   print('select first feature:')
#   select_p1 <- subset(AllWS, WSID == sc)
#   #plot(select_p1)
#   write_sf(select_p1, 'shape.shp')
  
#   print('now crop dem to polygon')
#   croppoly(dem_touse, 'shape.shp', crop_dem)
  
#   #now fill pits, hydro condition: 
#   pfd <- 'pitfilldepressions.tif'
  
#   fe <- exists('raster_streams.shp')
#   if (fe == FALSE){
#   make_streampoly(crop_dem, pfd)
#   }else{
#     print('streams and pit filing complete go to select')
#   }
#   #select subset of streams:
#   stream_select(streamrastertif, sttype, stval)
#   getdangles(streamshp, outppts)
  
#   make_watersheds(sr, outppts,  outws)
  
#   finalouthand <-paste('HAND_', sc , '_', sttype, stval, '.tif', sep = '')
#   makehand(pfd, finalouthand)
#   sc = sc + 1
#   setwd(goworkdir)
# }

from whitebox.whitebox_tools import WhiteboxTools, default_callback
import rasterio
import matplotlib.pyplot as plt


def strahler_stream_order(dem, streamsRaster, outputHAND):
    
    wbt = WhiteboxTools()
    wbt.set_working_dir('data')
    wbt.set_verbose_mode(True)
    wbt.set_compress_rasters(True) 
    
    # Fill the DEM
    filledDEM = r'C:\Users\abfernan\CrossCanFloodMapping\GISAutomation\data\filled_dem.tif'
    wbt.fill_single_cell_pits(dem,filledDEM)
    
    d8_pntr = r'C:\Users\abfernan\CrossCanFloodMapping\GISAutomation\data\D8pointer.tif'
    wbt.d8_pointer(filledDEM,d8_pntr,esri_pntr=False)
    
    streamVect = r'C:\Users\abfernan\CrossCanFloodMapping\GISAutomation\data\StreamVector.shp'
    wbt.raster_to_vector_lines(streamsRaster,d8_pntr,streamVect)
    
    fAccTif = r'C:\Users\abfernan\CrossCanFloodMapping\GISAutomation\data\flow_accum.tif'
    wbt.d8_flow_accumulation(filledDEM, fAccTif,out_type="cells",log=False, 
            clip=False, 
            pntr=False, 
            esri_pntr=False) 
    
    # Calculate Strahler stream order
    wbt.strahler_stream_order(fAccTif, streamVect, outputHAND)



import whitebox

# Create a new instance of the WhiteboxTools class
wbt = whitebox.WhiteboxTools()

# Call the stream_network_analysis tool
wbt.stream_network_analysis(
    '--dem="path/to/dem.tif"',
    '--out_file="path/to/stream_network.tif"'
)

# Call the raster_streams_to_vector tool
wbt.raster_streams_to_vector(
    '--i="path/to/stream_network.tif"',
    '--o="path/to/stream_network.shp"'
)

# Call the extract_streams tool
wbt.extract_streams(
    '--dem="path/to/dem.tif"',
    '--streams="path/to/stream_network.tif"',
    '--out_file="path/to/streams.tif"'
)

def main():
    
    wbt = WhiteboxTools()
    wbt.set_working_dir('data')
    wbt.set_verbose_mode(True)
    wbt.set_compress_rasters(True) 

    # Set input file paths
    dem = r'C:\Users\abfernan\CrossCanFloodMapping\GISAutomation\data\BC_Salmo_EffectiveBasin_Clip.tif'
    streams = r'C:\Users\abfernan\CrossCanFloodMapping\GISAutomation\data\BC_Salmo_EffectiveBasin_Clip_mainRiverPath_.tif'
    output = 'data\HAND_FromWbT.tif'

    # Calculate Strahler stream order
    strahler_stream_order(dem, streams, output)

    # Load resulting raster
    with rasterio.open(output) as src:
        strahler_raster = src.read(1)

    # Plot resulting raster
    plt.imshow(strahler_raster)
    plt.colorbar()
    plt.title('Strahler Stream Order')
    plt.show()
   
if __name__ == "__main__":
    main()  
