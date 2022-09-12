
library(readxl)
library(splitr)
library(lubridate)
library(magrittr)
library(tibble)
library(dplyr)
library(tidyverse)
library(devtools)
library(dplR) 　
library(doParallel)
library(iterators)
library(parallel)

library(ncdf4) 
library(raster) 
library(rgdal) 
library(ncdump)
library(sp)
library(stats)
library(leaflet) 
library(RColorBrewer)
library(ggplot2)
library(sf)
library(ggspatial)
library(maps)
library(ggmap)
library(spData)
library(spDataLarge)
library(rgeos)

## commit1

devtools::install_github("rich-iannone/splitr", force=TRUE)
setwd("C:/hysplit/working/HYSPLIT_R")





################### TRAJECTORY DF ######################

daily_hours = c(0:23)
#daily_hours = seq(0,23,6)
run_lon = 126.5720
run_lat = 37.62191

trajectory <-
  hysplit_trajectory(
    lon = run_lon,
    lat = run_lat,
    height = 1000,
    duration = 48,                  # number of points that are drawn on the map
    #days = "2022-05-26",
    days = seq(
      lubridate::ymd("2022-05-20"),
      lubridate::ymd("2022-05-27"),
      by = "1 day"
    ),
    daily_hours = c(0:23),
    vert_motion = 0,
    model_height = 20000,
    direction = "backward",
    met_type = "reanalysis",
    extended_met = TRUE)

trajectory_plot(trajectory)


trajectory_tbl <- as_tibble(trajectory)
trajectory_df <- as.data.frame(trajectory)  # dataframe-ize
trajectory_lonlat <- trajectory_df %>%
  dplyr::select(lon, lat)
write.csv(trajectory_lonlat, "C:/hysplit/working/HYSPLIT_R/outputs/trajectory_lonlat.csv")
df_all <- read.csv("C:/hysplit/working/HYSPLIT_R/outputs/trajectory_lonlat.csv")

# df_all <- df_all %>%
#   filter(!lon == run_lon & !lat == run_lat)





################### RASTERIZATION ######################

#OutDFile <- setwd("C:/hysplit/working/HYSPLIT_R")


# Open as netCDF data #

# nc_data <- ncdf4::nc_open("C:/hysplit/working/HYSPLIT_R/tot_prec_mon/tot_precp_mon_0.25_glb.nc")
# nc_attributes <- ncatt_get(nc_data, varid = 0)
# 
# lon  <- ncdf4::ncvar_get(nc_data, "lon")
# lat  <- ncdf4::ncvar_get(nc_data, "lat")
# time <- ncdf4::ncvar_get(nc_data, "time")
# time <- as.Date(time, origin = "1891-01-01", tz = "UTC")



# Open as netCDF data  in a different way that's working lol #

#nc_data <- Sys.glob("C:/hysplit/working/HYSPLIT_R/tot_prec_mon/tot_precp_mon_0.25_glb.nc")
nc_data <- Sys.glob("C:/hysplit/working/HYSPLIT_R/0.1degree_nc_kr/20210101_FRP_0.1deg.nc")
nc_info <- ncdump::NetCDF(nc_data)
#names(nc_info)
rst_data <- raster(nc_data)
#proj4string(rst_data) = CRS("+init=EPSG:4326")
plot(rst_data)
# rst_poly <- rasterToPolygons(rst_data)
# rst_shp <- raster::shapefile(rst_poly, "rst_poly.shp")    # write shp file
# writeOGR(obj = rst_poly,                                      
#          dsn = "C:/hysplit/working/HYSPLIT_R/outputs", 
#          layer = 'rst_shp_1',
#          driver = "ESRI Shapefile")

ext <- extent(115, 135, 31, 44)   # fixed 
rst_data <- crop(rst_data, ext)
plot(rst_data)

rst_att <- as.data.frame(rst_data, xy = TRUE)
colnames(rst_att) <- c("Lon", "Lat", "var")
rst_att


rst_att_tbl <- as_tibble(rst_att)

# rst_lon <- rst_att_tbl$Lon              # 4160
rst_lon_unq <- unique(rst_att_tbl$Lon)  # 80
# rst_lat <- rst_att_tbl$Lat              # 4160
rst_lat_unq <- unique(rst_att_tbl$Lat)  # 52

trj_lon <- df_all$lon   # 4200
trj_lat <- df_all$lat   # 4200




nhits <- c()

for(a in 1:length(rst_lat_unq)) {  # 52 numbers
  # a = 1
  if (a < length(rst_lat_unq)) {
    df_lat_range <- df_all[which(df_all$lat <  rst_lat_unq[a] &
                                   df_all$lat >= rst_lat_unq[a+1]),]
  } else {
    df_lat_range <- df_all[which(df_all$lat >=  rst_lat_unq[a]),]
  }
  
  for(b in 1:length(rst_lon_unq)) {    # 80 numbers repeat      
    # b = 1
    if (b < length(rst_lon_unq)) {
      df_lonlat_range <- df_lat_range[which(df_lat_range$lon >= rst_lon_unq[b] &
                                              df_lat_range$lon <  rst_lon_unq[b+1]),]
    } else {
      df_lonlat_range <- df_lat_range[which(df_lat_range$lon >= rst_lon_unq[b]),]
    }
    
    nhits <- c(nhits, nrow(df_lonlat_range))
    
  }
}

rst_att_tbl <- cbind(rst_att_tbl, nhits)
summary(rst_att_tbl)

df_nhits <-  select(rst_att_tbl, Lon, Lat, nhits)
#write.csv(rst_att_tbl, "C:/hysplit/working/HYSPLIT_R/outputs/ras_att_tbl.csv")
df_nhits <- df_nhits %>%
  mutate(q95 = quantile(nhits, 0.95, na.rm = TRUE)) %>%
  filter(nhits < q95)

rst_nhits <- rasterFromXYZ(df_nhits[, c("Lon", "Lat", "nhits")], crs = crs('+proj=longlat +datum=WGS84'))


# pal <- colorBin(palette = "Reds",
#                 domain = rst_att_tbl_plt$nhits,
#                 #bins = 5,
#                 na.color = "transparent")
# 
# pal <- colorFactor('Set1', rst_att_tbl_plt$nhits)

# pal <- colorNumeric(
#   palette = "Reds",
#   domain = rst_att_tbl_plt$nhits)

pal <- brewer.pal(10, "Reds")

traj_plot <-
  leaflet::leaflet() %>%
  leaflet::addProviderTiles(
    provider = "OpenStreetMap",
    group = "OpenStreetMap"
  ) %>%
  leaflet::addProviderTiles(
    provider = "CartoDB.DarkMatter",
    group = "CartoDB Dark Matter"
  ) %>%
  leaflet::addProviderTiles(
    provider = "CartoDB.Positron",
    group = "CartoDB Positron"
  ) %>%
  leaflet::addProviderTiles(
    provider = "Esri.WorldTerrain",
    group = "ESRI World Terrain"
  ) %>%
  leaflet::addProviderTiles(
    provider = "Stamen.Toner",
    group = "Stamen Toner"
  ) %>%
  # leaflet::fitBounds(
  #   lng1 = min(rst_att_tbl_plt[["Lon"]]),
  #   lat1 = min(rst_att_tbl_plt[["Lat"]]),
  #   lng2 = max(rst_att_tbl_plt[["Lon"]]),
  #   lat2 = max(rst_att_tbl_plt[["Lat"]])
  # ) %>%
  leaflet::setView(
    lng = 126.501,
    lat = 36.249,    # in the middle of the installed traps 
    zoom = 7
  ) %>%
  leaflet::addLayersControl(
    baseGroups = c(
      "CartoDB Positron", "CartoDB Dark Matter",
      "Stamen Toner", "ESRI World Terrain"
    ),
    overlayGroups = c("trajectory_points", "trajectory_paths"),
    position = "topright"
  ) %>%
  #addTiles() %>%
  addRasterImage(rst_nhits, colors = pal, opacity = 0.8)# %>%
#addLegend("bottomright", pal = pal, values = nhits,
#            title = "",
#            opacity = 0.7)

traj_plot

