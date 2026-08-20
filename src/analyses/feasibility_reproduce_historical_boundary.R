#!/usr/bin/env Rscript

# Reproduce historical treatment assignment and border distance without reading outcomes.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop(paste(
    "Usage: feasibility_reproduce_historical_boundary.R",
    "<main_rds> <zone_shp> <province_rds> <output_audit_csv> <output_segments_csv>"
  ))
}

suppressPackageStartupMessages(library(sf))

main_path <- args[[1]]
zone_path <- args[[2]]
province_path <- args[[3]]
output_path <- args[[4]]
segment_output_path <- args[[5]]

if (grepl("\\.zip$", zone_path, ignore.case = TRUE)) {
  zone_extract_dir <- tempfile("historical-zones-")
  dir.create(zone_extract_dir, recursive = TRUE)
  unzip(zone_path, exdir = zone_extract_dir)
  zone_candidates <- list.files(zone_extract_dir, pattern = "\\.shp$", recursive = TRUE, full.names = TRUE)
  if (length(zone_candidates) != 1) {
    stop(paste("Expected one shapefile in zone archive, found", length(zone_candidates)))
  }
  zone_path <- zone_candidates[[1]]
}

main <- readRDS(main_path)
zones <- st_read(zone_path, quiet = TRUE)
zones <- st_transform(zones, st_crs(main))
southwest <- zones[zones$ZONE_NAME == "Southwest", ]
west <- zones[zones$ZONE_NAME == "West", ]

shared_boundary <- st_intersection(st_geometry(southwest), st_geometry(west))
predicted_southwest <- lengths(st_within(main, southwest)) > 0
predicted_west <- lengths(st_within(main, west)) > 0
predicted_treatment <- ifelse(predicted_southwest, 1, ifelse(predicted_west, 0, NA_real_))

unsigned_distance_km <- as.numeric(st_distance(main, shared_boundary)) / 1000
predicted_signed_distance_km <- ifelse(predicted_treatment == 1, unsigned_distance_km, -unsigned_distance_km)

comparable_treatment <- !is.na(predicted_treatment) & !is.na(main$treat)
comparable_distance <- comparable_treatment & !is.na(main$dist_border)
distance_difference <- predicted_signed_distance_km[comparable_distance] - main$dist_border[comparable_distance]

provinces <- readRDS(province_path)
kampong_speu <- provinces[provinces$NAME_1 == "Kâmpóng Spœ", ]
kampong_speu <- st_transform(kampong_speu, st_crs(main))
boundary_vertices <- st_cast(shared_boundary, "POINT")
boundary_vertices <- boundary_vertices[lengths(st_within(boundary_vertices, kampong_speu)) > 0]
vertex_coordinates <- unique(as.data.frame(st_coordinates(boundary_vertices)))
segment_points <- st_as_sf(vertex_coordinates, coords = c("X", "Y"), crs = st_crs(main))
predicted_segment <- st_nearest_feature(main, segment_points)
segment_comparable <- !is.na(main$dist.segment)

audit <- data.frame(
  metric = c(
    "public_village_rows",
    "rows_with_reconstructed_zone_assignment",
    "treatment_assignment_agreement_share",
    "signed_distance_correlation",
    "signed_distance_mean_absolute_difference_km",
    "signed_distance_maximum_absolute_difference_km",
    "author_boundary_segment_levels",
    "reconstructed_boundary_segment_points",
    "boundary_segment_assignment_agreement_share"
  ),
  value = c(
    nrow(main),
    sum(comparable_treatment),
    mean(predicted_treatment[comparable_treatment] == main$treat[comparable_treatment]),
    cor(predicted_signed_distance_km[comparable_distance], main$dist_border[comparable_distance]),
    mean(abs(distance_difference)),
    max(abs(distance_difference)),
    nlevels(main$dist.segment),
    nrow(segment_points),
    mean(predicted_segment[segment_comparable] == as.numeric(main$dist.segment[segment_comparable]))
  ),
  stringsAsFactors = FALSE
)

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
write.csv(audit, output_path, row.names = FALSE)
segment_lonlat <- st_transform(segment_points, 4326)
segment_xy <- st_coordinates(segment_points)
segment_ll <- st_coordinates(segment_lonlat)
segment_table <- data.frame(
  `Historical Boundary Segment` = seq_len(nrow(segment_points)),
  `Easting EPSG 32648` = segment_xy[, "X"],
  `Northing EPSG 32648` = segment_xy[, "Y"],
  Longitude = segment_ll[, "X"],
  Latitude = segment_ll[, "Y"],
  check.names = FALSE
)
dir.create(dirname(segment_output_path), recursive = TRUE, showWarnings = FALSE)
write.csv(segment_table, segment_output_path, row.names = FALSE)
print(audit, row.names = FALSE)
