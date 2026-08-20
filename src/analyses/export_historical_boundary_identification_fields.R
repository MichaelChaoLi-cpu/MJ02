#!/usr/bin/env Rscript

# Export an outcome-blind subset of the public historical-boundary replication files.
# Contemporary poverty, education, night-light, and other outcome fields are deliberately
# excluded. The resulting CSV files form the enforceable input boundary for diagnostics.

suppressPackageStartupMessages(library(sf))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: export_historical_boundary_identification_fields.R SOURCE_DIR OUTPUT_DIR")
}

source_dir <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

main <- readRDS(file.path(source_dir, "main_0310.rds"))
allowed_main <- c(
  "vill_code", "comm", "treat", "dist_border", "dist.segment",
  "temp_mean", "temp_var", "prec_mean", "prec_var",
  "class_12", "class_40", "fertile_pct", "rug", "elev", "river_d",
  "dist_cap", "pop_75", "build_75_sum", "built_1975", "built_1975_sum",
  "road_d", "road_cell"
)
missing_main <- setdiff(allowed_main, names(main))
if (length(missing_main) > 0) {
  stop(paste("Missing expected main fields:", paste(missing_main, collapse = ", ")))
}
main_coordinates <- st_coordinates(main)
if (nrow(main_coordinates) != nrow(main)) {
  stop("Expected point geometry in main_0310.rds")
}
main_export <- st_drop_geometry(main[, allowed_main])
main_export$longitude <- main_coordinates[, "X"]
main_export$latitude <- main_coordinates[, "Y"]
write.csv(
  main_export,
  file.path(output_dir, "predetermined_and_alignment_fields.csv"),
  row.names = FALSE,
  na = ""
)

road <- readRDS(file.path(source_dir, "road_df2.rds"))
allowed_road <- c("vill_code", "south", "dist.NR3", "X", "Y")
missing_road <- setdiff(allowed_road, names(road))
if (length(missing_road) > 0) {
  stop(paste("Missing expected road-placebo fields:", paste(missing_road, collapse = ", ")))
}
road_export <- st_drop_geometry(road[, allowed_road])
write.csv(
  road_export,
  file.path(output_dir, "nr3_placebo_design_fields.csv"),
  row.names = FALSE,
  na = ""
)

message("Exported outcome-blind historical-boundary identification fields to ", output_dir)
