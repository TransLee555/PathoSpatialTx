#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(optparse)
  library(fs)
  library(readr)
  library(dplyr)
  library(tidyr)
  library(tibble)
})

`%||%` <- function(x, y) if (!is.null(x) && nzchar(x)) x else y

option_list <- list(
  make_option(c("--analysis-root"), type = "character", help = "Stage3_OS_archetypes_analysis output directory"),
  make_option(c("--gene-sig-dir"), type = "character", default = NULL, help = "Directory containing *_interface_de_full.csv files (default: <analysis-root>/06_interface_gene_de)"),
  make_option(c("--gene-mapping"), type = "character", default = NULL, help = "Optional TSV mapping (gene_symbol, ensembl_id) for resolving gene indices"),
  make_option(c("--output-dir"), type = "character", default = NULL, help = "Output directory for mapped signatures (default: <analysis-root>/sc_analysis/gene_signatures_mapped)"),
  make_option(c("--cell-counts-csv"), type = "character", default = NULL, help = "Prototype x cell_type counts CSV (prototype_id, cell_type, n_cells)"),
  make_option(c("--composition-csv"), type = "character", default = NULL, help = "Output CSV for prototype composition summary (default: <analysis-root>/sc_analysis/archetype_cellular_composition.csv)"),
  make_option(c("--only-map"), action = "store_true", default = FALSE, help = "Stop after mapping gene signatures")
)

opt <- parse_args(OptionParser(option_list = option_list))
if (is.null(opt$analysis_root)) {
  stop("--analysis-root is required")
}
analysis_root <- path_abs(opt$analysis_root)
if (!dir_exists(analysis_root)) {
  stop("analysis root does not exist: ", analysis_root)
}

gene_sig_dir <- `%||%`(opt$gene_sig_dir, path(analysis_root, "06_interface_gene_de"))
out_dir <- `%||%`(opt$output_dir, path(analysis_root, "sc_analysis", "gene_signatures_mapped"))
dir_create(out_dir, recurse = TRUE)
composition_csv <- `%||%`(opt$composition_csv, path(analysis_root, "sc_analysis", "archetype_cellular_composition.csv"))

read_gene_map <- function(mapping_path) {
  if (is.null(mapping_path) || !file_exists(mapping_path)) {
    return(NULL)
  }
  gm <- suppressMessages(read_tsv(mapping_path, col_names = FALSE, show_col_types = FALSE))
  colnames(gm) <- c("gene_symbol", "ensembl_id")[seq_len(ncol(gm))]
  gm %>% mutate(idx_0based = row_number() - 1L)
}

map_signature_file <- function(file_path, gene_map, out_dir) {
  message("Mapping ", file_path)
  dt <- suppressMessages(read_csv(file_path, show_col_types = FALSE))
  if (!"gene" %in% names(dt)) {
    warning("Skipping ", file_path, " (missing 'gene' column)")
    return(NULL)
  }
  gene_col_numeric <- suppressWarnings(all(!is.na(as.integer(dt$gene))))
  if (!is.null(gene_map) && gene_col_numeric) {
    dt <- dt %>%
      mutate(idx_0based = as.integer(gene)) %>%
      left_join(gene_map, by = "idx_0based") %>%
      mutate(gene_symbol = if_else(!is.na(gene_symbol), gene_symbol, as.character(gene)))
  } else if (!"gene_symbol" %in% names(dt)) {
    dt <- dt %>% mutate(gene_symbol = as.character(gene))
  }
  out_file <- path(out_dir, path_file(file_path))
  suppressMessages(write_csv(dt, out_file))
  out_file
}

map_gene_signatures <- function(sig_dir, gene_map_path, out_dir) {
  if (!dir_exists(sig_dir)) {
    stop("gene signature directory not found: ", sig_dir)
  }
  files <- dir_ls(sig_dir, recurse = TRUE, glob = "*_interface_de_full.csv", type = "file")
  if (!length(files)) {
    stop("no signature files found under ", sig_dir)
  }
  gene_map <- read_gene_map(gene_map_path)
  purrr::map(files, map_signature_file, gene_map = gene_map, out_dir = out_dir)
  message("Mapped ", length(files), " files to ", out_dir)
}

summarize_celltype_composition <- function(counts_csv, out_csv) {
  if (is.null(counts_csv) || !file_exists(counts_csv)) {
    stop("cell-type counts CSV not found: ", counts_csv)
  }
  counts <- suppressMessages(read_csv(counts_csv, show_col_types = FALSE))
  required_cols <- c("prototype_id", "cell_type", "n_cells")
  missing_cols <- setdiff(required_cols, names(counts))
  if (length(missing_cols)) {
    stop("counts CSV missing columns: ", paste(missing_cols, collapse = ", "))
  }
  comp <- counts %>%
    group_by(prototype_id) %>%
    mutate(frac = n_cells / sum(n_cells)) %>%
    ungroup() %>%
    select(prototype_id, cell_type, frac) %>%
    pivot_wider(names_from = cell_type, values_from = frac, values_fill = 0) %>%
    mutate(Archetype = paste0("Archetype", prototype_id)) %>%
    arrange(Archetype)
  suppressMessages(write_csv(comp, out_csv))
  message("Saved composition summary to ", out_csv)
}

map_gene_signatures(gene_sig_dir, opt$gene_mapping, out_dir)
if (isTRUE(opt$only_map)) {
  message("Mapping complete (--only-map)")
  quit(save = "no")
}

if (!is.null(opt$cell_counts_csv)) {
  summarize_celltype_composition(opt$cell_counts_csv, composition_csv)
} else {
  message("--cell-counts-csv not provided; prototype composition summary skipped")
}
