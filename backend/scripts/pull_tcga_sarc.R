#!/usr/bin/env Rscript
# TCGA-SARC metadata via TCGAbiolinks (real GDC) or deterministic stub.
#
# Usage:
#   Rscript backend/scripts/pull_tcga_sarc.R --out-dir data/intermediate [--patient-list ids.txt] [--stub]
#
# Non-stub: BiocManager::install(c("TCGAbiolinks","SummarizedExperiment","jsonlite","arrow"))
# Environment: GDC authentication follows TCGAbiolinks defaults (see package vignette).

suppressPackageStartupMessages({
  if (!requireNamespace("optparse", quietly = TRUE)) {
    stop("Install optparse: install.packages('optparse')")
  }
  library(optparse)
})

option_list <- list(
  make_option("--out-dir", type = "character", default = "data/intermediate",
              help = "Output directory"),
  make_option("--patient-list", type = "character", default = NA_character_,
              help = "Newline-separated TCGA ids for stub rows"),
  make_option("--stub", action = "store_true", default = FALSE,
              help = "Stub parquet (no GDC)"),
  make_option("--max-samples", type = "integer", default = 40L,
              help = "Max RNA samples in real mode")
)

opt <- parse_args(OptionParser(option_list = option_list))
out_dir <- opt[["out-dir"]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
meta_path <- file.path(out_dir, "tcga_sarc_metadata.parquet")

write_parquet_or_csv <- function(df) {
  if (requireNamespace("arrow", quietly = TRUE)) {
    arrow::write_parquet(df, meta_path)
    message("Wrote ", meta_path)
  } else {
    p <- file.path(out_dir, "tcga_sarc_metadata.csv")
    utils::write.csv(df, p, row.names = FALSE)
    message("Package 'arrow' not found; wrote ", p)
  }
}

case_from_pid <- function(pid) {
  parts <- strsplit(pid, "-", fixed = TRUE)[[1]]
  if (length(parts) >= 3L) paste(parts[1:3], collapse = "-") else pid
}

stub_rows <- function(pids) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Stub mode needs jsonlite: install.packages('jsonlite')")
  }
  genes <- c("TP53", "TTN", "MUC16", "BRCA1")
  do.call(rbind, lapply(pids, function(pid) {
    vals <- vapply(genes, function(g) {
      z <- utf8ToInt(paste0(pid, "|", g))
      h <- sum(z) %% 10000L
      0.05 + (as.numeric(h) / 9999) * 199.95
    }, numeric(1))
    data.frame(
      patient_id = pid,
      barcode = pid,
      case_barcode = case_from_pid(pid),
      sample_prefix16 = substr(pid, 1L, min(16L, nchar(pid))),
      rna_tpm_dict = as.character(jsonlite::toJSON(as.list(vals), auto_unbox = TRUE)),
      rna_long_path = "",
      purity = 0.82,
      purity_method = "stub_r_deterministic",
      purity_confidence = 0.55,
      cnv_segments = "[]",
      normal_bam_path = "",
      source = "stub_r",
      stringsAsFactors = FALSE
    )
  }))
}

if (isTRUE(opt$stub) || identical(Sys.getenv("TCGA_STUB", unset = ""), "1")) {
  message("TCGA_STUB: writing synthetic tcga_sarc_metadata")
  pids <- character(0)
  pl <- opt$`patient-list`
  if (!is.na(pl) && nzchar(pl) && file.exists(pl)) {
    pids <- trimws(readLines(pl, warn = FALSE))
    pids <- pids[nzchar(pids)]
  }
  if (length(pids) == 0L) {
    pids <- c("TCGA-DX-TEST-01A-01D-A1", "TCGA-DX-TEST2-01A-01D-A1")
  }
  write_parquet_or_csv(stub_rows(pids))
  quit(status = 0L)
}

# ---- Real mode (requires TCGAbiolinks + network) ----
message("Real mode: querying GDC for TCGA-SARC RNA-Seq (Gene Expression Quantification) …")
if (!requireNamespace("TCGAbiolinks", quietly = TRUE) ||
    !requireNamespace("SummarizedExperiment", quietly = TRUE) ||
    !requireNamespace("jsonlite", quietly = TRUE)) {
  message("Missing R packages. Install with:")
  message("  if (!requireNamespace('BiocManager', quietly=TRUE)) install.packages('BiocManager')")
  message("  BiocManager::install(c('TCGAbiolinks','SummarizedExperiment','jsonlite','arrow'))")
  quit(status = 3L)
}

ok <- tryCatch({
  query <- TCGAbiolinks::GDCquery(
    project = "TCGA-SARC",
    data.category = "Transcriptome Profiling",
    data.type = "Gene Expression Quantification",
    workflow.type = "STAR - Counts",
    access = "open"
  )
  TCGAbiolinks::GDCdownload(query, method = "api", files.per.chunk = 20)
  se <- TCGAbiolinks::GDCprepare(query, save = FALSE, summarizedExperiment = TRUE)

  an <- SummarizedExperiment::assayNames(se)
  pick <- if ("tpm_unstranded" %in% an) "tpm_unstranded" else an[[1L]]
  mat <- as.matrix(SummarizedExperiment::assay(se, pick))
  cd <- as.data.frame(SummarizedExperiment::colData(se))
  rd <- as.data.frame(SummarizedExperiment::rowData(se))
  gsym <- if ("gene_name" %in% names(rd)) as.character(rd$gene_name) else rownames(mat)
  if (length(gsym) != nrow(mat)) {
    gsym <- rownames(mat)
  }

  samples <- colnames(mat)
  ns <- min(length(samples), opt$`max-samples`)
  samples <- head(samples, ns)

  rows <- list()
  for (s in samples) {
    vec <- mat[, s]
    names(vec) <- gsym
    vec <- vec[is.finite(vec)]
    o <- order(-vec, na.last = TRUE)
    keep <- head(o, 400L)
    sub <- vec[keep]
    js <- jsonlite::toJSON(as.list(sub), auto_unbox = TRUE)
    pid <- if ("submitter_id" %in% names(cd) && s %in% rownames(cd)) {
      as.character(cd[s, "submitter_id"])
    } else if ("patient" %in% names(cd) && s %in% rownames(cd)) {
      as.character(cd[s, "patient"])
    } else {
      s
    }
    rows[[length(rows) + 1L]] <- data.frame(
      patient_id = pid,
      barcode = s,
      case_barcode = case_from_pid(pid),
      sample_prefix16 = substr(pid, 1L, min(16L, nchar(pid))),
      rna_tpm_dict = as.character(js),
      rna_long_path = "",
      purity = NA_real_,
      purity_method = NA_character_,
      purity_confidence = NA_real_,
      cnv_segments = "[]",
      normal_bam_path = "",
      source = "tcgabiolinks",
      stringsAsFactors = FALSE
    )
  }
  df <- do.call(rbind, rows)
  write_parquet_or_csv(df)
  TRUE
}, error = function(e) {
  message("Real pull failed: ", conditionMessage(e))
  message("Re-run with --stub for offline metadata, or fix GDC access.")
  FALSE
})

quit(status = if (isTRUE(ok)) 0L else 4L)
