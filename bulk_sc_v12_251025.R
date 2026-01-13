### =========================================================
## Functional Archetype (Prototype0–4) — full pipeline (FINAL, aligned-only, one-click)
## =========================================================

suppressPackageStartupMessages({
  reposCRAN <- "https://cloud.r-project.org"
  
  cran_pkgs <- c(
    "Seurat","Matrix","dplyr","tidyr","tibble","readr","stringr","ggplot2",
    "RColorBrewer","pheatmap","purrr","ggpubr","ggrepel","pROC","splines",
    "matrixStats","future","future.apply","digest","patchwork","ggsci","viridis","clue",
    "scales","grid","ggh4x","UCell","glmnet","msigdbr","mclust",
    "ggbeeswarm","ggridges","gridExtra","magick","png","cowplot"
  )
  
  # 先装，再载入 —— 用 requireNamespace/require（都返回 TRUE/FALSE）
  miss_cran <- cran_pkgs[!vapply(cran_pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(miss_cran)) install.packages(miss_cran, repos = reposCRAN)
  invisible(vapply(cran_pkgs, require, logical(1), quietly = TRUE, character.only = TRUE))
  
  if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager", repos = reposCRAN)
  
  bioc_pkgs <- c("GSVA","GSEABase","BiocParallel","SummarizedExperiment","ComplexHeatmap","circlize")
  miss_bioc <- bioc_pkgs[!vapply(bioc_pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(miss_bioc)) BiocManager::install(miss_bioc, ask = FALSE, update = FALSE)
  invisible(vapply(bioc_pkgs, require, logical(1), quietly = TRUE, character.only = TRUE))
  
  # CellChat 单独处理（有些环境装起来慢/需依赖）
  if (!requireNamespace("CellChat", quietly = TRUE)) {
    try(BiocManager::install("CellChat", update = FALSE, ask = FALSE), silent = TRUE)
  }
  if (requireNamespace("CellChat", quietly = TRUE)) {
    suppressPackageStartupMessages(library(CellChat))
  }
})
suppressPackageStartupMessages(library(Seurat))
options(stringsAsFactors = FALSE, bitmapType = "cairo")
set.seed(1234)

library(magrittr)





## --------------------------
## 基础路径
## --------------------------+
base_path <- "D:\\after_bib\\data_v2\\paper_img\\archetype_analysis_results_final_v11_fix1_250930"
cache_root <- file.path(base_path, "cache")
dir.create(cache_root, showWarnings = FALSE, recursive = TRUE)

my_tempdir <- "D:/after_bib/data_v2/paper_img/archetype_analysis_results_final_v11_fix1_250930/cache"

outdir    <- file.path(base_path, "PlanOverview_Prototype_ECI_RI")  # 可自定义
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

## scRNA
sc_dir               <- "D:\\after_bib\\data_v2\\paper_img\\archetype_analysis_results_final_v11_fix1\\bc_sc\\Wu_etal_2021_BRCA_scRNASeq"
sc_metadata_filename <- "metadata.csv"
sc_celltype_col      <- "celltype_subset"

## bulk
bulk_expr_file  <- "D:\\after_bib\\data_v2\\paper_img\\archetype_analysis_results_final_v11_fix1\\geo_val\\GSE243375_All_Samples_TMM_log2_CPM.csv"
bulk_pheno_file <- "D:\\after_bib\\data_v2\\paper_img\\archetype_analysis_results_final_v11_fix1\\geo_val\\GSE243375_sampleinfo_label.csv"
dataset_name    <- "GSE243375"
pheno_col       <- "Sample_characteristics_ch1"   # 含 “pcr: Yes/No”
pcr_prefix      <- "^pcr:\\s*"


## 原型方向性签名
proto_sig_file_raw <- "D:/after_bib/data_v2/paper_img/archetype_analysis_results_final_v11_fix1_250930/figures_publication/interface_core_signature_means.csv"

## 先验（样本级，可选）
# prior_file <- file.path(base_path, "Python_Validation_Analysis_with_UMAP_CellChat/archetype_true_composition_prototypes_x_celltypes.tsv")
## 原型×宏类 组成先验（必须）
prior_comp_file_raw <- file.path(
  base_path, "COHORT_Yale/2_cell_feature_analysis", "cell_prototype_composition.csv"
)


## 主题与颜色
theme_pub2 <- function(base_size = 12, base_family = NULL){
  ggplot2::theme_minimal(base_size = base_size, base_family = base_family) +
    ggplot2::theme(
      panel.grid.major = element_line(color = "#f2f2f2", linewidth = 0.3),
      panel.grid.minor = element_blank(),
      axis.title  = element_text(size = base_size + 1, face = "bold"),
      axis.text   = element_text(size = base_size),
      plot.title  = element_text(size = base_size + 3, face = "bold", hjust = 0.5, margin = margin(b = 6)),
      legend.title= element_text(face = "bold"),
      strip.text  = element_text(face = "bold")
    )
}
theme_set(theme_pub2())
pal_proto <- c(Prototype0="#7f7f7f", Prototype2="#1f77b4", Prototype4="#d62728")
pal_proto_plot <- c(pal_proto, Unknown = "#bdbdbd")
pal_resp  <- c(non_pCR="#B2182B", pCR="#2166AC")
proto_levels <- c("Archetype0","Archetype2","Archetype4")
proto_col_v1 <- "Archetype_aligned_v1"
proto_col_v2 <- "Archetype_aligned_v2"

`%nin%` <- function(x, y) !(x %in% y)
nz <- function(x) x[is.finite(x) & !is.na(x)]
safe_cor <- function(x, y) suppressWarnings(cor(x, y, method = "spearman", use = "complete.obs"))
rb_from_auc <- function(a) ifelse(is.finite(a), 2 * a - 1, NA_real_)



# 统一原型命名：支持 "Prototype 2" / "A2" / "proto=2" / "2" → "Prototype2"
std_proto_unified <- function(x){
  x <- as.character(x)
  k <- suppressWarnings(as.integer(sub("^.*?(\\d+).*?$", "\\1", x, perl = TRUE)))
  ifelse(is.finite(k), paste0("Prototype", k), NA_character_)
}


log_n <- function(tag, obj) {
  cat(sprintf("[%-22s] cells = %d\n", tag, ncol(obj)))
}


qc_check_alignment <- function(mapping, sim_mat, seu){
  message("\n=== QC Check: Archetype Alignment ===")
  if (any(mapping$similarity < 0.2, na.rm=TRUE)) {
    low <- mapping[mapping$similarity < 0.2, ]
    message("⚠️ 低相关映射 (<0.2):")
    print(low)
  } else {
    message("✅ 所有映射相似度 ≥ 0.2")
  }
  dup <- mapping$Prior_col[duplicated(mapping$Prior_col)]
  if (length(dup)) {
    message("⚠️ 存在多对一合并: ", paste(unique(dup), collapse=", "))
  } else {
    message("✅ 没有多对一合并")
  }
  if (any(is.na(seu$Prototype_aligned))) {
    message("⚠️ 存在 NA Prototype_aligned (未对齐的细胞)")
  } else {
    message("✅ 所有细胞都有对齐后的 Archetype 标签")
  }
}



chunked_read_table <- function(...){
  if ("read_table_chunked" %in% getNamespaceExports("readr")) {
    readr::read_table_chunked(...)
  } else {
    readr::read_delim_chunked(..., delim = " ")
  }
}


prefilter_raw_mtx_by_library_size <- function(
  mtx_file, genes_file, barcodes_file,
  keep_top = 120000, min_counts = 100
){
  stopifnot(file.exists(mtx_file), file.exists(genes_file), file.exists(barcodes_file))
  
  con <- file(mtx_file, "r")
  on.exit(close(con))
  header_lines <- 0L
  dims_line <- NULL
  repeat {
    ln <- readLines(con, n = 1)
    if (!length(ln)) stop("mtx 文件异常或为空：", mtx_file)
    if (!startsWith(ln, "%")) {
      dims_line <- ln
      break
    }
    header_lines <- header_lines + 1L
  }
  dims <- scan(text = dims_line, what = integer(), quiet = TRUE)
  if (length(dims) < 3) stop("无法从 mtx 文件解析矩阵维度：", mtx_file)
  nrow <- dims[1]; ncol <- dims[2]
  skip <- header_lines + 1L
  
  colsum <- numeric(ncol)
  cb1 <- function(x, pos){
    colsum[x$j] <<- colsum[x$j] + as.numeric(x$x)
  }
  chunked_read_table(
    file = mtx_file,
    callback = readr::SideEffectChunkCallback$new(cb1),
    col_names = c("i", "j", "x"),
    col_types = readr::cols(i = readr::col_integer(), j = readr::col_integer(), x = readr::col_double()),
    skip = skip,
    progress = interactive(),
    chunk_size = 200000
  )
  
  keep_idx <- which(colsum >= min_counts)
  if (length(keep_idx) > keep_top) {
    ord <- order(colsum[keep_idx], decreasing = TRUE)
    keep_idx <- keep_idx[ord[seq_len(keep_top)]]
  }
  if (!length(keep_idx)) {
    stop("筛选阈值过严，未保留任何细胞；请调低 min_counts 或增大 keep_top。")
  }
  
  map_idx <- integer(ncol)
  map_idx[keep_idx] <- seq_along(keep_idx)
  i_chunks <- list(); j_chunks <- list(); x_chunks <- list(); chunk_id <- 0L
  cb2 <- function(x, pos){
    jj <- map_idx[x$j]
    sel <- jj > 0L
    if (any(sel)) {
      chunk_id <<- chunk_id + 1L
      i_chunks[[chunk_id]] <<- as.integer(x$i[sel])
      j_chunks[[chunk_id]] <<- as.integer(jj[sel])
      x_chunks[[chunk_id]] <<- as.numeric(x$x[sel])
    }
  }
  chunked_read_table(
    file = mtx_file,
    callback = readr::SideEffectChunkCallback$new(cb2),
    col_names = c("i", "j", "x"),
    col_types = readr::cols(i = readr::col_integer(), j = readr::col_integer(), x = readr::col_double()),
    skip = skip,
    progress = interactive(),
    chunk_size = 200000
  )
  
  if (!length(i_chunks)) {
    stop("构建稀疏矩阵失败，保留的细胞没有非零计数；请调整过滤阈值。")
  }
  i_vec <- as.integer(unlist(i_chunks, use.names = FALSE))
  j_vec <- as.integer(unlist(j_chunks, use.names = FALSE))
  x_vec <- as.numeric(unlist(x_chunks, use.names = FALSE))
  
  mat <- Matrix::sparseMatrix(
    i = i_vec,
    j = j_vec,
    x = x_vec,
    dims = c(nrow, length(keep_idx)),
    giveCsparse = TRUE
  )
  
  barcodes <- readr::read_tsv(barcodes_file, col_names = FALSE, show_col_types = FALSE)[[1]]
  genes_df <- readr::read_tsv(genes_file, col_names = FALSE, show_col_types = FALSE)
  gene_symbols <- if (ncol(genes_df) >= 2) genes_df[[2]] else genes_df[[1]]
  gene_symbols <- make.unique(as.character(gene_symbols))
  stopifnot(length(barcodes) >= length(keep_idx))
  
  colnames(mat) <- barcodes[keep_idx]
  rownames(mat) <- gene_symbols
  
  message(sprintf(
    "✔ 预筛保留 %d/%d 个 barcodes (min UMI %d)，非零条目 %d",
    length(keep_idx), ncol, min_counts, length(i_vec)
  ))
  mat
}



## =========================================================
## 全局强制：所有图只允许使用对齐后的原型
## =========================================================
# .enforce_aligned_proto_only <- TRUE
# .require_aligned_proto <- function(seu){
#   if (.enforce_aligned_proto_only && !"Prototype_aligned" %in% colnames(seu@meta.data)) {
#     stop("未找到 `Prototype_aligned`。请先完成匈牙利对齐并回灌到单细胞对象后再作图。")
#   }
# }

## ========= 提前定义：读取“原型×宏类先验”的健壮函数 =========
read_prior_comp <- function(prior_comp_file, macro_levels){
  if (!file.exists(prior_comp_file)) stop("找不到先验文件：", prior_comp_file)
  
  ext <- tolower(tools::file_ext(prior_comp_file))
  df0 <- tryCatch(
    if (ext %in% c("tsv","txt")) read.delim(prior_comp_file, check.names=FALSE, quote="", comment.char="")
    else read.csv(prior_comp_file, check.names=FALSE, quote="", comment.char=""),
    error=function(e) NULL
  )
  if (is.null(df0) || !nrow(df0)) stop("读取先验失败或为空：", prior_comp_file)
  
  cn <- colnames(df0)
  
  # std_proto_unified 用于识别各种原型命名写法
  
  ## ========= 先试“长表”识别：prototype + macro + value =========
  proto_cand <- grep("^(proto|archetype|prototype|a_?id|prototype_?id)$", cn, ignore.case=TRUE, value=TRUE)
  macro_cand <- grep("^(macro|macroclass|class|celltype|category)$",  cn, ignore.case=TRUE, value=TRUE)
  val_cand   <- grep("^(frac|prop|proportion|percent|value|weight|score)$", cn, ignore.case=TRUE, value=TRUE)
  if (length(proto_cand)>=1 && length(macro_cand)>=1 && length(val_cand)>=1) {
    pc <- proto_cand[1]; mc <- macro_cand[1]; vc <- val_cand[1]
    long <- df0[, c(pc, mc, vc), drop=FALSE]
    colnames(long) <- c("prototype","macro","value")
    long$prototype <- std_proto_unified(long$prototype)
    long$macro     <- trimws(as.character(long$macro))
    long$value     <- suppressWarnings(as.numeric(long$value))
    long <- long[is.finite(long$value), , drop=FALSE]
    long <- long[long$macro %in% macro_levels, , drop=FALSE]
    if (!nrow(long)) stop("先验长表识别成功，但没有与 macro_levels 交集的行。")
    
    wide <- tidyr::pivot_wider(long, names_from="macro", values_from="value", values_fill=0) %>% as.data.frame()
    rownames(wide) <- wide$prototype
    wide$prototype <- NULL
    
    miss_cols <- setdiff(macro_levels, colnames(wide))
    for (m in miss_cols) wide[[m]] <- 0
    wide <- wide[, macro_levels, drop=FALSE]
    
    mat <- as.matrix(wide)
    rs <- rowSums(mat); rs[!is.finite(rs) | rs<=0] <- 1
    return(sweep(mat, 1, rs, "/"))
  }
  
  ## ========= 宽表路径 =========
  df <- df0
  
  # 1) 原型列：优先识别显式列；否则**优先取第一列**，再次失败才尝试 rownames
  proto_col_idx <- which(grepl("^\\s*(proto|archetype|prototype|a_?id|prototype_?id)\\b", cn, ignore.case = TRUE))
  
  if (length(proto_col_idx)) {
    df$prototype <- std_proto_unified(as.character(df[[proto_col_idx[1]]]))
  } else {
    df$prototype <- std_proto_unified(as.character(df[[1]]))
    if (all(is.na(df$prototype))) {
      rn_try <- std_proto_unified(rownames(df))
      if (sum(!is.na(rn_try)) > 0) df$prototype <- rn_try
    }
  }
  
  df <- df[!is.na(df$prototype) & nzchar(df$prototype), , drop = FALSE]
  rownames(df) <- df$prototype
  df$prototype <- NULL
  
  # 2) 宏类列：按 macro_levels 匹配（大小写/空格容错）
  macro_cols <- intersect(macro_levels, colnames(df))
  if (!length(macro_cols)) {
    # 放宽：大小写/空格容错映射
    rename_map <- sapply(macro_levels, function(m){
      hit <- grep(paste0("^\\s*", m, "\\s*$"), colnames(df), ignore.case=TRUE, value=TRUE)
      if (length(hit)) hit[1] else NA_character_
    }, USE.NAMES = TRUE)
    rename_map <- rename_map[!is.na(rename_map)]
    if (length(rename_map)) {
      for (m in names(rename_map)) colnames(df)[match(rename_map[[m]], colnames(df))] <- m
      macro_cols <- intersect(macro_levels, colnames(df))
    }
  }
  if (!length(macro_cols)) {
    stop("在先验文件中找不到任何宏类列；需要列名：", paste(macro_levels, collapse=", "))
  }
  
  sub <- as.data.frame(df[, macro_cols, drop=FALSE])
  miss <- setdiff(macro_levels, colnames(sub))
  for (m in miss) sub[[m]] <- 0
  sub <- sub[, macro_levels, drop=FALSE]
  
  mat <- as.matrix(sub); storage.mode(mat) <- "double"
  rs <- rowSums(mat); bad <- !is.finite(rs) | rs<=0
  if (any(bad)) mat[bad, ] <- 1/length(macro_levels)
  rs <- rowSums(mat)
  sweep(mat, 1, rs, "/")
}

## ===== A) 先验对齐：用宏类组成把本队列的 Prototype 映射到“标准原型名” =====
macro_levels_all   <- c("Connective","Dead","Epithelial","Inflammatory","Neoplastic","Unknown")
# macro_levels_align <- c("Epithelial","Inflammatory","Connective","Neoplastic")



# 计算“观测的 Prototype×MacroClass 组成”（行和=1）
obs_proto_macro <- function(seu, proto_col = "Prototype_refined",
                            macro_col = "MacroClass",
                            macro_levels = macro_levels_all){
  stopifnot(proto_col %in% colnames(seu@meta.data),
            macro_col %in% colnames(seu@meta.data))
  md <- seu@meta.data[, c(proto_col, macro_col), drop=FALSE]
  proto_levels <- c("Prototype0","Prototype2","Prototype4")
  base_proto <- std_proto_unified(gsub("^Interface_.*$","", gsub("_(Core|Edge)$","", md[[proto_col]])))
  base_proto <- factor(base_proto, levels = proto_levels)
  macro      <- factor(as.character(md[[macro_col]]), levels = macro_levels)
  keep <- !is.na(base_proto) & !is.na(macro)
  tab <- table(base_proto[keep], macro[keep], dnn = c("Prototype","Macro"), useNA = "no")
  tab <- as.matrix(tab)
  if (!length(tab)) stop("观测数据中没有可用的 Prototype×Macro 组合。")
  rs <- rowSums(tab)
  rs[rs == 0] <- 1
  sweep(tab, 1, rs, "/")
}

# 余弦相似度（行向量）
cosine_sim <- function(A, B){
  A <- as.matrix(A); B <- as.matrix(B)
  A <- A / pmax(sqrt(rowSums(A^2)), 1e-9)
  B <- B / pmax(sqrt(rowSums(B^2)), 1e-9)
  A %*% t(B)
}

# 把矩形相似度矩阵 pad 成方阵再用匈牙利匹配；返回行→列的一一匹配
hungarian_match <- function(S){
  nr <- nrow(S); nc <- ncol(S); n <- max(nr, nc)
  S_pad <- matrix(0, n, n); S_pad[seq_len(nr), seq_len(nc)] <- S
  cost  <- 1 - pmax(pmin(S_pad, 1), 0)
  assign <- clue::solve_LSAP(cost)
  sel    <- cbind(seq_len(nr), assign[seq_len(nr)])
  sel[sel[,2] <= nc, , drop=FALSE]
}

align_prototypes_with_prior <- function(
  seu,
  prior_comp_file = NULL,
  proto_col = "Prototype_refined",
  macro_col = "MacroClass",
  macro_levels = NULL,
  outdir = NULL
){
  # --------- 参数与先验读取 ----------
  if (is.null(outdir)) {
    outdir <- "D:/after_bib/data_v2/paper_img/archetype_analysis_results_final_v11_fix1_250930/cache/prior_align"
  }
  if (is.null(prior_comp_file)) stop("prior_comp_file 不能为空（请显式传入路径）")
  if (is.null(macro_levels)) {
    macro_levels <- c("Connective","Dead","Epithelial","Inflammatory","Neoplastic","Unknown")
  }
  
  stopifnot(file.exists(prior_comp_file))
  prior_mat <- read_prior_comp(prior_comp_file, macro_levels = macro_levels)
  
  # Unknown 列的稳健处理：若存在则用“已知列均值”，否则不强制添加（你的 read_prior_comp 里通常会给出）
  if ("Unknown" %in% colnames(prior_mat)) {
    known_cols <- setdiff(colnames(prior_mat), "Unknown")
    if (length(known_cols)) {
      prior_mat[, "Unknown"] <- apply(prior_mat[, known_cols, drop = FALSE], 1, function(v){
        v <- v[is.finite(v)]; if (!length(v)) 0.5 else mean(v)
      })
    } else {
      prior_mat[, "Unknown"] <- 0.5
    }
  }
  
  # 只保留允许的原型行
  allowed_proto <- c("Prototype0","Prototype2","Prototype4")
  prior_mat <- prior_mat[rownames(prior_mat) %in% allowed_proto, , drop = FALSE]
  if (!nrow(prior_mat)) stop("先验中未找到任何允许的原型：", paste(allowed_proto, collapse=", "))
  
  bad <- setdiff(rownames(prior_mat), allowed_proto)
  if (length(bad)) stop("先验包含未定义原型：", paste(bad, collapse=", "))
  
  message("先验原型保留：", paste(rownames(prior_mat), collapse = ", "))
  
  # --------- 观测（对齐前）原型×宏类组成，用于构建相似度 ----------
  obs_mat <- obs_proto_macro(seu, proto_col, macro_col, macro_levels)
  
  # 统一宏类列（缺的补 0，顺序一致）
  add_missing_cols <- setdiff(macro_levels, colnames(prior_mat))
  if (length(add_missing_cols)) {
    prior_mat <- cbind(prior_mat, matrix(0, nrow=nrow(prior_mat), ncol=length(add_missing_cols),
                                         dimnames=list(rownames(prior_mat), add_missing_cols)))
  }
  prior_mat <- prior_mat[, macro_levels, drop = FALSE]
  
  add_missing_cols2 <- setdiff(macro_levels, colnames(obs_mat))
  if (length(add_missing_cols2)) {
    obs_mat <- cbind(obs_mat, matrix(0, nrow=nrow(obs_mat), ncol=length(add_missing_cols2),
                                     dimnames=list(rownames(obs_mat), add_missing_cols2)))
  }
  obs_mat <- obs_mat[, macro_levels, drop = FALSE]
  
  # 与先验的交互版本（行为观测原型，列为宏类）
  obs_use   <- as.matrix(obs_mat[rownames(obs_mat), , drop = FALSE])
  prior_use <- as.matrix(prior_mat[rownames(prior_mat), , drop = FALSE])
  
  # --------- 余弦相似度矩阵（行=观测原型，列=先验原型） ----------
  row_cosine <- function(A, B){
    A <- as.matrix(A); B <- as.matrix(B)
    An <- sqrt(rowSums(A*A)); An[An == 0] <- 1
    Bn <- sqrt(rowSums(B*B)); Bn[Bn == 0] <- 1
    S <- A %*% t(B)
    S <- S / outer(An, Bn, "*")
    S[!is.finite(S)] <- 0
    S
  }
  S <- if (exists("cosine_sim", mode = "function")) {
    cosine_sim(obs_use, prior_use)
  } else {
    row_cosine(obs_use, prior_use)
  }
  
  # 映射得分缩放与匈牙利匹配（最大化相似度）
  S_scaled <- pmax(pmin(S, 1), -1)
  neg_pairs <- sum(S_scaled < 0, na.rm = TRUE)
  S_match <- (S_scaled + 1) / 2
  
  cost <- 1 - S_match
  n <- nrow(cost); m <- ncol(cost)
  if (n != m) {
    k <- max(n, m)
    cost_pad <- matrix(1, nrow=k, ncol=k)
    cost_pad[seq_len(n), seq_len(m)] <- cost
    rownames(cost_pad) <- c(rownames(cost), paste0(".pad_r", seq_len(k - n)))
    colnames(cost_pad) <- c(colnames(cost), paste0(".pad_c", seq_len(k - m)))
    cost <- cost_pad
  }
  
  used_solver <- "clue::solve_LSAP"; mapping_ok <- TRUE
  if (requireNamespace("clue", quietly = TRUE)) {
    sol <- try(clue::solve_LSAP(cost), silent = TRUE)
    if (inherits(sol, "try-error")) { mapping_ok <- FALSE; used_solver <- "greedy" }
  } else { mapping_ok <- FALSE; used_solver <- "greedy" }
  
  if (mapping_ok) {
    pick <- as.integer(sol)
    rows_real <- rownames(cost)[!startsWith(rownames(cost), ".pad_r")]
    cols_real <- colnames(cost)[!startsWith(colnames(cost), ".pad_c")]
    pick_real <- pick[seq_along(rows_real)]
    to <- cols_real[pick_real]
    map_vec <- setNames(to, rows_real)
  } else {
    rows_real <- rownames(cost)[!startsWith(rownames(cost), ".pad_r")]
    cols_real <- colnames(cost)[!startsWith(colnames(cost), ".pad_c")]
    left_cols <- cols_real
    map_vec <- setNames(rep(NA_character_, length(rows_real)), rows_real)
    for (r in rows_real) {
      cc <- cost[r, left_cols]; j <- names(which.min(cc))
      if (length(j)) { map_vec[r] <- j; left_cols <- setdiff(left_cols, j) }
    }
  }
  
  map_vec <- map_vec[!startsWith(map_vec, ".pad_c")]
  map_vec <- map_vec[intersect(names(map_vec), rownames(obs_use))]
  map_vec <- map_vec[!is.na(map_vec)]
  
  if (length(map_vec)) {
    map_df <- data.frame(
      Source_proto = names(map_vec),
      Prior_col    = unname(map_vec),
      similarity   = S[cbind(names(map_vec), unname(map_vec))],
      stringsAsFactors = FALSE
    )
    map_df <- map_df[order(-map_df$similarity), ]
    rownames(map_df) <- NULL
  } else {
    map_df <- data.frame(Source_proto=character(0), Prior_col=character(0), similarity=numeric(0))
  }
  
  # --------- 回写对齐后的标签（Prototype_aligned） ----------
  # 尽量将 proto_col 的原始取值映射到 obs_mat 的行名（Source_proto 的定义域）
  proto_raw  <- as.character(seu[[proto_col]][, 1])
  src_levels <- rownames(obs_mat)
  
  # 简单规范化：去掉 Interface_* 前缀和 _Core/_Edge 后缀，然后尝试匹配
  proto_norm <- gsub("^Interface_.*$", "", gsub("_(Core|Edge)$", "", proto_raw))
  # 如果规范化后能命中行名就用规范化值，否则先用原始值，最后仍未命中则走 Unknown
  proto_src  <- ifelse(proto_norm %in% src_levels, proto_norm,
                       ifelse(proto_raw %in% src_levels, proto_raw, NA_character_))
  
  if (length(map_vec)) {
    dict <- setNames(map_df$Prior_col, map_df$Source_proto)
    aligned <- unname(dict[proto_src])
    aligned[is.na(aligned) | aligned==""] <- "Unknown"
  } else {
    message("⚠️ 对齐未得到有效映射，保持原标签（或置 Unknown）。")
    aligned <- proto_src
    aligned[is.na(aligned) | aligned==""] <- "Unknown"
  }
  
  levels_std  <- intersect(rownames(prior_mat), paste0("Prototype", 0:9))
  level_order <- unique(c(levels_std, map_df$Prior_col, "Unknown"))
  seu$Prototype_aligned <- droplevels(factor(aligned, levels = level_order))
  
  # --------- 可视化输出（已改为“对齐后标签”作图） ----------
  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
  
  # 1) 相似度热图（可按先验列顺序 + 映射列顺序重排行列，更直观）
  S_plot <- S
  col_ord <- rownames(prior_mat)  # 先验原型顺序作为列顺序
  # 行按其被映射到的先验列顺序排序
  if (nrow(map_df)) {
    row_ord <- map_df$Source_proto[order(match(map_df$Prior_col, col_ord))]
    row_ord <- row_ord[row_ord %in% rownames(S_plot)]
    S_plot  <- S_plot[row_ord, col_ord, drop = FALSE]
  } else {
    S_plot  <- S_plot[, col_ord, drop = FALSE]
  }
  
  pheatmap::pheatmap(
    S_plot,
    cluster_rows = FALSE, cluster_cols = FALSE,
    color = colorRampPalette(c("#2166AC", "#F7F7F7", "#B2182B"))(101),
    display_numbers = TRUE, number_format = "%.2f",
    filename = file.path(outdir, "FigureA_prior_alignment_similarity.png"),
    width = 6 + 0.28 * ncol(S_plot), height = 4.2 + 0.28 * nrow(S_plot)
  )
  
  # 2) 组成条形图：Observed 用“对齐后的标签”重算；Prior 用 prior_mat
  obs_mat_aln <- obs_proto_macro(seu, "Prototype_aligned", macro_col, macro_levels)
  
  obs_long <- as.data.frame(as.table(as.matrix(obs_mat_aln)), stringsAsFactors = FALSE)
  colnames(obs_long) <- c("Prototype", "Macro", "Frac")
  obs_long$Source <- "Observed (aligned)"
  
  prior_long <- as.data.frame(as.table(as.matrix(prior_mat)), stringsAsFactors = FALSE)
  colnames(prior_long) <- c("Prototype", "Macro", "Frac")
  prior_long$Source <- "Prior"
  
  # 统一两个面板的 x 轴顺序（严格沿用先验原型顺序）
  x_levels <- rownames(prior_mat)
  obs_long$Prototype   <- factor(obs_long$Prototype,   levels = x_levels)
  prior_long$Prototype <- factor(prior_long$Prototype, levels = x_levels)
  
  gg <- dplyr::bind_rows(obs_long, prior_long) %>%
    dplyr::mutate(Macro = factor(Macro, levels = macro_levels)) %>%
    ggplot2::ggplot(ggplot2::aes(x = Prototype, y = Frac, fill = Macro)) +
    ggplot2::geom_bar(stat = "identity", width = 0.85) +
    ggplot2::facet_wrap(~Source, ncol = 1) +
    ggplot2::scale_fill_manual(values = c(
      Connective = "#1f78b4", Dead = "#6a3d9a", Epithelial = "#33a02c",
      Inflammatory = "#e31a1c", Neoplastic = "#ff7f00", Unknown = "#bdbdbd"
    )) +
    theme_pub2() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 35, hjust = 1)) +
    ggplot2::labs(title = "Prototype×Macro composition: observed (aligned) vs prior",
                  x = NULL, y = "Fraction")
  
  try(ggplot2::ggsave(
    file.path(outdir, "FigureA_prior_alignment_composition_bars.png"),
    gg, width = 10, height = 7.5, dpi = 220, bg = "white"
  ), silent = TRUE)
  
  # 导出映射表
  utils::write.csv(map_df, file.path(outdir, "prototype_prior_alignment_mapping.csv"), row.names = FALSE)
  
  # QC 提示
  align_res <- list(mapping = map_df, sim_mat = S, levels_std = levels_std, neg_pairs = neg_pairs)
  try(qc_check_alignment(mapping = map_df, sim_mat = S, seu = seu), silent = TRUE)
  if (neg_pairs > 0) message(sprintf("⚠️ 先验对齐出现 %d 个负相似度配对（已按 rescaled cost 处理）", neg_pairs))
  
  # 返回
  list(seu = seu, align_res = align_res)
}




# 用 marker_bank 在 sc 上打分（identity_up/down + core/edge/pcr → UCell），
# 结合 MacroClass 先验（可调 beta_prior），挑 best 原型生成 Prototype_initial，
# 再按先验做匈牙利对齐 → Prototype_aligned
# 用 marker_bank 在 sc 上打分（identity_up/down + core/edge/pcr → UCell），
# 结合 MacroClass 先验（可调 beta_prior），挑 best 原型生成 Prototype_initial，
# 再按先验做匈牙利对齐 → Prototype_aligned
score_sc_prototypes_and_align <- function(
  seu, marker_bank, prior_comp_file,
  macro_col = "MacroClass",
  w_spatial = 0.4, min_margin = 0.30,     # 严格：0.25~0.35
  w_func = 0.35, beta_prior = 0.5,        # 适中先验权重：0.4~0.6
  use_hvg = FALSE, hvg_n = 3000,
  trim_per_set = 200,
  ucell_batch = 4000,
  ucell_parallel = FALSE,
  cache_dir = file.path("D:/after_bib/data_v2/paper_img/archetype_analysis_results_final_v11_fix1_250930/cache", "ucell_cache"),
  # —— 严格门控 —— 
  use_consensus_gate = TRUE,              # 三路一致性门（identity/core-edge/pcr）
  consensus_k = 1,                        # 至少 3/3 同向（最严）
  prior_gate = 0.20,                      # 先验绝对阈值（0~1）；NA 表示关闭
  prior_gate_q = NA_real_                 # 行内分位阈值，如 0.4；NA 表示关闭
){
  stopifnot(inherits(seu, "Seurat"))
  if ("RNA" %in% Seurat::Assays(seu)) {
    Seurat::DefaultAssay(seu) <- "RNA"
  } else {
    Seurat::DefaultAssay(seu) <- Seurat::Assays(seu)[1]
  }
  
  ## —— 确保 data 层（CPM×1e4 → log1p；稀疏）——
  ok_data <- TRUE
  suppressWarnings(
    tryCatch({
      dd <- Seurat::GetAssayData(seu, slot="data")
      ok_data <- inherits(dd, "dgCMatrix") && ncol(dd) == ncol(seu)
    }, error = function(e) ok_data <<- FALSE)
  )
  if (!ok_data) {
    message("⚙️ 用 counts 生成 data 层（CPM×1e4 → log1p）")
    M <- Seurat::GetAssayData(seu, slot="counts")
    lib <- Matrix::colSums(M); lib[lib == 0] <- 1
    D <- Matrix::t(Matrix::t(M) / lib) * 1e4
    D <- methods::as(D, "dgCMatrix"); D@x <- log1p(D@x)
    seu <- Seurat::SetAssayData(seu, slot="data", new.data=D)
    rm(M, D); gc(FALSE)
  }
  
  ## —— 基因名映射（大小写鲁棒）——
  rn <- rownames(seu)
  map_upper_to_orig <- setNames(rn, toupper(rn))
  present <- function(genes_chr){
    if (is.null(genes_chr) || !length(genes_chr)) return(character(0))
    g_up <- unique(toupper(as.character(genes_chr)))
    hit <- map_upper_to_orig[g_up]
    unname(unique(hit[!is.na(hit)]))
  }
  
  ## —— 可选 HVG —— 
  hvgs <- NULL
  if (isTRUE(use_hvg)) {
    if (length(Seurat::VariableFeatures(seu)) < 10) {
      seu <- Seurat::FindVariableFeatures(seu, selection.method="vst", nfeatures=hvg_n, verbose=FALSE)
    }
    hvgs <- toupper(Seurat::VariableFeatures(seu))
  }
  
  ## —— 构建 gene sets —— 
  gs <- list()
  for (pid in names(marker_bank)) {
    mb <- marker_bank[[pid]]
    for (tag in c("identity_up","identity_down","core","edge","pcr_up","pcr_down")) {
      g <- present(mb[[tag]])
      if (!is.null(hvgs)) g <- g[toupper(g) %in% hvgs]
      if (length(g) < 5) next
      if (is.finite(trim_per_set) && trim_per_set > 0 && length(g) > trim_per_set) {
        g <- g[seq_len(trim_per_set)]
      }
      gs[[paste0(toupper(sub("_","",tag)), "_", pid)]] <- g
    }
  }
  if (!length(gs)) stop("用于原型评分的 gene sets 过小或为空，请检查 marker_bank/基因名。")
  
  ## —— 缓存 —— 
  dir.create(cache_dir, showWarnings=FALSE, recursive=TRUE)
  sig_key <- digest::digest(list(ncell=ncol(seu), genesets=lapply(gs, length), names=names(gs)))
  cache_file <- file.path(cache_dir, paste0("UCell_scores_", sig_key, ".rds"))
  
  ## —— 分批 UCell 打分（稀疏，预分配框架，缺列补 NA）——
  get_ucell_scores <- function(){
    mat_all <- Seurat::GetAssayData(seu, slot = "data")
    cells <- colnames(mat_all)
    sets  <- names(gs)
    exp_cols <- paste0(sets, "_UCell")
    
    out <- matrix(NA_real_, nrow = length(cells), ncol = length(sets),
                  dimnames = list(cells, exp_cols))
    
    chunks <- split(cells, ceiling(seq_along(cells) / max(1, ucell_batch)))
    message("⏱ UCell 分批：", length(chunks), " 批；batch≈", ucell_batch, " 细胞")
    
    for (cc in chunks) {
      M  <- mat_all[, cc, drop = FALSE]
      sc <- suppressWarnings(UCell::ScoreSignatures_UCell(M, features = gs))
      sc <- as.data.frame(sc, check.names = FALSE)
      common <- intersect(colnames(sc), exp_cols)
      if (length(common)) out[cc, common] <- as.matrix(sc[, common, drop = FALSE])
    }
    
    out_df <- as.data.frame(out, check.names = FALSE)
    rownames(out_df) <- cells
    
    miss <- setdiff(exp_cols, colnames(out_df))
    if (length(miss)) {
      message("ℹ 有 ", length(miss), " 个 gene set 未返回 UCell 列（集合太小/HVG裁剪都可能）：",
              paste(head(miss, 12), collapse = ", "),
              if (length(miss) > 12) paste0(" …(+", length(miss) - 12, ")") else "")
      for (m in miss) out_df[[m]] <- NA_real_
      out_df <- out_df[, exp_cols, drop = FALSE]
    }
    out_df
  }
  
  ## —— 读取/计算 —— 
  if (file.exists(cache_file)) {
    message("✔ 读取缓存 UCell 打分：", cache_file)
    sc_scores <- readRDS(cache_file)
    rn_ok <- sum(colnames(seu) %in% rownames(sc_scores))
    if (is.null(rownames(sc_scores)) || rn_ok < ncol(seu)) {
      message("⚠️ 缓存与当前细胞不匹配，将重新计算 UCell 分数。")
      sc_scores <- get_ucell_scores()
      sc_scores <- sc_scores[colnames(seu), , drop=FALSE]
      saveRDS(sc_scores, cache_file)
    } else {
      sc_scores <- sc_scores[colnames(seu), , drop=FALSE]
    }
  } else {
    sc_scores <- get_ucell_scores()
    sc_scores <- sc_scores[colnames(seu), , drop=FALSE]
    saveRDS(sc_scores, cache_file)
  }
  if (is.matrix(sc_scores)) sc_scores <- as.data.frame(sc_scores)
  if (is.null(rownames(sc_scores))) rownames(sc_scores) <- colnames(seu)
  
  ## —— 列存在性提示（不强制报错）——
  expected_cols <- paste0(names(gs), "_UCell")
  missing <- setdiff(expected_cols, colnames(sc_scores))
  if (length(missing) > 0) {
    message("ℹ 有 ", length(missing), " 个 gene set 没有 UCell 列（集合太小或被 HVG/trim 删掉都正常）：",
            paste(head(missing, 12), collapse = ", "),
            if (length(missing) > 12) paste0(" …(+", length(missing) - 12, ")") else "")
    saveRDS(sc_scores, cache_file)
  }
  
  ## —— 取分工具 —— 
  get_col <- function(nm){
    cn <- paste0(nm, "_UCell")
    v <- if (cn %in% colnames(sc_scores)) sc_scores[, cn] else rep(NA_real_, nrow(sc_scores))
    s <- suppressWarnings(sd(v, na.rm=TRUE))
    if (!is.finite(s) || s < 1e-8) message("⚠️ 列 ", cn, " 方差≈0（可能没取到/全 0）")
    v
  }
  
  ## —— 固定原型顺序 —— 
  P_all <- sort(unique(sub("^.*_", "", names(gs))))
  P <- c("Prototype0","Prototype2","Prototype4")
  P <- P[P %in% P_all]
  if (!length(P)) stop("未在 gene sets 名称里识别到 Prototype0/2/4。")
  
  ## —— 汇总得分 + 三路分解 —— 
  ncell <- ncol(seu)
  S     <- matrix(0, nrow=ncell, ncol=length(P), dimnames=list(colnames(seu), P))
  S_id  <- matrix(0, nrow=ncell, ncol=length(P), dimnames=list(colnames(seu), P))
  S_sp  <- matrix(0, nrow=ncell, ncol=length(P), dimnames=list(colnames(seu), P))
  S_fn  <- matrix(0, nrow=ncell, ncol=length(P), dimnames=list(colnames(seu), P))
  
  for (pid in P) {
    idup <- get_col(paste0("IDENTITYUP_", pid))
    iddn <- get_col(paste0("IDENTITYDOWN_", pid))
    core <- get_col(paste0("CORE_", pid))
    edge <- get_col(paste0("EDGE_", pid))
    pup  <- get_col(paste0("PCRUP_", pid))
    pdn  <- get_col(paste0("PCRDOWN_", pid))
    
    idup[!is.finite(idup)] <- 0; iddn[!is.finite(iddn)] <- 0
    core[!is.finite(core)] <- 0; edge[!is.finite(edge)] <- 0
    pup[!is.finite(pup)]  <- 0; pdn[!is.finite(pdn)]  <- 0
    
    S_id[, pid] <- (idup - iddn)
    S_sp[, pid] <- (core - edge)
    S_fn[, pid] <- (pup  - pdn)
    S[,   pid]  <- S_id[, pid] + w_spatial * S_sp[, pid] + w_func * S_fn[, pid]
  }
  
  ## —— 先验矩阵 —— 
  prior_mat <- read_prior_comp(
    prior_comp_file,
    macro_levels = c("Connective","Dead","Epithelial","Inflammatory","Neoplastic","Unknown")
  )
  if (!("Unknown" %in% colnames(prior_mat))) {
    prior_mat <- cbind(prior_mat, Unknown=1)
  }
  missing_rows <- setdiff(P, rownames(prior_mat))
  if (length(missing_rows)) {
    add <- matrix(1, nrow=length(missing_rows), ncol=ncol(prior_mat),
                  dimnames=list(missing_rows, colnames(prior_mat)))
    prior_mat <- rbind(prior_mat, add)
  }
  mac <- if (macro_col %in% colnames(seu@meta.data)) as.character(seu[[macro_col]][,1]) else rep("Unknown", ncol(seu))
  mac[is.na(mac) | mac==""] <- "Unknown"
  
  ## —— 应用先验加权（可选）—— 
  if (beta_prior > 0) {
    inter_pid <- intersect(P, rownames(prior_mat))
    if (length(inter_pid)) {
      W <- matrix(1, nrow=nrow(S), ncol=ncol(S))
      ok <- mac %in% colnames(prior_mat)
      for (pj in inter_pid) {
        wj <- rep(1, length(mac))
        if (any(ok)) {
          ridx <- match(pj, rownames(prior_mat))
          cidx <- match(mac[ok], colnames(prior_mat))
          w_proto <- prior_mat[ridx, cidx]
          if (all(!is.finite(w_proto) | w_proto <= 0)) {
            w_proto[] <- 1
          } else {
            w_proto[!is.finite(w_proto)] <- 0
            w_proto <- pmax(w_proto, 0)
          }
          wj[ok] <- pmax(w_proto, 1e-6) ^ beta_prior
        }
        W[, colnames(S)==pj] <- wj
      }
      S <- S * W
    }
  }
  
  ## —— 标准化 & margin —— 
  col_mean <- apply(S, 2, function(x) mean(x[is.finite(x)], na.rm=TRUE)); col_mean[!is.finite(col_mean)] <- 0
  col_sd   <- apply(S, 2, function(x) stats::sd(x[is.finite(x)],  na.rm=TRUE)); col_sd[!is.finite(col_sd) | col_sd < 1e-6] <- 1
  S <- sweep(S, 2, col_mean, "-")
  S <- sweep(S, 2, col_sd, "/")
  
  best_idx <- max.col(S, ties.method="first")
  top1 <- S[cbind(seq_len(nrow(S)), best_idx)]
  S[cbind(seq_len(nrow(S)), best_idx)] <- -Inf
  top2 <- matrixStats::rowMaxs(S)
  S[cbind(seq_len(nrow(S)), best_idx)] <- top1
  margin <- top1 - top2
  
  margin_finite <- margin[is.finite(margin)]
  if (length(margin_finite)) {
    qfun <- function(p) stats::quantile(margin_finite, probs=p, na.rm=TRUE, names=FALSE, type=7)
    message(sprintf("UCell→prototype margin stats: n=%d, median=%.4f, p1=%.4f, p5=%.4f, p25=%.4f",
                    length(margin_finite), stats::median(margin_finite), qfun(0.01), qfun(0.05), qfun(0.25)))
  } else {
    message("UCell→prototype margin stats: no finite margins; check gene sets / normalization.")
  }
  
  ## —— 严格门控：一致性 + 先验 —— 
  P_len <- length(P)
  best  <- P[best_idx]
  irow  <- seq_len(nrow(S))
  pick_col <- function(M, best, P){
    idx <- match(best, P)
    out <- rep(0, length(best))
    ok  <- is.finite(idx)
    if (any(ok)) out[ok] <- M[cbind(irow[ok], idx[ok])]
    out[!is.finite(out)] <- 0
    out
  }
  
  # 组件一致性：三路 >0 计同向
  if (isTRUE(use_consensus_gate)) {
    k_consensus <- (pick_col(S_id,  best, P) > 0) +
      (pick_col(S_sp,  best, P) > 0) +
      (pick_col(S_fn,  best, P) > 0)
    cons_ok <- k_consensus >= consensus_k
  } else {
    cons_ok <- rep(TRUE, length(best))
  }
  
  # 先验一致性：该细胞宏类在被选原型的先验 >= 阈值
  prior_cell <- rep(NA_real_, length(best))
  ok_pm <- (best %in% rownames(prior_mat)) & (mac %in% colnames(prior_mat))
  if (any(ok_pm)) {
    ridx <- match(best[ok_pm], rownames(prior_mat))
    cidx <- match(mac[ok_pm],  colnames(prior_mat))
    prior_cell[ok_pm] <- prior_mat[cbind(ridx, cidx)]
  }
  prior_cell[!is.finite(prior_cell)] <- -Inf
  
  if (is.finite(prior_gate)) {
    prior_ok <- prior_cell >= prior_gate
  } else if (is.finite(prior_gate_q)) {
    row_q <- sapply(best, function(p) stats::quantile(prior_mat[p, ], probs=prior_gate_q, na.rm=TRUE))
    prior_ok <- prior_cell >= row_q
  } else {
    prior_ok <- rep(TRUE, length(best))
  }
  
  assign_mask <- is.finite(top1) & (margin >= min_margin) & cons_ok & prior_ok
  
  # 绝不放宽：不过线的保持 NA
  best[!is.finite(top1)] <- NA
  best[!assign_mask]     <- NA
  
  # 日志统计
  message(sprintf("Gate stats: margin-pass=%d, consensus-pass=%d, prior-pass=%d, final(non-NA)=%d / %d",
                  sum(is.finite(top1) & (margin >= min_margin)),
                  sum(cons_ok, na.rm=TRUE),
                  sum(prior_ok, na.rm=TRUE),
                  sum(!is.na(best)), length(best)))
  
  # 不做“单一原型→整体回填”。宁缺毋滥，保留 NA。
  seu$Prototype_initial <- factor(best, levels=P)
  
  ## —— 匈牙利对齐（稳妥包装）—— 
  seu_out <- tryCatch({
    aln <- align_prototypes_with_prior(
      seu,
      prior_comp_file = prior_comp_file,
      proto_col = "Prototype_initial",     # 注意：里面会把 NA 在最终写回时变为 "Unknown"
      macro_col = macro_col,
      macro_levels = c("Connective","Dead","Epithelial","Inflammatory","Neoplastic","Unknown")
    )
    if (is.list(aln) && !is.null(aln$seu)) aln$seu else aln
  }, error = function(e){
    message("⚠️ align_prototypes_with_prior 失败：", conditionMessage(e),
            "；暂用 Prototype_initial 作为 Prototype_aligned。")
    if (!("Prototype_aligned" %in% colnames(seu@meta.data))) {
      seu$Prototype_aligned <- seu$Prototype_initial
    }
    seu
  })
  
  seu_out
}




ensure_macroclass <- function(
  seu, src_col,                 # 细粒度细胞类型所在列名，比如 "cell_type" / "annot"
  mapping,                      # 你给的 named vector
  levels_macro = c("Connective","Dead","Epithelial","Inflammatory","Neoplastic","Unknown"),
  out_col = "MacroClass"
){
  stopifnot(inherits(seu, "Seurat"))
  if (out_col %in% colnames(seu@meta.data)) {
    message("MacroClass 已存在：跳过映射（使用现有列：", out_col, "）。")
    # 规范 level
    cur <- as.character(seu[[out_col]][,1])
    cur[is.na(cur) | cur == ""] <- "Unknown"
    seu[[out_col]] <- factor(cur, levels = levels_macro)
    return(seu)
  }
  stopifnot(src_col %in% colnames(seu@meta.data))
  ori <- as.character(seu[[src_col]][,1])
  mm  <- unname(mapping[ori])
  mm[is.na(mm) | mm == ""] <- "Unknown"
  cov <- mean(mm != "Unknown")
  message(sprintf("MacroClass 映射覆盖率：%.1f%%  (n=%d/%d)，其余设为 Unknown",
                  100*cov, sum(mm!="Unknown"), length(mm)))
  seu[[out_col]] <- factor(mm, levels = levels_macro)
  
  # 小检查（可选）
  tb <- sort(table(seu[[out_col]][,1]), decreasing = TRUE)
  message("MacroClass 分布："); print(tb)
  seu
}





derive_spatiallike_from_ucell <- function(
  seu, marker_bank, proto_col = "Prototype_aligned", delta_thr = 0.10,
  maxRank = 30000, chunk.size = 2000, trim_per_set = 200
){
  stopifnot(inherits(seu, "Seurat"))
  
  get_set <- function(pid, tag) {
    g <- marker_bank[[pid]][[tag]]
    if (is.null(g)) g <- character(0)
    g <- unique(toupper(g))
    if (is.finite(trim_per_set) && trim_per_set > 0 && length(g) > trim_per_set) g <- g[seq_len(trim_per_set)]
    g
  }
  
  gs <- list(
    CORE_P0 = get_set("Prototype0","core"),
    EDGE_P0 = get_set("Prototype0","edge"),
    CORE_P2 = get_set("Prototype2","core"),
    EDGE_P2 = get_set("Prototype2","edge"),
    CORE_P4 = get_set("Prototype4","core"),
    EDGE_P4 = get_set("Prototype4","edge")
  )
  gs <- gs[vapply(gs, length, integer(1)) >= 3]
  if (!length(gs)) stop("Spatial-like 打分的签名为空，请检查 marker_bank$<pid>$core/edge")
  
  M <- Seurat::GetAssayData(seu, slot = "data")
  sc <- UCell::ScoreSignatures_UCell(M, features = gs, maxRank = maxRank, chunk.size = chunk.size)
  
  getc <- function(nm) {
    cn <- paste0(nm, "_UCell")
    if (cn %in% colnames(sc)) sc[, cn] else rep(0, ncol(M))
  }
  
  d0 <- getc("CORE_P0") - getc("EDGE_P0")
  d2 <- getc("CORE_P2") - getc("EDGE_P2")
  d4 <- getc("CORE_P4") - getc("EDGE_P4")
  
  DEL <- cbind(Prototype0 = d0, Prototype2 = d2, Prototype4 = d4)
  rownames(DEL) <- colnames(seu)
  
  idx <- max.col(abs(DEL), ties.method = "first")
  best_val <- DEL[cbind(seq_len(nrow(DEL)), idx)]
  
  lab <- rep("Other", nrow(DEL))
  lab[best_val >=  delta_thr] <- "Core"
  lab[best_val <= -delta_thr] <- "Edge"
  # 夹在中间的一些给 "Interface"
  eps <- max(0.05, 0.5 * delta_thr)
  lab[abs(best_val) < delta_thr & abs(best_val) >= eps] <- "Interface"
  
  seu$SpatialLike <- factor(lab, levels = c("Core","Interface","Edge","Other"))
  seu
}



## --------------------------
## 并行
## --------------------------
n_cores <- max(1, parallel::detectCores() - 2)
future::plan(future::multisession, workers = n_cores)
options(future.globals.maxSize = 50 * 1024^3)
BPPARAM_gsva <- BiocParallel::SnowParam(workers = min(max(1, n_cores), 4), progressbar = TRUE)
BiocParallel::register(BPPARAM_gsva, default = TRUE)

# 一个小助手：确保某段代码强制顺序执行，执行完再恢复原 plan
with_sequential <- function(expr) {
  old <- future::plan()
  on.exit(try(future::plan(old), silent = TRUE), add = TRUE)
  future::plan(future::sequential)
  force(expr)
}





## =========================================================
## 1) 读入方向性签名（pCR>0 / non_pCR<0）
## =========================================================
## 参数
## =========================================================
## BULK：严格方向性集合 + 秩打分(up−down) + 分层CV + 导出
## 只需：proto_sig_file_raw / bulk_expr_file / bulk_pheno_file / pheno_col / pcr_prefix / outdir / dataset_name
## =========================================================

suppressPackageStartupMessages({
  if (!requireNamespace("matrixStats", quietly=TRUE)) install.packages("matrixStats", repos="https://cloud.r-project.org")
  if (!requireNamespace("pROC",         quietly=TRUE)) install.packages("pROC",         repos="https://cloud.r-project.org")
  library(matrixStats); library(pROC)
})

## ---------- 工具函数 ----------
to_num   <- function(v) suppressWarnings(as.numeric(v))
canon    <- function(v){ v <- gsub("\\s+", " ", v); v <- gsub("\u00A0", " ", v, fixed=TRUE); trimws(v) }
clean_gene <- function(x){
  x <- as.character(x)
  x <- gsub("(\\||_|\\()+ENSG[0-9]+(\\.[0-9]+)?\\)?$", "", x)
  x <- gsub("ENSG[0-9]+(\\.[0-9]+)?$", "", x)
  x <- gsub("\\|.*$", "", x)
  x <- trimws(x)
  toupper(ifelse(nchar(x)==0, NA_character_, x))
}

## ---------- 0) 参数：原型与列名配置（按你的文件修改这里） ----------
use_pvalue_discovery <- FALSE
p_thr_discovery      <- 0.50
min_abs_log2fc       <- 0.00
proto_set   <- c("P0","P2","P4")           # 你要保留的原型（可改）
pairs_proto <- c("P0_vs_P2","P0_vs_P4","P2_vs_P4")

## A) core vs edge 的列名模板（按原始 CSV 实际列名调整）
col_coreedge_lfc <- function(p) paste0(p, "_core_vs_edge_log2fc")
col_coreedge_p   <- function(p) paste0(p, "_core_vs_edge_pvalue")
col_coreedge_q   <- function(p) paste0(p, "_core_vs_edge_qvalue")   # 可无

## B) pCR vs non-pCR 的列名模板
col_pcr_lfc <- function(p) paste0(p, "_pCR_log2fc")
col_pcr_p   <- function(p) paste0(p, "_pCR_pvalue")
col_pcr_q   <- function(p) paste0(p, "_pCR_qvalue")                 # 可无

## C) pair（同层面）的列名模板（layer = "core" / "edge"）
col_pair_lfc <- function(layer, pp) sprintf("%s_pair_%s_log2fc", layer, pp)
col_pair_p   <- function(layer, pp) sprintf("%s_pair_%s_pvalue", layer, pp)
col_pair_q   <- function(layer, pp) sprintf("%s_pair_%s_qvalue", layer, pp)  # 可无

## ---------- 1) 读取签名并“严格用 log2fc 正负定向” ----------
stopifnot(file.exists(proto_sig_file_raw))
sig <- read.csv(proto_sig_file_raw, check.names = FALSE)
stopifnot("gene" %in% names(sig))
sig$gene <- clean_gene(sig$gene)

map_proto_name <- function(p) paste0("Prototype", gsub("\\D","", p))

## A) core vs edge：p≤阈值 + log2fc 正负决定 Up_in_core / Up_in_edge
make_coreedge_block <- function(p){
  c_lfc <- col_coreedge_lfc(p); c_p <- col_coreedge_p(p); c_q <- col_coreedge_q(p)
  stopifnot(c_p %in% names(sig))
  lfc_vec <- if (c_lfc %in% names(sig)) to_num(sig[[c_lfc]]) else NA_real_
  p_vec   <- to_num(sig[[c_p]])
  if (use_pvalue_discovery) {
    keep <- is.finite(p_vec) & p_vec <= p_thr_discovery
  } else {
    keep <- is.finite(lfc_vec) & abs(lfc_vec) >= min_abs_log2fc
  }
  keep <- keep & !is.na(sig$gene) & nzchar(sig$gene)
  if (!any(keep)) return(data.frame())
  df <- data.frame(
    Prototype = map_proto_name(p),
    gene      = sig$gene[keep],
    log2fc    = lfc_vec[keep],
    p_value   = p_vec[keep],
    stringsAsFactors = FALSE
  )
  if (c_q %in% names(sig)) df$q_value <- to_num(sig[[c_q]])[keep]
  df$Direction <- ifelse(is.finite(df$log2fc) & df$log2fc > 0, "Up_in_core",
                         ifelse(is.finite(df$log2fc) & df$log2fc < 0, "Up_in_edge", NA_character_))
  df <- df[!is.na(df$Direction), , drop=FALSE]
  df
}
proto_keep_coreedge <- do.call(rbind, lapply(proto_set, make_coreedge_block))



## B) pCR vs non-pCR：同理 Up_in_pCR / Up_in_non_pCR
make_pcr_block <- function(p){
  c_lfc <- col_pcr_lfc(p); c_p <- col_pcr_p(p); c_q <- col_pcr_q(p)
  stopifnot(c_p %in% names(sig))
  lfc_vec <- if (c_lfc %in% names(sig)) to_num(sig[[c_lfc]]) else NA_real_
  p_vec   <- to_num(sig[[c_p]])
  if (use_pvalue_discovery) {
    keep <- is.finite(p_vec) & p_vec <= p_thr_discovery
  } else {
    keep <- is.finite(lfc_vec) & abs(lfc_vec) >= min_abs_log2fc
  }
  keep <- keep & !is.na(sig$gene) & nzchar(sig$gene)
  if (!any(keep)) return(data.frame())
  df <- data.frame(
    Prototype = map_proto_name(p),
    gene      = sig$gene[keep],
    log2fc    = lfc_vec[keep],
    p_value   = p_vec[keep],
    stringsAsFactors = FALSE
  )
  if (c_q %in% names(sig)) df$q_value <- to_num(sig[[c_q]])[keep]
  df$Direction <- ifelse(is.finite(df$log2fc) & df$log2fc > 0, "Up_in_pCR",
                         ifelse(is.finite(df$log2fc) & df$log2fc < 0, "Up_in_non_pCR", NA_character_))
  df <- df[!is.na(df$Direction), , drop=FALSE]
  df
}
proto_keep_pCR <- do.call(rbind, lapply(proto_set, make_pcr_block))



## C) pair（同层面）：lfc>0 表示 left>right
make_pair_block <- function(layer = c("core","edge")) {
  layer <- match.arg(layer)
  out_list <- list()
  for (pp in pairs_proto) {
    c_lfc <- col_pair_lfc(layer, pp); c_p <- col_pair_p(layer, pp); c_q <- col_pair_q(layer, pp)
    if (!(c_p %in% names(sig))) next
    lfc_vec <- if (c_lfc %in% names(sig)) to_num(sig[[c_lfc]]) else NA_real_
    p_vec   <- to_num(sig[[c_p]])
    if (use_pvalue_discovery) {
      keep <- is.finite(p_vec) & p_vec <= p_thr_discovery
    } else {
      keep <- is.finite(lfc_vec) & abs(lfc_vec) >= min_abs_log2fc
    }
    keep <- keep & !is.na(sig$gene) & nzchar(sig$gene)
    if (!any(keep)) next
    left  <- sub("_vs_.*$", "", pp)
    right <- sub("^.*_vs_", "", pp)
    sub <- data.frame(
      Prototype   = paste0("Prototype", gsub("\\D","", left)),  # 视角=left
      Group_left  = left, Group_right = right, Layer = layer,
      gene        = sig$gene[keep],
      log2fc      = lfc_vec[keep],
      p_value     = p_vec[keep],
      stringsAsFactors = FALSE
    )
    sub$Direction <- ifelse(is.finite(sub$log2fc) & sub$log2fc > 0, paste0("Up_in_", left, "_", layer),
                            ifelse(is.finite(sub$log2fc) & sub$log2fc < 0, paste0("Up_in_", right, "_", layer),
                                   NA_character_))
    sub <- sub[!is.na(sub$Direction), , drop=FALSE]
    if (nrow(sub)) out_list[[pp]] <- sub
  }
  if (length(out_list)) do.call(rbind, out_list) else data.frame()
}
proto_keep_corepair <- make_pair_block("core")
proto_keep_edgepair <- make_pair_block("edge")

message("✅ core_edge rows = ", nrow(proto_keep_coreedge))
message("✅ pCR rows      = ", nrow(proto_keep_pCR))
message("✅ core_pair rows= ", nrow(proto_keep_corepair))
message("✅ edge_pair rows= ", nrow(proto_keep_edgepair))



cell_type_mapping <- c(
  'CAFs MSC iCAF-like s1'='Connective','CAFs MSC iCAF-like s2'='Connective',
  'CAFs Transitioning s3'='Connective','CAFs myCAF like s4'='Connective',
  'CAFs myCAF like s5'='Connective','Cycling PVL'='Connective',
  'Endothelial ACKR1'='Connective','Endothelial CXCL12'='Connective',
  'Endothelial Lymphatic LYVE1'='Connective','Endothelial RGS5'='Connective',
  'Myoepithelial'='Connective','PVL Differentiated s3'='Connective',
  'PVL Immature s1'='Connective','PVL_Immature s2'='Connective',
  'Myeloid_c5_Macrophage_3_SIGLEC1'='Inflammatory','Myeloid_c7_Monocyte_3_FCGR3A'='Inflammatory',
  'Luminal Progenitors'='Epithelial','Mature Luminal'='Epithelial',
  'B cells Memory'='Inflammatory','B cells Naive'='Inflammatory','Cycling_Myeloid'='Inflammatory',
  'Myeloid_c0_DC_LAMP3'='Inflammatory','Myeloid_c10_Macrophage_1_EGR1'='Inflammatory',
  'Myeloid_c11_cDC2_CD1C'='Inflammatory','Myeloid_c12_Monocyte_1_IL1B'='Inflammatory',
  'Myeloid_c1_LAM1_FABP5'='Inflammatory','Myeloid_c2_LAM2_APOE'='Inflammatory',
  'Myeloid_c3_cDC1_CLEC9A'='Inflammatory','Myeloid_c4_DCs_pDC_IRF7'='Inflammatory',
  'Myeloid_c8_Monocyte_2_S100A9'='Inflammatory','Myeloid_c9_Macrophage_2_CXCL10'='Inflammatory',
  'Plasmablasts'='Inflammatory','T_cells_c0_CD4+_CCR7'='Inflammatory','T_cells_c10_NKT_cells_FCGR3A'='Inflammatory',
  'T_cells_c11_MKI67'='Inflammatory','T_cells_c1_CD4+_IL7R'='Inflammatory','T_cells_c2_CD4+_T-regs_FOXP3'='Inflammatory',
  'T_cells_c3_CD4+_Tfh_CXCL13'='Inflammatory','T_cells_c4_CD8+_ZFP36'='Inflammatory','T_cells_c5_CD8+_GZMK'='Inflammatory',
  'T_cells_c6_IFIT1'='Inflammatory','T_cells_c7_CD8+_IFNG'='Inflammatory','T_cells_c8_CD8+_LAG3'='Inflammatory',
  'T_cells_c9_NK_cells_AREG'='Inflammatory',
  'Cancer Basal SC'='Neoplastic','Cancer Cycling'='Neoplastic','Cancer Her2 SC'='Neoplastic',
  'Cancer LumA SC'='Neoplastic','Cancer LumB SC'='Neoplastic'
)



suppressPackageStartupMessages({ library(pROC) })

tune_combo_weights_fixeddir <- function(
  B_identity, B_spatial, y_vec,
  pid_id = "Prototype4",   # 身份分量用哪个原型
  pid_sp = "Prototype2",   # 空间分量用哪个原型
  sp_sign = -1,            # 空间分量固定方向：-1 表示取负
  step = 0.01, K = 5, seed = 1,
  robust = TRUE            # 是否用稳健标准化：median/MAD
){
  stopifnot(requireNamespace("pROC", quietly = TRUE))
  auc_safely <- function(y, s) {
    out <- try(pROC::auc(pROC::roc(y, s, quiet = TRUE, direction = "auto")), silent = TRUE)
    if (inherits(out, "try-error")) NA_real_ else as.numeric(out)
  }
  zfit <- function(x, robust=TRUE){
    if (robust) {
      m <- stats::median(x, na.rm = TRUE)
      s <- stats::mad(x, constant = 1.4826, na.rm = TRUE)
      if (!is.finite(s) || s == 0) s <- stats::sd(x, na.rm = TRUE)
    } else {
      m <- mean(x, na.rm = TRUE); s <- sd(x, na.rm = TRUE)
    }
    if (!is.finite(s) || s == 0) s <- 1
    list(mu=m, sd=s)
  }
  zapply <- function(x, fit) (x - fit$mu) / fit$sd
  
  # 统一样本并取两个分量（空间分量锁定方向）
  smp <- Reduce(intersect, list(rownames(B_identity), rownames(B_spatial), names(y_vec)))
  y   <- as.numeric(y_vec[smp])
  X   <- data.frame(
    P4_id = B_identity[smp, pid_id],
    P2_sp = sp_sign * B_spatial[smp, pid_sp],
    row.names = smp, check.names = FALSE
  )
  keep <- stats::complete.cases(X, y) & is.finite(y)
  X <- X[keep, , drop = FALSE]; y <- y[keep]
  if (length(unique(y)) < 2 || nrow(X) < 30) stop("有效样本不足。")
  
  set.seed(seed)
  folds <- sample(rep(1:K, length.out = nrow(X)))
  ws <- seq(0, 1, by = step)
  auc_mat <- matrix(NA_real_, nrow = K, ncol = length(ws))
  colnames(auc_mat) <- paste0("w=", ws)
  
  for (i in 1:K) {
    tr <- folds != i; te <- !tr
    xtr <- X[tr, , drop = FALSE]; ytr <- y[tr]
    xte <- X[te, , drop = FALSE]; yte <- y[te]
    
    # 训练折拟合标准化，再作用于测试折（防泄露）
    f_id <- zfit(xtr$P4_id, robust); f_sp <- zfit(xtr$P2_sp, robust)
    z_id_te <- zapply(xte$P4_id, f_id)
    z_sp_te <- zapply(xte$P2_sp, f_sp)
    
    for (j in seq_along(ws)) {
      w <- ws[j]
      s <- w * z_id_te + (1 - w) * z_sp_te
      auc_mat[i, j] <- auc_safely(yte, s)
    }
  }
  
  auc_cv <- colMeans(auc_mat, na.rm = TRUE)
  jbest  <- which.max(auc_cv)
  w_best <- ws[jbest]
  
  # 全体样本一次性标准化（仅报告用，偏乐观）
  f_id_all <- zfit(X$P4_id, robust); f_sp_all <- zfit(X$P2_sp, robust)
  Z_id_all <- zapply(X$P4_id, f_id_all)
  Z_sp_all <- zapply(X$P2_sp, f_sp_all)
  S_best   <- w_best * Z_id_all + (1 - w_best) * Z_sp_all
  
  list(
    w_best       = w_best,                # α：给 P4_id 的权重
    auc_cv_best  = as.numeric(auc_cv[jbest]),
    auc_full     = auc_safely(y, S_best),
    score_all    = setNames(as.numeric(S_best), rownames(X)),
    grid         = ws,
    cv_auc_curve = auc_cv
  )
}

stopifnot(requireNamespace("pROC", quietly = TRUE))

.tz_fit   <- function(x, robust=TRUE){ 
  if (robust){ m <- stats::median(x, na.rm=TRUE); s <- stats::mad(x, 1.4826, na.rm=TRUE); if(!is.finite(s)||s==0) s <- stats::sd(x,na.rm=TRUE)
  } else { m <- mean(x,na.rm=TRUE); s <- sd(x,na.rm=TRUE) }
  if(!is.finite(s)||s==0) s <- 1
  list(mu=m, sd=s)
}
.tz_apply <- function(x, f) (x - f$mu) / f$sd
.auc_safe <- function(y, s){
  ro <- try(pROC::roc(y, s, quiet = TRUE, direction = "auto"), silent = TRUE)
  if (inherits(ro, "try-error")) list(roc=NULL, auc=NA_real_, ci=c(NA,NA,NA))
  else list(roc=ro, auc=as.numeric(pROC::auc(ro)), ci=as.numeric(pROC::ci.auc(ro)))
}
.rdirichlet <- function(n, k, alpha = rep(1, k)){
  M <- matrix(stats::rgamma(n * k, shape = alpha), nrow = n, ncol = k)
  M / rowSums(M)
}

tune_combo_weights_9_fixeddir <- function(
  B_identity, B_spatial, B_functional, y_vec,
  pids   = c("Prototype0","Prototype2","Prototype4"),
  signs_id = c(+1, +1, +1),    # +1同向，-1反向
  signs_sp = c(+1, -1, +1),    # 通常P2空间取负，其它同向
  signs_fn = c(+1, +1, +1),
  K = 5, seed = 1, robust = TRUE,
  search = c("dirichlet","grid"),
  n_try = 2000,        # Dirichlet 候选数
  grid_step = 0.25     # grid很粗；仅调试时用
){
  search <- match.arg(search)
  
  ## 对齐样本
  smp <- Reduce(intersect, list(rownames(B_identity), rownames(B_spatial), rownames(B_functional), names(y_vec)))
  if (!length(smp)) stop("B_identity/B_spatial/B_functional 与 y_vec 无共同样本。")
  y <- as.numeric(y_vec[smp])
  
  ## 只用存在于三条轨道的共同原型
  pid_ok <- Reduce(intersect, list(colnames(B_identity), colnames(B_spatial), colnames(B_functional)))
  pids <- intersect(pids, pid_ok)
  if (length(pids) < 2) stop("可用原型不足2个。")
  
  ## 对齐 sign 向量长度（支持长度1回收，也支持具名向量）
  norm_signs <- function(sig){
    if (is.null(names(sig))) {
      if (length(sig) == 1) rep(sig, length(pids)) else sig
    } else {
      sig[pids]
    }
  }
  signs_id <- norm_signs(signs_id); if (length(signs_id) != length(pids)) stop("signs_id 长度与 pids 不匹配")
  signs_sp <- norm_signs(signs_sp); if (length(signs_sp) != length(pids)) stop("signs_sp 长度与 pids 不匹配")
  signs_fn <- norm_signs(signs_fn); if (length(signs_fn) != length(pids)) stop("signs_fn 长度与 pids 不匹配")
  
  mk_block <- function(B, sgn, tag){
    out <- do.call(cbind, lapply(seq_along(pids), function(i) sgn[i] * as.numeric(B[smp, pids[i]])))
    colnames(out) <- paste0(tag, ":", pids)
    out
  }
  X_id <- mk_block(B_identity,   signs_id, "id")
  X_sp <- mk_block(B_spatial,    signs_sp, "sp")
  X_fn <- mk_block(B_functional, signs_fn, "fn")
  X <- cbind(X_id, X_sp, X_fn)
  
  keep <- stats::complete.cases(X, y) & is.finite(y)
  X <- X[keep,,drop=FALSE]; y <- y[keep]
  if (nrow(X) < 40 || length(unique(y)) < 2) stop("有效样本不足。")
  
  ## 候选权重（单纯形）
  d <- ncol(X)
  W <- if (search == "dirichlet") {
    .rdirichlet(n_try, d, alpha = rep(1, d))
  } else {
    ws <- seq(0, 1, by = grid_step)
    G  <- expand.grid(rep(list(ws), d))
    G  <- as.matrix(G[rowSums(G) > 0 & abs(rowSums(G) - 1) < 1e-9, , drop = FALSE])
    if (nrow(G) == 0) stop("grid_step 太粗，生成不了单纯形点。调小 grid_step。")
    G
  }
  
  ## K 折 CV
  set.seed(seed)
  folds <- sample(rep(1:K, length.out = nrow(X)))
  auc_mat <- matrix(NA_real_, nrow = K, ncol = nrow(W))
  
  for (i in 1:K) {
    tr <- folds != i; te <- !tr
    Xtr <- X[tr,,drop=FALSE]; ytr <- y[tr]
    Xte <- X[te,,drop=FALSE]; yte <- y[te]
    
    fits <- lapply(seq_len(ncol(Xtr)), function(j) .tz_fit(Xtr[, j], robust))
    Zte  <- sapply(seq_len(ncol(Xte)), function(j) .tz_apply(Xte[, j], fits[[j]]))
    Zte  <- matrix(Zte, nrow = sum(te), ncol = ncol(Xte))
    
    S <- Zte %*% t(W)        # n_te × n_try
    for (j in seq_len(ncol(S))) {
      auc_mat[i, j] <- .auc_safe(yte, S[, j])$auc
    }
  }
  
  auc_cv <- colMeans(auc_mat, na.rm = TRUE)
  jbest  <- which.max(auc_cv)
  w_best <- W[jbest, ]
  
  ## Full-data 展示分数
  fits_all <- lapply(seq_len(ncol(X)), function(j) .tz_fit(X[, j], robust))
  Zall     <- sapply(seq_len(ncol(X)), function(j) .tz_apply(X[, j], fits_all[[j]]))
  Zall     <- matrix(Zall, nrow = nrow(X), ncol = ncol(X), dimnames = list(rownames(X), colnames(X)))
  S_best   <- as.numeric(Zall %*% w_best)
  
  list(
    w_best       = stats::setNames(w_best, colnames(X)),
    auc_cv_best  = as.numeric(auc_cv[jbest]),
    auc_full     = .auc_safe(y, S_best)$auc,
    score_all    = stats::setNames(S_best, rownames(X)),
    candidates   = W,
    cv_auc_curve = auc_cv
  )
}


tune_combo_glmnet <- function(
  Z, y, seed = 1,
  alphas = seq(0, 1, by = 0.1),   # Ridge→Lasso
  nonneg = TRUE,                       # 是否强制系数非负
  balance = TRUE                       # 是否用类权重平衡阳/阴样本
){
  stopifnot(requireNamespace("glmnet", quietly = TRUE),
            requireNamespace("pROC",   quietly = TRUE))
  set.seed(seed)
  Zm <- as.matrix(Z); ym <- as.numeric(y)
  # 去掉无方差/全NA列，避免数值问题
  keepcol <- apply(Zm, 2, function(v) sd(v, na.rm = TRUE) > 0 && sum(is.finite(v)) >= 5)
  Zm <- Zm[, keepcol, drop = FALSE]
  if (ncol(Zm) == 0) stop("特征全被过滤了。")
  # 类别权重（阳性/阴性各占总权重 0.5）
  ww <- NULL
  if (isTRUE(balance)) {
    n1 <- sum(ym == 1, na.rm = TRUE); n0 <- sum(ym == 0, na.rm = TRUE)
    ww <- ifelse(ym == 1, 0.5 / max(n1, 1), 0.5 / max(n0, 1))
  }
  best <- list(auc = -Inf)
  for (a in alphas) {
    cv <- glmnet::cv.glmnet(
      Zm, ym, family = "binomial",
      alpha = a,
      type.measure = "auc",
      standardize = FALSE,             # 你传入的是稳健 z，已标准化
      lower.limits = if (nonneg) 0 else -Inf,
      upper.limits = Inf,
      weights = ww
    )
    auc <- max(cv$cvm, na.rm = TRUE)
    if (auc > best$auc) best <- list(auc = auc, cv = cv, alpha = a)
  }
  beta <- as.numeric(stats::coef(best$cv, s = "lambda.min"))[-1]  # 去截距
  names(beta) <- colnames(Zm)
  if (isTRUE(nonneg)) beta[beta < 0] <- 0
  # 归一到“权重和=1”，以便接到你的 reporter
  if (sum(beta) == 0) {
    w_best <- rep(1 / ncol(Zm), ncol(Zm)); names(w_best) <- colnames(Zm)
  } else {
    w_best <- beta / sum(beta)
  }
  S <- as.numeric(Zm %*% w_best)
  ro <- pROC::roc(ym, S, quiet = TRUE, direction = "auto")
  list(
    w_best = w_best,
    auc_cv_best = best$auc,
    auc_full = as.numeric(pROC::auc(ro)),
    score_all = stats::setNames(S, rownames(Zm))
  )
}



## ============ 9 权重自动报告（画 ROC + 出 AUC 表） ============
auto_combo_report_9 <- function(X_mat, y, w_best, outdir=".", prefix="Bulk_combo_9feat", top_single=6){
  stopifnot(requireNamespace("readr", quietly = TRUE))
  ## 单分量
  one_rocs <- lapply(colnames(X_mat), function(nm) .auc_safe(y, X_mat[, nm]))
  names(one_rocs) <- colnames(X_mat)
  df_one <- do.call(rbind, lapply(names(one_rocs), function(nm){
    data.frame(Feature=nm, AUC=one_rocs[[nm]]$auc,
               CI_low=one_rocs[[nm]]$ci[1], CI_high=one_rocs[[nm]]$ci[3], stringsAsFactors = FALSE)
  }))
  df_one <- df_one[order(-df_one$AUC), , drop = FALSE]
  
  ## 等权
  weq <- rep(1/ncol(X_mat), ncol(X_mat)); names(weq) <- colnames(X_mat)
  Seq <- as.numeric(as.matrix(X_mat) %*% weq)
  roc_eq <- .auc_safe(y, Seq)
  
  ## tuned
  w_best <- w_best / sum(w_best)
  St <- as.numeric(as.matrix(X_mat) %*% as.numeric(w_best))
  roc_tuned <- .auc_safe(y, St)
  
  ## 出图
  fn_png <- file.path(outdir, paste0(prefix, "_ROC.png"))
  png(fn_png, 980, 760, res=150)
  plot(roc_tuned$roc, col="#e41a1c", lwd=2.8, legacy.axes=TRUE,
       main=sprintf("9-Feature Combo (tuned); AUC=%.3f", roc_tuned$auc))
  plot(roc_eq$roc,   col="#ff7f00", lwd=2.2, add=TRUE)
  
  ## 只叠加 top_single 个单分量，避免图例太长
  topN <- head(df_one$Feature, min(top_single, nrow(df_one)))
  pal <- setNames(colorRampPalette(c("#377eb8","#4daf4a","#984ea3","#e41a1c","#ff7f00","#a65628","#f781bf","#999999"))(length(topN)), topN)
  for (nm in topN) plot(one_rocs[[nm]]$roc, col=pal[[nm]], lwd=2.0, add=TRUE)
  
  leg <- c(sprintf("Tuned AUC=%.3f", roc_tuned$auc),
           sprintf("Equal AUC=%.3f", roc_eq$auc),
           paste0(topN, " AUC=", sprintf("%.3f", df_one$AUC[match(topN, df_one$Feature)])))
  leg_col <- c("#e41a1c", "#ff7f00", pal[topN])
  leg_lwd <- c(2.8, 2.2, rep(2.0, length(topN)))
  legend("bottomright", leg, col=leg_col, lwd=leg_lwd, bty="n")
  dev.off()
  
  ## AUC 表
  rows <- rbind(
    df_one,
    data.frame(Feature="Combo_equal(9)", AUC=roc_eq$auc, CI_low=roc_eq$ci[1], CI_high=roc_eq$ci[3]),
    data.frame(Feature=paste0("Combo_tuned[", paste(sprintf("%s=%.3f", names(w_best), w_best), collapse=", "), "]"),
               AUC=roc_tuned$auc, CI_low=roc_tuned$ci[1], CI_high=roc_tuned$ci[3])
  )
  fn_csv <- file.path(outdir, paste0(prefix, "_AUC_table.csv"))
  readr::write_csv(rows, fn_csv)
  
  list(file_png=fn_png, file_csv=fn_csv, table=rows,
       score_equal=stats::setNames(Seq, rownames(X_mat)),
       score_tuned=stats::setNames(St,  rownames(X_mat)))
}








## ---------- 3) 读取 bulk + pCR 并对齐 ----------
stopifnot(file.exists(bulk_expr_file), file.exists(bulk_pheno_file))
bulk_expr <- read.csv(bulk_expr_file, row.names=1, check.names=FALSE)
pheno_raw <- read.csv(bulk_pheno_file, row.names=1, check.names=FALSE)
pheno     <- if (nrow(pheno_raw) < ncol(pheno_raw)) as.data.frame(t(pheno_raw)) else pheno_raw



## pCR 解析（使用提供的 pheno_col / pcr_prefix）
parse_pcr_from_pheno <- function(pheno, pheno_col, pcr_prefix="^pcr:\\s*",
                                 yes_pat="^(yes|y|pcr)$", no_pat="^(no|n|nonpcr)$"){
  stopifnot(pheno_col %in% colnames(pheno))
  v <- tolower(as.character(pheno[[pheno_col]]))
  v <- sub(pcr_prefix, "", v, ignore.case=TRUE)
  v <- sub("[;|,].*$", "", v); v <- trimws(v)
  ifelse(grepl(yes_pat, v), "pCR", ifelse(grepl(no_pat, v), "non_pCR", NA_character_))
}
pheno$pcr_response <- parse_pcr_from_pheno(pheno, pheno_col, pcr_prefix)

## 名称规范化并对齐
colnames(bulk_expr) <- canon(colnames(bulk_expr))
rownames(pheno)     <- canon(rownames(pheno))
common_samples <- intersect(colnames(bulk_expr), rownames(pheno))
if (length(common_samples) < 10) stop("❌ bulk 与 pheno 交集样本太少。")
bulk_expr <- as.matrix(bulk_expr[, common_samples, drop=FALSE])
pheno     <- pheno[common_samples, , drop=FALSE]
message("✅ 对齐样本数 = ", length(common_samples),
        "；pCR 分布：", paste(capture.output(print(table(pheno$pcr_response, useNA="ifany"))), collapse=" "))



## ---------- 4) 基因名标准化、合并重复符号（行均值） ----------
rn_raw   <- rownames(bulk_expr)
sym      <- clean_gene(rn_raw)
keep_row <- !is.na(sym) & nzchar(sym)
bulk_expr2 <- bulk_expr[keep_row, , drop=FALSE]
sym2        <- sym[keep_row]
grp <- factor(sym2)
sum_mat <- rowsum(bulk_expr2, group = grp, reorder = FALSE)
cnt_vec <- as.numeric(rowsum(matrix(1, nrow=nrow(bulk_expr2), ncol=1), group=grp, reorder=FALSE)[,1])
bulk_mat <- sum_mat / cnt_vec
bulk_mat <- bulk_mat[rowSums(is.finite(bulk_mat))>0 & rowSums(bulk_mat!=0, na.rm=TRUE)>0, , drop=FALSE]
message("✅ bulk 维度（gene × sample）= ", nrow(bulk_mat), " × ", ncol(bulk_mat))



# —— 构建 data 层（不依赖并行、不调用 NormalizeData），稀疏友好 —— 
ensure_logdata <- function(obj) {
  assy <- Seurat::DefaultAssay(obj)
  
  # 若已有 data 层就直接返回
  M <- try(Seurat::GetAssayData(obj, assay = assy, slot = "data"), silent = TRUE)
  if (!inherits(M, "try-error") && !is.null(M) && nrow(M) > 0) return(obj)
  
  message("⚙️ 手工从 counts 生成 data 层（CPM×1e4 后 log1p），不使用并行 ...")
  C <- Seurat::GetAssayData(obj, assay = assy, slot = "counts")
  if (inherits(C, "try-error") || is.null(C) || nrow(C) == 0)
    stop("counts 层为空，无法构建 data 层。")
  
  # 稀疏列归一化：每列 / colSums ×1e4，再只对非零条目做 log1p（保持稀疏）
  C <- methods::as(C, "dgCMatrix")
  cs <- Matrix::colSums(C)
  cs[!is.finite(cs) | cs <= 0] <- 1
  D <- Matrix::t(Matrix::t(C) / cs) * 1e4
  D@x <- log1p(D@x)  # 仅对非零 log1p，内存友好
  
  obj <- Seurat::SetAssayData(obj, assay = assy, slot = "data", new.data = D)
  return(obj)
}



## ---------- 5) 打分与相关：最宽松发现 → 单细胞补强 → bulk 关联 ----------
## =========================
## 1) 三层 marker bank（用方向，不限 p）
## =========================
build_marker_bank_from_dfs <- function(
  proto_keep_corepair, proto_keep_edgepair, proto_keep_coreedge, proto_keep_pCR,
  proto_ids = c("Prototype0","Prototype2","Prototype4")
){
  bank <- lapply(proto_ids, function(pid){
    k <- gsub("\\D", "", pid)
    up_core <- proto_keep_corepair$gene[
      proto_keep_corepair$Direction == paste0("Up_in_P", k, "_core")
    ]
    up_edge <- proto_keep_edgepair$gene[
      proto_keep_edgepair$Direction == paste0("Up_in_P", k, "_edge")
    ]
    identity_up <- sort(unique(toupper(c(up_core, up_edge))))
    others <- setdiff(proto_ids, pid)
    other_up <- unique(unlist(lapply(others, function(op){
      kk <- gsub("\\D", "", op)
      c(
        proto_keep_corepair$gene[proto_keep_corepair$Direction == paste0("Up_in_P", kk, "_core")],
        proto_keep_edgepair$gene[proto_keep_edgepair$Direction == paste0("Up_in_P", kk, "_edge")]
      )
    })))
    identity_down <- sort(unique(setdiff(toupper(other_up), identity_up)))
    ce <- proto_keep_coreedge[proto_keep_coreedge$Prototype == pid, , drop = FALSE]
    core_markers <- sort(unique(toupper(ce$gene[ce$Direction == "Up_in_core"])))
    edge_markers <- sort(unique(toupper(ce$gene[ce$Direction == "Up_in_edge"])))
    pcr <- proto_keep_pCR[proto_keep_pCR$Prototype == pid, , drop = FALSE]
    pcr_up <- sort(unique(toupper(pcr$gene[pcr$Direction == "Up_in_pCR"])))
    pcr_down <- sort(unique(toupper(pcr$gene[pcr$Direction == "Up_in_non_pCR"])))
    list(
      identity_up = identity_up,
      identity_down = identity_down,
      core = core_markers,
      edge = edge_markers,
      pcr_up = pcr_up,
      pcr_down = pcr_down
    )
  })
  names(bank) <- proto_ids
  bank
}

## =========================
## 2) 单细胞补强/扩充 + 先验加权（label-free）
## =========================
refine_bank_with_sc <- function(
  seu, marker_bank, sc_celltype_col, proto_col = "Prototype_aligned",
  prior_comp_file = prior_comp_file_raw,
  top_n_per_ct = 80, min_cells = 40, min_pct = 0.10, logfc_th = 0.15
){
  stopifnot(inherits(seu, "Seurat"))
  stopifnot(sc_celltype_col %in% colnames(seu@meta.data))
  stopifnot(proto_col %in% colnames(seu@meta.data))
  
  Idents(seu) <- factor(seu[[proto_col]][,1])
  
  prior_mat <- read_prior_comp(
    prior_comp_file,
    macro_levels = c("Connective","Dead","Epithelial","Inflammatory","Neoplastic","Unknown")
  )
  
  ct_vec <- as.character(seu[[sc_celltype_col]][,1])
  mac_vec <- if ("MacroClass" %in% colnames(seu@meta.data)) as.character(seu$MacroClass) else rep("Unknown", ncol(seu))
  macro_map <- tapply(mac_vec, ct_vec, function(v){
    v <- v[!is.na(v)]
    if (!length(v)) return("Unknown")
    names(sort(table(v), decreasing = TRUE))[1]
  })
  macro_map <- unlist(macro_map)
  macro_map[!nzchar(macro_map)] <- "Unknown"
  
  add_w <- list()
  
  for (pid in names(marker_bank)) {
    for (set_tag in c("identity_up","identity_down","core","edge")) {
      genes_seed <- marker_bank[[pid]][[set_tag]]
      if (is.null(genes_seed)) genes_seed <- character()
      genes_seed <- unique(toupper(genes_seed))
      set_label <- switch(set_tag,
                          identity_up   = "IDENTITY_UP",
                          identity_down = "IDENTITY_DOWN",
                          core          = "CORE",
                          edge          = "EDGE"
      )
      set_name <- paste0(set_label, ":", pid)
      w_g <- numeric(0)
      
      for (ct_i in names(macro_map)) {
        cells_ct <- colnames(seu)[ct_vec == ct_i]
        if (length(cells_ct) < min_cells) next
        obj <- tryCatch(subset(seu, cells = cells_ct), error = function(e) NULL)
        if (is.null(obj) || ncol(obj) < min_cells) next
        Idents(obj) <- factor(obj[[proto_col]][,1])
        if (!(pid %in% levels(Idents(obj)))) next
        expr_mat <- NULL
        mk <- tryCatch(
          Seurat::FindMarkers(obj, ident.1 = pid, only.pos = FALSE,
                              min.pct = min_pct, logfc.threshold = logfc_th,
                              verbose = FALSE),
          error = function(e) NULL
        )
        if (is.null(mk) || !nrow(mk)) next
        mk$gene <- toupper(rownames(mk))
        if (set_tag == "identity_down") {
          cand <- mk[is.finite(mk$avg_log2FC) & mk$avg_log2FC < 0, , drop = FALSE]
          if (!nrow(cand)) next
          cand$score_sc <- abs(cand$avg_log2FC)
        } else {
          cand <- mk[is.finite(mk$avg_log2FC) & mk$avg_log2FC > 0, , drop = FALSE]
          if (!nrow(cand)) next
          cand$score_sc <- cand$avg_log2FC
        }
        cand <- cand[order(cand$score_sc, decreasing = TRUE), , drop = FALSE]
        cand <- head(cand, top_n_per_ct)
        w <- as.numeric(cand$score_sc)
        if (length(w) == 1L) {
          w[] <- 1
        } else {
          mn <- min(w, na.rm = TRUE)
          mx <- max(w, na.rm = TRUE)
          rng <- mx - mn
          if (!is.finite(rng) || rng <= 1e-6) {
            w[] <- 1
          } else {
            w <- (w - mn) / rng
          }
        }
        names(w) <- cand$gene
        mac <- macro_map[[ct_i]]
        if (is.null(mac) || !nzchar(mac)) mac <- "Unknown"
        if (!(mac %in% colnames(prior_mat))) mac <- "Unknown"
        if (pid %in% rownames(prior_mat) && mac %in% colnames(prior_mat)) {
          w <- w * prior_mat[pid, mac]
        }
        if (!length(w)) next
        w <- w[is.finite(w) & w > 0]
        if (!length(w)) next
        
        if (set_tag %in% c("core","edge") && length(genes_seed) >= 3) {
          if (is.null(expr_mat)) expr_mat <- as.matrix(Seurat::GetAssayData(obj, slot = "data"))
          seed_present <- intersect(genes_seed, rownames(expr_mat))
          if (length(seed_present) >= 3) {
            seed_signal <- colMeans(expr_mat[seed_present, , drop = FALSE])
            if (stats::sd(seed_signal) > 0) {
              gene_expr <- expr_mat[cand$gene, , drop = FALSE]
              cor_vals <- apply(gene_expr, 1, function(x){
                sx <- stats::sd(x)
                if (!is.finite(sx) || sx == 0) return(0)
                stats::cor(x, seed_signal, method = "pearson")
              })
              cor_vals[!is.finite(cor_vals)] <- 0
              cor_vals <- pmax(cor_vals, 0)
              w <- w * cor_vals[names(w)]
              w <- w[is.finite(w) & w > 0]
              if (!length(w)) next
            }
          }
        }
        
        w <- rowsum(matrix(w, ncol = 1, dimnames = list(names(w), "w")), group = names(w))[,1]
        if (!length(w_g)) {
          w_g <- w
        } else {
          for (g in names(w)) {
            if (g %in% names(w_g)) {
              w_g[g] <- w_g[g] + w[g]
            } else {
              w_g[g] <- w[g]
            }
          }
        }
      }
      
      if (length(w_g)) {
        w_g <- w_g[order(w_g, decreasing = TRUE)]
        add_w[[set_name]] <- data.frame(
          set = set_name,
          gene = names(w_g),
          weight = as.numeric(w_g),
          stringsAsFactors = FALSE
        )
        marker_bank[[pid]][[set_tag]] <- sort(unique(c(genes_seed, names(w_g))))
      }
    }
  }
  
  w_sc <- if (length(add_w)) dplyr::bind_rows(add_w) else data.frame(set = character(), gene = character(), weight = numeric())
  if (nrow(w_sc)) {
    w_sc$gene <- toupper(w_sc$gene)
  }
  list(bank = marker_bank, w_sc = w_sc)
}



fuse_three_marker_sources <- function(
  bank_seed,
  bank_sc,
  w_sc,
  proto_ids = c("Prototype0","Prototype2","Prototype4"),
  pcr_bonus = 0.6,
  pcr_malus = 0.5,
  cross_malus = 0.3,
  uniq_top_n = 200,
  min_core = 60,
  min_edge = 60
){
  stopifnot(length(proto_ids) > 0)
  
  bank <- bank_sc
  for (pid in proto_ids) {
    for (tg in c("identity_up","identity_down","core","edge","pcr_up","pcr_down")) {
      seed_vec <- unique(toupper(bank_seed[[pid]][[tg]]))
      if (length(seed_vec)) {
        cur <- unique(toupper(bank[[pid]][[tg]]))
        bank[[pid]][[tg]] <- sort(unique(c(cur, seed_vec)))
      } else if (is.null(bank[[pid]][[tg]])) {
        bank[[pid]][[tg]] <- character()
      }
    }
  }
  
  w_tab <- if (!is.null(w_sc) && nrow(w_sc)) {
    wtmp <- w_sc
    wtmp$gene <- toupper(wtmp$gene)
    wtmp
  } else data.frame(set=character(), gene=character(), weight=numeric())
  
  base_med <- if (nrow(w_tab)) stats::median(w_tab$weight, na.rm = TRUE) else 1
  add_up <- list()
  for (pid in proto_ids) {
    set_name <- paste0("IDENTITY_UP:", pid)
    present  <- toupper(w_tab$gene[w_tab$set == set_name])
    need     <- setdiff(toupper(bank[[pid]]$identity_up), present)
    if (length(need)) {
      add_up[[length(add_up)+1]] <- data.frame(
        set = set_name,
        gene = need,
        weight = rep(base_med * 0.6, length(need)),
        stringsAsFactors = FALSE
      )
    }
  }
  if (length(add_up)) w_tab <- rbind(w_tab, do.call(rbind, add_up))
  
  pcr_map_up   <- lapply(proto_ids, function(pid) unique(toupper(bank[[pid]]$pcr_up)))
  pcr_map_down <- lapply(proto_ids, function(pid) unique(toupper(bank[[pid]]$pcr_down)))
  names(pcr_map_up) <- names(pcr_map_down) <- proto_ids
  
  adj_chunks <- list()
  freq_all <- table(toupper(w_tab$gene))
  for (pid in proto_ids) {
    set_up <- paste0("IDENTITY_UP:", pid)
    sub <- w_tab[w_tab$set == set_up, , drop = FALSE]
    if (!nrow(sub)) next
    g <- toupper(sub$gene)
    w <- sub$weight
    
    if (length(pcr_map_up[[pid]]))   w[g %in% pcr_map_up[[pid]]]   <- w[g %in% pcr_map_up[[pid]]]   * (1 + pcr_bonus)
    if (length(pcr_map_down[[pid]])) w[g %in% pcr_map_down[[pid]]] <- w[g %in% pcr_map_down[[pid]]] * pmax(0, 1 - pcr_malus)
    
    others_up <- unique(unlist(pcr_map_up[setdiff(proto_ids, pid)], use.names = FALSE))
    if (length(others_up)) w[g %in% others_up] <- w[g %in% others_up] * pmax(0, 1 - cross_malus)
    
    rarity <- 1 / as.numeric(freq_all[g])
    rarity[!is.finite(rarity)] <- 1
    score <- w * rarity
    ord <- order(score, decreasing = TRUE)
    keep <- if (length(ord) > uniq_top_n) ord[seq_len(uniq_top_n)] else ord
    sub <- sub[keep, , drop = FALSE]
    adj_chunks[[length(adj_chunks)+1]] <- sub
    bank[[pid]]$identity_up <- sort(unique(sub$gene))
  }
  if (length(adj_chunks)) {
    kept_sets <- vapply(adj_chunks, function(x) unique(x$set), character(1))
    w_tab <- rbind(do.call(rbind, adj_chunks), w_tab[!(w_tab$set %in% kept_sets), , drop = FALSE])
  }
  
  for (pid in proto_ids) {
    others <- setdiff(proto_ids, pid)
    other_up <- unique(unlist(lapply(others, function(op) bank[[op]]$identity_up), use.names = FALSE))
    bank[[pid]]$identity_down <- sort(unique(setdiff(other_up, bank[[pid]]$identity_up)))
    
    cc <- unique(toupper(bank[[pid]]$core))
    if (length(cc) < min_core) {
      add <- setdiff(bank[[pid]]$identity_up, cc)
      bank[[pid]]$core <- sort(unique(c(cc, head(add, min_core - length(cc)))))
    }
    ee <- unique(toupper(bank[[pid]]$edge))
    if (length(ee) < min_edge) {
      add <- setdiff(setdiff(bank[[pid]]$identity_up, bank[[pid]]$core), ee)
      bank[[pid]]$edge <- sort(unique(c(ee, head(add, min_edge - length(ee)))))
    }
  }
  
  w_tab$gene <- toupper(w_tab$gene)
  w_func <- list()
  for (pid in proto_ids) {
    if (length(bank[[pid]]$pcr_up)) {
      w_func[[length(w_func)+1]] <- data.frame(set = paste0("PCR_UP:", pid),
                                               gene = toupper(bank[[pid]]$pcr_up),
                                               weight = 1, stringsAsFactors = FALSE)
    }
    if (length(bank[[pid]]$pcr_down)) {
      w_func[[length(w_func)+1]] <- data.frame(set = paste0("PCR_DOWN:", pid),
                                               gene = toupper(bank[[pid]]$pcr_down),
                                               weight = 1, stringsAsFactors = FALSE)
    }
  }
  if (length(w_func)) w_tab <- rbind(w_tab, do.call(rbind, w_func))
  
  list(bank = bank, gene_weight = w_tab)
}



ensure_umap_fast <- function(
  obj,
  red = "umap",
  dims_use = 1:15,
  subsample = 50000,
  n_neighbors = 20,
  min_dist = 0.5,
  n_epochs = 200,
  nn_method = c("annoy","rann","hnsw"),
  metric = "cosine",
  cache_dir = file.path(my_tempdir, "UMAP_cache"),
  seed = 123
){
  stopifnot(inherits(obj, "Seurat"))
  if (red %in% names(obj@reductions)) return(obj)
  
  nn_method <- match.arg(nn_method)
  dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)
  set.seed(seed)
  
  npcs_need <- max(30, max(dims_use))  # 30 够用了
  
  if (!"pca" %in% names(obj@reductions)) {
    if (length(Seurat::VariableFeatures(obj)) < 2000) {
      obj <- Seurat::FindVariableFeatures(
        obj, selection.method = "vst", nfeatures = 2000, verbose = FALSE
      )
    }
    
    ## 先尝试：分块 ScaleData，内存友好
    ok_pca <- FALSE
    try({
      obj <- Seurat::ScaleData(
        obj,
        features = Seurat::VariableFeatures(obj),
        verbose = FALSE,
        block.size = 1000
      )
      obj <- Seurat::RunPCA(
        obj,
        features = Seurat::VariableFeatures(obj),
        npcs = npcs_need,
        approx = TRUE,
        verbose = FALSE
      )
      ok_pca <- TRUE
    }, silent = TRUE)
    
    ## OOM 回退：直接在 data 上用 irlba 做 PCA
    if (!ok_pca) {
      if (!requireNamespace("irlba", quietly = TRUE))
        install.packages("irlba", repos = "https://cloud.r-project.org")
      feats <- Seurat::VariableFeatures(obj)
      D <- Seurat::GetAssayData(obj, slot = "data")
      D <- D[intersect(feats, rownames(D)), , drop = FALSE]
      X <- Matrix::t(D)
      Xd <- as.matrix(X)
      pc <- irlba::prcomp_irlba(Xd, n = npcs_need, center = TRUE, scale. = FALSE)
      emb <- pc$x
      rownames(emb) <- colnames(obj)
      colnames(emb) <- paste0("PC_", seq_len(ncol(emb)))
      obj[["pca"]] <- Seurat::CreateDimReducObject(
        embeddings = emb, key = "PC_", assay = Seurat::DefaultAssay(obj)
      )
      rm(Xd, X, pc); gc(FALSE)
    }
  }
  
  ## === 取 PCA，并做有限值检查（新增） ===
  pca_mat <- Seurat::Embeddings(obj, "pca")
  dims_use <- dims_use[dims_use <= ncol(pca_mat)]
  pca_mat <- pca_mat[, dims_use, drop = FALSE]
  stopifnot(ncol(pca_mat) >= 2)
  if (!all(is.finite(pca_mat))) {
    bad <- sum(!is.finite(pca_mat))
    stop("PCA 矩阵存在 NA/Inf（", bad, " 个元素），请先检查 ScaleData/VariableFeatures 或剔除异常细胞后再运行。")
  }
  
  if (!requireNamespace("uwot", quietly = TRUE)) install.packages("uwot", repos = "https://cloud.r-project.org")
  if (nn_method == "annoy" && !requireNamespace("RcppAnnoy", quietly = TRUE)) {
    try(utils::install.packages("RcppAnnoy", repos = "https://cloud.r-project.org"), silent = TRUE)
  }
  
  fkey <- paste(nrow(pca_mat), length(dims_use), subsample, n_neighbors, min_dist, n_epochs, nn_method, metric, sep = "_")
  coord_rds <- file.path(cache_dir, paste0("uwot_coords_", fkey, ".rds"))
  
  ## 读缓存
  if (file.exists(coord_rds)) {
    Y_cache <- try(readRDS(coord_rds), silent = TRUE)
    if (!inherits(Y_cache, "try-error") && is.matrix(Y_cache) &&
        nrow(Y_cache) == nrow(pca_mat) && ncol(Y_cache) == 2) {
      obj[[red]] <- Seurat::CreateDimReducObject(embeddings = Y_cache, key = "UMAP_", assay = Seurat::DefaultAssay(obj))
      return(obj)
    }
  }
  
  ## 训练子集 UMAP
  n_cells <- nrow(pca_mat)
  n_sub   <- min(subsample, n_cells)
  sub_idx <- if (n_sub < n_cells) sample.int(n_cells, n_sub) else seq_len(n_cells)
  X_sub   <- pca_mat[sub_idx, , drop = FALSE]
  
  message(sprintf("⏱ Training UMAP (sub %d/%d cells, method=%s, metric=%s)...",
                  n_sub, n_cells, nn_method, metric))
  umap_train <- uwot::umap(
    X_sub,
    n_neighbors = n_neighbors,
    min_dist = min_dist,
    n_components = 2,
    metric = metric,
    nn_method = nn_method,
    init = "random",
    n_epochs = n_epochs,
    ret_model = TRUE,
    n_threads = max(1, parallel::detectCores() - 1),
    verbose = TRUE
  )
  
  ## === 新增：兼容两种返回形式，并在无 model 时回退全量拟合 ===
  get_umap_model <- function(x) {
    if (is.list(x) && !is.null(x$model)) return(x$model)
    m <- attr(x, "model")
    if (!is.null(m)) return(m)
    NULL
  }
  get_umap_embedding <- function(x) {
    if (is.list(x) && !is.null(x$embedding)) return(x$embedding)
    x
  }
  
  umap_model     <- get_umap_model(umap_train)
  umap_embedding <- get_umap_embedding(umap_train)
  
  if (is.null(umap_model)) {
    warning("uwot 未返回可用 model；改为对全量 PCA 直接拟合 UMAP（不走 transform）。")
    Y_full <- uwot::umap(
      pca_mat,
      n_neighbors = n_neighbors,
      min_dist = min_dist,
      n_components = 2,
      metric = metric,
      nn_method = nn_method,
      init = "spectral",
      n_epochs = n_epochs,
      n_threads = max(1, parallel::detectCores() - 1),
      verbose = TRUE
    )
    Y_full <- as.matrix(Y_full)
    if (!all(is.finite(Y_full))) stop("UMAP 全量拟合结果存在 NA/Inf。")
    rownames(Y_full) <- rownames(pca_mat)
    saveRDS(Y_full, coord_rds)
    obj[[red]] <- Seurat::CreateDimReducObject(
      embeddings = Y_full, key = "UMAP_", assay = Seurat::DefaultAssay(obj)
    )
    return(obj)
  }
  
  ## 正常路径：transform 全量
  message("⏱ Projecting all cells with trained UMAP model...")
  transform_fn <- if ("umap_transform" %in% getNamespaceExports("uwot")) uwot::umap_transform else uwot::transform
  Y_full <- transform_fn(pca_mat, umap_model, n_threads = max(1, parallel::detectCores() - 1))
  Y_full <- as.matrix(Y_full)
  if (!all(is.finite(Y_full))) stop("UMAP 投影结果存在 NA/Inf。")
  rownames(Y_full) <- rownames(pca_mat)
  saveRDS(Y_full, coord_rds)
  
  obj[[red]] <- Seurat::CreateDimReducObject(
    embeddings = Y_full, key = "UMAP_", assay = Seurat::DefaultAssay(obj)
  )
  obj
}



score_cells_by_bank_simple <- function(seu, marker_bank, gene_weight) {
  stopifnot(inherits(seu, "Seurat"))
  mat <- Seurat::GetAssayData(seu, slot = "data")
  rownames(mat) <- toupper(rownames(mat))
  
  wpick <- function(setnm) {
    if (is.null(gene_weight) || !nrow(gene_weight)) return(NULL)
    sub <- subset(gene_weight, set == setnm)
    if (!nrow(sub)) return(NULL)
    w <- sub$weight
    names(w) <- toupper(sub$gene)
    w
  }
  
  wmean <- function(genes, w = NULL) {
    genes <- unique(toupper(genes))
    g <- intersect(genes, rownames(mat))
    if (!length(g)) return(rep(0, ncol(mat)))
    M <- mat[g, , drop = FALSE]
    if (is.null(w)) return(colMeans(M))
    w <- w[g]
    if (!length(w)) return(colMeans(M))
    w[!is.finite(w) | w < 0] <- 0
    if (sum(w) <= 0) return(colMeans(M))
    w <- w / sum(w)
    as.numeric(Matrix::t(w) %*% M)
  }
  
  P <- names(marker_bank)
  res_id <- res_sp <- res_fn <- matrix(0, nrow = ncol(seu), ncol = length(P), dimnames = list(colnames(seu), P))
  for (pid in P) {
    entry <- marker_bank[[pid]]
    res_id[, pid] <- wmean(entry$identity_up, wpick(paste0("IDENTITY_UP:", pid))) -
      wmean(entry$identity_down, wpick(paste0("IDENTITY_DOWN:", pid)))
    res_sp[, pid] <- wmean(entry$core, wpick(paste0("CORE:", pid))) -
      wmean(entry$edge, wpick(paste0("EDGE:", pid)))
    res_fn[, pid] <- wmean(entry$pcr_up, wpick(paste0("PCR_UP:", pid))) -
      wmean(entry$pcr_down, wpick(paste0("PCR_DOWN:", pid)))
  }
  list(identity = res_id, spatial = res_sp, functional = res_fn)
}

# run_alignment_sensitivity <- function(
#   seu, marker_bank, prior_comp_file, gene_weight,
#   bulk_mat, pheno,
#   w_spatial_vals = c(0, 0.4),
#   w_func_vals    = c(0, 0.35),
#   beta_vals      = c(0.5, 0.9),
#   cache_dir = file.path(my_tempdir, "ucell_alignment_sensitivity"),
#   seed = 123
# ){
#   stopifnot(inherits(seu, "Seurat"))
#   dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)
#   combos <- expand.grid(w_spatial = w_spatial_vals, w_func = w_func_vals, beta = beta_vals)
#   if (!nrow(combos)) stop("w_spatial_vals / w_func_vals / beta_vals 为空")
#   
#   y <- ifelse(pheno$pcr_response == "pCR", 1,
#               ifelse(pheno$pcr_response == "non_pCR", 0, NA))
#   keep <- is.finite(y)
#   bulk_lab <- y[keep]
#   sample_order <- colnames(bulk_mat)
#   pheno_ordered <- pheno[match(sample_order, rownames(pheno)), , drop = FALSE]
#   bulk_lab <- ifelse(pheno_ordered$pcr_response == "pCR", 1,
#                      ifelse(pheno_ordered$pcr_response == "non_pCR", 0, NA))
#   
#   res_list <- list()
#   ref_labels <- NULL
#   set.seed(seed)
#   
#   for (i in seq_len(nrow(combos))) {
#     ws <- combos$w_spatial[i]
#     wf <- combos$w_func[i]
#     bt <- combos$beta[i]
#     cache_i <- file.path(cache_dir, sprintf("ws%.2f_wf%.2f_bt%.2f", ws, wf, bt))
#     seu_tmp <- score_sc_prototypes_and_align(
#       seu,
#       marker_bank     = marker_bank,
#       prior_comp_file = prior_comp_file,
#       macro_col       = "MacroClass",
#       w_spatial       = ws,
#       min_margin      = 0.3,
#       w_func          = wf,
#       use_hvg         = FALSE,
#       hvg_n           = 3000,
#       trim_per_set    = 300,
#       ucell_batch     = 8000,
#       ucell_parallel  = FALSE,
#       cache_dir       = cache_i,
#       beta_prior      = bt
#     )
#     labels <- as.character(seu_tmp$Prototype_aligned)
#     names(labels) <- colnames(seu_tmp)  # 绑定细胞名
#     
#     if (is.null(ref_labels)) {
#       ref_labels <- labels
#     } else {
#       common <- intersect(names(ref_labels), names(labels))
#       ari <- if (length(common) > 0) {
#         mclust::adjustedRandIndex(ref_labels[common], labels[common])
#       } else NA_real_
#     }
#     ari <- tryCatch(mclust::adjustedRandIndex(labels, ref_labels), error = function(e) NA_real_)
#     
#     res_list[[i]] <- data.frame(
#       w_spatial = ws,
#       w_func    = wf,
#       beta      = bt,
#       ari_ref   = ari,
#       min_count = if (length(counts)) min(counts) else NA_integer_,
#       max_count = if (length(counts)) max(counts) else NA_integer_,
#       rho_proto0 = pick_rho("Prototype0"),
#       rho_proto2 = pick_rho("Prototype2"),
#       rho_proto4 = pick_rho("Prototype4"),
#       stringsAsFactors = FALSE
#     )
#   }
#   
#   res_df <- do.call(rbind, res_list)
#   message("Alignment sensitivity summary:\n",
#           paste(capture.output(print(res_df)), collapse = "\n"))
#   invisible(res_df)
# }


## =========================
## 3) bulk 打分：identity/spatial（label-free）与 functional（解释用）
## =========================
score_bulk_by_bank <- function(bulk_mat, marker_bank, mode = c("identity","spatial","functional"),
                               gene_weight = NULL, scale_scores = FALSE) {
  mode <- match.arg(mode)
  stopifnot(nrow(bulk_mat) > 0, ncol(bulk_mat) > 0)
  
  # ✅ 统一大写行名（与 gene set/权重一致）
  rownames(bulk_mat) <- toupper(rownames(bulk_mat))
  
  ranked <- matrixStats::colRanks(as.matrix(bulk_mat), ties.method = "average", preserveShape = TRUE)
  ranked <- (ranked - 1) / pmax(nrow(bulk_mat) - 1, 1)
  R <- ranked
  rownames(R) <- rownames(bulk_mat)   # 已是大写
  colnames(R) <- colnames(bulk_mat)
  
  wmean <- function(mat, genes, w = NULL){
    genes <- unique(toupper(genes))
    g <- intersect(genes, rownames(mat))
    if (!length(g)) return(rep(0, ncol(mat)))
    M <- mat[g, , drop = FALSE]
    if (nrow(M) == 0) return(rep(0, ncol(mat)))
    if (is.null(w) || !length(w)) {
      return(colMeans(M, na.rm = TRUE))
    }
    names(w) <- toupper(names(w))
    w <- w[g]
    if (!length(w)) return(colMeans(M, na.rm = TRUE))
    w[!is.finite(w) | w < 0] <- 0
    if (sum(w) <= 0) return(colMeans(M, na.rm = TRUE))
    w <- w / sum(w)
    as.numeric(t(w) %*% M)
  }
  
  fetch_w <- function(set_name){
    if (is.null(gene_weight) || !nrow(gene_weight)) return(NULL)
    sub <- gene_weight[gene_weight$set == set_name, , drop = FALSE]
    if (!nrow(sub)) return(NULL)
    w <- sub$weight
    names(w) <- toupper(sub$gene)
    w
  }
  
  proto_ids <- names(marker_bank)
  if (!length(proto_ids)) {
    res <- matrix(0, nrow = ncol(bulk_mat), ncol = 0)
    rownames(res) <- colnames(bulk_mat)
    return(res)
  }
  
  B <- sapply(proto_ids, function(pid){
    entry <- marker_bank[[pid]]
    if (mode == "identity") {
      up <- entry$identity_up
      down <- entry$identity_down
      w_up <- fetch_w(paste0("IDENTITY_UP:", pid))
      w_dn <- fetch_w(paste0("IDENTITY_DOWN:", pid))
      s_up <- wmean(R, up, w_up)
      s_dn <- wmean(R, down, w_dn)
      s_up - s_dn
    } else if (mode == "spatial") {
      core <- entry$core
      edge <- entry$edge
      w_core <- fetch_w(paste0("CORE:", pid))
      w_edge <- fetch_w(paste0("EDGE:", pid))
      wmean(R, core, w_core) - wmean(R, edge, w_edge)
    } else {
      up <- entry$pcr_up
      down <- entry$pcr_down
      w_up <- fetch_w(paste0("PCR_UP:", pid))
      w_dn <- fetch_w(paste0("PCR_DOWN:", pid))
      wmean(R, up, w_up) - wmean(R, down, w_dn)
    }
  }, simplify = "matrix")
  
  if (!is.matrix(B)) {
    B <- matrix(B, ncol = 1)
    colnames(B) <- proto_ids[1]
  }
  rownames(B) <- colnames(bulk_mat)
  colnames(B) <- proto_ids
  
  B[!is.finite(B)] <- 0
  if (isTRUE(scale_scores)) {
    B <- scale(B)
    B <- as.matrix(B)
    B[!is.finite(B)] <- 0
  }
  B
}

marker_bank <- build_marker_bank_from_dfs(
  proto_keep_corepair, proto_keep_edgepair, proto_keep_coreedge, proto_keep_pCR,
  proto_ids = c("Prototype0","Prototype2","Prototype4")
)
marker_bank_seed <- marker_bank





## === 加载并准备 Seurat 对象 (必须先有 `seu`) ===
# 1) 读入 / 构建 Seurat 对象
## ===== 1) 直接读三件套，不预筛，不做 QC 过滤 =====
if (!exists("seu")) {
  message(">> Direct ReadMtx: matrix.mtx + genes.tsv + barcodes.tsv (no prefilter)")
  
  mtx_file <- file.path(sc_dir, "matrix.mtx")
  genes_ts <- file.path(sc_dir, "genes.tsv")
  bcs_ts   <- file.path(sc_dir, "barcodes.tsv")
  stopifnot(file.exists(mtx_file), file.exists(genes_ts), file.exists(bcs_ts))
  
  ## genes.tsv 常见为两列(ID, SYMBOL)；只有一列就用第1列
  feat_col <- tryCatch({
    ncol(utils::read.delim(genes_ts, header = FALSE, nrows = 1, check.names = FALSE))
  }, error = function(e) 2)
  if (is.na(feat_col) || feat_col < 1) feat_col <- 2
  feat_col <- ifelse(feat_col >= 2, 2, 1)
  
  ## 不做任何预筛
  mat <- Seurat::ReadMtx(
    mtx = mtx_file,
    features = genes_ts,
    cells = bcs_ts,
    feature.column = feat_col
  )
  message(sprintf("ReadMtx 维度 = %d genes × %d barcodes", nrow(mat), ncol(mat)))
  
  ## 不要在这里过滤
  seu <- Seurat::CreateSeuratObject(
    counts = mat,
    project = basename(sc_dir),
    min.cells = 0,
    min.features = 0
  )
  rm(mat); gc(FALSE)
  
  ## 合并 metadata：只对齐，不做交集（没对上的填 NA，不丢细胞）
  meta_path <- file.path(sc_dir, sc_metadata_filename)  # 通常是 "metadata.csv"
  if (file.exists(meta_path)) {
    md <- read.csv(meta_path, row.names = 1, check.names = FALSE)
    
    ## 如果条码在一列里（如 "barcode"），把它挪到行名
    if (!any(rownames(md) %in% colnames(seu)) && "barcode" %in% colnames(md)) {
      rownames(md) <- as.character(md$barcode)
    }
    
    ## 处理 -1 后缀不一致
    fix_bc <- function(x) sub("-1$", "", x)
    if (!any(rownames(md) %in% colnames(seu))) {
      colnames(seu) <- fix_bc(colnames(seu))
    }
    if (!any(rownames(md) %in% colnames(seu))) {
      rownames(md) <- fix_bc(rownames(md))
    }
    
    idx <- match(colnames(seu), rownames(md))
    seu <- Seurat::AddMetaData(seu, md[idx, , drop = FALSE])
    message(sprintf("Merged meta with NA padding. Matched cells = %d / %d",
                    sum(!is.na(idx)), ncol(seu)))
  } else {
    message("⚠ 未找到 metadata.csv，先跳过合并。")
  }
  
  cat(sprintf("Loaded: features=%d, cells=%d\n", nrow(seu), ncol(seu)))
}







#####  画原始celltype_minor的umap
plan("sequential")
# 1. 标准化
seu <- NormalizeData(seu, normalization.method = "LogNormalize", scale.factor = 10000)

# 2. 寻找高变基因
seu <- FindVariableFeatures(seu, selection.method = "vst", nfeatures = 2000)

# 3. 缩放
all.genes <- rownames(seu)
seu <- ScaleData(seu, features = VariableFeatures(object = seu))

# 4. 运行 PCA
seu <- RunPCA(seu, features = VariableFeatures(object = seu))

# 5. 运行 UMAP (关键步骤)
#    (注意: 'dims' 的数量应根据您的 PCA 'ElbowPlot' 来确定，这里先用 30)
seu <- RunUMAP(seu, dims = 1:30)


# 1. (可选，但推荐) 先把您的绘图命令赋值给一个变量
p <- DimPlot(seu, 
             reduction = "umap", 
             group.by = "celltype_minor", 
             label = TRUE, 
             repel = TRUE, 
             label.size = 3)

# (可选) 确保图像在您的 R 绘图窗口中显示
print(p) 

# 2. 将变量 p 保存为 PNG 文件
ggsave(filename = "D:/after_bib/data_v2/paper_img/archetype_analysis_results_final_v11_fix1_250930/umap_celltype_minor.png",  # <-- 文件名
       plot = p,                           # <-- 要保存的图
       width = 12,                         # 宽度 (英寸)
       height = 8,                         # 高度 (英寸)
       dpi = 300)                          # 分辨率 (300 dpi 是出版标准)






############### 映射宏类

if (!exists("seu")) stop("Seurat 对象 `seu` 未成功加载，请检查上游读入步骤。")
seu_raw <- seu
rm(seu)

sc_celltype_col <- "celltype_minor"

## 2) MacroClass：优先用映射，其次用正则兜底
seu_raw <- ensure_macroclass(
  seu_raw,
  src_col = sc_celltype_col,
  mapping = cell_type_mapping
)

if (!("MacroClass" %in% colnames(seu_raw@meta.data))) {
  ct <- tolower(as.character(seu_raw[[sc_celltype_col]][,1]))
  macro <- ifelse(grepl("tumou|tumor|malig|carcin|neoplas", ct), "Neoplastic",
                  ifelse(grepl("epi|duct|luminal|basal|keratin", ct), "Epithelial",
                         ifelse(grepl("fibro|stromal|endo|endothelial|pericyte|connect", ct), "Connective",
                                ifelse(grepl("t cell|b cell|myeloid|mono|macro|dend|nk|mast|plasma|immune", ct), "Inflammatory",
                                       ifelse(grepl("dead|apop|debris", ct), "Dead", "Unknown")))))
  seu_raw$MacroClass <- factor(macro, levels=c("Connective","Dead","Epithelial","Inflammatory","Neoplastic","Unknown"))
}

# 4. 绘图 (按 MacroClass)
#    (这将自动使用您在第一个流程中计算好的 UMAP 坐标)
p_macro <- DimPlot(seu_raw, 
                   reduction = "umap", 
                   group.by = "MacroClass",  # <-- 只需更改这个标签
                   label = TRUE, 
                   repel = TRUE)

# 5. 显示和保存
print(p_macro)

ggsave(filename = "D:/after_bib/data_v2/paper_img/archetype_analysis_results_final_v11_fix1_250930/umap_MacroClass.png", 
       plot = p_macro, 
       width = 12, 
       height = 8, 
       dpi = 300)


















marker_bank_v1 <- marker_bank_seed
marker_bank    <- marker_bank_v1

strict_score <- function(obj, bank, cache_dir, w_spatial = 0.35) {
  with_sequential({
    score_sc_prototypes_and_align(
      obj,
      marker_bank     = bank,
      prior_comp_file = prior_comp_file_raw,
      macro_col       = "MacroClass",
      w_spatial       = w_spatial,
      min_margin      = 0.15,
      w_func          = 0.25,
      use_hvg         = FALSE,
      hvg_n           = 3000,
      trim_per_set    = 600,
      ucell_batch     = 4000,
      ucell_parallel  = FALSE,
      cache_dir       = cache_dir,
      beta_prior      = 0.45,
      use_consensus_gate = TRUE,
      consensus_k        = 1,
      prior_gate         = NA,
      prior_gate_q       = 0.6
    )
  })
}

message(">> Pass 1: 使用 v1 bank（种子）打分并对齐 …")
seu_full <- strict_score(
  seu_full,
  bank = marker_bank_v1,
  cache_dir = file.path(cache_root, "ucell_cache_pass1")
)


if (!("Prototype_aligned" %in% colnames(seu_full@meta.data))) {
  if ("Prototype_initial" %in% colnames(seu_full@meta.data)) {
    warning("Prototype_aligned 缺失，临时退回 Prototype_initial。")
    seu_full$Prototype_aligned <- seu_full$Prototype_initial
  } else if ("Prototype_refined" %in% colnames(seu_full@meta.data)) {
    lab_tmp <- std_proto_unified(seu_full$Prototype_refined)
    lab_tmp[is.na(lab_tmp) | !(lab_tmp %in% proto_levels)] <- "Unknown"
    seu_full$Prototype_aligned <- factor(lab_tmp, levels = c(proto_levels, "Unknown"))
    message("Prototype_aligned 缺失，已根据 Prototype_refined 兜底。")
  } else {
    stop("自动评分失败且无备用列（Prototype_initial/refined），请先修复打分步骤。")
  }
}

if (!(proto_col_v1 %in% colnames(seu_full@meta.data))) {
  seu_full[[proto_col_v1]] <- seu_full$Prototype_aligned
}


lab_v1 <- std_proto_unified(seu_full[[proto_col_v1]])
lab_v1[is.na(lab_v1) | !(lab_v1 %in% proto_levels)] <- "Unknown"
seu_full[[proto_col_v1]] <- factor(lab_v1, levels = c(proto_levels, "Unknown"))
seu_full$Prototype_plot_v1 <- seu_full[[proto_col_v1]]
log_n("after_pass1", seu_full)

## SpatialLike（v1）→ 再 UMAP（全量）
seu_full <- derive_spatiallike_from_ucell(
  seu_full,
  marker_bank_v1,
  proto_col = proto_col_v1,
  delta_thr = 0.10
)

visual_dir_umap <- file.path(outdir, "UMAP")
visual_dir_heat <- file.path(outdir, "ProtoHeatmap")
visual_dir_enrich <- file.path(outdir, "Enrich")
dir.create(visual_dir_umap, showWarnings = FALSE, recursive = TRUE)
dir.create(visual_dir_heat, showWarnings = FALSE, recursive = TRUE)
dir.create(visual_dir_enrich, showWarnings = FALSE, recursive = TRUE)

umap_cache <- file.path(outdir, "seurat_with_umap_full.rds")
need_recalc_umap <- TRUE
if (file.exists(umap_cache)) {
  try({
    seu_cached <- readRDS(umap_cache)
    same_n  <- ncol(seu_cached) == ncol(seu_full)
    same_id <- same_n && setequal(colnames(seu_cached), colnames(seu_full))
    if (same_id && "umap" %in% names(seu_cached@reductions)) {
      seu_full[["umap"]] <- seu_cached[["umap"]]
      need_recalc_umap <- FALSE
      message("✔ UMAP loaded from cache (matching full cell set).")
    } else {
      message("⚠️ UMAP cache exists but cell set changed; will recompute.")
    }
  }, silent = TRUE)
}

if (need_recalc_umap) {
  seu_full <- ensure_umap_fast(
    seu_full,
    red = "umap",
    dims_use = 1:15,
    subsample = 20000,
    n_neighbors = 20,
    min_dist = 0.2,
    n_epochs = 200,
    nn_method = "annoy",
    metric = "cosine",
    cache_dir = file.path(cache_root, "UMAP_cache_full")
  )
  saveRDS(seu_full, umap_cache)
}











## Pass 2：sc 细调 + 融合，得到 v2 bank
sc_ref <- refine_bank_with_sc(
  seu = seu_full,
  marker_bank = marker_bank_v1,
  sc_celltype_col = sc_celltype_col,
  proto_col = proto_col_v1,
  prior_comp_file = prior_comp_file_raw,
  top_n_per_ct = 120,
  min_cells = 30,
  min_pct = 0.1,
  logfc_th = 0.10
)

fusion <- fuse_three_marker_sources(
  bank_seed = marker_bank_seed,
  bank_sc   = sc_ref$bank,
  w_sc      = sc_ref$w_sc,
  proto_ids = proto_levels
)
marker_bank_v2 <- fusion$bank
gene_weight    <- fusion$gene_weight
marker_bank    <- marker_bank_v2
# w_sc <- gene_weight

## Pass 2b：用 v2 bank 重对齐，并回写 SpatialLike（v2）
message(">> Pass 2b: 使用 v2 bank 重新打分 + 对齐 …")
seu_full <- strict_score(
  seu_full,
  bank = marker_bank_v2,
  cache_dir = file.path(cache_root, "ucell_cache_pass2b"),
  w_spatial = 0.35
)



if (!(proto_col_v2 %in% colnames(seu_full@meta.data))) {
  seu_full[[proto_col_v2]] <- seu_full$Prototype_aligned
}


## 用 Prototype_aligned 直接生成 v2 列（不再调用 std_proto_unified）
lab_v2 <- as.character(seu_full$Prototype_aligned)
lab_v2 <- trimws(lab_v2)
# 如有简写，统一映射（可选）
lab_v2 <- sub("^P0$", "Prototype0", lab_v2, ignore.case = TRUE)
lab_v2 <- sub("^P2$", "Prototype2", lab_v2, ignore.case = TRUE)
lab_v2 <- sub("^P4$", "Prototype4", lab_v2, ignore.case = TRUE)

# 非法/缺失 → Unknown
lab_v2[is.na(lab_v2) | !(lab_v2 %in% proto_levels)] <- "Unknown"

# 写回并检查
seu_full[[proto_col_v2]] <- factor(lab_v2, levels = c(proto_levels, "Unknown"))
table(seu_full[[proto_col_v2]], useNA = "ifany")



seu_full <- derive_spatiallike_from_ucell(
  seu_full,
  marker_bank_v2,
  proto_col = proto_col_v2,
  delta_thr = 0.10
)

## seu_analysis：仅保留目标原型细胞，用于所有计算
mask_keep <- lab_v2 %in% proto_levels
seu_analysis <- subset(seu_full, cells = colnames(seu_full)[mask_keep])
seu_analysis <- ensure_logdata(seu_analysis)
log_n("seu_analysis", seu_analysis)
if (!(proto_col_v2 %in% colnames(seu_analysis@meta.data))) {
  stop("对齐后未生成 `", proto_col_v2, "`，请检查单细胞元数据与先验。")
}



# 到这里 `seu` 一定存在，且含有所需列
deduplicate_seurat_features <- function(seu_obj, assay = "RNA") {
  collapse_matrix <- function(mat) {
    if (is.null(mat) || nrow(mat) == 0) return(mat)
    rn <- toupper(rownames(mat))
    idx_list <- split(seq_along(rn), rn)
    uniq <- names(idx_list)
    if (length(uniq) == length(rn)) {
      rownames(mat) <- rn
      return(mat)
    }
    if (inherits(mat, "sparseMatrix")) {
      mat_t <- methods::as(mat, "dgTMatrix")
      new_i <- match(rn[mat_t@i + 1], uniq)
      res_sum <- Matrix::sparseMatrix(i = new_i, j = mat_t@j + 1, x = mat_t@x,
                                      dims = c(length(uniq), ncol(mat)))
      dup_counts <- vapply(idx_list, length, numeric(1))
      scale_vec <- 1 / dup_counts
      res <- Matrix::Diagonal(x = scale_vec) %*% res_sum
      rownames(res) <- uniq
      colnames(res) <- colnames(mat)
      return(methods::as(res, "dgCMatrix"))
    } else {
      df <- stats::aggregate(as.matrix(mat), by = list(gene = rn), FUN = mean)
      m <- as.matrix(df[,-1, drop=FALSE])
      rownames(m) <- df$gene
      colnames(m) <- colnames(mat)
      return(m)
    }
  }
  assays <- Seurat::Assays(seu_obj)
  for (ass in assays) {
    mat_counts <- try(Seurat::GetAssayData(seu_obj, assay = ass, slot = "counts"), silent = TRUE)
    if (!inherits(mat_counts, "try-error") && !is.null(mat_counts)) {
      new_counts <- collapse_matrix(mat_counts)
      seu_obj <- Seurat::SetAssayData(seu_obj, assay = ass, slot = "counts", new.data = new_counts)
    }
    mat_data <- try(Seurat::GetAssayData(seu_obj, assay = ass, slot = "data"), silent = TRUE)
    if (!inherits(mat_data, "try-error") && !is.null(mat_data)) {
      new_data <- collapse_matrix(mat_data)
      seu_obj <- Seurat::SetAssayData(seu_obj, assay = ass, slot = "data", new.data = new_data)
    }
  }
  seu_obj
}
seu_analysis <- deduplicate_seurat_features(seu_analysis)




gc(verbose = FALSE)

B_identity <- score_bulk_by_bank(bulk_mat, marker_bank, mode = "identity",   gene_weight = gene_weight)
B_spatial  <- score_bulk_by_bank(bulk_mat, marker_bank, mode = "spatial",    gene_weight = gene_weight)
B_functional <- score_bulk_by_bank(bulk_mat, marker_bank, mode = "functional", gene_weight = gene_weight)

write.csv(B_identity,   file.path(outdir, sprintf("%s_bulk_proto_identity_scores.csv",   dataset_name)))
write.csv(B_spatial,    file.path(outdir, sprintf("%s_bulk_proto_spatial_scores.csv",    dataset_name)))
write.csv(B_functional, file.path(outdir, sprintf("%s_bulk_proto_functional_scores.csv", dataset_name)))










## =========================
## 4) Bulk signatures：v2 bank 固定评估
## =========================
bulk_status <- pheno$pcr_response
names(bulk_status) <- rownames(pheno)
bulk_y <- ifelse(bulk_status == "pCR", 1,
                 ifelse(bulk_status == "non_pCR", 0, NA_real_))

## 统一交集样本（把三路 + y 都纳入）
common_samples <- Reduce(intersect, list(
  rownames(B_identity),
  rownames(B_spatial),
  rownames(B_functional),
  names(bulk_y)
))
if (length(common_samples) == 0) stop("bulk scores与表型没有交集样本，无法评估。")

ensure_proto_cols <- function(M, pids = c("Prototype0","Prototype2","Prototype4")){
  M <- as.data.frame(M, check.names = FALSE)
  miss <- setdiff(pids, colnames(M))
  for (m in miss) M[[m]] <- 0
  as.matrix(M[, pids, drop = FALSE])
}

## 显式按 common_samples 对齐三路矩阵
B_id_use <- ensure_proto_cols(B_identity   [common_samples, , drop = FALSE])
B_sp_use <- ensure_proto_cols(B_spatial    [common_samples, , drop = FALSE])
B_fn_use <- ensure_proto_cols(B_functional [common_samples, , drop = FALSE])

## 基本一致性检查
stopifnot(
  identical(rownames(B_id_use), common_samples),
  identical(rownames(B_sp_use), common_samples),
  identical(rownames(B_fn_use), common_samples),
  identical(colnames(B_id_use), c("Prototype0","Prototype2","Prototype4")),
  identical(colnames(B_sp_use), c("Prototype0","Prototype2","Prototype4")),
  identical(colnames(B_fn_use), c("Prototype0","Prototype2","Prototype4"))
)

## y 与矩阵同序对齐（不要用未定义的 keep）
y_vec <- bulk_y[common_samples]
names(y_vec) <- common_samples   # 保险：显式设置名字

## 9特征组合寻优
res9 <- tune_combo_weights_9_fixeddir(
  B_identity   = B_id_use,
  B_spatial    = B_sp_use,
  B_functional = B_fn_use,
  y_vec        = y_vec,
  pids         = c("Prototype0","Prototype2","Prototype4"),
  signs_id     = c(+1, +1, +1),
  signs_sp     = c(+1, -1, +1),   # P2 空间取负（按你的先验）
  signs_fn     = c(+1, +1, +1),
  K = 5, seed = 1, robust = TRUE,
  search = "dirichlet", n_try = 2000
)


res9$w_best       # 9 个权重
res9$auc_cv_best  # K 折 AUC（建议汇报）
res9$auc_full     # 全数据 AUC（偏乐观，仅展示）
head(res9$score_all)



## 2) 构造Z矩阵用于报告
##    ——与调参一致的稳健z：对全体样本一次性拟合，仅用于“展示/导出”
make_Z9 <- function(){
  smp <- Reduce(intersect, list(rownames(B_id_use), rownames(B_sp_use), rownames(B_fn_use), names(y_vec)))
  pids <- c("Prototype0","Prototype2","Prototype4")
  pids <- intersect(pids, Reduce(intersect, list(colnames(B_id_use), colnames(B_sp_use), colnames(B_fn_use))))
  X_id <- do.call(cbind, lapply(pids, function(p) +1 * as.numeric(B_id_use[smp, p]))); colnames(X_id) <- paste0("id:", pids)
  X_sp <- do.call(cbind, lapply(pids, function(p) c(Prototype0=+1, Prototype2=-1, Prototype4=+1)[p] * as.numeric(B_sp_use[smp, p]))); colnames(X_sp) <- paste0("sp:", pids)
  X_fn <- do.call(cbind, lapply(pids, function(p) +1 * as.numeric(B_fn_use[smp, p]))); colnames(X_fn) <- paste0("fn:", pids)
  X <- cbind(X_id, X_sp, X_fn)
  keep <- stats::complete.cases(X, y_vec[smp])
  X <- X[keep,,drop=FALSE]; y <- as.numeric(y_vec[smp][keep])
  fits <- lapply(seq_len(ncol(X)), function(j) .tz_fit(X[,j], robust=TRUE))
  Z   <- sapply(seq_len(ncol(X)), function(j) .tz_apply(X[,j], fits[[j]]))
  Z   <- matrix(Z, nrow=nrow(X), ncol=ncol(X), dimnames=list(rownames(X), colnames(X)))
  list(Z=Z, y=y)
}
tmp9 <- make_Z9()

## 3) 报告/出图
# 1) 用 glmnet 寻优（非负 + 类平衡），返回的是“可直接用于 reporter 的权重”
res9 <- tune_combo_glmnet(tmp9$Z, tmp9$y, nonneg = TRUE, balance = TRUE, seed = 1)

# 2) 复用你现成的 reporter（会画单分量/等权/以及 tuned 组合）
rep9 <- auto_combo_report_9(
  X_mat  = tmp9$Z, y = tmp9$y,
  w_best = res9$w_best,
  outdir = outdir, prefix = "Bulk_combo_9feat_glmnet", top_single = 6
)

# 3) 导出分数（名称保持一致风格，避免覆盖旧文件）
readr::write_csv(
  data.frame(sample = names(res9$score_all),
             combo_score = as.numeric(res9$score_all)),
  file.path(outdir, "Bulk_scores_combo_weight_tuned_9feat_glmnet.csv")
)

message("✅ 9feat ROC：", rep9$file_png)
message("✅ 9feat AUC表：", rep9$file_csv)



## ================================
## 三分量（id/sp/fn）权重寻优 + 报告
##  - 固定方向：sp_sign = -1（Edge 取负）、fn_sign = +1
##  - 先寻优(w_best)，再自动出图/出表
##  - 导出 tuned 组合分数
## ================================
stopifnot(requireNamespace("pROC",  quietly = TRUE))
stopifnot(requireNamespace("readr", quietly = TRUE))


## --- 3分量固定方向寻优：id/sp/fn ---
tune_combo_weights_3_fixeddir <- function(
  B_identity, B_spatial, B_functional, y_vec,
  pid_id = "Prototype4",     # 身份原型
  pid_sp = "Prototype2",     # 空间原型
  pid_fn = "Prototype4",     # 功能原型（常用 P4）
  sp_sign = -1,              # 空间固定方向
  fn_sign = +1,              # 功能固定方向
  step = 0.01, K = 5, seed = 1,
  robust = TRUE
){
  stopifnot(requireNamespace("pROC", quietly = TRUE))
  
  ## 统一样本
  smp <- Reduce(intersect, list(rownames(B_identity), rownames(B_spatial), rownames(B_functional), names(y_vec)))
  y   <- as.numeric(y_vec[smp])
  
  X <- data.frame(
    id = B_identity[smp,  pid_id],
    sp = sp_sign * B_spatial[smp,   pid_sp],
    fn = fn_sign * B_functional[smp, pid_fn],
    row.names = smp, check.names = FALSE
  )
  keep <- stats::complete.cases(X, y) & is.finite(y)
  X <- X[keep, , drop = FALSE]; y <- y[keep]
  if (length(unique(y)) < 2 || nrow(X) < 30) stop("有效样本不足。")
  
  set.seed(seed)
  folds <- sample(rep(1:K, length.out = nrow(X)))
  
  ## 网格：w_id ∈ [0,1], w_sp ∈ [0,1-w_id], w_fn = 1 - w_id - w_sp
  ws <- seq(0, 1, by = step)
  grid <- expand.grid(w_id = ws, w_sp = ws)
  grid <- subset(grid, w_id + w_sp <= 1)
  grid$w_fn <- 1 - grid$w_id - grid$w_sp
  
  auc_mat <- matrix(NA_real_, nrow = K, ncol = nrow(grid))
  
  for (i in 1:K) {
    tr <- folds != i; te <- !tr
    xtr <- X[tr, , drop = FALSE]; ytr <- y[tr]
    xte <- X[te, , drop = FALSE]; yte <- y[te]
    
    ## 训练折稳健 z 标准化
    f_id <- .tz_fit(xtr$id, robust); f_sp <- .tz_fit(xtr$sp, robust); f_fn <- .tz_fit(xtr$fn, robust)
    z_id <- .tz_apply(xte$id, f_id)
    z_sp <- .tz_apply(xte$sp, f_sp)
    z_fn <- .tz_apply(xte$fn, f_fn)
    
    for (j in seq_len(nrow(grid))) {
      w1 <- grid$w_id[j]; w2 <- grid$w_sp[j]; w3 <- grid$w_fn[j]
      s  <- w1 * z_id + w2 * z_sp + w3 * z_fn
      auc_mat[i, j] <- .auc_safe(yte, s)$auc
    }
  }
  
  auc_cv <- colMeans(auc_mat, na.rm = TRUE)
  jbest  <- which.max(auc_cv)
  w_best <- as.numeric(grid[jbest, c("w_id","w_sp","w_fn")])
  
  ## 全体样本一次性 z 标准化并输出“展示分数”
  f_id_all <- .tz_fit(X$id, robust); f_sp_all <- .tz_fit(X$sp, robust); f_fn_all <- .tz_fit(X$fn, robust)
  Z_id_all <- .tz_apply(X$id, f_id_all)
  Z_sp_all <- .tz_apply(X$sp, f_sp_all)
  Z_fn_all <- .tz_apply(X$fn, f_fn_all)
  S_best   <- w_best[1] * Z_id_all + w_best[2] * Z_sp_all + w_best[3] * Z_fn_all
  
  list(
    w_best       = stats::setNames(w_best, c("id","sp","fn")), # α, β, γ
    auc_cv_best  = as.numeric(auc_cv[jbest]),
    auc_full     = .auc_safe(y, S_best)$auc,
    score_all    = stats::setNames(as.numeric(S_best), rownames(X)),
    grid         = grid,
    cv_auc_curve = auc_cv
  )
}

## --- 报告/出图函数（用寻优的 w_best 出 Tuned 曲线，同时给等权&单分量）---
auto_combo_report <- function(
  B_id, B_sp, y_vec,
  B_fn = NULL,
  pid_id, pid_sp, pid_fn = NULL,   # 指定参与的原型
  sp_sign = -1, fn_sign = +1,      # 固定方向
  w_best = NULL,                   # 传入 id/sp/fn 的权重；若为空只画等权+单分量
  robust = TRUE,
  outdir = ".", prefix = "Bulk_combo"
){
  getv <- function(M, col, smp) if (!is.null(M) && col %in% colnames(M)) as.numeric(M[smp, col]) else rep(NA_real_, length(smp))
  
  pools <- list(rownames(B_id), rownames(B_sp), names(y_vec))
  if (!is.null(B_fn)) pools <- c(pools, list(rownames(B_fn)))
  smp <- Reduce(intersect, pools)
  y   <- as.numeric(y_vec[smp])
  
  v_id <- getv(B_id, pid_id, smp)
  v_sp <- sp_sign * getv(B_sp, pid_sp, smp)
  vecs <- list(id = v_id, sp = v_sp)
  
  with_fn <- FALSE
  if (!is.null(B_fn) && !is.null(pid_fn) && pid_fn %in% colnames(B_fn)) {
    vecs$fn <- fn_sign * getv(B_fn, pid_fn, smp)
    with_fn <- TRUE
  }
  
  X <- as.data.frame(vecs)
  keep <- stats::complete.cases(X, y)
  X <- X[keep,,drop=FALSE]; y <- y[keep]
  if (length(unique(y)) < 2 || nrow(X) < 20) stop("有效样本不足，无法绘制/评估。")
  
  Z <- as.data.frame(lapply(X, function(col){ f <- .tz_fit(col, robust); .tz_apply(col, f) }))
  
  labs_comp <- c(
    id = sprintf("id:%s", pid_id),
    sp = sprintf("sp(%s):%s", ifelse(sp_sign<0,"−","+"), pid_sp)
  )
  if (with_fn) labs_comp["fn"] <- sprintf("fn(%s):%s", ifelse(fn_sign<0,"−","+"), pid_fn)
  
  rocs <- lapply(names(Z), function(nm) .auc_safe(y, Z[[nm]]))
  names(rocs) <- names(Z)
  
  ## 等权
  w_eq <- rep(1/ncol(Z), ncol(Z)); names(w_eq) <- colnames(Z)
  S_eq <- as.numeric(as.matrix(Z) %*% w_eq)
  roc_eq <- .auc_safe(y, S_eq)
  
  ## tuned（若给了 w_best，就用；缺失分量忽略）
  roc_tuned <- NULL; S_tuned <- NULL; w_tuned_used <- NULL
  if (!is.null(w_best)) {
    w_tuned_used <- w_best[intersect(names(w_best), colnames(Z))]
    w_tuned_used <- w_tuned_used / sum(w_tuned_used)
    S_tuned <- as.numeric(as.matrix(Z[, names(w_tuned_used), drop=FALSE]) %*% as.numeric(w_tuned_used))
    roc_tuned <- .auc_safe(y, S_tuned)
  }
  
  ## AUC 表
  rows <- lapply(names(Z), function(nm){
    data.frame(Feature = labs_comp[[nm]],
               AUC = rocs[[nm]]$auc,
               CI_low = rocs[[nm]]$ci[1],
               CI_high = rocs[[nm]]$ci[3],
               stringsAsFactors = FALSE)
  })
  rows <- do.call(rbind, rows)
  rows <- rows[order(-rows$AUC), , drop=FALSE]
  rows <- rbind(
    rows,
    data.frame(Feature = sprintf("Combo_equal(%s)", paste(names(Z), collapse="+")),
               AUC = roc_eq$auc, CI_low = roc_eq$ci[1], CI_high = roc_eq$ci[3])
  )
  if (!is.null(roc_tuned)) {
    rows <- rbind(
      rows,
      data.frame(Feature = sprintf("Combo_tuned[%s]", paste(sprintf("%s=%.2f", names(w_tuned_used), w_tuned_used), collapse=", ")),
                 AUC = roc_tuned$auc, CI_low = roc_tuned$ci[1], CI_high = roc_tuned$ci[3])
    )
  }
  
  ## 画 ROC
  fn_png <- file.path(outdir, paste0(prefix, "_ROC.png"))
  png(fn_png, width=980, height=760, res=150)
  if (!is.null(roc_tuned)) {
    plot(roc_tuned$roc, col="#e41a1c", lwd=2.8, legacy.axes=TRUE,
         main=sprintf("Combo ROC (tuned %s)", paste(sprintf("%s=%.2f", names(w_tuned_used), w_tuned_used), collapse=", ")))
  } else {
    plot(roc_eq$roc, col="#ff7f00", lwd=2.4, legacy.axes=TRUE, main="Combo ROC (equal weights)")
  }
  cols <- c(id="#377eb8", sp="#4daf4a", fn="#984ea3")[names(Z)]
  for (nm in names(Z)) plot(rocs[[nm]]$roc, col=cols[[nm]], lwd=2.0, add=TRUE)
  if (!is.null(roc_tuned)) plot(roc_eq$roc, col="#ff7f00", lwd=2.2, add=TRUE)
  leg_txt <- c(
    if (!is.null(roc_tuned)) sprintf("Tuned AUC=%.3f", roc_tuned$auc),
    sprintf("Equal AUC=%.3f", roc_eq$auc),
    sprintf("%s AUC=%.3f", labs_comp["id"], rocs$id$auc),
    sprintf("%s AUC=%.3f", labs_comp["sp"], rocs$sp$auc),
    if (with_fn) sprintf("%s AUC=%.3f", labs_comp["fn"], rocs$fn$auc)
  )
  leg_col <- c(if (!is.null(roc_tuned)) "#e41a1c", "#ff7f00", cols)
  leg_lwd <- c(if (!is.null(roc_tuned)) 2.8, 2.2, rep(2.0, length(cols)))
  legend("bottomright", leg_txt, col=leg_col, lwd=leg_lwd, bty="n")
  dev.off()
  
  ## 导出
  fn_csv <- file.path(outdir, paste0(prefix, "_AUC_table.csv"))
  readr::write_csv(rows, fn_csv)
  
  list(
    table = rows,
    file_csv = fn_csv,
    file_png = fn_png,
    weights_equal = w_eq,
    weights_tuned = w_tuned_used,
    score_equal = stats::setNames(S_eq, rownames(Z)),
    score_tuned = if (!is.null(S_tuned)) stats::setNames(S_tuned, rownames(Z)) else NULL
  )
}



## ==========================
## 单细胞 × 原型 映射热图
##  - track: "identity" / "spatial" / "functional"
##  - per-cell 版会对每个原型最多抽 cells_per_proto 个细胞（总上限 max_cells）
##  - per-cluster 版对 group.by（默认 seurat_clusters）聚合求均值
## ==========================
stopifnot(requireNamespace("ComplexHeatmap", quietly = TRUE))

## 若你已经有 cs_simple，就直接用；没有就重算一次
if (!exists("cs_simple")) {
  cs_simple <- score_cells_by_bank_simple(seu_analysis, marker_bank, gene_weight)
}


cell_by_proto_heatmap_percell <- function(
  seu, cs_simple, track = c("identity","spatial","functional"),
  pids = c("Prototype0","Prototype2","Prototype4"),
  cells_per_proto = Inf,            # ← 不限每原型细胞数（设具体数就抽样）
  max_cells = Inf,                  # ← 不限总细胞数（设具体数就限额）
  clip_q = c(0.02, 0.98),           # ← 量化裁剪抬对比
  palette = c("#313695","#ffffbf","#a50026"), # ← 强对比配色
  out_png = file.path(visual_dir_heat, "Heatmap_identity_percell.png")
){
  track <- match.arg(track)
  S <- cs_simple[[track]][, intersect(pids, colnames(cs_simple[[track]])), drop=FALSE]
  keep <- Matrix::rowSums(is.finite(S)) == ncol(S); S <- S[keep,,drop=FALSE]
  
  # 选择细胞：默认按每列分数降序并集，不随机
  pick <- unique(unlist(lapply(colnames(S), function(pp){
    o <- order(S[, pp], decreasing = TRUE)
    head(rownames(S)[o], if (is.finite(cells_per_proto)) cells_per_proto else nrow(S))
  })))
  if (is.finite(max_cells) && length(pick) > max_cells) pick <- pick[seq_len(max_cells)]
  S <- S[pick,,drop=FALSE]
  
  # 列内稳健 z 标准化
  zcol <- function(v){ f <- .tz_fit(v, TRUE); .tz_apply(v, f) }
  S_z <- apply(S, 2, zcol)
  
  # 量化裁剪，提升对比
  lo <- quantile(S_z, clip_q[1], na.rm=TRUE); hi <- quantile(S_z, clip_q[2], na.rm=TRUE)
  S_z[S_z < lo] <- lo; S_z[S_z > hi] <- hi
  
  ann <- data.frame(
    MacroClass = seu$MacroClass[rownames(S_z)],
    Prototype  = seu[[proto_col_v2]][rownames(S_z),1,drop=TRUE],
    SpatialLike = if ("SpatialLike" %in% colnames(seu@meta.data)) seu$SpatialLike[rownames(S_z)] else factor("Other", levels=c("Core","Interface","Edge","Other"))
  )
  ha <- ComplexHeatmap::HeatmapAnnotation(
    df = ann, which = "row",
    col = list(Prototype = pal_proto_plot,
               MacroClass = c(Connective="#1f78b4", Dead="#6a3d9a", Epithelial="#33a02c",
                              Inflammatory="#e31a1c", Neoplastic="#ff7f00", Unknown="#bdbdbd"),
               SpatialLike = c(Core="#08306b", Interface="#6baed6", Edge="#fc8d59", Other="#bdbdbd"))
  )
  
  png(out_png, 2400, 3000, res=240)
  ComplexHeatmap::Heatmap(
    S_z, name = paste0("z(", track, ")"),
    right_annotation = ha, show_row_names=FALSE, cluster_rows=FALSE, cluster_columns=TRUE,
    col = circlize::colorRamp2(c(lo, (lo+hi)/2, hi), palette),
    column_title = paste0("Cells × Prototypes (", track, ")"),
    row_title = paste0("Cells (n=", nrow(S_z), ")")
  ) %>% ComplexHeatmap::draw()
  dev.off()
}



cell_by_proto_heatmap_percluster <- function(
  seu, cs_simple, track = c("identity","spatial","functional"),
  pids = c("Prototype0","Prototype2","Prototype4"),
  group.by = NULL,
  out_png = file.path(visual_dir_heat, "Heatmap_cells_by_proto_percluster.png")
){
  track <- match.arg(track)
  stopifnot(track %in% names(cs_simple))
  S <- cs_simple[[track]]        # cells × prototypes
  pids <- intersect(pids, colnames(S))
  if (!length(pids)) stop("所选原型在 ", track, " 轨道中不存在。")
  S <- S[, pids, drop=FALSE]
  
  if (is.null(group.by)) {
    group.by <- if ("seurat_clusters" %in% colnames(seu@meta.data)) "seurat_clusters" else "MacroClass"
  }
  grp <- factor(seu[[group.by]][rownames(S), 1, drop=TRUE])
  M <- rowsum(S, group = grp, reorder = TRUE) / as.numeric(table(grp))
  
  ## 行注释
  ann <- data.frame(Group = rownames(M))
  rownames(ann) <- rownames(M)
  ha <- ComplexHeatmap::rowAnnotation(df = ann)
  
  ## 列 z-scale
  zcol <- function(v){ f <- .tz_fit(v, TRUE); .tz_apply(v, f) }
  M_z <- apply(M, 2, zcol)
  
  png(out_png, width = 1800, height = 1200, res = 220)
  ComplexHeatmap::Heatmap(
    M_z, name = paste0("z(", track, ")"),
    show_row_names = TRUE, show_column_names = TRUE,
    cluster_rows = TRUE, cluster_columns = TRUE,
    col = circlize::colorRamp2(c(-2.5, 0, 2.5), c("#2166ac","#f7f7f7","#b2182b")),
    column_title = paste0("Cluster × Prototypes (", track, ")"),
    row_title = paste0("Group: ", group.by)
  ) %>% ComplexHeatmap::draw()
  dev.off()
  message("✅ per-cluster 热图：", out_png)
}

## 调用示例：
dir.create(visual_dir_heat, showWarnings = FALSE, recursive = TRUE)
cell_by_proto_heatmap_percell(
  seu = seu_analysis, cs_simple = cs_simple,
  track = "identity",
  pids = c("Prototype0","Prototype2","Prototype4"),
  cells_per_proto = 2000, max_cells = 12000,
  out_png = file.path(visual_dir_heat, "Heatmap_identity_percell.png")
)
cell_by_proto_heatmap_percluster(
  seu = seu_analysis, cs_simple = cs_simple,
  track = "identity",
  pids = c("Prototype0","Prototype2","Prototype4"),
  group.by = NULL,
  out_png = file.path(visual_dir_heat, "Heatmap_identity_percluster.png")
)



## ================= 调用（关键：全程只用 res3 / rep3 这两个对象名） =================
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

## 1) 三分量权重寻优
res3 <- tune_combo_weights_3_fixeddir(
  B_identity   = B_id_use,
  B_spatial    = B_sp_use,
  B_functional = B_fn_use,
  y_vec        = y_vec,
  pid_id = "Prototype4",
  pid_sp = "Prototype2",
  pid_fn = "Prototype4",
  sp_sign = -1,   # P2 spatial 取负
  fn_sign = +1,   # P4 functional 同向
  step = 0.01, K = 5, seed = 1,
  robust = TRUE
)

## 2) 自动画图/出表（等权 + tuned + 单分量）
rep3 <- auto_combo_report(
  B_id = B_id_use, B_sp = B_sp_use, B_fn = B_fn_use,
  y_vec = y_vec,
  pid_id = "Prototype4", pid_sp = "Prototype2", pid_fn = "Prototype4",
  sp_sign = -1, fn_sign = +1,
  w_best = res3$w_best,             # 用刚寻优到的 w_best
  robust = TRUE,
  outdir = outdir, prefix = "Bulk_combo_3feat"
)

## 3) 导出“tuned 组合分数”（用于你后续下游图/表）
readr::write_csv(
  data.frame(sample = names(res3$score_all),
             combo_score = as.numeric(res3$score_all)),
  file.path(outdir, "Bulk_scores_combo_weight_tuned_3feat.csv")
)

message("✅ ROC 图：", rep3$file_png)
message("✅ AUC 表：", rep3$file_csv)







assoc_mat <- function(B, track_label) {
  if (is.null(B) || !nrow(B)) return(data.frame())
  cols <- intersect(colnames(B), proto_levels)
  if (!length(cols)) return(data.frame())
  res <- lapply(cols, function(proto) {
    score <- as.numeric(B[, proto])
    if (sum(is.finite(score)) < 5 || length(unique(score)) < 2) {
      return(data.frame(Prototype = proto))
    }
    r_s <- safe_cor(score, y_vec)
    roc_obj <- try(pROC::roc(response = y_vec, predictor = score, quiet = TRUE, direction = "auto"), silent = TRUE)
    if (inherits(roc_obj, "try-error")) {
      auc <- ci_low <- ci_high <- p_delong <- dir_flag <- NA_real_
    } else {
      auc <- as.numeric(pROC::auc(roc_obj))
      ci <- as.numeric(pROC::ci.auc(roc_obj))
      ci_low <- ci[1]; ci_high <- ci[3]
      p_delong <- try(pROC::roc.test(roc_obj, auc = 0.5, method = "delong")$p.value, silent = TRUE)
      if (inherits(p_delong, "try-error")) p_delong <- NA_real_
      dir_flag <- roc_obj$direction
    }
    data.frame(
      Track = track_label,
      Prototype = proto,
      spearman = r_s,
      auc = auc,
      ci_low = ci_low,
      ci_high = ci_high,
      p_value = p_delong,
      direction = dir_flag,
      n_pos = sum(y_vec == 1, na.rm = TRUE),
      n_neg = sum(y_vec == 0, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  })
  dplyr::bind_rows(res)
}

assoc_identity <- assoc_mat(B_id_use, "identity(label-free)")
assoc_spatial  <- assoc_mat(B_sp_use, "spatial(label-free)")
assoc_all <- dplyr::bind_rows(assoc_identity, assoc_spatial)
if (!nrow(assoc_all)) stop("AUC 结果为空，请检查 bulk 打分。")
assoc_all$rb <- rb_from_auc(as.numeric(assoc_all$auc))
assoc_all$FDR <- p.adjust(assoc_all$p_value, method = "BH")
write.csv(assoc_all,
          file.path(outdir, sprintf("%s_bulk_assoc_auc_rankbiserial_v2.csv", dataset_name)),
          row.names = FALSE)

safe_max <- function(x){
  x <- as.numeric(x)
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  max(x)
}

assoc_proto <- assoc_all %>%
  dplyr::group_by(Prototype) %>%
  dplyr::reframe(
    evidence = safe_max(abs(rb)),
    auc_max  = safe_max(auc),
    rho_max  = safe_max(abs(spearman))
  ) %>%
  dplyr::arrange(dplyr::desc(evidence), dplyr::desc(auc_max), dplyr::desc(rho_max))

thr_rb <- 0.05
proto_whitelist <- assoc_proto$Prototype[assoc_proto$evidence >= thr_rb]
if (!length(proto_whitelist)) proto_whitelist <- head(assoc_proto$Prototype, 3)
proto_whitelist <- proto_whitelist[proto_whitelist %in% proto_levels]
if (!length(proto_whitelist)) proto_whitelist <- proto_levels
proto_whitelist <- unique(proto_whitelist)
message("[Info] 聚焦原型（AUC/Rank-biserial 基于 label-free）：", paste(proto_whitelist, collapse=", "))



## 保证 Prototype_plot 列存在且 levels 完整
if (!"Prototype_plot" %in% colnames(seu_full@meta.data)) {
  seu_full$Prototype_plot <- seu_full[[proto_col_v2]][, 1, drop = TRUE]
}
seu_full$Prototype_plot <- factor(
  seu_full$Prototype_plot,
  levels = c(proto_levels, "Unknown")
)

## 防御：颜色板兜底（避免 levels 比颜色多）
make_proto_palette <- function(levels_proto,
                               reserved = c(Prototype0 = "#1b9e77",  # 你原来的青绿
                                            Prototype2 = "#d95f02",  # 你原来的橙
                                            Prototype4 = "#7570b3",  # 你原来的紫
                                            Unknown    = "#bdbdbd"), # 灰
                               seed = 1) {
  lv <- as.character(levels_proto)
  pal <- setNames(rep(NA_character_, length(lv)), lv)
  
  ## 先填保留色（若该 level 存在）
  for (nm in names(reserved)) {
    if (nm %in% lv) pal[nm] <- reserved[[nm]]
  }
  
  ## 还没有颜色的 level
  missing <- names(pal)[!nzchar(pal)]
  k <- length(missing)
  if (k > 0) {
    cols <- NULL
    
    # 1) 最优先：Glasbey（极高区分度）
    if (requireNamespace("pals", quietly = TRUE)) {
      set.seed(seed)
      cols <- pals::glasbey(n = k)
      # 2) 次选：colorspace 的 qualitative_hcl（色盲友好）
    } else if (requireNamespace("colorspace", quietly = TRUE)) {
      # "Dark 3" 对比度高；也可换 "Set 2"/"Set 3"/"Tableau 20"
      cols <- colorspace::qualitative_hcl(k, palette = "Dark 3")
      # 3) 兜底：基础 HCL 均分色环（保证亮度/饱和度一致）
    } else {
      hues <- seq(15, 375, length.out = k + 1)[1:k]
      cols <- grDevices::hcl(h = hues, c = 70, l = 55)
    }
    
    pal[missing] <- cols
  }
  
  pal
}

## 使用：替换你原来的 if 块
lv <- levels(seu_full$Prototype_plot)
pal_proto_plot <- make_proto_palette(lv)


## ====== UMAP 与 Feature 输出 ======
if ("umap" %in% names(seu_full@reductions)) {
  common_cells <- intersect(colnames(seu_analysis), colnames(seu_full))
  if (length(common_cells)) {
    emb <- seu_full[["umap"]]@cell.embeddings[common_cells, , drop = FALSE]
    seu_analysis[["umap"]] <- Seurat::CreateDimReducObject(
      embeddings = emb,
      key = seu_full[["umap"]]@key,
      global = seu_full[["umap"]]@global,
      assay = Seurat::DefaultAssay(seu_analysis)
    )
  }
  p1 <- Seurat::DimPlot(seu_full, reduction = "umap",
                        group.by = sc_celltype_col, label = FALSE) + ggtitle("Celltype UMAP")
  p2 <- Seurat::DimPlot(seu_full, reduction = "umap",
                        group.by = "MacroClass", label = FALSE) + ggtitle("MacroClass UMAP")
  p3 <- Seurat::DimPlot(seu_full, reduction = "umap",
                        group.by = "Prototype_plot", cols = pal_proto_plot, label = TRUE, repel = TRUE) +
    ggtitle("UMAP (Full cells; Prototype v2)")
  if (!"SpatialLike" %in% colnames(seu_full@meta.data)) {
    seu_full$SpatialLike <- factor("Other", levels = c("Core","Interface","Edge","Other"))
  }
  p4 <- Seurat::DimPlot(seu_full, reduction = "umap",
                        group.by = "SpatialLike", label = FALSE) + ggtitle("Spatial-like UMAP (v2)")
  ggsave(file.path(visual_dir_umap, "UMAP_celltype.png"),    p1, width = 6.5, height = 5.5, dpi = 300, bg = "white")
  ggsave(file.path(visual_dir_umap, "UMAP_macroclass.png"),  p2, width = 6.5, height = 5.5, dpi = 300, bg = "white")
  ggsave(file.path(visual_dir_umap, "UMAP_prototype_v2.png"),   p3, width = 6.5, height = 5.5, dpi = 300, bg = "white")
  ggsave(file.path(visual_dir_umap, "UMAP_spatialLike_v2.png"), p4, width = 6.5, height = 5.5, dpi = 300, bg = "white")
} else {
  message("⛔ 未能生成 UMAP：跳过 UMAP 作图。")
}

cs_simple <- score_cells_by_bank_simple(seu_analysis, marker_bank, gene_weight)
getcol <- function(M, nm) if (!is.null(M) && ncol(M) && (nm %in% colnames(M))) M[, nm] else rep(0, ncol(seu_analysis))
seu_analysis <- Seurat::AddMetaData(seu_analysis, list(
  Score_P4_id = getcol(cs_simple$identity,   "Prototype4"),
  Score_P4_sp = getcol(cs_simple$spatial,    "Prototype4"),
  Score_P2_sp = getcol(cs_simple$spatial,    "Prototype2"),
  Score_P2_id = getcol(cs_simple$identity,   "Prototype2"),
  Score_P4_fn = getcol(cs_simple$functional, "Prototype4"),
  Score_P2_fn = getcol(cs_simple$functional, "Prototype2")
))
for (feat in c("Score_P4_id","Score_P4_sp","Score_P2_sp","Score_P2_id","Score_P4_fn","Score_P2_fn")) {
  fp <- Seurat::FeaturePlot(seu_analysis, reduction = "umap", features = feat, cols = c("#f7fbff","#08306b")) +
    ggtitle(paste("Feature:", feat))
  ggsave(file.path(visual_dir_umap, paste0("UMAP_feature_", feat, ".png")), fp, width = 6.5, height = 5.5, dpi = 300, bg = "white")
}

plot_proto_heat <- function(pid, top_n = 50, cells_per = 1500, obj = seu_analysis) {
  gpool <- unique(toupper(marker_bank[[pid]]$identity_up))
  get_expr <- function(obj_inner) {
    M <- try(Seurat::GetAssayData(obj_inner, slot = "data"), silent = TRUE)
    if (!inherits(M, "try-error") && !is.null(M) && nrow(M) > 0) return(M)
    M <- Seurat::GetAssayData(obj_inner, slot = "counts")
    cs <- Matrix::colSums(M); cs[!is.finite(cs) | cs <= 0] <- 1
    log1p(Matrix::t(Matrix::t(M) / cs) * 1e4)
  }
  dat <- get_expr(obj)
  rownames(dat) <- toupper(rownames(dat))
  gpool <- intersect(gpool, rownames(dat))
  if (!length(gpool)) {
    extra <- gene_weight[gene_weight$set == paste0("IDENTITY_UP:", pid), , drop = FALSE]
    extra <- extra[order(extra$weight, decreasing = TRUE), , drop = FALSE]
    gpool <- unique(toupper(extra$gene))
    gpool <- intersect(gpool, rownames(dat))
    if (!length(gpool)) {
      message("[ProtoHeatmap] 跳过 ", pid, "：缺少 identity_up 基因。")
      return(invisible(NULL))
    }
  }
  tgt_cells <- colnames(obj)[obj[[proto_col_v2]] == pid]
  if (!length(tgt_cells)) {
    if (exists("cs_simple") && is.list(cs_simple) && "identity" %in% names(cs_simple) && pid %in% colnames(cs_simple$identity)) {
      ord <- order(cs_simple$identity[, pid], decreasing = TRUE)
      tgt_cells <- colnames(obj)[head(ord, 2000)]
    } else {
      message("[ProtoHeatmap] 跳过 ", pid, "：没有对应细胞。")
      return(invisible(NULL))
    }
  } else if (length(tgt_cells) > 2000) {
    set.seed(123)
    tgt_cells <- sample(tgt_cells, 2000)
  }
  M <- dat[gpool, tgt_cells, drop = FALSE]
  if (!nrow(M) || !ncol(M)) return(invisible(NULL))
  M <- as.matrix(M)
  storage.mode(M) <- "double"
  gmean <- rowMeans(M)
  keepg <- names(sort(gmean, decreasing = TRUE))[seq_len(min(top_n, length(gmean)))]
  M <- M[keepg, , drop = FALSE]
  if (ncol(M) > cells_per) {
    if (exists("cs_simple") && is.list(cs_simple) && "identity" %in% names(cs_simple) && pid %in% colnames(cs_simple$identity)) {
      ordc <- order(cs_simple$identity[colnames(M), pid], decreasing = TRUE)
      M <- M[, colnames(M)[head(ordc, cells_per)], drop = FALSE]
    } else {
      M <- M[, seq_len(cells_per), drop = FALSE]
    }
  }
  sp_like <- if ("SpatialLike" %in% colnames(obj@meta.data)) {
    factor(obj$SpatialLike, levels = c("Core","Interface","Edge","Other"))
  } else {
    factor(rep("Other", ncol(obj)), levels = c("Core","Interface","Edge","Other"))
  }
  names(sp_like) <- colnames(obj)
  ann <- data.frame(
    MacroClass  = factor(obj$MacroClass[colnames(M)]),
    Prototype   = factor(obj[[proto_col_v2]][colnames(M), 1, drop = TRUE],
                         levels = c(proto_levels, "Unknown")),
    SpatialLike = sp_like[colnames(M)]
  )
  rownames(ann) <- colnames(M)
  
  ha <- ComplexHeatmap::HeatmapAnnotation(
    df = ann,
    col = list(
      Prototype = pal_proto_plot,
      MacroClass = c(Connective = "#1f78b4", Dead = "#6a3d9a", Epithelial = "#33a02c",
                     Inflammatory = "#e31a1c", Neoplastic = "#ff7f00", Unknown = "#bdbdbd"),
      SpatialLike = c(Core = "#08306b", Interface = "#6baed6", Edge = "#fc8d59", Other = "#bdbdbd")
    )
  )
  png(file.path(visual_dir_heat, sprintf("ProtoHeat_%s.png", pid)), width = 2600, height = 1500, res = 220)
  ht <- ComplexHeatmap::Heatmap(
    M,
    name = "Expr",
    top_annotation = ha,
    show_row_names = TRUE,
    show_column_names = FALSE,
    col = circlize::colorRamp2(c(min(M), median(M), max(M)), c("#f7fbff","#6baed6","#08306b")),
    cluster_rows = TRUE,
    cluster_columns = TRUE,
    column_title = paste0(pid, " identity_up Top", top_n, " (cells≤", ncol(M), ")")
  )
  ComplexHeatmap::draw(ht)
  dev.off()
}




invisible(lapply(c("Prototype4","Prototype2","Prototype0"), plot_proto_heat, top_n = 50))

## 固定组合：P4 identity / P2 spatial (edge 方向取负)
combo_samples <- Reduce(intersect, list(rownames(B_identity), rownames(B_spatial), names(bulk_y)))
combo_scores <- list(
  P4_id = B_identity[combo_samples, "Prototype4"],
  P2_sp = B_spatial[combo_samples, "Prototype2"]
)
if (all(vapply(combo_scores, function(v) sum(is.finite(v)) >= 5, logical(1)))) {
  df_combo <- data.frame(sample = combo_samples,
                         P4_id = combo_scores$P4_id,
                         P2_sp = combo_scores$P2_sp,
                         stringsAsFactors = FALSE)
  df_combo$label <- bulk_status[df_combo$sample]
  df_combo$y <- ifelse(df_combo$label == "pCR", 1,
                       ifelse(df_combo$label == "non_pCR", 0, NA_real_))
  keep_combo <- stats::complete.cases(df_combo[, c("P4_id","P2_sp","y")])
  df_combo <- df_combo[keep_combo, , drop = FALSE]
  if (nrow(df_combo) >= 20 && length(unique(df_combo$y)) > 1) {
    X <- scale(df_combo[, c("P4_id","P2_sp")])
    X <- as.data.frame(X)
    colnames(X) <- c("P4_id_z","P2_sp_z")
    X$P2_sp_z <- -X$P2_sp_z
    S_combo_eq <- rowSums(X)
    roc_combo <- pROC::roc(df_combo$y, S_combo_eq, quiet = TRUE)
    auc_combo <- as.numeric(pROC::auc(roc_combo))
    ci_combo <- as.numeric(pROC::ci.auc(roc_combo))
    roc_P4 <- pROC::roc(df_combo$y, X$P4_id_z, quiet = TRUE)
    auc_P4 <- as.numeric(pROC::auc(roc_P4)); ci_P4 <- as.numeric(pROC::ci.auc(roc_P4))
    roc_P2 <- pROC::roc(df_combo$y, X$P2_sp_z, quiet = TRUE)
    auc_P2 <- as.numeric(pROC::auc(roc_P2)); ci_P2 <- as.numeric(pROC::ci.auc(roc_P2))
    png(file.path(outdir, "Fig_ROC_fixed_combo.png"), 900, 750, res = 150)
    plot(roc_combo, col="#e41a1c", lwd=2.5, legacy.axes=TRUE,
         main=sprintf("Fixed combo ROC (AUC=%.3f)", auc_combo))
    plot(roc_P4,   col="#377eb8", lwd=2, add=TRUE)
    plot(roc_P2,   col="#4daf4a", lwd=2, add=TRUE)
    legend("bottomright",
           c(sprintf("Combo(eq) AUC=%.3f", auc_combo),
             sprintf("P4_id AUC=%.3f", auc_P4),
             sprintf("P2_sp(edge−) AUC=%.3f", auc_P2)),
           col=c("#e41a1c","#377eb8","#4daf4a"), lwd=c(2.5,2,2), bty="n")
    dev.off()
    readr::write_csv(data.frame(
      sample = df_combo$sample,
      P4_id_z = X$P4_id_z,
      P2_sp_z = X$P2_sp_z,
      Combo_eq = S_combo_eq,
      y = df_combo$y
    ), file.path(outdir, "Bulk_scores_fixed_combo.csv"))
    readr::write_csv(data.frame(
      Feature = c("P4_id_z","P2_sp_z","Combo_eq"),
      AUC = c(auc_P4, auc_P2, auc_combo),
      CI_low = c(ci_P4[1], ci_P2[1], ci_combo[1]),
      CI_high = c(ci_P4[3], ci_P2[3], ci_combo[3])
    ), file.path(outdir, "Bulk_AUC_fixed_combo.csv"))
  } else {
    message("⚠️ 固定组合有效样本不足，跳过 ROC 绘制。")
  }
} else {
  message("⚠️ 缺少 P4/P2 bulk 打分，无法生成固定组合 ROC。")
}

if (exists("B_identity") && exists("B_spatial") && exists("B_functional")) {
  score_long <- list(
    as.data.frame(B_identity) %>%
      tibble::rownames_to_column("sample") %>%
      tidyr::pivot_longer(-sample, names_to = "Prototype", values_to = "score") %>%
      dplyr::mutate(Track = "identity(label-free)"),
    as.data.frame(B_spatial) %>%
      tibble::rownames_to_column("sample") %>%
      tidyr::pivot_longer(-sample, names_to = "Prototype", values_to = "score") %>%
      dplyr::mutate(Track = "spatial(label-free)"),
    as.data.frame(B_functional) %>%
      tibble::rownames_to_column("sample") %>%
      tidyr::pivot_longer(-sample, names_to = "Prototype", values_to = "score") %>%
      dplyr::mutate(Track = "functional(pCR-informed)")
  ) %>% dplyr::bind_rows()
  score_long$label <- pheno[score_long$sample, "pcr_response"]
  score_long <- subset(score_long, !is.na(label))
  score_long$label <- factor(score_long$label, levels = c("non_pCR","pCR"))
  p_box <- ggplot(score_long, aes(x = label, y = score, fill = label)) +
    geom_boxplot(alpha = 0.6, outlier.shape = NA) +
    ggbeeswarm::geom_quasirandom(width = 0.15, alpha = 0.5, size = 1.4) +
    facet_grid(Track ~ Prototype, scales = "free_y") +
    scale_fill_manual(values = pal_resp) +
    theme_pub2() + labs(x = "pCR status", y = "Score")
  # 计算每个 (Track, Prototype) 的 Wilcoxon，并 FDR 矫正
  p_tab <- score_long %>%
    dplyr::group_by(Track, Prototype) %>%
    dplyr::summarise(p = tryCatch(wilcox.test(score ~ label)$p.value, error = function(e) NA_real_), .groups = "drop") %>%
    dplyr::mutate(p_adj = p.adjust(p, "BH"),
                  lab = dplyr::case_when(
                    !is.finite(p_adj) ~ "NA",
                    p_adj < 0.001 ~ "***",
                    p_adj < 0.01  ~ "**",
                    p_adj < 0.05  ~ "*",
                    TRUE ~ "ns"))
  
  
  # 计算每个面板的顶端 y 位置，留一点空间
  ypos <- score_long %>%
    dplyr::group_by(Track, Prototype) %>%
    dplyr::summarise(y = max(score, na.rm = TRUE), .groups = "drop") %>%
    dplyr::mutate(y = y + 0.05 * abs(y))
  
  p_box_sig <- p_box +
    geom_text(
      data = dplyr::left_join(p_tab, ypos, by = c("Track","Prototype")),
      aes(x = 1.5, y = y, label = lab),    # 1.5 介于 non_pCR 与 pCR 中间
      inherit.aes = FALSE, size = 3.2
    )
  
  ggsave(file.path(outdir, "Fig_Boxplots_scores_sig.png"),
         p_box_sig, width = 10, height = 7, dpi = 320, bg = "white")
  
}














if (exists("assoc_all") && nrow(assoc_all)) {
  tab_plot <- assoc_all %>%
    dplyr::mutate(
      rb_low = rb_from_auc(as.numeric(ci_low)),
      rb_high = rb_from_auc(as.numeric(ci_high)),
      Track = factor(Track, levels = c("identity(label-free)","spatial(label-free)")),
      Prototype = factor(Prototype, levels = proto_levels)
    ) %>%
    dplyr::arrange(Track, Prototype)
  if (nrow(tab_plot)) {
    pB <- ggplot(tab_plot, aes(x = rb, y = forcats::fct_rev(interaction(Track, Prototype, sep = " • ")), color = Prototype)) +
      geom_vline(xintercept = 0, linewidth = 0.6, linetype = 2, color = "#999999") +
      geom_errorbarh(aes(xmin = rb_low, xmax = rb_high), height = 0.15, linewidth = 0.9, alpha = 0.9) +
      geom_point(size = 2.8) +
      geom_text(aes(label = sprintf("AUC=%.3f | ρ=%+.3f", as.numeric(auc), as.numeric(spearman))),
                nudge_x = 0.04, hjust = 0, size = 3.0, color = "black") +
      scale_color_manual(values = pal_proto_plot) +
      scale_x_continuous(labels = scales::number_format(accuracy = 0.01)) +
      theme_pub2() + theme(axis.title.y = element_blank()) +
      labs(x = "Rank-biserial (2*AUC−1)", title = "Effect sizes by track & prototype")
    ggsave(file.path(outdir, "Fig_forest_auc_rb_spearman.png"), pB, width = 10.5, height = 5.8, dpi = 320, bg = "white")
  }
}


if (exists("marker_bank") && exists("bulk_mat")) {
  g_p4_up <- intersect(toupper(marker_bank[["Prototype4"]]$identity_up), toupper(rownames(bulk_mat)))
  g_p2_edge <- intersect(toupper(marker_bank[["Prototype2"]]$edge), toupper(rownames(bulk_mat)))
  background_genes <- toupper(rownames(bulk_mat))
  
  run_kegg_cp <- function(gene_set, label) {
    if (!length(gene_set)) return(NULL)
    out_csv <- file.path(visual_dir_enrich, paste0("KEGG_", label, ".csv"))
    out_png <- file.path(visual_dir_enrich, paste0("KEGG_", label, "_dotplot.png"))
    if (requireNamespace("clusterProfiler", quietly = TRUE) && requireNamespace("org.Hs.eg.db", quietly = TRUE)) {
      library(clusterProfiler)
      library(org.Hs.eg.db)
      eg <- suppressWarnings(clusterProfiler::bitr(gene_set, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db))$ENTREZID
      eg_bg <- suppressWarnings(clusterProfiler::bitr(background_genes, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db))$ENTREZID
      ek <- try(clusterProfiler::enrichKEGG(eg, universe = eg_bg, organism = "hsa", pAdjustMethod = "BH", qvalueCutoff = 0.2), silent = TRUE)
      if (!inherits(ek, "try-error") && !is.null(ek) && nrow(as.data.frame(ek))) {
        df <- as.data.frame(ek)
        readr::write_csv(df, out_csv)
        p <- clusterProfiler::dotplot(ek, showCategory = 20) + ggtitle(paste0("KEGG ORA: ", label))
        ggsave(out_png, p, width = 7.5, height = 6.2, dpi = 300, bg = "white")
      }
    } else {
      if (!requireNamespace("msigdbr", quietly = TRUE)) install.packages("msigdbr", repos = "https://cloud.r-project.org")
      library(msigdbr)
      m_kegg <- msigdbr(species = "Homo sapiens", category = "C2", subcategory = "CP:KEGG") %>%
        dplyr::select(gs_name, gene_symbol) %>%
        dplyr::mutate(gene_symbol = toupper(gene_symbol))
      ora <- m_kegg %>% dplyr::group_by(gs_name) %>% dplyr::summarise(
        k = sum(gene_symbol %in% gene_set),
        K = dplyr::n(),
        n = length(gene_set),
        N = length(background_genes),
        p = phyper(k - 1, K, N - K, n, lower.tail = FALSE),
        .groups = "drop"
      ) %>% dplyr::mutate(padj = p.adjust(p, "BH")) %>% dplyr::arrange(padj)
      readr::write_csv(ora, out_csv)
      top <- head(ora, 20)
      if (nrow(top)) {
        top$gs_name <- factor(top$gs_name, levels = rev(top$gs_name))
        p <- ggplot(top, aes(x = gs_name, y = -log10(padj), size = k / n)) +
          geom_point() + coord_flip() + theme_pub2() + labs(title = paste0("KEGG ORA (Fisher): ", label), x = NULL, y = "-log10(FDR)")
        ggsave(out_png, p, width = 7.5, height = 6.2, dpi = 300, bg = "white")
      }
    }
  }
  
  run_kegg_cp(g_p4_up, "P4_identity_up")
  run_kegg_cp(g_p2_edge, "P2_edge")
}

## 基因层显著性（可选）
if (exists("bulk_mat") && exists("marker_bank")) {
  mat_gene <- bulk_mat
  rownames(mat_gene) <- toupper(rownames(mat_gene))
  sample_order <- colnames(mat_gene)
  label_gene <- bulk_status[sample_order]
  y_gene <- ifelse(label_gene == "pCR", 1, ifelse(label_gene == "non_pCR", 0, NA_real_))
  keep_gene <- is.finite(y_gene)
  mat_gene <- mat_gene[, keep_gene, drop = FALSE]
  y_gene <- y_gene[keep_gene]
  if (length(unique(y_gene)) > 1) {
    collect_gene_info <- function(tag, proto, genes) {
      if (!length(genes)) return(NULL)
      track <- if (tag %in% c("identity_up","identity_down")) "identity" else "spatial"
      direction <- if (tag %in% c("identity_up","core")) "positive" else "negative"
      data.frame(
        Prototype = proto,
        Signature = tag,
        Track = track,
        Direction = direction,
        Gene = toupper(genes),
        stringsAsFactors = FALSE
      )
    }
    gene_meta <- lapply(names(marker_bank), function(pid){
      mb <- marker_bank[[pid]]
      do.call(rbind, lapply(c("identity_up","identity_down","core","edge"), function(tag){
        collect_gene_info(tag, pid, mb[[tag]])
      }))
    })
    gene_meta <- dplyr::bind_rows(gene_meta)
    gene_meta <- gene_meta[dplyr::between(nchar(gene_meta$Gene), 1, Inf), ]
    gene_meta <- gene_meta[gene_meta$Gene %in% rownames(mat_gene), , drop = FALSE]
    gene_meta <- dplyr::distinct(gene_meta)
    if (nrow(gene_meta)) {
      gene_stats <- apply(gene_meta, 1, function(info){
        gene <- info[["Gene"]]
        expr <- as.numeric(mat_gene[gene, ])
        if (sum(is.finite(expr)) < 5 || length(unique(expr)) < 2) {
          return(c(AUC=NA, CI_low=NA, CI_high=NA, p_value=NA, Spearman=NA, RB=NA))
        }
        r_s <- safe_cor(expr, y_gene)
        roc_obj <- try(pROC::roc(y_gene, expr, quiet = TRUE, direction = "auto"), silent = TRUE)
        if (inherits(roc_obj, "try-error")) {
          auc <- ci_low <- ci_high <- p_delong <- NA_real_
        } else {
          auc <- as.numeric(pROC::auc(roc_obj))
          ci <- as.numeric(pROC::ci.auc(roc_obj))
          ci_low <- ci[1]; ci_high <- ci[3]
          p_delong <- try(pROC::roc.test(roc_obj, auc = 0.5, method = "delong")$p.value, silent = TRUE)
          if (inherits(p_delong, "try-error")) p_delong <- NA_real_
        }
        rb <- rb_from_auc(auc)
        c(AUC = auc, CI_low = ci_low, CI_high = ci_high, p_value = p_delong, Spearman = r_s, RB = rb)
      })
      gene_stats <- as.data.frame(t(gene_stats))
      gene_table <- cbind(gene_meta, gene_stats)
      gene_table$FDR <- p.adjust(as.numeric(gene_table$p_value), method = "BH")
      expected_sign <- ifelse(gene_table$Direction == "positive", 1, -1)
      gene_table$direction_match <- ifelse(is.finite(gene_table$RB), gene_table$RB * expected_sign >= 0, NA)
      readr::write_csv(gene_table, file.path(outdir, "bulk_marker_gene_auc_v2.csv"))
      sig_gene <- subset(gene_table, is.finite(AUC) & AUC >= 0.60 & is.finite(FDR) & FDR < 0.05 & isTRUE(direction_match))
      if (nrow(sig_gene)) {
        readr::write_csv(sig_gene, file.path(outdir, "significant_markers_v2.csv"))
      }
    }
  }
}





























## =========================================================
## 5) CellChat：基于 Prototype/Spatial × Macro 的群组通信分析
## =========================================================
stopifnot(exists("seu_analysis"), inherits(seu_analysis, "Seurat"))
if (!requireNamespace("CellChat", quietly = TRUE)) {
  stop("未检测到 CellChat，请先安装：BiocManager::install('CellChat')")
}
suppressPackageStartupMessages(library(CellChat))


# ============ 安全写入/保存工具 ============
.norm_path <- function(p) normalizePath(p, winslash = "/", mustWork = FALSE)
.ensure_dir <- function(p){ dir.create(p, recursive = TRUE, showWarnings = FALSE); p }

.safe_write_rds <- function(obj, file){
  file <- .norm_path(file)
  tryCatch({ saveRDS(obj, file); TRUE }, error = function(e){
    tf <- tempfile(fileext = ".rds"); saveRDS(obj, tf)
    if (!file.copy(tf, file, overwrite = TRUE)) stop("saveRDS 兜底复制失败：", conditionMessage(e))
    unlink(tf); TRUE
  })
}

.safe_write_csv <- function(df, file){
  file <- .norm_path(file)
  tryCatch({ readr::write_csv(df, file); TRUE }, error = function(e){
    tf <- tempfile(fileext = ".csv"); readr::write_csv(df, tf)
    if (!file.copy(tf, file, overwrite = TRUE)) stop("write_csv 兜底复制失败：", conditionMessage(e))
    unlink(tf); TRUE
  })
}

.safe_png <- function(file, width, height, res, draw_expr){
  file <- .norm_path(file)
  ok <- FALSE
  try({
    grDevices::png(file, width = width, height = height, res = res, bg = "white")
    print(draw_expr)  # 对非ggplot对象同样可用（grid/grob也行）
    grDevices::dev.off()
    ok <- TRUE
  }, silent = TRUE)
  if (!ok){
    # 兜底：先画到临时文件，再复制
    tf <- tempfile(fileext = ".png")
    grDevices::png(tf, width = width, height = height, res = res, bg = "white")
    print(draw_expr)
    grDevices::dev.off()
    if (!file.copy(tf, file, overwrite = TRUE)) stop("png 兜底复制失败：", file)
    unlink(tf)
  }
  invisible(TRUE)
}

.safe_ggsave <- function(file, plot, width, height, dpi){
  file <- .norm_path(file)
  ok <- FALSE
  if (inherits(plot, "ggplot")){
    try({
      ggplot2::ggsave(file, plot, width = width, height = height, dpi = dpi, bg = "white")
      ok <- TRUE
    }, silent = TRUE)
  }
  if (!ok){
    # 兜底：先写到临时 png 再复制
    tf <- tempfile(fileext = ".png")
    ggplot2::ggsave(tf, plot, width = width, height = height, dpi = dpi, bg = "white")
    if (!file.copy(tf, file, overwrite = TRUE)) stop("ggsave 兜底复制失败：", file)
    unlink(tf)
  }
  invisible(TRUE)
}



## ---- 5.1：准备两种分组（proto_macro / spatial_macro）----
prepare_cellchat_object <- function(obj){
  stopifnot(inherits(obj, "Seurat"))
  if (!("SpatialLike" %in% colnames(obj@meta.data))) {
    stop("seu@meta.data 缺少 SpatialLike，请先在 seu_full 上计算。")
  }
  need_cols <- c("MacroClass", proto_col_v2, "SpatialLike")
  miss <- setdiff(need_cols, colnames(obj@meta.data))
  if (length(miss)) stop("seu@meta.data 缺少列：", paste(miss, collapse=", "))
  proto_vals <- as.character(obj[[proto_col_v2]][,1])
  proto_factor <- factor(proto_vals, levels = proto_levels, exclude = NA)
  obj[[proto_col_v2]] <- proto_factor
  obj$proto_macro   <- droplevels(interaction(proto_factor, obj$MacroClass, sep="|", drop=TRUE))
  obj$spatial_macro <- droplevels(interaction(obj$SpatialLike,       obj$MacroClass, sep="|", drop=TRUE))
  obj
}

seu_analysis <- prepare_cellchat_object(seu_analysis)

## ---- 5.2：通用运行函数（含下采样、计算、作图、保存）----
run_cellchat_batched_downsampled <- function(
  seu, outdir, species = c("human","mouse"),
  grouping_mode = "proto_macro",
  max_cells_per_group = 1200, min_cells_group = 50,
  do_pathways = TRUE, generate_plots = TRUE,
  max_total_cells = 8000,
  workers = 1, future_timeout = 600, globals_maxsize_gb = 8,
  filter_lr_genes = TRUE,
  group_whitelist = NULL,
  out_name = "proto_macro"
){
  stopifnot(inherits(seu, "Seurat"))
  species <- match.arg(species)
  
  if (species == "human") {
    if (!exists("CellChatDB.human")) data("CellChatDB.human", package = "CellChat")
    DB <- CellChatDB.human
  } else {
    if (!exists("CellChatDB.mouse")) data("CellChatDB.mouse", package = "CellChat")
    DB <- CellChatDB.mouse
  }
  
  ## — 分组与白名单 —
  lab_chr <- as.character(seu[[grouping_mode]][, 1])
  tab_all <- sort(table(lab_chr), decreasing = TRUE)
  keep_groups <- names(tab_all)[tab_all >= min_cells_group]
  if (!is.null(group_whitelist)) keep_groups <- intersect(keep_groups, group_whitelist)
  if (!length(keep_groups)) stop("[CellChat] 有效分组为 0：请放宽阈值或检查白名单。")
  if (length(keep_groups) < 2)  stop("[CellChat] 分组数量不足 2。")
  
  pick_cells <- unlist(lapply(keep_groups, function(g){
    idx <- which(lab_chr == g)
    if (length(idx) > max_cells_per_group) idx <- sample(idx, max_cells_per_group)
    colnames(seu)[idx]
  }), use.names = FALSE)
  if (length(pick_cells) > max_total_cells) {
    set.seed(123); pick_cells <- sample(pick_cells, max_total_cells)
  }
  
  obj <- subset(seu, cells = pick_cells)
  labels_chr <- as.character(obj[[grouping_mode]][, 1])
  labels <- droplevels(factor(labels_chr))
  
  ## — 表达矩阵，尽量保持稀疏 —
  data.input <- Seurat::GetAssayData(obj, slot = "data")
  if (inherits(data.input, "dgCMatrix")) {
    if (!is.double(data.input@x)) data.input@x <- as.numeric(data.input@x)
  } else {
    suppressWarnings({
      data.input <- try(Matrix::Matrix(data.input, sparse = TRUE), silent = TRUE)
    })
    if (inherits(data.input, "try-error")) {
      data.input <- as.matrix(data.input); storage.mode(data.input) <- "double"
    }
  }
  if (isTRUE(filter_lr_genes)) {
    lr_upper <- unique(toupper(c(DB$interaction$ligand, DB$interaction$receptor)))
    rn_upper <- toupper(rownames(data.input))
    keep_lr  <- rn_upper %in% lr_upper
    if (sum(keep_lr) > 0) data.input <- data.input[keep_lr, , drop = FALSE]
    nz <- Matrix::rowSums(data.input) > 0
    if (any(!nz)) data.input <- data.input[nz, , drop = FALSE]
  }
  
  meta <- data.frame(labels = labels, row.names = colnames(obj))
  
  ## — 计划/内存 —
  op0 <- list(plan = future::plan(),
              gm   = getOption("future.globals.maxSize"),
              ct   = getOption("future.makeNodePSOCK.connectTimeout"))
  on.exit({
    future::plan(op0$plan)
    options(future.globals.maxSize = op0$gm,
            future.makeNodePSOCK.connectTimeout = op0$ct)
  }, add = TRUE)
  future::plan(sequential)
  options(future.globals.maxSize = globals_maxsize_gb * 1024^3,
          future.makeNodePSOCK.connectTimeout = future_timeout)
  
  ## — 主流程 —
  cellchat <- CellChat::createCellChat(object = data.input, meta = meta, group.by = "labels")
  cellchat@DB <- DB
  cellchat <- CellChat::subsetData(cellchat)
  cellchat <- CellChat::identifyOverExpressedGenes(cellchat)
  cellchat <- CellChat::identifyOverExpressedInteractions(cellchat)
  if (workers > 1) try(future::plan(multisession, workers = workers), silent = TRUE)
  cellchat <- CellChat::computeCommunProb(cellchat, population.size = TRUE)
  cellchat <- CellChat::filterCommunication(cellchat, min.cells = 10)
  if (isTRUE(do_pathways)) cellchat <- CellChat::computeCommunProbPathway(cellchat)
  cellchat <- CellChat::aggregateNet(cellchat)
  
  ## — 输出目录/表 —
  outd <- file.path(outdir, paste0("CellChat_", out_name))
  dir.create(outd, showWarnings = FALSE, recursive = TRUE)
  saveRDS(cellchat, file.path(outd, "cellchat.rds"))
  comm_all  <- try(CellChat::subsetCommunication(cellchat), silent = TRUE)
  comm_path <- if (isTRUE(do_pathways)) try(CellChat::subsetCommunication(cellchat, slot.name = "netP"), silent = TRUE) else NULL
  if (!inherits(comm_all, "try-error")  && !is.null(comm_all)  && nrow(comm_all))  utils::write.csv(comm_all,  file.path(outd, "communication_table.csv"), row.names = FALSE)
  if (!inherits(comm_path, "try-error") && !is.null(comm_path) && nrow(comm_path)) utils::write.csv(comm_path, file.path(outd, "communication_table_pathway.csv"), row.names = FALSE)
  
  ## — 圈图（可选） —
  if (isTRUE(generate_plots)) {
    grSize <- as.numeric(table(cellchat@idents))
    png(file.path(outd, "net_circle_overall.png"), 1600, 1600, res = 200)
    CellChat::netVisual_circle(cellchat@net$count, weight.scale = TRUE, label.edge = FALSE,
                               vertex.weight = grSize, title.name = "Number of interactions")
    dev.off()
    png(file.path(outd, "net_circle_overall_weighted.png"), 1600, 1600, res = 200)
    CellChat::netVisual_circle(cellchat@net$weight, weight.scale = TRUE, label.edge = FALSE,
                               vertex.weight = grSize, title.name = "Interaction weights")
    dev.off()
  }
  
  ## — FigD1 / FigD2：发送/接收强度 —
  W <- cellchat@net$weight
  gr <- colnames(W)
  df_out <- data.frame(group = gr, strength = as.numeric(rowSums(W)))
  df_in  <- data.frame(group = gr, strength = as.numeric(colSums(W)))
  p1 <- ggplot2::ggplot(df_out, ggplot2::aes(x = reorder(group, strength), y = strength)) +
    ggplot2::geom_col() + ggplot2::coord_flip() + ggplot2::theme_bw() +
    ggplot2::labs(x = NULL, y = "Outgoing strength", title = "Sender strength")
  ggplot2::ggsave(file.path(outd, "FigD1_Sender_Strength.png"), p1, width = 7.5, height = 6.2, dpi = 300, bg = "white")
  p2 <- ggplot2::ggplot(df_in, ggplot2::aes(x = reorder(group, strength), y = strength)) +
    ggplot2::geom_col() + ggplot2::coord_flip() + ggplot2::theme_bw() +
    ggplot2::labs(x = NULL, y = "Incoming strength", title = "Receiver strength")
  ggplot2::ggsave(file.path(outd, "FigD2_Receiver_Strength.png"), p2, width = 7.5, height = 6.2, dpi = 300, bg = "white")
  
  ## — FigG：Pathway bubbles —
  if (isTRUE(do_pathways)) {
    ws <- vapply(cellchat@netP$netP, function(m) if (is.null(m)) 0 else sum(m, na.rm = TRUE), numeric(1))
    pathways.show <- head(names(sort(ws, decreasing = TRUE)), 12)
    if (length(pathways.show)) {
      p <- CellChat::netVisual_bubble(cellchat, pathways.show = pathways.show, remove.isolate = TRUE)
      ggplot2::ggsave(file.path(outd, "FigG_Pathway_Bubbles.png"), p, width = 14, height = 10, dpi = 300, bg = "white")
    }
    cellchat <- CellChat::netAnalysis_computeCentrality(cellchat, slot.name = "netP")
    pnet <- CellChat::netAnalysis_signalingRole_network(cellchat, slot.name = "netP", width = 10, height = 8)
    ggplot2::ggsave(file.path(outd, "role_network_sender_receiver.png"), pnet, width = 10, height = 8, dpi = 300, bg = "white")
  }
  
  ## 角色散点 & 热图（用设备保存最稳）
  png(file.path(outd, "role_scatter_sender_receiver.png"), 1600, 1200, res = 200)
  print(CellChat::netAnalysis_signalingRole_scatter(cellchat, slot.name = "netP", title = "Sender vs Receiver roles"))
  dev.off()
  png(file.path(outd, "net_heatmap_pathway_group.png"), 2000, 1300, res = 220)
  print(CellChat::netVisual_heatmap(cellchat, slot.name = "netP"))
  dev.off()
  
  message("✅ CellChat 完成：", out_name, "；目录：", outd)
  invisible(cellchat)
}
run_cellchat_batched_downsampled <- function(
  seu, outdir, species = c("human","mouse"),
  grouping_mode = "proto_macro",
  max_cells_per_group = 1200, min_cells_group = 50,
  do_pathways = TRUE, generate_plots = TRUE,
  max_total_cells = 8000,
  workers = 1, future_timeout = 600, globals_maxsize_gb = 8,
  filter_lr_genes = TRUE,
  group_whitelist = NULL,
  out_name = "proto_macro"
){
  stopifnot(inherits(seu, "Seurat"))
  species <- match.arg(species)
  
  # ---- DB ----
  DB <- switch(species,
               human = { if (!exists("CellChatDB.human")) data("CellChatDB.human", package = "CellChat"); CellChatDB.human },
               mouse = { if (!exists("CellChatDB.mouse")) data("CellChatDB.mouse", package = "CellChat"); CellChatDB.mouse }
  )
  
  # ---- 分组+白名单 ----
  lab_chr  <- as.character(seu[[grouping_mode]][, 1])
  tab_all  <- sort(table(lab_chr), decreasing = TRUE)
  keep_groups <- names(tab_all)[tab_all >= min_cells_group]
  if (!is.null(group_whitelist)) keep_groups <- intersect(keep_groups, group_whitelist)
  if (!length(keep_groups)) stop("[CellChat] 有效分组为 0（阈值过严或白名单为空）。")
  if (length(keep_groups) < 2) stop("[CellChat] 分组数量不足 2。")
  
  pick_cells <- unlist(lapply(keep_groups, function(g){
    idx <- which(lab_chr == g)
    if (length(idx) > max_cells_per_group) idx <- sample(idx, max_cells_per_group)
    colnames(seu)[idx]
  }), use.names = FALSE)
  if (length(pick_cells) > max_total_cells) { set.seed(123); pick_cells <- sample(pick_cells, max_total_cells) }
  
  obj <- subset(seu, cells = pick_cells)
  labels_chr <- as.character(obj[[grouping_mode]][, 1])
  labels <- droplevels(factor(labels_chr))
  
  # ---- 表达矩阵（稀疏）----
  data.input <- Seurat::GetAssayData(obj, slot = "data")
  if (inherits(data.input, "dgCMatrix")) {
    if (!is.double(data.input@x)) data.input@x <- as.numeric(data.input@x)
  } else {
    suppressWarnings({ data.input <- try(Matrix::Matrix(data.input, sparse = TRUE), silent = TRUE) })
    if (inherits(data.input, "try-error")) { data.input <- as.matrix(data.input); storage.mode(data.input) <- "double" }
  }
  if (isTRUE(filter_lr_genes)) {
    lr_upper <- unique(toupper(c(DB$interaction$ligand, DB$interaction$receptor)))
    rn_upper <- toupper(rownames(data.input))
    keep_lr  <- rn_upper %in% lr_upper
    if (sum(keep_lr) > 0) data.input <- data.input[keep_lr, , drop = FALSE]
    nz <- Matrix::rowSums(data.input) > 0
    if (any(!nz)) data.input <- data.input[nz, , drop = FALSE]
  }
  
  meta <- data.frame(labels = labels, row.names = colnames(obj))
  
  # ---- 计划/内存 ----
  op0 <- list(plan = future::plan(), gm = getOption("future.globals.maxSize"),
              ct = getOption("future.makeNodePSOCK.connectTimeout"))
  on.exit({ future::plan(op0$plan); options(future.globals.maxSize = op0$gm,
                                            future.makeNodePSOCK.connectTimeout = op0$ct) }, add = TRUE)
  future::plan(sequential)
  options(future.globals.maxSize = globals_maxsize_gb * 1024^3,
          future.makeNodePSOCK.connectTimeout = future_timeout)
  
  # ---- 主流程 ----
  cellchat <- CellChat::createCellChat(object = data.input, meta = meta, group.by = "labels")
  cellchat@DB <- DB
  cellchat <- CellChat::subsetData(cellchat)
  cellchat <- CellChat::identifyOverExpressedGenes(cellchat)
  cellchat <- CellChat::identifyOverExpressedInteractions(cellchat)
  if (workers > 1) try(future::plan(multisession, workers = workers), silent = TRUE)
  cellchat <- CellChat::computeCommunProb(cellchat, population.size = TRUE)
  cellchat <- CellChat::filterCommunication(cellchat, min.cells = 10)
  if (isTRUE(do_pathways)) cellchat <- CellChat::computeCommunProbPathway(cellchat)
  cellchat <- CellChat::aggregateNet(cellchat)
  
  # ---- 输出目录 ----
  outd <- file.path(.norm_path(outdir), paste0("CellChat_", out_name))
  .ensure_dir(outd)
  
  .safe_write_rds(cellchat, file.path(outd, "cellchat.rds"))
  comm_all  <- try(CellChat::subsetCommunication(cellchat), silent = TRUE)
  comm_path <- if (isTRUE(do_pathways)) try(CellChat::subsetCommunication(cellchat, slot.name = "netP"), silent = TRUE) else NULL
  if (!inherits(comm_all, "try-error")  && !is.null(comm_all)  && nrow(comm_all))  .safe_write_csv(comm_all,  file.path(outd, "communication_table.csv"))
  if (!inherits(comm_path, "try-error") && !is.null(comm_path) && nrow(comm_path)) .safe_write_csv(comm_path, file.path(outd, "communication_table_pathway.csv"))
  
  # ---- 圈图（两张必须产出）----
  if (isTRUE(generate_plots)) {
    grSize <- as.numeric(table(cellchat@idents))
    .safe_png(file.path(outd, "net_circle_overall.png"),        1600, 1600, 200,
              CellChat::netVisual_circle(cellchat@net$count,  weight.scale = TRUE, label.edge = FALSE,
                                         vertex.weight = grSize, title.name = "Number of interactions"))
    .safe_png(file.path(outd, "net_circle_overall_weighted.png"),1600, 1600, 200,
              CellChat::netVisual_circle(cellchat@net$weight, weight.scale = TRUE, label.edge = FALSE,
                                         vertex.weight = grSize, title.name = "Interaction weights"))
  }
  
  # ---- FigD1 / FigD2：发送/接收强度（两张必须产出）----
  W  <- cellchat@net$weight
  gr <- colnames(W)
  df_out <- data.frame(group = gr, strength = as.numeric(rowSums(W)))
  df_in  <- data.frame(group = gr, strength = as.numeric(colSums(W)))
  p1 <- ggplot2::ggplot(df_out, ggplot2::aes(x = reorder(group, strength), y = strength)) +
    ggplot2::geom_col() + ggplot2::coord_flip() + ggplot2::theme_bw() +
    ggplot2::labs(x = NULL, y = "Outgoing strength", title = "Sender strength")
  p2 <- ggplot2::ggplot(df_in,  ggplot2::aes(x = reorder(group, strength), y = strength)) +
    ggplot2::geom_col() + ggplot2::coord_flip() + ggplot2::theme_bw() +
    ggplot2::labs(x = NULL, y = "Incoming strength", title = "Receiver strength")
  .safe_ggsave(file.path(outd, "FigD1_Sender_Strength.png"),   p1, 7.5, 6.2, 300)
  .safe_ggsave(file.path(outd, "FigD2_Receiver_Strength.png"), p2, 7.5, 6.2, 300)
  
  # ---- FigG：Pathway bubbles（必须产出，如无通路则跳过）----
  if (isTRUE(do_pathways)) {
    ws <- vapply(cellchat@netP$netP, function(m) if (is.null(m)) 0 else sum(m, na.rm = TRUE), numeric(1))
    pathways.show <- head(names(sort(ws, decreasing = TRUE)), 12)
    if (length(pathways.show)) {
      p_bub <- try(CellChat::netVisual_bubble(cellchat, pathways.show = pathways.show, remove.isolate = TRUE), silent = TRUE)
      if (inherits(p_bub, "ggplot")) {
        .safe_ggsave(file.path(outd, "FigG_Pathway_Bubbles.png"), p_bub, 14, 10, 300)
      } else {
        .safe_png(file.path(outd, "FigG_Pathway_Bubbles.png"), 1800, 1000, 200,
                  CellChat::netVisual_bubble(cellchat, pathways.show = pathways.show, remove.isolate = TRUE))
      }
    }
    # 角色网络图（必须产出）
    cellchat <- CellChat::netAnalysis_computeCentrality(cellchat, slot.name = "netP")
    .safe_png(file.path(outd, "role_network_sender_receiver.png"), 1600, 1200, 220,
              CellChat::netAnalysis_signalingRole_network(cellchat, slot.name = "netP", width = 10, height = 8))
  }
  
  # ---- 角色散点 & 热图（两张必须产出）----
  .safe_png(file.path(outd, "role_scatter_sender_receiver.png"), 1600, 1200, 200,
            CellChat::netAnalysis_signalingRole_scatter(cellchat, slot.name = "netP", title = "Sender vs Receiver roles"))
  .safe_png(file.path(outd, "net_heatmap_pathway_group.png"),    2000, 1300, 220,
            CellChat::netVisual_heatmap(cellchat, slot.name = "netP"))
  
  message("✅ CellChat 完成：", out_name, "；目录：", outd)
  invisible(cellchat)
}





















## ---- 5.3：分别按两种分组运行 ----
# ——阈值：AUC 或 RB（=2*AUC−1 的绝对值）——
thr_auc <- 0.60
thr_rb  <- 0.15

proto_focus <- assoc_proto$Prototype[
  (is.finite(assoc_proto$auc_max) & assoc_proto$auc_max >= thr_auc) |
    (is.finite(assoc_proto$evidence) & assoc_proto$evidence >= thr_rb)
]
if (!length(proto_focus)) proto_focus <- head(assoc_proto$Prototype, 2)  # 兜底
proto_focus <- intersect(proto_focus, proto_levels)

# ——把 Proto×Macro 的组合里，原型属于 proto_focus 的 group 选出来——
min_cells_group <- 50
grp_all <- droplevels(seu_analysis$proto_macro)
tab_grp <- sort(table(grp_all), decreasing = TRUE)
group_whitelist <- names(tab_grp)[
  tab_grp >= min_cells_group &
    grepl(paste0("^(", paste(proto_focus, collapse="|"), ")\\|"), names(tab_grp))
]
message("[CellChat] 选用 Proto×Macro 组：", paste(group_whitelist, collapse=", "))



# Proto × Macro
cellchat_proto <- run_cellchat_batched_downsampled(
  seu = seu_analysis, outdir = outdir,
  species = "human",
  grouping_mode = "proto_macro",
  max_cells_per_group = 1200, min_cells_group = min_cells_group,
  do_pathways = TRUE, generate_plots = TRUE,
  max_total_cells = 8000,
  workers = 1, globals_maxsize_gb = 8,
  filter_lr_genes = TRUE,
  group_whitelist = group_whitelist,   # 只分析强证据原型对应的组
  out_name = "proto_macro"             # 统一输出到 CellChat_proto_macro/
)







# Spatial × Macro
cellchat_spatial <- run_cellchat_batched_downsampled(
  seu_analysis, outdir,
  species = "human",
  grouping_mode = "spatial_macro",
  max_cells_per_group = 1200, min_cells_group = 50,
  do_pathways = TRUE, generate_plots = TRUE,
  chord_topN = 120, lollipop_topN = 60, radial_topE = 200,
  max_total_cells = 8000,
  workers = 1,
  globals_maxsize_gb = 8,
  filter_lr_genes = TRUE
)

message("✅ CellChat 全量分析完成（proto_macro & spatial_macro）。")


if (exists("cellchat_proto") && !is.null(cellchat_proto)) {
  cc <- cellchat_proto
  glabels <- colnames(cc@net$weight)
  is_p4 <- grepl("^Prototype4\\|", glabels)
  W <- cc@net$weight
  rownames(W) <- colnames(W) <- glabels
  
  # 圆图（已在函数里输出过整体图；这里是你原有的 P4 强调版）
  if (length(glabels)) {
    send_P4 <- if (any(is_p4)) rowSums(W[is_p4, , drop = FALSE]) else numeric(0)
    recv_P4 <- if (any(is_p4)) colSums(W[, is_p4, drop = FALSE]) else numeric(0)
  }
  
  # 中心性（需要 do_pathways=TRUE）
  if (!is.null(cc@netP$centr)) {
    cc_cent <- CellChat::netAnalysis_computeCentrality(cc, slot.name = "netP")
    cent_df <- as.data.frame(cc_cent@netP$centr)
    cent_df$group <- rownames(cent_df)
    cent_df$Prototype <- sub("\\|.*$", "", cent_df$group)
    cent_df$Macro <- sub("^.*\\|", "", cent_df$group)
    cent_p4 <- subset(cent_df, Prototype == "Prototype4")
    if (nrow(cent_p4)) {
      pC2 <- cent_p4 %>%
        tidyr::pivot_longer(cols = c("out.degree","in.degree","betweenness","eigen"),
                            names_to = "metric", values_to = "value") %>%
        ggplot(aes(x = metric, y = value, group = group)) +
        geom_line(alpha = 0.6, linewidth = 0.7, color = "#d62728") +
        geom_point(size = 1.8, color = "#d62728") +
        coord_polar() +
        theme_pub2() + theme(axis.title = element_blank()) +
        labs(title = "Centrality profile of Prototype4 groups (proto_macro)")
      ggsave(file.path(outdir, "FigC2_cellchat_proto_centrality_P4.png"),
             pC2, width = 6.8, height = 6.2, dpi = 320, bg = "white")
    }
  }
  
  pull_top_lr <- function(cc_obj, mode = c("senderP4","edgeRecv"), top_n = 30) {
    mode <- match.arg(mode)
    tbl <- try(CellChat::subsetCommunication(cc_obj), silent = TRUE)
    if (inherits(tbl, "try-error") || is.null(tbl) || !nrow(tbl)) return(NULL)
    if (mode == "senderP4") {
      sub <- subset(tbl, grepl("^Prototype4\\|", source))
    } else {
      sub <- subset(tbl, grepl("^Edge\\|", target))
    }
    if (!nrow(sub)) return(NULL)
    agg <- sub %>%
      dplyr::group_by(ligand, receptor, source, target) %>%
      dplyr::summarise(weight = sum(prob, na.rm = TRUE), .groups = "drop") %>%
      dplyr::arrange(dplyr::desc(weight)) %>%
      head(top_n)
    agg$pair <- paste0(agg$ligand, "→", agg$receptor)
    agg
  }
  
  topP4   <- pull_top_lr(cellchat_proto,   "senderP4", 30)
  topEdge <- pull_top_lr(cellchat_spatial, "edgeRecv", 30)
  
  plot_lr_bar <- function(df, title, file) {
    if (is.null(df) || !nrow(df)) return()
    df$pair <- factor(df$pair, levels = rev(df$pair))
    p <- ggplot(df, aes(x = pair, y = weight, fill = target)) +
      geom_col(width = 0.85) + coord_flip() +
      theme_pub2() + labs(y = "Aggregated LR weight", x = NULL, title = title)
    ggsave(file.path(outdir, file), p, width = 7.2, height = 6.8, dpi = 320, bg = "white")
  }
  plot_lr_bar(topP4,  "Top ligand→receptor pairs sent by P4 groups (proto_macro)",  "FigE1_topLR_P4_sender.png")
  plot_lr_bar(topEdge, "Top ligand→receptor pairs received by Edge groups (spatial_macro)", "FigE2_topLR_Edge_receiver.png")
}

if (exists("cellchat_spatial") && !is.null(cellchat_spatial) &&
    !is.null(cellchat_spatial@netP$pathways)) {
  ccs <- cellchat_spatial
  groups <- names(ccs@idents)
  is_edge <- grepl("^Edge\\|", groups)
  if (any(is_edge)) {
    path_names <- rownames(ccs@netP$pathways)
    recv_mat <- sapply(path_names, function(pw) {
      M <- ccs@netP$netP[[pw]]
      if (is.null(M)) return(rep(0, sum(is_edge)))
      colSums(M[, is_edge, drop = FALSE])
    })
    if (!is.null(dim(recv_mat))) {
      rownames(recv_mat) <- groups[is_edge]
      colnames(recv_mat) <- path_names
      col_keep <- names(sort(colSums(recv_mat), decreasing = TRUE))[seq_len(min(30, ncol(recv_mat)))]
      row_keep <- names(sort(rowSums(recv_mat[, col_keep, drop = FALSE]), decreasing = TRUE))[seq_len(min(15, sum(is_edge)))]
      H <- recv_mat[row_keep, col_keep, drop = FALSE]
      png(file.path(outdir, "FigD_spatial_edge_recv_pathways_heatmap.png"), width = 2200, height = 1200, res = 250)
      ComplexHeatmap::Heatmap(
        H,
        name = "weight",
        col = circlize::colorRamp2(c(0, median(H), max(H)), c("#f7fbff","#6baed6","#08306b")),
        cluster_rows = TRUE,
        cluster_columns = TRUE,
        show_row_names = TRUE,
        show_column_names = TRUE,
        column_title = "Pathway activity received by Edge groups (spatial_macro)",
        row_title = "Edge|Macro groups"
      ) %>% print()
      dev.off()
    }
  }
}





# 
# alignment_sens_file <- file.path(outdir, "alignment_sensitivity_summary.csv")
# sens_res <- run_alignment_sensitivity(
#   seu = seu_analysis,
#   marker_bank = marker_bank,
#   prior_comp_file = prior_comp_file_raw,
#   gene_weight = gene_weight,
#   bulk_mat = bulk_mat,
#   pheno = pheno,
#   cache_dir = file.path(cache_root, "ucell_alignment_sensitivity")
# )
# if (!is.null(sens_res) && nrow(sens_res)) {
#   readr::write_csv(sens_res, alignment_sens_file)
# }
# 
# bulk_eval_n <- if (exists("y_vec")) length(y_vec) else NA_integer_
# run_meta <- data.frame(
#   key = c("proto_col_v1","proto_col_v2","spatial_delta","n_cells_full","n_cells_analysis","bulk_samples_evaluated"),
#   value = c(proto_col_v1, proto_col_v2, 0.10, ncol(seu_full), ncol(seu_analysis), bulk_eval_n),
#   stringsAsFactors = FALSE
# )
# readr::write_csv(run_meta, file.path(outdir, "run_meta.csv"))
# capture.output(sessionInfo(), file = file.path(outdir, "sessionInfo.txt"))
# 

