#!/usr/bin/env python3
"""Extract nuclei features from CellViT outputs using sliding windows."""

###### Accelerated v2


import argparse
import json
import geopandas
import shapely
from shapely.geometry import Polygon, MultiPoint, Point, LineString, MultiPolygon
from PIL import Image, ImageDraw
import os
from pathlib import Path
import openslide
import numpy as np
import pandas as pd
import skimage.io 
import histomicstk as htk
import shutil
from skimage.transform import resize
from sklearn.preprocessing import RobustScaler
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from scipy.spatial import KDTree 
import warnings
warnings.filterwarnings('ignore')

from typing import Optional

# logging configuration is set in main()
import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract nuclei features from CellViT outputs and WSIs using sliding windows."
    )
    parser.add_argument(
        "--cell-output-dir",
        type=Path,
        required=True,
        help="Directory containing per-sample CellViT outputs (each sample is a subdirectory).",
    )
    parser.add_argument(
        "--svs-dir",
        type=Path,
        required=True,
        help="Directory containing the corresponding SVS files.",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        help="Optional list of sample names to process. Defaults to all subdirectories in --cell-output-dir.",
    )
    parser.add_argument(
        "--physical-patch-size-um",
        type=int,
        default=2048,
        help="Patch size (in level-0 pixels/um units) used for sliding windows.",
    )
    parser.add_argument(
        "--overlap-level0",
        type=int,
        default=0,
        help="Overlap between adjacent patches in level-0 pixels.",
    )
    parser.add_argument(
        "--upsample-factor",
        type=int,
        default=2,
        help="Upsampling factor applied before nuclei feature extraction.",
    )
    parser.add_argument(
        "--svs-level",
        type=int,
        default=3,
        help="SVS level used when reading image patches.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Number of worker processes. Defaults to CPU count - 2.",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=5.0,
        help="Centroid distance threshold (level-0 pixels) for deduplicating nuclei of the same type.",
    )
    parser.add_argument(
        "--output-subdir",
        default="full_slide_nuclei_features_sliding_window_debug",
        help="Subdirectory (within each sample) where features will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute features even if the combined CSV already exists.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


shared_gdf = None

def init_worker(gdf_data):
    """
    Initializer executed inside each worker process.
    Stores the large GeoDataFrame in a process-local global variable so it
    does not need to be pickled and transferred for every patch.
    """
    global shared_gdf
    shared_gdf = gdf_data


def extract_seg_features(im_input_rgb, mask_dir_path_temp, barcode, upsample_factor=1):
    """
    Compute nuclei-level features for a single patch using HistomicsTK.

    The RGB patch and nuclei mask can optionally be upsampled before feature
    extraction. Returns a DataFrame of nuclei features or None if no nuclei
    are present.
    """
    try:
        # --- Perform upsampling of the IMAGE if factor > 1 ---
        if upsample_factor > 1:
            new_height = int(im_input_rgb.shape[0] * upsample_factor)
            new_width = int(im_input_rgb.shape[1] * upsample_factor)

            im_input = resize(im_input_rgb, (new_height, new_width),
                              anti_aliasing=True, preserve_range=True).astype(np.uint8)
            logger.debug(f"Patch {barcode} image upsampled from {im_input_rgb.shape} to {im_input.shape}")
        else:
            im_input = im_input_rgb


        stainColorMap = {
            'hematoxylin': [0.65, 0.70, 0.29],
            'eosin':       [0.07, 0.99, 0.11],
            'dab':         [0.27, 0.57, 0.78],
            'null':        [0.0, 0.0, 0.0]
        }
        stain_1 = 'hematoxylin'
        stain_2 = 'eosin'
        stain_3 = 'null'
        W = np.array([stainColorMap[stain_1], stainColorMap[stain_2], stainColorMap[stain_3]]).T
        im_stains = htk.preprocessing.color_deconvolution.color_deconvolution(im_input, W).Stains

        npy_path = os.path.join(mask_dir_path_temp, 'instances.npy')
        if not os.path.exists(npy_path):
            logger.error(f"instances.npy is missing for patch {barcode} at {npy_path}; expected segmentation mask not found.")
            return None

        # --- Upsample the mask as well (CRUCIAL: use nearest-neighbor interpolation) ---
        im_nuclei_seg_mask_original = np.load(npy_path)
        if upsample_factor > 1:
            new_mask_height = int(im_nuclei_seg_mask_original.shape[0] * upsample_factor)
            new_mask_width = int(im_nuclei_seg_mask_original.shape[1] * upsample_factor)

            im_nuclei_seg_mask = resize(im_nuclei_seg_mask_original,
                                        (new_mask_height, new_mask_width),
                                        order=0, # Nearest-neighbor interpolation
                                        preserve_range=True, # Preserve original label values
                                        anti_aliasing=False).astype(np.int32) # Ensure integer labels
            logger.debug(f"Patch {barcode} mask upsampled from {im_nuclei_seg_mask_original.shape} to {im_nuclei_seg_mask.shape}")
        else:
            im_nuclei_seg_mask = im_nuclei_seg_mask_original

        if im_input.shape[:2] != im_nuclei_seg_mask.shape[:2]:
            logger.warning(
                f"Patch {barcode} image size {im_input.shape[:2]} does not match mask {im_nuclei_seg_mask.shape[:2]} after upsampling; resizing mask."
            )
            im_nuclei_seg_mask = resize(
                im_nuclei_seg_mask,
                im_input.shape[:2],
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            ).astype(np.int32)
            if im_input.shape[:2] != im_nuclei_seg_mask.shape[:2]:
                logger.error(
                    f"Patch {barcode} dimensions {im_input.shape[:2]} still do not match mask {im_nuclei_seg_mask.shape[:2]} after resizing; skipping patch."
                )
                return None


        im_nuclei_stain = im_stains[:, :, 0] # Hematoxylin channel

        unique_labels = np.unique(im_nuclei_seg_mask)
        if len(unique_labels) <= 1 and 0 in unique_labels: # Only background (0) or no labels
            return None

        nuclei_features = htk.features.compute_nuclei_features(im_nuclei_seg_mask, im_nuclei_stain)

        expected_htk_features = [
            "Identifier.CentroidX", "Identifier.CentroidY", "Label",
        ]

        for col in expected_htk_features:
            if col not in nuclei_features.columns:
                nuclei_features[col] = np.nan

        if "Label" in nuclei_features.columns:
            nuclei_features["Label"] = nuclei_features["Label"].astype(int)
        else:
            logger.error(f"Missing 'Label' column in nuclei features for patch {barcode}; unable to continue.")
            return None

        nuclei_num = len(nuclei_features["Label"])
        if nuclei_num == 0:
            return None

        json_path = os.path.join(mask_dir_path_temp, 'nuclei_dict.json')
        if not os.path.exists(json_path):
            logger.error(f"nuclei_dict.json is missing for patch {barcode} at {json_path}; expected CellViT metadata.")
            return None

        with open(json_path) as json_file:
            json_data = json.load(json_file)

        type_df = pd.DataFrame.from_dict(json_data, orient='index', columns=['type'])
        type_df = type_df.reset_index()
        type_df.columns = ['nuclei_id_str', 'type']
        type_df['Label'] = type_df['nuclei_id_str'].astype(int)

        merge_df = pd.merge(type_df, nuclei_features, on="Label", how='inner')

        cols_to_drop_if_present = [
            'Identifier.Xmin', 'Identifier.Ymin',
            'Identifier.Xmax', 'Identifier.Ymax',
            'Identifier.WeightedCentroidX', 'Identifier.WeightedCentroidY'
        ]

        existing_cols_to_drop = [col for col in cols_to_drop_if_present if col in merge_df.columns]

        if existing_cols_to_drop:
            merge_df = merge_df.drop(columns=existing_cols_to_drop)
        return merge_df

    except Exception as e:
        logger.error(f"Failed to extract nuclei features for patch {barcode}: {e}")
        return None


def process_single_patch(
    patch_params,
    svs_file_path,
    full_slide_width_level0,
    full_slide_height_level0,
    desired_svs_level,
    downsample_factor,
    PATCH_SIZE_LEVEL0,
    OVERLAP_LEVEL0,
    upsample_for_htk,
    output_features_root_dir,
):
    global shared_gdf

    x_level0, y_level0 = patch_params['x'], patch_params['y']
    
    slide = None
    try:
        slide = openslide.OpenSlide(svs_file_path)
    except openslide.OpenSlideError as e:
        logger.error(f"Worker failed to open SVS file {svs_file_path}: {e}. Skipping patch ({x_level0},{y_level0}).")
        return None

    process_temp_mask_dir = os.path.join(output_features_root_dir, f'temp_masks_pid{os.getpid()}_{x_level0}_{y_level0}')
    os.makedirs(process_temp_mask_dir, exist_ok=True)

    try:
        window_x_end_level0 = min(x_level0 + PATCH_SIZE_LEVEL0, full_slide_width_level0)
        window_y_end_level0 = min(y_level0 + PATCH_SIZE_LEVEL0, full_slide_height_level0)
        current_patch_width_level0 = window_x_end_level0 - x_level0
        current_patch_height_level0 = window_y_end_level0 - y_level0
        if current_patch_width_level0 <= 0 or current_patch_height_level0 <= 0: return None
        current_patch_width_current_level = int(current_patch_width_level0 / downsample_factor)
        current_patch_height_current_level = int(current_patch_height_level0 / downsample_factor)
        if current_patch_width_current_level == 0 or current_patch_height_current_level == 0: return None
        barcode = f"patch_{y_level0}_{x_level0}"
        try:
            patch_img_pil = slide.read_region((x_level0, y_level0), desired_svs_level,
                                             (current_patch_width_current_level, current_patch_height_current_level))
            im_input_rgb_np = np.array(patch_img_pil.convert("RGB"))
        except Exception as e:
            logger.warning(f"Failed to read patch {barcode}: {e}. Skipping patch.")
            return None

        patch_bbox_level0 = Polygon([
            (x_level0, y_level0),
            (x_level0 + current_patch_width_level0, y_level0),
            (x_level0 + current_patch_width_level0, y_level0 + current_patch_height_level0),
            (x_level0, y_level0 + current_patch_height_level0),
            (x_level0, y_level0)
        ])
        
        if not hasattr(shared_gdf, 'sindex') or shared_gdf.sindex is None:
            shared_gdf.sindex

        possible_matches_index = list(shared_gdf.sindex.intersection(patch_bbox_level0.bounds))
        potential_nuclei = shared_gdf.iloc[possible_matches_index]
        nuclei_in_patch_gdf = potential_nuclei[potential_nuclei.geometry.intersects(patch_bbox_level0, align=False)]
        
        if nuclei_in_patch_gdf.empty: return None

        temp_instance_mask = np.zeros((current_patch_height_current_level, current_patch_width_current_level), dtype=np.int32)
        temp_nuclei_dict = {}
        current_nuclei_id_in_patch = 1
        for _, row in nuclei_in_patch_gdf.iterrows():
            poly_geom_level0 = row.geometry
            cell_type_name = row['classification']['name']
            try:
                if not poly_geom_level0.is_valid:
                    poly_geom_level0 = poly_geom_level0.buffer(0)
                    if not poly_geom_level0.is_valid: continue
                clipped_poly_level0 = poly_geom_level0.intersection(patch_bbox_level0).buffer(0)
                if clipped_poly_level0.is_empty: continue
                geometries_to_draw = []
                if clipped_poly_level0.geom_type == 'Polygon': geometries_to_draw.append(clipped_poly_level0)
                elif clipped_poly_level0.geom_type == 'MultiPolygon':
                    for single_poly in clipped_poly_level0.geoms:
                        if single_poly.geom_type == 'Polygon' and not single_poly.is_empty: geometries_to_draw.append(single_poly)
                else: continue
                for geom_to_draw in geometries_to_draw:
                    if hasattr(geom_to_draw, 'exterior') and geom_to_draw.geom_type == 'Polygon':
                        exterior_coords_scaled_translated = [
                            ((coord[0] - x_level0) / downsample_factor, (coord[1] - y_level0) / downsample_factor)
                            for coord in geom_to_draw.exterior.coords]
                        single_nucleus_mask_pil = Image.new('L', (current_patch_width_current_level, current_patch_height_current_level), 0)
                        ImageDraw.Draw(single_nucleus_mask_pil).polygon([c for p in exterior_coords_scaled_translated for c in p], fill=current_nuclei_id_in_patch)
                        current_drawn_mask = np.array(single_nucleus_mask_pil)
                        temp_instance_mask = np.where(temp_instance_mask == 0, current_drawn_mask, temp_instance_mask)
                        temp_nuclei_dict[str(current_nuclei_id_in_patch)] = {"type": cell_type_name}
                        current_nuclei_id_in_patch += 1
            except Exception as e:
                logger.warning(f"Unexpected error for nucleus {row.name} in patch {barcode}: {e}. Skipping nucleus.")
                continue
        if current_nuclei_id_in_patch == 1: return None
        
        temp_npy_path = os.path.join(process_temp_mask_dir, 'instances.npy')
        temp_json_path = os.path.join(process_temp_mask_dir, 'nuclei_dict.json')
        np.save(temp_npy_path, temp_instance_mask)
        with open(temp_json_path, 'w') as f:
            json.dump(temp_nuclei_dict, f)

        patch_df = extract_seg_features(im_input_rgb_np, process_temp_mask_dir, barcode, upsample_factor=upsample_for_htk)

        if patch_df is not None and not patch_df.empty:
            patch_df['Identifier.CentoidX_Global'] = (patch_df['Identifier.CentroidX'] / upsample_for_htk * downsample_factor) + x_level0
            patch_df['Identifier.CentoidY_Global'] = (patch_df['Identifier.CentroidY'] / upsample_for_htk * downsample_factor) + y_level0
            return patch_df
        else:
            return None
    finally:
        if slide: slide.close()
        if os.path.exists(process_temp_mask_dir):
            try:
                shutil.rmtree(process_temp_mask_dir)
            except OSError as e:
                logger.error(f"Failed to clean up temporary directory {process_temp_mask_dir}: {e}")

def process_slide_with_parallel(
    cellvit_path,
    svs_path,
    sample_name,
    PHYSICAL_PATCH_SIZE_UM=2048,
    OVERLAP_LEVEL0=0,
    FORCED_UPSAMPLE_FACTOR=2,
    desired_svs_level=3,
    output_subdir="full_slide_nuclei_features_sliding_window_debug",
    num_processes: Optional[int] = None,
    deduplication_threshold_pixels_level0: float = 5.0,
    overwrite: bool = False,
):
    svs_file_path = os.path.join(svs_path, sample_name + ".svs")
    geojson_file_path = os.path.join(cellvit_path, sample_name, 'cell_detection', 'cells.geojson')
    output_features_dir = os.path.join(cellvit_path, sample_name, output_subdir)

    combined_csv_path = os.path.join(output_features_dir, 'all_nuclei_features_full_slide_sliding_window_robust_scaled.csv')
    if not overwrite and os.path.exists(combined_csv_path):
        logger.info("Features already exist at %s; skipping %s", output_features_dir, sample_name)
        return

    os.makedirs(output_features_dir, exist_ok=True)
    logger.info("Launching full-slide nuclei feature extraction for %s", svs_file_path)
    logger.info("Loading GeoJSON cells from %s", geojson_file_path)
    all_cells_gdf = geopandas.GeoDataFrame()
    try:
        with open(geojson_file_path, 'r', encoding='utf-8') as f: raw_geojson_data = json.load(f)
        if isinstance(raw_geojson_data, list):
            geojson_data_for_geopandas = {"type": "FeatureCollection", "features": raw_geojson_data}
        else:
            geojson_data_for_geopandas = raw_geojson_data
        all_cells_gdf = geopandas.GeoDataFrame.from_features(geojson_data_for_geopandas['features'])
        all_cells_gdf = all_cells_gdf[all_cells_gdf.geometry.apply(lambda g: isinstance(g, (Polygon, MultiPolygon)) and not g.is_empty)]
        logger.info("Validating and repairing GeoJSON geometries (buffer(0) heuristic)...")
        initial_invalid_count = all_cells_gdf[~all_cells_gdf.geometry.is_valid].shape[0]
        if initial_invalid_count > 0:
            logger.warning(f"Found %d invalid geometries before buffering.", initial_invalid_count)
            all_cells_gdf['geometry'] = all_cells_gdf['geometry'].apply(lambda g: g.buffer(0) if not g.is_valid else g)
            post_buffer_invalid_count = all_cells_gdf[~all_cells_gdf.geometry.is_valid].shape[0]
            if post_buffer_invalid_count > 0:
                logger.error(
                    "%d geometries remain invalid after buffer(0); downstream feature extraction may be affected.",
                    post_buffer_invalid_count,
                )
            else:
                logger.info("All invalid geometries were repaired via buffer(0).")
        else:
            logger.info("All GeoJSON geometries were valid before buffering.")
        all_cells_gdf = all_cells_gdf.explode(ignore_index=True)
        all_cells_gdf = all_cells_gdf[all_cells_gdf.geometry.apply(lambda g: isinstance(g, Polygon) and not g.is_empty)]
        all_cells_gdf.sindex 
        logger.info("Loaded %d valid nuclei polygons from GeoJSON after explode().", len(all_cells_gdf))
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Failed to load or process GeoJSON: {e}")
        return
    
    slide = None
    try:
        slide = openslide.OpenSlide(svs_file_path)
        full_slide_width_level0, full_slide_height_level0 = slide.dimensions
        PATCH_SIZE_LEVEL0_CALC = int(round(PHYSICAL_PATCH_SIZE_UM))
        if PATCH_SIZE_LEVEL0_CALC < 64: PATCH_SIZE_LEVEL0_CALC = 64
        if desired_svs_level >= slide.level_count:
            desired_svs_level = slide.level_count - 1
            logger.warning("Requested SVS level exceeds available levels. Using highest available level %d.", desired_svs_level)
        downsample_factor = slide.level_downsamples[desired_svs_level]
        patch_size_current_level = int(PATCH_SIZE_LEVEL0_CALC / downsample_factor)
        upsample_for_htk = FORCED_UPSAMPLE_FACTOR
        logger.info(
            "Slide %s | level-0 dims %dx%d | level %d downsample %.4f | patch size level0 %d | patch size level%d %d | overlap %d | upsample %d",
            svs_file_path,
            full_slide_width_level0,
            full_slide_height_level0,
            desired_svs_level,
            downsample_factor,
            PATCH_SIZE_LEVEL0_CALC,
            desired_svs_level,
            patch_size_current_level,
            OVERLAP_LEVEL0,
            upsample_for_htk,
        )
    except openslide.OpenSlideError as e:
        logger.error(f"Failed to open SVS file {svs_file_path}: {e}")
        return
    finally:
        if slide: slide.close()

    patch_tasks = []
    for y_level0 in range(0, full_slide_height_level0, PATCH_SIZE_LEVEL0_CALC - OVERLAP_LEVEL0):
        for x_level0 in range(0, full_slide_width_level0, PATCH_SIZE_LEVEL0_CALC - OVERLAP_LEVEL0):
            patch_tasks.append({'x': x_level0, 'y': y_level0})
    if not patch_tasks:
        logger.warning("No patch tasks generated for sample %s; check patch size/overlap configuration.", sample_name)
        return
    logger.info("Sample %s will be processed with %d patch tasks.", sample_name, len(patch_tasks))

    func_for_map = partial(
        process_single_patch,
        svs_file_path=svs_file_path,
        full_slide_width_level0=full_slide_width_level0,
        full_slide_height_level0=full_slide_height_level0,
        desired_svs_level=desired_svs_level,
        downsample_factor=downsample_factor,
        PATCH_SIZE_LEVEL0=PATCH_SIZE_LEVEL0_CALC,
        OVERLAP_LEVEL0=OVERLAP_LEVEL0,
        upsample_for_htk=upsample_for_htk,
        output_features_root_dir=output_features_dir
    )

    if num_processes is None:
        num_processes = max(1, cpu_count() - 2)
    logger.info("Spawning %d worker processes for patch extraction.", num_processes)

    all_extracted_nuclei_features = []
    with Pool(processes=num_processes,
              initializer=init_worker,
              initargs=(all_cells_gdf,)) as pool:
        
        for patch_df in tqdm(pool.imap_unordered(func_for_map, patch_tasks), total=len(patch_tasks), desc="Processing patches"):
            if patch_df is not None and not patch_df.empty:
                all_extracted_nuclei_features.append(patch_df)

    if all_extracted_nuclei_features:
        final_combined_df = pd.concat(all_extracted_nuclei_features, ignore_index=True)
        final_combined_df['Global_Nuclei_ID'] = range(len(final_combined_df))

        initial_rows_before_dropna = len(final_combined_df)
        final_combined_df.dropna(inplace=True)
        rows_after_dropna = len(final_combined_df)
        logger.info(
            "Dropped rows with NaNs: kept %d rows (removed %d).",
            rows_after_dropna,
            initial_rows_before_dropna - rows_after_dropna,
        )

        logger.info(
            "Deduplicating nuclei by centroid distance < %.2f pixels at level-0 within each cell type.",
            deduplication_threshold_pixels_level0,
        )
        unique_nuclei_list = []
        for nuclei_type, group_df in final_combined_df.groupby('type'):
            centroids = group_df[['Identifier.CentoidX_Global', 'Identifier.CentoidY_Global']].values
            if len(centroids) == 0: continue
            tree = KDTree(centroids)
            processed_centroid_indices = set()
            for i_local, (original_df_index, row) in enumerate(group_df.iterrows()):
                if i_local in processed_centroid_indices: continue
                nearby_indices_in_tree = tree.query_ball_point(centroids[i_local], deduplication_threshold_pixels_level0)
                for tree_idx in nearby_indices_in_tree: processed_centroid_indices.add(tree_idx)
                unique_nuclei_list.append(row)
        final_combined_df = pd.DataFrame(unique_nuclei_list).reset_index(drop=True)
        deduplicated_count = len(final_combined_df)
        logger.info(
            "Deduplication finished: kept %d nuclei (removed %d).",
            deduplicated_count,
            rows_after_dropna - deduplicated_count,
        )

        logger.info("Dropping helper columns and applying RobustScaler normalization.")
        cols_to_drop = ['Identifier.CentroidX', 'Identifier.CentroidY', 'Label', 'nuclei_id_str']
        cols_to_drop_existing = [col for col in cols_to_drop if col in final_combined_df.columns]
        if cols_to_drop_existing:
            final_combined_df.drop(columns=cols_to_drop_existing, inplace=True)
        non_numeric_cols = ['Global_Nuclei_ID', 'type', 'Identifier.CentoidX_Global', 'Identifier.CentoidY_Global']
        cols_for_normalization = [col for col in final_combined_df.columns if pd.api.types.is_numeric_dtype(final_combined_df[col]) and col not in non_numeric_cols]
        if cols_for_normalization:
            final_combined_df[cols_for_normalization] = RobustScaler().fit_transform(final_combined_df[cols_for_normalization])

        combined_output_path = os.path.join(output_features_dir, 'all_nuclei_features_full_slide_sliding_window_robust_scaled.csv')
        final_combined_df.to_csv(combined_output_path, header=True, index=False)
        logger.info("Extracted features for %d nuclei; saved to %s", len(final_combined_df), combined_output_path)

    else:
        logger.warning("No nuclei features were extracted for %s; check GeoJSON and SVS inputs.", sample_name)


def main() -> None:
    args = parse_args()
    args.cell_output_dir = args.cell_output_dir.expanduser()
    args.svs_dir = args.svs_dir.expanduser()
    if args.samples:
        args.samples = sorted(set(args.samples))
    if not args.cell_output_dir.exists():
        raise FileNotFoundError(f"Cell output directory {args.cell_output_dir} does not exist.")
    if not args.svs_dir.exists():
        raise FileNotFoundError(f"SVS directory {args.svs_dir} does not exist.")

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format='%(asctime)s - %(levelname)s - %(message)s')

    if args.samples:
        sample_names = args.samples
    else:
        sample_names = sorted(
            [p.name for p in args.cell_output_dir.iterdir() if p.is_dir() and not p.name.startswith('.')]
        )

    if not sample_names:
        logger.warning("No samples found under %s; nothing to process.", args.cell_output_dir)
        return

    for sample in sample_names:
        sample_dir = args.cell_output_dir / sample
        if not sample_dir.is_dir():
            logger.warning("Sample %s not found under %s; skipping.", sample, args.cell_output_dir)
            continue
        logger.info("Processing sample %s", sample)
        process_slide_with_parallel(
            cellvit_path=str(args.cell_output_dir),
            svs_path=str(args.svs_dir),
            sample_name=sample,
            PHYSICAL_PATCH_SIZE_UM=args.physical_patch_size_um,
            OVERLAP_LEVEL0=args.overlap_level0,
            FORCED_UPSAMPLE_FACTOR=args.upsample_factor,
            desired_svs_level=args.svs_level,
            output_subdir=args.output_subdir,
            num_processes=args.num_processes,
            deduplication_threshold_pixels_level0=args.dedup_threshold,
            overwrite=args.overwrite,
        )

    logger.info("Completed nuclei feature extraction for all requested samples.")


if __name__ == '__main__':
    main()
