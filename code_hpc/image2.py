import os
import pydicom
import numpy as np
import SimpleITK as sitk

DATA_DIR = "/user/home/ms13525/scratch/mshds-ml-data-2025"

# get 3D volume
def read_dicom_series(folder_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder_path)
    reader.SetFileNames(dicom_names)
    return reader.Execute()

# get CT Series UID from segmentation
def get_series_uid_from_seg(seg_path):
    return pydicom.dcmread(seg_path).ReferencedSeriesSequence[0].SeriesInstanceUID

# matching UID
def find_ct_folder_by_uid(folder, uid):
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".dcm"):
                try:
                    if pydicom.dcmread(os.path.join(root, f), stop_before_pixels=True).SeriesInstanceUID == uid:
                        return root
                except: continue
    return None

# calculate Z-coordinate
def get_z_coords(sitk_img):
    sz, sp, org, dir3 = sitk_img.GetSize(), sitk_img.GetSpacing(), sitk_img.GetOrigin(), sitk_img.GetDirection()
    return [org[2] + i * sp[2] * dir3[8] for i in range(sz[2])]

# resample（1mm*3）
def resample_to_spacing(image_sitk, new_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear):
    original_spacing = image_sitk.GetSpacing()
    original_size = image_sitk.GetSize()
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image_sitk.GetDirection())
    resampler.SetOutputOrigin(image_sitk.GetOrigin())
    resampler.SetInterpolator(interpolator)
    return resampler.Execute(image_sitk)

# main
def load_aligned_ct_and_mask(patient_folder, target_spacing=(1.5, 1.5, 1.5)):
    # Find asegmentation
    seg_path = next((os.path.join(root, f) for root, _, files in os.walk(patient_folder)
                     for f in files if "segmentation" in root.lower() and f.endswith(".dcm")), None)
    if not seg_path:
        return None, None, None

    # Matched CT
    uid = get_series_uid_from_seg(seg_path)
    ct_folder = find_ct_folder_by_uid(patient_folder, uid)
    if not ct_folder:
        return None, None, None

    # reading
    ct_img = read_dicom_series(ct_folder)
    seg_img = sitk.ReadImage(seg_path)

    # resample
    ct_img = resample_to_spacing(ct_img, new_spacing=target_spacing)

    # resample segmentation
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    seg_img = resampler.Execute(seg_img)

    # get mask
    mask = sitk.GetArrayFromImage(seg_img)
    mask = (mask > 0)

    # get CT image
    image = sitk.GetArrayFromImage(ct_img)

    return image, mask, target_spacing

def find_segmentation_dcm(patient_folder):
    """
    Recursively search under patient_folder to find the first subfolder containing the Segmentation keyword,
    And there is DICOM file inside, return this dcm file path.
    """
    for root, dirs, files in os.walk(patient_folder):
        if "segmentation" in root.lower():
            dcm_files = [os.path.join(root, f) for f in files if f.lower().endswith(".dcm")]
            if dcm_files:
                return dcm_files[0]
    return None


base_dir = os.path.join(DATA_DIR, "dataset2")

for patient_id in sorted(os.listdir(base_dir)):
    if not patient_id.startswith("R01-"):
        continue  # Only process the patient folders

    patient_path = os.path.join(base_dir, patient_id)
    if not os.path.isdir(patient_path):
        continue

    # Automatically find Segmentation DICOM
    seg_dcm_path = find_segmentation_dcm(patient_path)
    if seg_dcm_path is None:
        print(f"{patient_id} skipped: No segmentation found")
        continue

    # load continued only when Segmentation was found
    image, mask, spacing = load_aligned_ct_and_mask(patient_path)
    if image is None or mask is None:
        print(f"{patient_id} skipped: failed to load image/mask")
        continue

    # Standardized mask
    mask = (mask > 0)

    if np.sum(mask) < 100:
        print(f"{patient_id} skipped: ROI too small ({np.sum(mask)} voxels)")
        continue

    roi = image * mask
    print(f"{patient_id} ROI volume shape: {roi.shape}")


import os
import numpy as np
import pandas as pd
from skimage import measure
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

# Data reading and preprocessing
DATA_DIR = "/user/home/ms13525/scratch/mshds-ml-data-2025"
base_dir = os.path.join(DATA_DIR, "dataset2")

# Normalize gray scale based on ROI
def normalize_gray_roi_adaptive(image_np, mask_np, min_percentile=1, max_percentile=99):
    roi = image_np[mask_np > 0]
    if roi.size == 0:
        return np.zeros_like(image_np)
    vmin, vmax = np.percentile(roi, [min_percentile, max_percentile])
    image_clipped = np.clip(image_np, vmin, vmax)
    return (image_clipped - vmin) / (vmax - vmin + 1e-6)

# get shape features
def extract_shape_features(mask, spacing=(1.0, 1.0, 1.0)):
    mask = mask.astype(np.float32)

    if np.sum(mask) == 0:
        return {"volume_mm3": 0, "surface_mm2": 0, "max_diameter_mm": 0, "compactness": 0}
        
    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5, spacing=spacing)

    surface_area = measure.mesh_surface_area(verts, faces)
    voxel_volume = np.prod(spacing)
    volume = np.sum(mask > 0) * voxel_volume

    if len(verts) > 2000:
        idx = np.random.choice(len(verts), size=2000, replace=False)
        verts_sampled = verts[idx].astype(np.float32)
    else:
        verts_sampled = verts.astype(np.float32)

    if len(verts_sampled) >= 2:
        max_diameter = pdist(verts_sampled).max()
    else:
        max_diameter = 0.0

    if volume > 0:
        compactness = (surface_area ** 3) / (volume ** 2)
    else:
        compactness = 0.0

    return {
        "volume_mm3": volume,
        "surface_mm2": surface_area,
        "max_diameter_mm": max_diameter,
        "compactness": compactness
    }

# get Texture Features: contrast, correlation, dissimilarity, homogeneity
def gray_level_cooccurrence_features(img, mask, levels=32):
    bin_img = (img * (levels - 1)).astype(np.uint8)
    glcm = _calculate_glcm2(bin_img, mask, levels)
    glcm = glcm / np.sum(glcm)

    ix = np.arange(1, levels+1)[:, None, None].astype(np.float64)
    iy = np.arange(1, levels+1)[None, :, None].astype(np.float64)

    ux = np.mean(glcm, axis=0, keepdims=True)
    uy = np.mean(glcm, axis=1, keepdims=True)
    sigma_x = np.std(glcm, axis=0, keepdims=True)
    sigma_y = np.std(glcm, axis=1, keepdims=True)

    sigma_x[sigma_x < 1e-3] = 1e-3
    sigma_y[sigma_y < 1e-3] = 1e-3

    features = {
        "contrast": np.mean(np.sum((ix - iy) ** 2 * glcm, axis=(0, 1))),
        "correlation": np.mean(np.sum((ix * iy * glcm - ux * uy) / (sigma_x * sigma_y + 1e-6), axis=(0, 1))),
        "dissimilarity": np.mean(np.sum(np.abs(ix - iy) * glcm, axis=(0, 1))),
        "homogeneity": np.mean(np.sum(glcm / (1 + np.abs(ix - iy)), axis=(0, 1))),
    }

    return features

# get Texture Features: GLCM in 13 directions
def _calculate_glcm2(img, mask, nbins):
    out = np.zeros((nbins, nbins, 13))
    offsets = [
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (1, 1, 0), (-1, 1, 0), (1, 0, 1),
        (-1, 0, 1), (0, 1, 1), (0, -1, 1),
        (1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)
    ]
    matrix = np.array(img)
    matrix[mask <= 0] = nbins
    s = matrix.shape
    bins = np.arange(0, nbins + 1)

    for i, offset in enumerate(offsets):
        matrix1 = np.ravel(matrix[
            max(offset[0], 0):s[0]+min(offset[0], 0),
            max(offset[1], 0):s[1]+min(offset[1], 0),
            max(offset[2], 0):s[2]+min(offset[2], 0)
        ])
        matrix2 = np.ravel(matrix[
            max(-offset[0], 0):s[0]+min(-offset[0], 0),
            max(-offset[1], 0):s[1]+min(-offset[1], 0),
            max(-offset[2], 0):s[2]+min(-offset[2], 0)
        ])

        try:
            out[:, :, i] = np.histogram2d(matrix1, matrix2, bins=bins)[0]
        except Exception as e:
            print(f"GLCM histogram failed for offset {offset}: {e}")
            continue

    return out

# Main processing
all_features = []

for patient_id in sorted(os.listdir(base_dir)):
    if not patient_id.startswith("R01-"):
        continue

    patient_path = os.path.join(base_dir, patient_id)
    if not os.path.isdir(patient_path):
        continue

    image, mask, spacing = load_aligned_ct_and_mask(patient_path)
    if image is None or mask is None:
        print(f"{patient_id} skip (no image or mask)")
        continue

    mask = (mask > 0)  # Only the tumor area was retained

    # Skip the minimal ROI
    if np.sum(mask) < 100:
        print(f"{patient_id} skip: ROI too small ({np.sum(mask)} voxels)")
        continue

    image = normalize_gray_roi_adaptive(image, mask)

    # get segmentation
    shape_feats = extract_shape_features(mask, spacing)
    glcm_feats = gray_level_cooccurrence_features(image, mask)

    features = {"patient_id": patient_id}
    features.update(shape_feats)
    features.update(glcm_feats)
    all_features.append(features)

# save result
output_path = "../result/features2.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df = pd.DataFrame(all_features)
df.to_csv(output_path, index=False)
print(f"Done, {len(df)} patients processed.")