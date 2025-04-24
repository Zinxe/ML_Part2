import os, pydicom, numpy as np, SimpleITK as sitk

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

# matching z
def align_segmentation_by_z(seg_img, seg_z, ct_z):
    zset = set(round(z, 1) for z in ct_z)
    return sitk.GetArrayFromImage(seg_img)[[i for i, z in enumerate(seg_z) if round(z, 1) in zset]]

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

# normalise to [0,1]
def normalize_gray_roi_adaptive(image_np, mask_np, min_percentile=1, max_percentile=99):
    roi = image_np[mask_np > 0]
    if roi.size == 0:
        return np.zeros_like(image_np)
    vmin, vmax = np.percentile(roi, [min_percentile, max_percentile])
    image_clipped = np.clip(image_np, vmin, vmax)
    return (image_clipped - vmin) / (vmax - vmin + 1e-6)

# main
def load_aligned_ct_and_mask(patient_folder, target_spacing=(1.5, 1.5, 1.5)):
    seg_path = next((os.path.join(root, f) for root, _, files in os.walk(patient_folder)
                     for f in files if "segmentation" in root.lower() and f.endswith(".dcm")), None)
    if not seg_path: return None, None, None

    uid = get_series_uid_from_seg(seg_path)
    ct_folder = find_ct_folder_by_uid(patient_folder, uid)
    if not ct_folder: return None, None, None

    ct_img = read_dicom_series(ct_folder)
    seg_img = sitk.ReadImage(seg_path)

    # Resample to same spacing (1mm)
    ct_img = resample_to_spacing(ct_img, new_spacing=target_spacing)
    
    # resample segmentation
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    seg_img = resampler.Execute(seg_img)

    # z coordinate alignment
    ct_z = get_z_coords(ct_img)
    seg_z = get_z_coords(seg_img)
    mask = align_segmentation_by_z(seg_img, seg_z, ct_z)

    image = sitk.GetArrayFromImage(ct_img)
    image = normalize_gray_roi_adaptive(image, mask)

    # output
    return image, mask, target_spacing

base_dir = "../dataset1"

for patient_id in sorted(os.listdir(base_dir)):
    patient_path = os.path.join(base_dir, patient_id)
    if not os.path.isdir(patient_path):
        continue  # skip non-folders

    image, mask, spacing = load_aligned_ct_and_mask(patient_path)

    if image is not None:
        roi = image * (mask > 0)
        print(f"{patient_id} ROI volume shape: {roi.shape}")
    else:
        print(f"{patient_id} fail")

# Extraction of tumour 3D shape features based on marching cubes algorithm
import os
import numpy as np
import pandas as pd
import skimage as ski
from skimage import io, data, filters, measure, morphology
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import skimage.measure
from scipy.spatial.distance import  pdist

def extract_shape_features(mask, spacing=(1.0, 1.0, 1.0)):
    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5, spacing=spacing)

    # surface area
    surface_area = measure.mesh_surface_area(verts, faces)

    # volumetric
    voxel_volume = np.prod(spacing)
    volume = np.sum(mask > 0) * voxel_volume

    # maximum diameter
    if len(verts) >= 2:
        max_diameter = pdist(verts).max()
    else:
        max_diameter = 0.0

    # Compactness
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

for patient_id in sorted(os.listdir(base_dir)):
    patient_path = os.path.join(base_dir, patient_id)
    if not os.path.isdir(patient_path):
        continue

    image, mask, spacing = load_aligned_ct_and_mask(patient_path)
    if image is None or mask is None:
        print(f"{patient_id} skip")
        continue

    features = extract_shape_features(mask, spacing)
    print(f"{patient_id} features:", features)

# GLCM
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

    # downscale protection
    sigma_x[sigma_x < 1e-3] = 1e-3
    sigma_y[sigma_y < 1e-3] = 1e-3

    features = {
        "contrast": np.mean(np.sum((ix - iy) ** 2 * glcm, axis=(0, 1))),
        "correlation": np.mean(np.sum((ix * iy * glcm - ux * uy) / (sigma_x * sigma_y + 1e-6), axis=(0, 1))),
        "dissimilarity": np.mean(np.sum(np.abs(ix - iy) * glcm, axis=(0, 1))),
        "homogeneity": np.mean(np.sum(glcm / (1 + np.abs(ix - iy)), axis=(0, 1))),
    }

    return features

# GLCM2
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
        matrix1 = np.ravel(matrix[max(offset[0], 0):s[0]+min(offset[0], 0),
                                  max(offset[1], 0):s[1]+min(offset[1], 0),
                                  max(offset[2], 0):s[2]+min(offset[2], 0)])

        matrix2 = np.ravel(matrix[max(-offset[0], 0):s[0]+min(-offset[0], 0),
                                  max(-offset[1], 0):s[1]+min(-offset[1], 0),
                                  max(-offset[2], 0):s[2]+min(-offset[2], 0)])

        try:
            out[:, :, i] = np.histogram2d(matrix1, matrix2, bins=bins)[0]
        except Exception as e:
            print(f"GLCM histogram failed for offset {offset}: {e}")
            continue

    return out

import os
import pandas as pd

base_dir = "../dataset1"
all_features = []

for patient_id in sorted(os.listdir(base_dir)):
    if not patient_id.startswith("LUNG1-"):
        continue

    patient_path = os.path.join(base_dir, patient_id)
    if not os.path.isdir(patient_path):
        continue

    image, mask, spacing = load_aligned_ct_and_mask(patient_path)

    if image is not None and mask is not None:
        image = normalize_gray_roi_adaptive(image, mask)
        glcm = gray_level_cooccurrence_features(image, mask)

        glcm["patient_id"] = patient_id
        all_features.append(glcm)
    else:
        print(f"{patient_id} skip")

import os
import pandas as pd
import numpy as np

base_dir = "../dataset1"
all_features = []

for patient_id in sorted(os.listdir(base_dir)):
    if not patient_id.startswith("LUNG1-"):
        continue

    patient_path = os.path.join(base_dir, patient_id)
    if not os.path.isdir(patient_path):
        continue

    image, mask, spacing = load_aligned_ct_and_mask(patient_path)
    if image is None or mask is None:
        print(f"{patient_id} skip")
        continue

    image = normalize_gray_roi_adaptive(image, mask)
    shape_feats = extract_shape_features(mask, spacing)
    glcm_feats = gray_level_cooccurrence_features(image, mask)

    features = {"patient_id": patient_id}
    features.update(shape_feats)
    features.update(glcm_feats)
    all_features.append(features)

# Save the result as a csv file
output_path = "../result/combined_features.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df = pd.DataFrame(all_features)
df.to_csv("../result/features1.csv", index=False)
print(f"Done,{len(df)} in total