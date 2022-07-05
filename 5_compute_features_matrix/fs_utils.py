import os
import csv
import re
from pathlib import Path
import nibabel as nib

# CONSTANTS
VALID_FILTERS = {
    'glcm': 8,
    'rlm': 10,
    'lbp': 4
}

ASEG_LABELS_FILE = "./labels/aseg_labels.csv"
HP_SUBFIELDS_LABELS_FILE = "./labels/hpsubfields_labels.csv"


def get_subjects_names_and_paths(fs_subjects):
    # scripts/hippocampal-subfields-T1.log check if needed

    # subjects_dir = Path(base_folder)
    subjects_dir = Path(fs_subjects)

    folders = [f for f in subjects_dir.iterdir() if f.is_dir()
               and re.match(r'^[0-9]{3}_S_[0-9]{4}', f.name)]

    # subjects = [(sub.name, sub.resolve().as_posix()) for sub in folders]

    subjects_names = [path.name for path in folders]
    subjects_paths = [path.resolve().as_posix() for path in folders]

    return subjects_names, subjects_paths


def subjects_as_list_of_tuple(s_names, s_paths):
    subs = [sub for sub in zip(s_names, s_paths)]
    subs.sort()
    return subs


def get_subjects_from_args(args):
    subjects = list()
    for sd in args:
        if not os.path.exists(sd):
            raise ValueError(
                "Directory do not exist: {}".format(sd))

        temp = subjects_as_list_of_tuple(
            *get_subjects_names_and_paths(sd))
        subjects.extend(temp)
    return subjects

# Filters
# Check Filters Values


def check_filter_name(filter_name):
    if filter_name not in VALID_FILTERS:
        raise ValueError(
            "result: filter_name must be one of {}".format(VALID_FILTERS.keys()))

    return True


def check_feature_number(filter_name, feature_number):
    if feature_number not in range(1, VALID_FILTERS.get(filter_name) + 1):
        raise ValueError(
            "feature_number do not exist for {}".format(filter_name))
    return True

# Get Feature-related


def get_features_path(subject):
    return f"{subject[1]}/nifti/filtered/"


def get_nifti_path(subject):
    return f"{subject[1]}/nifti/"


def get_glcm_path(subject):
    return f"{subject[1]}/nifti/filtered/glcm"


def get_rlm_path(subject):
    return f"{subject[1]}/nifti/filtered/rlm"


def get_filter_path(subject_path, filter_name):
    check_filter_name(filter_name)
    return f"{subject_path}/nifti/filtered/{filter_name}"


def get_filter_feature(subject, filter_name, feature_number):
    check_filter_name(filter_name)
    check_feature_number(filter_name, feature_number)

    filter_path = get_filter_path(subject[1], filter_name)

    return f"{filter_path}/{subject[0]}_{filter_name}_feature_{feature_number}.nii.gz"


def get_all_filter_features(subject, filter_name):
    check_filter_name(filter_name)
    features = []

    for feature_number in range(1, VALID_FILTERS.get(filter_name) + 1):
        filter_path = get_filter_path(subject[1], filter_name)
        features.append(
            f"{filter_path}/{subject[0]}_{filter_name}_feature_{feature_number}.nii.gz")

    return features


def get_glcm_feature(subject, feature_number):
    check_feature_number('glcm', feature_number)
    return f"{subject[1]}/nifti/filtered/glcm/{subject[0]}_glcm_feature_{feature_number}.nii.gz"


def get_rlm_feature(subject, feature_number):
    check_feature_number('rlm', feature_number)
    return f"{subject[1]}/nifti/filtered/rlm/{subject[0]}_rlm_feature_{feature_number}.nii.gz"


def get_group_type_from_subject(subject):
    group = ['AD', 'CN', 'MCI']
    if '/AD/' in subject[1]:
        return group[0]
    elif '/CN/' in subject[1]:
        return group[1]
    elif '/MCI/' in subject[1]:
        return group[2]
    return None


def load_features_as_images(features_paths):
    features_as_images = []
    print("Loading features images")
    for feature_path in features_paths:
        features_as_images.append(load_nifti_image(feature_path))
    return features_as_images

# Labels


def get_extracted_labels_path(subject):
    return f"{subject[1]}/nifti/extracted_labels/"


def get_extracted_label(subject, label, hemisphere=None):
    if hemisphere == None:
        return f"{subject[1]}/nifti/extracted_labels/{subject[0]}_label_{label}.nii.gz"

    return f"{subject[1]}/nifti/extracted_labels/{subject[0]}_{hemisphere}_label_{label}.nii.gz"


def read_labels_from_subject_csv(subject):
    path = f"{subject[1]}/{subject[0]}_volumes.csv"
    csv_file = csv.DictReader(
        open(path, "r"), delimiter=",")
    labels = [label for label in csv_file]
    return labels


def read_labels_from_csv(csv_file):
    csv_file = csv.DictReader(
        open(csv_file, "r"), delimiter=",")
    labels = [label for label in csv_file]
    return labels


def write_labels_count_to_csv(csv_arr, s_name, dest_path):
    with open(f'{dest_path}/{s_name}_voxels_count.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_arr)


# Brain


def get_denoised_brain(subject):
    return f"{subject[1]}/nifti/{subject[0]}_brain_denoised.nii.gz"


def get_brain_mask(subject):
    return f"{subject[1]}/nifti/{subject[0]}_brain_mask_dilated.nii.gz"

# Nibabel


def load_nifti_image(im_path):
    im = nib.load(im_path)
    im_data = im.get_fdata()
    return (im_data, im.affine)


def save_nifti1_image(im_data, output_file, affine):
    nib.save(nib.Nifti1Image(im_data, affine), output_file)
