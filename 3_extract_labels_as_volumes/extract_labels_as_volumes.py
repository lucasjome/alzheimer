import argparse as ap
import csv
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

FREESURFER_HOME = f''
n_threads = os.cpu_count()

# Environment Variables
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
os.environ["FREESURFER_HOME"] = FREESURFER_HOME
os.environ["FREESURFER_SUBJECTS"] = f"{FREESURFER_HOME}/subjects"
os.environ["FREESURFER_FUNCTIONALS_DIR"] = f"{FREESURFER_HOME}/sessions"

# FreeSurfer Configuration
freesurfer_env_prefix = f"source {FREESURFER_HOME}/FreeSurferEnv.sh"


# LookUpTable File
lut_file = "./labels/FreeSurferColorLUT.txt"

# CSV Files
ASEG_LABELS_FILE = "./labels/aseg_labels.csv"
HP_SUBFIELDS_LABELS_FILE = "./labels/hpsubfields_labels.csv"


def compose_fs_cmd(cmd):
    return f"bash -c '{freesurfer_env_prefix} && {cmd}'"


def get_subjects_names_and_paths():
    # scripts/hippocampal-subfields-T1.log check if needed

    # subjects_dir = Path(base_folder)
    subjects_dir = Path(os.getenv('FREESURFER_SUBJECTS'))

    folders = [f for f in subjects_dir.iterdir() if f.is_dir()
               and re.match(r'^[0-9]{3}_S_[0-9]{4}', f.name)]

    # subjects = [(sub.name, sub.resolve().as_posix()) for sub in folders]

    subjects_names = [path.name for path in folders]
    subjects_paths = [path.resolve().as_posix() for path in folders]

    return subjects_names, subjects_paths


def check_necessary_dirs(subject_path, new_dirs):
    # Create new folders inside subject directory
    for new_dir in new_dirs:
        if not os.path.exists(subject_path + new_dir):
            os.mkdir(subject_path + new_dir)

    return subject_path + new_dirs[-1]


def extract_hp_labels(s_names, s_paths, p_pool, hp_labels):
    print(extract_hp_labels.__name__)

    #p_pool = ProcessPoolExecutor(n_threads)
    results = []

    output_sufix = '.nii.gz'
    hp_filenames = ['lh.hippoAmygLabels-T1.v21.FS60.FSvoxelSpace.mgz',
                    'rh.hippoAmygLabels-T1.v21.FS60.FSvoxelSpace.mgz']

    # Read labels from CSV file
    csv_file = csv.DictReader(
        open(hp_labels, "r"), delimiter=",")

    labels = [label for label in csv_file]

    for subj_name, subj_path in zip(s_names, s_paths):
        # Fazer uma função do callback que adicione ao fim do arquivo de log, o tempo por imagem
        # ou checar a última execução e pegar o tempo de finish dela no callback
        check_necessary_dirs(
            subj_path, ['/nifti', '/nifti/extracted_labels'])

        output_dir = subj_path + "/nifti/extracted_labels"

        for label in labels:
            # Left hemisphere
            output_filename_lh = f"{output_dir}/{subj_name}_lh_label_{label['label_index']}{output_sufix}"
            cmd_lh = compose_fs_cmd(
                f"mri_binarize --i {subj_path + '/mri/' + hp_filenames[0]} --o {output_filename_lh} --match {label['label_index']}")

            future_lh = p_pool.submit(os.system, cmd_lh)
            results.append(future_lh)

            # Right hemisphere
            output_filename_rh = f"{output_dir}/{subj_name}_rh_label_{label['label_index']}{output_sufix}"
            cmd_rh = compose_fs_cmd(
                f"mri_binarize --i {subj_path + '/mri/' + hp_filenames[1]} --o {output_filename_rh} --match {label['label_index']}")

            future_rh = p_pool.submit(os.system, cmd_rh)
            results.append(future_rh)


def extract_dkt_aseg_labels(s_names, s_paths, p_pool, aseg_labels):
    print(extract_dkt_aseg_labels.__name__)

    #p_pool = ProcessPoolExecutor(n_threads)
    results = []

    output_sufix = '.nii.gz'
    aseg_dkt_filename = 'aparc.DKTatlas+aseg.mgz'
    aseg_filename = 'aseg.mgz'

    # Read labels from CSV file
    csv_file = csv.DictReader(
        open(aseg_labels, "r"), delimiter=",")

    labels = [label for label in csv_file]

    for subj_name, subj_path in zip(s_names, s_paths):
        check_necessary_dirs(
            subj_path, ['/nifti', '/nifti/extracted_labels'])

        output_dir = subj_path + "/nifti/extracted_labels"

        for label in labels:
            output_filename = f"{output_dir}/{subj_name}_label_{label['label_index']}{output_sufix}"
            input_volume_file = aseg_filename if int(
                label['label_index']) in [3, 42] else aseg_dkt_filename

            cmd = compose_fs_cmd(
                f"mri_binarize --i {subj_path + '/mri/' + input_volume_file} --o {output_filename} --match {label['label_index']}")
            future = p_pool.submit(os.system, cmd)
            results.append(future)


def log_processing(start_time, end_time, output_list):
    # maybe change to __file__?
    if not os.path.exists("logs"):
        os.mkdir("logs")

    script_name = sys.argv[0].split(".")[0]

    with open(f"./logs/{script_name}_processing_{end_time.strftime('%d_%m_%Y_%H_%M')}", 'w') as f:
        f.writelines([f"{script_name}",
                      f"\n{end_time.strftime('%d/%m/%Y')}",
                      f"\nInicio: {start_time}",
                      f"\nFim: {end_time}",
                      f"\nTempo Total: {end_time-start_time}",
                      f"\nImages:\n"])
        f.writelines(output_list)


def main():

    # Parse Argument
    parser = ap.ArgumentParser(description='Extract Labels as NIFTI Volumes')
    parser.add_argument('--sd', action='store', type=str,
                        required=True, help='Subjects directory')
    parser.add_argument('--labels_aseg', action='store', type=str,
                        required=True, help='ASEG Labels CSV file')
    parser.add_argument('--labels_hp', action='store', type=str,
                        required=True, help='HP subfields Labels CSV file')

    args = parser.parse_args()
    if not args.sd == None:
        os.environ["FREESURFER_SUBJECTS"] = args.sd
        os.environ["SUBJECTS_DIR"] = args.sd

    s_names, s_paths = get_subjects_names_and_paths()

    # Parse Argument

    aseg_labels = args.labels_aseg
    hp_labels = args.labels_hp

    p_pool = ProcessPoolExecutor(n_threads)

    start_time = datetime.now()
    extract_hp_labels(s_names, s_paths, p_pool, hp_labels)
    extract_dkt_aseg_labels(s_names, s_paths, p_pool, aseg_labels)
    end_time = datetime.now()

    log_processing(start_time, end_time, [
                   f"{x}, {y}\n" for x, y in zip(s_names, s_paths)])


if __name__ == "__main__":
    main()

    pass
