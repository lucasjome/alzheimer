import argparse as ap
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

openmp_threads = 1

FREESURFER_HOME = f""
n_threads = os.cpu_count()
base_folder = ''


# Environment Variables
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
os.environ["FREESURFER_HOME"] = FREESURFER_HOME
os.environ["FREESURFER_SUBJECTS"] = f"{FREESURFER_HOME}/subjects"
os.environ["FREESURFER_FUNCTIONALS_DIR"] = f"{FREESURFER_HOME}/sessions"

# FreeSurfer Configuration
freesurfer_env_prefix = f"source {FREESURFER_HOME}/FreeSurferEnv.sh"


def compose_fs_cmd(cmd):
    return f"bash -c '{freesurfer_env_prefix} && {cmd}'"


def check_necessary_dirs(subject_path, new_dirs):
    for new_dir in new_dirs:
        if not os.path.exists(subject_path + new_dir):
            os.mkdir(subject_path + new_dir)

    return subject_path + new_dirs[-1]


def get_images_paths_from_csv(csv_file, images_folder):
    images_paths = []

    for row in csv_file:
        s_id = row['Subject ID']
        s_visit = row['Visit'].replace(" ", "_").replace("/", "")
        filename = f"{s_id}_{s_visit}.nii.gz"

        images_paths.append(
            f"{images_folder}/{row['Research Group']}/{filename}")
    return images_paths


def run_recon_all(images_paths):

    check_necessary_dirs(os.getenv('FREESURFER_SUBJECTS'),
                         ['/AD', '/CN', '/MCI'])

    p_pool = ProcessPoolExecutor(n_threads)
    results = []

    for image_path in images_paths:
        s_filename = image_path.split("/")[-1].split(".nii.gz")[0]
        s_group = Path(image_path).parent.name
        s_dir = f"{os.getenv('FREESURFER_SUBJECTS')}/{s_group}/"

        cmd = compose_fs_cmd(
            f"recon-all -s {s_filename} -i {image_path} -sd {s_dir} -all -openmp {openmp_threads}")
        # print(cmd)

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
    parser = ap.ArgumentParser(
        description='Run FreeSurfer\'s recon-all')
    parser.add_argument('--images_folder', action='store', type=str,
                        required=True, help='Subjects Input images base directory')
    parser.add_argument('--sd', action='store', type=str,
                        required=True, help='Subjects output base directory')
    parser.add_argument('--csv', action='store', type=str,
                        required=True, help='Subject\'s CSV')

    args = parser.parse_args()
    if not args.sd == None:
        os.environ["FREESURFER_SUBJECTS"] = args.sd
        os.environ["SUBJECTS_DIR"] = args.sd
        Path(args.sd).mkdir(exist_ok=True)

    images_folder = args.images_folder
    csv_file = csv.DictReader(
        open(args.csv, "r"), delimiter=",")

    images_paths = get_images_paths_from_csv(csv_file, images_folder)

    start_time = datetime.now()
    run_recon_all(images_paths)
    end_time = datetime.now()

    log_processing(start_time, end_time, [f"{x}\n" for x in images_paths])


if __name__ == "__main__":
    main()

    pass
