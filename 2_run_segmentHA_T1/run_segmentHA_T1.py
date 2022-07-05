import argparse as ap
import csv
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

#n_threads = psutil.cpu_count()

FREESURFER_HOME = f''
n_threads = os.cpu_count()

# Environment Variables
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
os.environ["FREESURFER_HOME"] = FREESURFER_HOME
os.environ["FREESURFER_SUBJECTS"] = f"{FREESURFER_HOME}/subjects"
os.environ["FREESURFER_FUNCTIONALS_DIR"] = f"{FREESURFER_HOME}/sessions"

# FreeSurfer Configuration
freesurfer_env_prefix = f"source {FREESURFER_HOME}/FreeSurferEnv.sh"


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


def run_ha_segment(subjects_names, p_pool):

    results = []

    for subject_name in subjects_names:
        cmd = compose_fs_cmd(
            f"segmentHA_T1.sh {subject_name} {os.getenv('FREESURFER_SUBJECTS')}")

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
    parser = ap.ArgumentParser(description='Segment Hippocampus Subfields')
    parser.add_argument('--sd', action='store', type=str,
                        required=True, help='Subjects directory')

    args = parser.parse_args()
    if not args.sd == None:
        os.environ["FREESURFER_SUBJECTS"] = args.sd
        os.environ["SUBJECTS_DIR"] = args.sd

    s_names, s_paths = get_subjects_names_and_paths()

    # print(s_names)
    # print(s_paths)

    p_pool = ProcessPoolExecutor(n_threads)

    start_time = datetime.now()
    run_ha_segment(s_names, p_pool)
    end_time = datetime.now()

    log_processing(start_time, end_time, [
                   f"{x}, {y}\n" for x, y in zip(s_names, s_paths)])


if __name__ == "__main__":
    main()

    pass
