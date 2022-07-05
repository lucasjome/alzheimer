import argparse as ap
import csv
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from subprocess import Popen
import SimpleITK as sitk
import radiomics.imageoperations as radio

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
file_path = Path(sys.argv[0])
file_path = file_path.parent.absolute().as_posix()
GLCM_BINARY = f'{file_path}/TextureFeatures/glcm_itk/build/computeGLCMFeatures'
RLM_BINARY = f'{file_path}/TextureFeatures/rlm_itk/build/computeRLMFeatures'


def compose_fs_cmd(cmd):
    return f"bash -c '{freesurfer_env_prefix} && {cmd}'"


def check_necessary_dirs(subject_path, new_dirs):
    # Create new folders inside subject directory
    for new_dir in new_dirs:
        if not os.path.exists(subject_path + new_dir):
            os.mkdir(subject_path + new_dir)

    return subject_path + new_dirs[-1]


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


def create_necessary_nifti_files(subjects_names, subjects_paths):

    subjects = list(tuple(zip(subjects_names, subjects_paths)))

    subjects_len = len(subjects)
    subjects_in_chunks = np.array_split(
        subjects, np.ceil(subjects_len/(n_threads)))
    print(subjects_in_chunks)

    for subject in subjects_in_chunks:
        commands = []
        for s_name, s_path in subject:
            check_necessary_dirs(s_path, ['/nifti'])

            nifti_path = f"{s_path}/nifti"

            # Convert denoised brain to nifti
            cmd_1 = compose_fs_cmd(
                # f"mri_convert {s_path}/mri/antsdn.brain.mgz {nifti_path}/{s_name}_brain_denoised.nii.gz")
                f"mri_convert {s_path}/mri/orig.mgz {nifti_path}/{s_name}_orig.nii.gz")
            commands.append(cmd_1)

            # Binarize and dilate brain mask to nifti
            cmd_2 = compose_fs_cmd(
                f"mri_binarize --i {s_path}/mri/brainmask.mgz --o {nifti_path}/{s_name}_brain_mask_dilated.nii.gz --min 0 --max 1 --binval 0 --binvalnot 1 --dilate 3")
            commands.append(cmd_2)
            print(commands)
        processes = [Popen(cmd, shell=True) for cmd in commands]
        exit_codes = [p.wait() for p in processes]


def run_glcm_filter(subjects_names, subjects_paths, p_pool):
    print(run_glcm_filter.__name__)

    for s_name, s_path in zip(subjects_names, subjects_paths):
        check_necessary_dirs(
            s_path, ['/nifti', '/nifti/filtered', '/nifti/filtered/glcm'])

        nifti_path = f"{s_path}/nifti"

        cmd = f"{GLCM_BINARY} -i {nifti_path}/{s_name}_orig.nii.gz -m {nifti_path}/{s_name}_brain_mask_dilated.nii.gz -o {nifti_path}/filtered/glcm/{s_name}.nii.gz -sepfeat -nr 1 -nb 16"
        print(cmd)

        p_pool.submit(os.system, cmd)


def run_rlm_filter(subjects_names, subjects_paths, p_pool):
    print(run_rlm_filter.__name__)

    for s_name, s_path in zip(subjects_names, subjects_paths):
        check_necessary_dirs(
            s_path, ['/nifti', '/nifti/filtered', '/nifti/filtered/rlm'])

        nifti_path = f"{s_path}/nifti"

        cmd = f"{RLM_BINARY} -i {nifti_path}/{s_name}_orig.nii.gz -m {nifti_path}/{s_name}_brain_mask_dilated.nii.gz -o {nifti_path}/filtered/rlm/{s_name}.nii.gz -sepfeat -nr 1 -nb 16"
        print(cmd)
        p_pool.submit(os.system, cmd)


def run_lbp(lbp_params, s_name, nifti_path):
    print(f"LBP Filter for: {s_name}")
    img_path = f"{nifti_path}/{s_name}_orig.nii.gz"
    mask_path = f"{nifti_path}/{s_name}_brain_mask_dilated.nii.gz"
    output_path = f"{nifti_path}/filtered/lbp/"

    imageType = "NiftiImageIO"

    img = sitk.ReadImage(img_path, imageIO=imageType)
    mask = sitk.ReadImage(mask_path, imageIO=imageType)

    generator = radio.getLBP3DImage(img, mask,
                                    lbp3DLevels=lbp_params['lbp3DLevels'],
                                    lbp3DIcosphereRadius=lbp_params['lbp3DIcosphereRadius'],
                                    lbp3DIcosphereSubdivision=lbp_params['lbp3DIcosphereSubdivision'])

    for i, (f_image, f_name, f_params) in enumerate(generator):
        output_name = f"{s_name}_lbp_feature_{i+1}"
        print(f"Writing LBP Feature {i+1}")
        output_full = output_path + output_name
        print(output_full)
        sitk.WriteImage(f_image, output_full + '.nii.gz', imageIO=imageType)


def run_lbp_filter(subjects_names, subjects_paths, p_pool, lbp_params):
    print(run_lbp_filter.__name__)

    for s_name, s_path in zip(subjects_names, subjects_paths):
        check_necessary_dirs(
            s_path, ['/nifti', '/nifti/filtered', '/nifti/filtered/lbp'])

        nifti_path = f"{s_path}/nifti"

        #p_pool.submit(run_lbp, lbp_params, s_name, nifti_path)
        run_lbp(lbp_params, s_name, nifti_path)


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
        description='Convert, Binarize Brain Regions and Run GLCM, RLM and LBP filters')
    parser.add_argument('--sd', action='store', type=str,
                        required=True, help='Subjects directory')
    parser.add_argument('--lbp_params', nargs=3, type=int, default=[3, 2, 1],
                        help="lbp3DLevels lbp3DIcosphereRadius lbp3DIcosphereSubdivision")

    parser.add_argument('-create', action='store_true',
                        help='Create NIFTI files from FreeSurfer Subjects')
    parser.add_argument('-glcm', action='store_true', help='Run GLCM Filter')
    parser.add_argument('-rlm', action='store_true', help='Run RLM Filter')
    parser.add_argument('-lbp', action='store_true', help='Run LBP Filter')

    args = parser.parse_args()
    if not args.sd == None:
        os.environ["FREESURFER_SUBJECTS"] = args.sd
        os.environ["SUBJECTS_DIR"] = args.sd

    s_names, s_paths = get_subjects_names_and_paths()

    p_pool = ProcessPoolExecutor(n_threads)

    start_time = datetime.now()

    # Check and run arguments
    if args.create:
        create_necessary_nifti_files(s_names, s_paths)
    if args.glcm:
        run_glcm_filter(s_names, s_paths, p_pool)
    if args.rlm:
        run_rlm_filter(s_names, s_paths, p_pool)
    if args.lbp:
        # LBP-3D
        lbp_params = {}
        lbp_params['lbp3DLevels'] = args.lbp_params[0]
        lbp_params['lbp3DIcosphereRadius'] = args.lbp_params[1]
        lbp_params['lbp3DIcosphereSubdivision'] = args.lbp_params[2]

        run_lbp_filter(s_names, s_paths, p_pool, lbp_params=lbp_params)

    end_time = datetime.now()

    #log_processing(start_time, end_time, [f"{x}, {y}\n" for x, y in zip(s_names, s_paths)])


if __name__ == "__main__":
    main()

    pass
