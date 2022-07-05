import argparse as ap
import csv
import os
import re
from pathlib import Path

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


def extract_labels_from_csv(labels_file):
    print(extract_labels_from_csv.__name__)

    csv_file = csv.DictReader(
        open(labels_file, "r"), delimiter=",")

    all_labels = [label for label in csv_file]

    aseg_labels = []
    dkt_lh_labels = []
    dkt_rh_labels = []

    # populate each label array
    for label in all_labels:
        if label['label_name'].startswith('ctx'):
            if label['label_name'].startswith('ctx-lh'):
                dkt_lh_labels.append(label)
            else:
                dkt_rh_labels.append(label)
        else:
            aseg_labels.append(label)

    return aseg_labels, dkt_lh_labels, dkt_rh_labels


def extract_all_volumes_as_csv(s_dir, s_names):
    print(extract_all_volumes_as_csv.__name__)

    cmd_subjects = ' '.join(s_names)

    out_aseg = f'{s_dir}/out_aseg.csv'
    out_dkt_lh = f'{s_dir}/out_dkt_lh.csv'
    out_dkt_rh = f'{s_dir}/out_dkt_rh.csv'

    aseg_cmd = f'asegstats2table --subjects {cmd_subjects} --meas volume --all-segs --delimiter=comma --tablefile {out_aseg}'
    dkt_lh_cmd = f'aparcstats2table --subjects {cmd_subjects} --hemi lh --meas volume --parc=aparc.DKTatlas --delimiter=comma --tablefile {out_dkt_lh}'
    dkt_rh_cmd = f'aparcstats2table --subjects {cmd_subjects} --hemi rh --meas volume --parc=aparc.DKTatlas --delimiter=comma --tablefile {out_dkt_rh}'

    cmds = [aseg_cmd, dkt_lh_cmd, dkt_rh_cmd]
    run_cmds = [os.system(compose_fs_cmd(cmd)) for cmd in cmds]

    return out_aseg, out_dkt_lh, out_dkt_rh


def write_subjects_label_volumes(subjects, out_aseg, out_dkt_lh, out_dkt_rh, arg_labels):
    csv_aseg = csv.DictReader(open(out_aseg, "r"), delimiter=",")
    csv_dkt_lh = csv.DictReader(open(out_dkt_lh, "r"), delimiter=",")
    csv_dkt_rh = csv.DictReader(open(out_dkt_rh, "r"), delimiter=",")

    aseg_labels, dkt_lh_labels, dkt_rh_labels = extract_labels_from_csv(
        arg_labels)

    aseg_vol = [l for l in csv_aseg]
    dkt_lh_vol = [l for l in csv_dkt_lh]
    dkt_rh_vol = [l for l in csv_dkt_rh]

    for i, sub in enumerate(subjects):
        print(f'Writing CSV files for {sub[0]}')

        # initialize output
        s_aseg = []
        s_dkt_lh = []
        s_dkt_rh = []

        # get subject's line from each csv
        s_aseg_volumes = aseg_vol[i]
        s_dkt_lh_volumes = dkt_lh_vol[i]
        s_dkt_rh_volumes = dkt_rh_vol[i]

        # non-necessary check
        if not s_aseg_volumes['Measure:volume'] == sub[0]:
            # return None
            print("not equal")

        # get aseg volumes
        for label in aseg_labels:
            l_vol = s_aseg_volumes[label['label_name']]
            s_aseg.append((label['label_index'], label['label_name'], l_vol))

        # get dkt_lh volumes
        for label in dkt_lh_labels:
            l_name = label['label_name'].replace(
                '-', '_').replace('ctx_', '') + '_volume'
            l_vol = s_dkt_lh_volumes[l_name]
            s_dkt_lh.append((label['label_index'], label['label_name'], l_vol))

        # get dkt_rh volumes
        for label in dkt_rh_labels:
            l_name = label['label_name'].replace(
                '-', '_').replace('ctx_', '') + '_volume'
            l_vol = s_dkt_rh_volumes[l_name]
            s_dkt_rh.append((label['label_index'], label['label_name'], l_vol))

        # write each set of labels vol as a csv file
        file_header = 'label_index,label_name,label_volume\n'
        with open(f'{sub[1]}/{sub[0]}_aseg_volumes.csv', 'w') as f:
            s_lines = [f','.join(label) for label in s_aseg]
            f.write(file_header)
            f.writelines('\n'.join(s_lines))

        with open(f'{sub[1]}/{sub[0]}_dkt_lh_volumes.csv', 'w') as f:
            s_lines = [f','.join(label) for label in s_dkt_lh]
            f.write(file_header)
            f.writelines('\n'.join(s_lines))

        with open(f'{sub[1]}/{sub[0]}_dkt_rh_volumes.csv', 'w') as f:
            s_lines = [f','.join(label) for label in s_dkt_rh]
            f.write(file_header)
            f.writelines('\n'.join(s_lines))

        # write all labels in the same order as input
        with open(f'{sub[1]}/{sub[0]}_volumes.csv', 'w') as f:
            s_aseg_lines = [f','.join(label) for label in s_aseg]
            s_dkt_lh_lines = [f','.join(label) for label in s_dkt_lh]
            s_dkt_rh_lines = [f','.join(label) for label in s_dkt_rh]

            f.write(file_header)
            f.writelines('\n'.join(s_aseg_lines))
            f.write('\n')
            f.writelines('\n'.join(s_dkt_lh_lines))
            f.write('\n')
            f.writelines('\n'.join(s_dkt_rh_lines))


def main():

    # Parse Argument
    parser = ap.ArgumentParser(
        description='Extract segmentation volume metrics for each subject')
    parser.add_argument('--sd', action='store', type=str,
                        required=True, help='Subjects directory')
    parser.add_argument('--labels', action='store', type=str,
                        required=True, help='Labels CSV file')

    args = parser.parse_args()

    # Get arguments
    s_dir = args.sd
    os.environ["FREESURFER_SUBJECTS"] = s_dir
    os.environ["SUBJECTS_DIR"] = s_dir

    s_names, s_paths = get_subjects_names_and_paths()
    subjects = [sub for sub in zip(s_names, s_paths)]

    out_aseg, out_dkt_lh, out_dkt_rh = extract_all_volumes_as_csv(
        s_dir, s_names)

    write_subjects_label_volumes(
        subjects, out_aseg, out_dkt_lh, out_dkt_rh, args.labels)


if __name__ == "__main__":
    main()

    pass
