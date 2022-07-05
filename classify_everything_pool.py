import argparse as ap
from datetime import datetime
from pathlib import Path
from subprocess import Popen
from concurrent.futures import ProcessPoolExecutor

# GLOBALS
nodeattrs = ['histogram', 'volume_based']
kernels = ['P2K', 'HGK_WL', 'HGK_SP', 'HGK_WSP']
distances = ['wasserstein', 'hellinger', 'kl']
classes = ['AD', 'MCI', 'CN']

python_path = ''
base_code = ''
path_prefix = f"{python_path} {base_code}/6_graph_classification/graphs_classif.py"

ad_folder_path = ''
mci_folder_path = ''
cn_folder_path = ''

result_dir = f"{base_code}/results/"

n_threads = 2


def execute_command(cmd, log_file, result_dir):
    print(log_file)
    # print(cmd)
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    log_file = open(log_file, 'a')
    p = Popen(cmd, shell=True,
              stdout=log_file, stderr=log_file).wait()
    # p.wait()
    log_file.close()

    return True


def classify(class_1, class_2, dir_name, p_pool):
    cmds = list()
    class1_name, class2_name = dir_name.split('x')
    for node_attr in nodeattrs:
        for kernel in kernels:
            tmp_result_dir = f"{result_dir}/{dir_name}/results_{node_attr}_{kernel}"

            for dist in distances:
                log_file_path = f"{tmp_result_dir}/{datetime.now().strftime('%d_%m_%Y_%H_%M')}_{dist}_log"
                temp_cmd = f"{path_prefix} --class1 {class_1} --class2 {class_2} --results {tmp_result_dir} --dist {dist} --nodeattr {node_attr} --kernel {kernel} --shortened_names {class1_name} {class2_name}"
                final_cmd = f"cd {base_code}/6_graph_classification/ && " + temp_cmd
                cmds.append((final_cmd, log_file_path))

                p_pool.submit(execute_command, final_cmd,
                              log_file_path, tmp_result_dir)


def main():
    print("main")
    p_pool = ProcessPoolExecutor(n_threads)

    try:
        # CN x AD
        classify(cn_folder_path, ad_folder_path, 'CNxAD', p_pool)

        # MCI x CN
        classify(cn_folder_path, mci_folder_path, 'CNxMCI', p_pool)

        # AD x MCI
        classify(mci_folder_path, ad_folder_path, 'MCIxAD', p_pool)
    except KeyboardInterrupt:
        p_pool.shutdown()
        print("end")


if __name__ == "__main__":
    main()
