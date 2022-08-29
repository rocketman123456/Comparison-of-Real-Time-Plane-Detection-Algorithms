import argparse
import matplotlib.pyplot as plt
import os
from typing import Dict, List
from classes import Result
from evaluate import evaluate
import pandas as pd
from fileio import create_pcd
import seaborn as sb
import numpy as np


# globals
ALGOS = ['RSPD', 'OPS', '3DKHT']
ALGO_ext = {'RSPD': '.geo', 'OPS': '', '3DKHT': ''}
ALGO_in = {'RSPD': '.txt', 'OPS': '.pcd', '3DKHT': '.txt'}


def get_df(results_folder: str):
    # load results
    results = [Result.from_file(os.path.join(results_folder, file))
               for file in os.listdir(results_folder) if file.endswith('.out')]
    fig, axs = plt.subplots(2, len(ALGOS))
    for ax, algo in zip(axs[0], ALGOS):
        ax.set_title(algo)

        # filter results by algorithm
        algo_data = [res for res in results if res.algorithm == algo]
        algo_data.sort(key=lambda x: x.dataset.lower())
        # create algo dataframe
        algo_df = pd.DataFrame(algo_data).drop(
            columns=['detected', 'out_of', 'time_total', 'time_per_plane', 'time_per_sample'])
        algo_df = algo_df.rename(columns={'dataset': 'Scene Types'})
        algo_df.plot.bar(x='Scene Types', ax=ax)  # , marker='o',label='rspd')


    for ax, algo in zip(axs[1], ALGOS):
        algo_data = [res for res in results if res.algorithm == algo]
        algo_data.sort(key=lambda x: x.dataset)
        df = pd.DataFrame(algo_data).drop(
            columns=['precision', 'recall', 'f1', 'detected', 'out_of'])
        sb.violinplot(data=df,ax=ax)
    plt.ylim([0.0, 1.0])
    plt.show()
    # TODO save fig to /$root_folder/figures


def collect_results(root_folder: str):
    # scene -> [Result_RSPD, Result_OPS, ...]
    results_per_scene: Dict[str, List[Result]] = dict()
    # algo -> [scene_type -> [scene_specific_result]]
    results_per_algo: Dict[str, Dict[str, List[Result]]] = {
        'RSPD': {}, 'OPS': {}, '3DKHT': {}}
    datasets = os.listdir(root_folder)
    for dataset in datasets:
        if os.path.isfile(os.path.join(root_folder, dataset)):
            # skip files
            continue
        if dataset.startswith('nope_') or dataset == 'results':
            # skip non-datasets (without GT or results directory)
            continue
        results_per_scene[dataset] = []
        result_path = os.path.join(root_folder, dataset, 'results')
        scene_type = dataset.split('_')[0]

        # populate result dicts, per_scene is currently unused, might be useful later
        for algorithm_resultfile in os.listdir(result_path):
            result = Result.from_file(os.path.join(
                result_path, algorithm_resultfile))
            results_per_scene[dataset].append(result)
            if scene_type not in results_per_algo[result.algorithm].keys():
                results_per_algo[result.algorithm][scene_type] = []
            results_per_algo[result.algorithm][scene_type].append(result)

    # calculate average results for each algorithm w.r.t scene type
    algo_results: List[Result] = []
    for algorithm, scenes_results in results_per_algo.items():
        for scene_type, scene in scenes_results.items():
            scene_p = scene_r = scene_f1 = 0.0
            scene_found = scene_all = 0
            total = per_plane = per_sample = 0.0
            for r in scene:
                scene_p += r.precision
                scene_r += r.recall
                scene_f1 += r.f1
                scene_found += r.detected
                scene_all += r.out_of
                total += r.time_total
                per_plane += r.time_per_plane
                per_sample += r.time_per_sample
            scene_p /= len(scene)
            scene_r /= len(scene)
            scene_f1 /= len(scene)
            total /= len(scene)
            per_plane /= len(scene)
            per_sample /= len(scene)
            scene_type_average = Result(
                scene_p, scene_r, scene_f1, scene_found, scene_all, f'{scene_type}_avg', algorithm, total, per_plane, per_sample)
            algo_results.append(scene_type_average)
    # save results to file
    for result in algo_results:
        if 'results' not in os.listdir(rootFolder):
            os.mkdir(os.path.join(rootFolder, 'results'))
        filepath = os.path.join(
            root_folder, 'results', f'{result.algorithm}-{result.dataset}.out')
        result.to_file(filepath)


def batch_evaluate(root_folder: str):
    datasets = os.listdir(root_folder)
    for dataset in datasets:
        # ignore files, results and datasets without GT
        if os.path.isfile(os.path.join(root_folder, dataset)):
            continue
        if dataset.startswith('nope_') or dataset.startswith('results'):
            continue
        dataset_path = os.path.join(root_folder, dataset)
        gt_path = os.path.join(dataset_path, "GT")
        methods = []
        cloud_filename = dataset + '.txt'
        cloud_path = os.path.join(dataset_path, cloud_filename)
        for algo in ALGOS:
            if algo in os.listdir(dataset_path):
                methods.append(os.path.join(dataset_path, algo))
        for algo_path in methods:
            evaluate(cloud_path, gt_path, algo_path)


def batch_detect(rootfolder: str, binaries_path: str) -> None:
    for dataset in os.listdir(rootfolder):
        dataset_path = os.path.join(rootfolder, dataset)
        # again, ignore files, results and datasets without GT
        if not os.path.isdir(dataset_path):
            continue
        if 'nope_' in dataset or dataset == 'results':
            continue
        for algo in ALGOS:
            if algo != '3DKHT':
                continue
            # get input params for given algorithm
            binary = os.path.join(binaries_path, algo)
            cloud_file = os.path.join(
                dataset_path, f'{dataset}{ALGO_in[algo]}')
            result_file = os.path.join(
                dataset_path, algo, f'{dataset}{ALGO_ext[algo]}')
            # create pcd file if needed (OPS)
            if cloud_file not in os.listdir(dataset_path):
                create_pcd(cloud_file.replace('.pcd', '.txt'))
            # create output folder if not already existing
            # TODO clear directory if already exists?
            if algo not in os.listdir(dataset_path):
                os.mkdir(os.path.join(dataset_path, algo))
            else:
                for file in os.listdir(os.path.join(dataset_path, algo)):
                    os.remove(os.path.join(dataset_path, algo, file))
            # run PDA on dataset
            print(f'Calling {algo} on {dataset}!')
            command = f'{binary} {cloud_file} {result_file}'
            os.system(command)


if __name__ == '__main__':
    fallback_root = "/home/pedda/Documents/uni/BA/Thesis/catkin_ws/src/plane-detection/src/EVAL/Stanford3dDataset_v1.2_Aligned_Version/TEST"
    fallback_algo_binaries = "/home/pedda/Documents/uni/BA/Thesis/catkin_ws/src/plane-detection/src/EVAL/AlgoBinaries"

    # input argument handling
    parser = argparse.ArgumentParser('BatchEvaluation')
    parser.add_argument('-R', '--root-folder', default=fallback_root,
                        help='Path to root directory which includes datasets to be tested')
    parser.add_argument('-A', '--algo-binaries',
                        default=fallback_algo_binaries)
    args = parser.parse_args()

    rootFolder = args.root_folder
    algorithm_binaries = args.algo_binaries

    # batch_detect(rootFolder, algorithm_binaries)
    # batch_evaluate(rootFolder)
    # collect_results(rootFolder)
    get_df(os.path.join(rootFolder, 'results'))
