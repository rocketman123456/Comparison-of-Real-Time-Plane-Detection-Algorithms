import argparse
import os
from typing import List
import pandas as pd
import open3d as o3d
from batchEvaluation import ALGOS, ALGO_ext, ALGO_IN, get_df
from classes import Plane, Result
from fileio import IOHelper, create_pcd, create_txt
from visualizer import draw_bb_planes, draw_compare, draw_planes, draw_voxel_correspondence
from evaluator import Evaluator
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/pedda/Documents/coding/OBRG/')
import obrg


def dyn_eval(path_to_subclouds: str, binaries_path: str):
    subcloud_paths: List[str] = [file for file in os.listdir(
        path_to_subclouds) if file.endswith('.pcd')]
    subcloud_paths.sort()
    print(subcloud_paths)
    for subcloud in subcloud_paths:
        for algo in ALGOS:
            if algo != 'RSPD':
                continue
            # get input params for given algorithm
            binary = os.path.join(binaries_path, algo)
            cloud_file = os.path.join(
                path_to_subclouds, f'{subcloud.rsplit(".", 1)[0]}{ALGO_IN[algo]}')
            result_file = os.path.join(
                path_to_subclouds, algo, f'{subcloud.rsplit(".", 1)[0]}{ALGO_ext[algo]}')
            # create txt file if needed (RSPD or 3DKHT)
            if cloud_file not in os.listdir(path_to_subclouds):
                create_txt(cloud_file.replace('.txt', '.pcd'))
            # create output folder if not already existing
            if algo not in os.listdir(path_to_subclouds):
                os.mkdir(os.path.join(path_to_subclouds, algo))
            # else:
            #     for file in os.listdir(os.path.join(path_to_subclouds, algo)):
            #         os.remove(os.path.join(path_to_subclouds, algo, file))
            # run PDA on subcloud
            print(f'Calling {algo} on {subcloud}!')
            command = f'{binary} {cloud_file} {result_file}'
            os.system(command)


def evaluate_timeframe(subcloud, subgt, subalgo, time):
    global voxel_grid, iohelper
    if iohelper.method == '3DKHT':
        print('3DKHT, translating')
        for algo_plane in subalgo:
            algo_plane.translate(subcloud.get_center())

    kdtree = o3d.geometry.KDTreeFlann(subcloud)

    if subgt[0].indices == []:
        for plane in subgt:
            plane.get_indices_from_kdtree(kdtree)

    if subalgo[0].indices == []:
        for plane in subalgo:
            plane.get_indices_from_kdtree(kdtree)

    voxel_evaluator = Evaluator.create(np.empty(0), subgt, subalgo, voxel_grid)
    print('calculating correspondence')
    voxel_evaluator.correspondence()

    print('done calculating correspondence')
    voxel_evaluator.calc_voxels(sub_cloud)
    p, r, f1 = voxel_evaluator.get_metrics()
    f = set()
    # draw_voxel_correspondence(ground_truth, sub_algo, sub_cloud)

    for gtp in voxel_evaluator.correspondences.values():
        if gtp != None:
            f.add(gtp)
    print(f'{p}, {r}, {f1} at {time = }')
    total, per_plane, per_sample = iohelper.get_times()

    iohelper.save_results(p, r, f1, len(f), len(
        ground_truth), total, per_plane, per_sample, time=time)


def dynamic_detection(dataset_path: str, binaries_path: str, algos=ALGOS):
    files = os.listdir(dataset_path)
    calculated = []
    for file in files:
        if file.endswith('.bag') or 'nope' in file or os.path.isdir(os.path.join(dataset_path, file)):
            continue
        if file.split('.')[0] in calculated:
            continue
        for algo in algos:
            algo_path = os.path.join(dataset_path, algo)
            binary = os.path.join(binaries_path, algo)
            cloud_file = os.path.join(
                dataset_path, f'{file.rsplit(".",1)[0]}{ALGO_IN[algo]}')
            result_file = os.path.join(
                dataset_path, algo, f'{file.rsplit(".",1)[0]}{ALGO_ext[algo]}')
            # create txt file if needed (non-ops)
            if cloud_file not in os.listdir(dataset_path):
                create_txt(cloud_file.replace('.txt', '.pcd'))

            # create output folder if not already existing
            if algo not in os.listdir(dataset_path):
                os.mkdir(algo_path)
            # run PDA on dataset
            if algo == 'OBRG':
                obrg.calculate(cloud_file, dataset_path)
            else:
                print(f'Calling {algo} on {cloud_file}!')
                command = f'{binary} {cloud_file} {result_file}'
                os.system(command)
                calculated.append(file.split('.')[0])


def dynamic_collection(dataset_path: str, algos=ALGOS):
    results_folder = os.path.join(dataset_path, 'results')
    for algo in algos:
        results = [Result.from_file(os.path.join(results_folder, file))
                   for file in os.listdir(results_folder) if file.endswith('.out') and not 'avg' in file and algo in file]
        if len(results) == 0:
            continue
        avg_p = avg_r = avg_f1 = 0.0
        avg_t = 0
        for res in results:
            avg_p += res.precision
            avg_r += res.recall
            avg_f1 += res.f1
            avg_t += res.time_total
        avg_p /= len(results)
        avg_r /= len(results)
        avg_f1 /= len(results)
        avg_t /= len(results)
        avg = Result(avg_p, avg_r, avg_f1,0,0,dataset_path.rsplit("/")[-1], algo, avg_t,-1,-1)
        filepath = os.path.join(
            dataset_path, 'results', f'{algo}-{dataset_path.rsplit("/")[-1]}_avg.out')
        avg.to_file(filepath)

def get_dyn_df(results_folder: str, algos=ALGOS):
    # load results
    results = [Result.from_file(os.path.join(results_folder, file))
               for file in os.listdir(results_folder) if file.endswith('.out') and 'avg' in file]
    fig, axs = plt.subplots(1, len(algos))
    fig.set_size_inches(20, 15)
    for ax, algo in zip(axs, algos):
        ax.set_title(algo)

        # filter results by algorithm
        algo_data = [res for res in results if res.algorithm == algo]
        if len(algo_data) == 0:
            continue
        algo_data.sort(key=lambda x: x.dataset.lower())
        # create algo dataframe
        algo_df = pd.DataFrame(algo_data).drop(
            columns=['detected', 'out_of', 'time_total', 'time_per_plane', 'time_per_sample'])
        algo_df = algo_df.rename(columns={'dataset': 'Scene Types'})
        algo_df.plot.bar(x='Scene Types', ax=ax)  # , marker='o',label='rspd')
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("")
        ax.get_xaxis().set_label("")
        ax.legend().remove()

    fig.autofmt_xdate(rotation=45)
    fig.supxlabel('Scene Types')
    area = results_folder.rsplit('/', 2)[1].lower()
    fname = f'{area}_acc.png'

    # plt.savefig(os.path.join('/home/pedda/Documents/uni/BA/Thesis/Document/images',fname))
    plt.show()
    # plt.close()
    fig, axs = plt.subplots(1, len(algos))
    fig.set_size_inches(20, 15)

    for ax, algo in zip(axs, algos):
        ax.set_title(algo)

        algo_data = [res for res in results if res.algorithm == algo]
        if len(algo_data) == 0:
            continue
        algo_data.sort(key=lambda x: x.dataset.lower())
        df = pd.DataFrame(algo_data).drop(
            columns=['precision', 'recall', 'f1', 'detected', 'out_of', 'time_per_plane', 'time_per_sample'])
        # sb.violinplot(data=df,ax=ax)
        df.plot.bar(x='dataset', ax=ax)  # , marker='o',label='rspd')
        ax.set_xlabel("")
        ax.get_xaxis().set_label("")
        ax.legend().remove()
    fig.autofmt_xdate(rotation=45)
    fig.supxlabel('Scene Type')
    fname = f'{area}_time.png'
    plt.show()
    # TODO save fig to /$root_folder/figures


if __name__ == '__main__':
    fallback_cloud = "FIN-Dataset/TEST/1663834492.251202345.pcd"
    fallback_clouds_path = "FIN-Dataset/TEST"
    binaries = 'AlgoBinaries/'

    parser = argparse.ArgumentParser('Dynamic Evaluation')
    parser.add_argument('-D', '--dataset', default=fallback_clouds_path,
                        help='Path to root directory which includes datasets to be tested')
    parser.add_argument('-N', '--last-cloud',
                        default=fallback_cloud)
    args = parser.parse_args()

    dataset = args.dataset
    last_cloud = args.last_cloud
    gt_path = f"{dataset}/GT"

    # dynamic_detection(dataset, binaries, ['RSPD'])
    # dynamic_evaluation()
    # for algo in ['RSPD']:
    #     algo_path = os.path.join(dataset, algo)

    #     iohelper = IOHelper(last_cloud, gt_path, algo_path)
    #     complete_cloud: o3d.geometry.PointCloud = iohelper.read_pcd(last_cloud)
    #     ground_truth = iohelper.read_gt()
    #     voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
    #         complete_cloud, voxel_size=0.3)

    #     timeframes = iohelper.get_frames(dataset)
    #     for timeframe in timeframes:
    #         sub_cloud, sub_gt, sub_algo = iohelper.get_frame_data(
    #             timeframe, voxel_grid, complete_cloud)
    #         # draw_bb_planes(sub_gt, sub_cloud)
    #         evaluate_timeframe(sub_cloud, sub_gt, sub_algo, timeframe)
    dynamic_collection(dataset,['RSPD'])
    # get_dynamic_results()
    get_dyn_df(os.path.join(dataset, 'results'))
