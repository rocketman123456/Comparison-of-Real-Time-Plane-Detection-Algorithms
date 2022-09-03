import os
from typing import List
import open3d as o3d
from batchEvaluation import ALGOS, ALGO_ext, ALGO_IN
from classes import Plane
from fileio import IOHelper, create_pcd, create_txt
from visualizer import draw_bb_planes, draw_compare, draw_planes, draw_voxel_correspondence
from evaluator import Evaluator
import numpy as np

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
    draw_voxel_correspondence(ground_truth, sub_algo, sub_cloud)

    for gtp in voxel_evaluator.correspondences.values():
        if gtp != None:
            f.add(gtp)
    print(f'{p}, {r}, {f1} at {time = }')
    # total, per_plane, per_sample = iohelper.get_times()

    # iohelper.save_results(p, r, f1, len(f), len(
    #     ground_truth), total, per_plane, per_sample)



if __name__ == '__main__':
    complete_cloud='bags/1661773765.311562777.pcd'
    cloud_path="bags"
    gt_path="bags/.GT"
    binaries='AlgoBinaries/'
    algo_path="/home/pedda/Documents/uni/BA/Thesis/catkin_ws/src/plane-detection/src/EVAL/bags/RSPD"

    iohelper=IOHelper(complete_cloud, gt_path, algo_path)
    complete_cloud=iohelper.read_pcd(complete_cloud)
    ground_truth=iohelper.read_gt()
    voxel_grid=o3d.geometry.VoxelGrid.create_from_point_cloud(
        complete_cloud, voxel_size = 0.3)

    # timeframes = iohelper.get_frames(cloud_path)
    # for timeframe in timeframes[-1:]:
    #     sub_cloud, sub_gt, sub_algo = iohelper.get_frame_data(timeframe, voxel_grid, complete_cloud)
    #     evaluate_timeframe(sub_cloud, sub_gt, sub_algo, timeframe)
    p = iohelper.read_planes_geo('here.geo')
    draw_planes(p, complete_cloud)
