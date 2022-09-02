import os
from typing import List
import open3d as o3d
from batchEvaluation import ALGOS, ALGO_ext, ALGO_IN
from fileio import IOHelper, create_pcd, create_txt
from visualizer import draw_planes


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


if __name__ == '__main__':
    # bag = rb.Bag('bags/dynamic_data.bag','r')
    # for topic, msg, t in bag.read_messages(topics=[]):
    #     print(topic, msg, t)
    # bag.close()
    complete_cloud = 'bags/1661773765.311562777.pcd'
    cloud_path = "bags/"
    gt_path = "bags/.GT"
    binaries = 'AlgoBinaries/'
    algo_path = "/home/pedda/Documents/uni/BA/Thesis/catkin_ws/src/plane-detection/src/EVAL/bags/RSPD"
    iohelper = IOHelper(complete_cloud, gt_path, algo_path)
    complete_cloud = iohelper.read_pcd(complete_cloud)
    ground_truth = iohelper.read_gt()
    # algo_planes = iohelper.read_algo()
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        complete_cloud, voxel_size=0.05)
    # dyn_eval(cloud_path, binaries)
    for file in sorted(os.listdir(cloud_path)):
        if not file.endswith('.pcd'):
            continue
        subcloud = iohelper.read_pcd()
        sub_gt = list(map(lambda plane: plane.crop(
            subcloud, voxel_grid, complete_cloud), ground_truth))
        # draw_planes(sub_gt, complete_cloud)
        
