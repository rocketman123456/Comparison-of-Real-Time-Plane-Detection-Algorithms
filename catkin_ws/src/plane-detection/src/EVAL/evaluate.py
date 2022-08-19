from matplotlib import pyplot as plt
import numpy as np
from evaluator import Evaluator, VoxelEvaluator
from visualizer import draw_bb_planes, draw_compare, draw_planes, draw_voxel_correspondence
from fileio import Reader
import open3d as o3d
if __name__ == '__main__':
    # cloud_path = sys.argv[1]
    # gt_path = sys.argv[2]
    # algo_path = sys.argv[3]

    # cloud_path = "/home/pedda/Documents/uni/BA/clones/datasets/RSPD/pointclouds/boiler_room.pcl"
    # gt_path = "/home/pedda/Documents/uni/BA/clones/datasets/RSPD/detections/boiler_room_ground_truth.geo"
    # algo_path = "/home/pedda/Documents/uni/BA/clones/datasets/RSPD/detections/boiler_room_ransac_schnabel.geo"
    cloud_path = "/home/pedda/Documents/uni/BA/Thesis/catkin_ws/src/plane-detection/src/EVAL/Stanford3dDataset_v1.2_Aligned_Version/Area_1/hallway_6/hallway_6.txt"
    gt_path = "/home/pedda/Documents/uni/BA/Thesis/catkin_ws/src/plane-detection/src/EVAL/Stanford3dDataset_v1.2_Aligned_Version/Area_1/hallway_6/GT"
    # algo_path = "/home/pedda/Documents/uni/BA/Thesis/catkin_ws/src/plane-detection/src/EVAL/here.geo"
    algo_path = "/home/pedda/Documents/uni/BA/clones/PlaneDetection/CommandLineOption/hallway_6.geo"

    reader = Reader(cloud_path, gt_path, algo_path)

    points = reader.read_pcd()
    colors  = np.loadtxt(cloud_path, dtype=int, usecols=(3,4,5))
    colors  = colors * (1/255)
    ground_truth = reader.read_gt()
    test = reader.read_algo()
    # draw_compare(ground_truth,test)

    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)
    pointcloud.colors = o3d.utility.Vector3dVector(colors)

    draw_planes(test, pointcloud)

    kdtree = o3d.geometry.KDTreeFlann(pointcloud)

    o3d.visualization.draw_geometries([pointcloud])

    # if own datasets, find corresponding indices for planes in xyz format
    if ground_truth[0].indices == []:
        for plane in ground_truth:
            plane.get_indices_from_kdtree(kdtree)

    if test[0].indices == []:
        for plane in test:
            plane.get_indices_from_kdtree(kdtree)

    octree = o3d.geometry.Octree(max_depth=8)
    octree.convert_from_point_cloud(pointcloud)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pointcloud, voxel_size=0.13)

    # o3d.visualization.draw_geometries([octree])

    inlier_evaluator = Evaluator.create(points, ground_truth, test)
    print('calculating correspondence')
    inlier_evaluator.correspondence()
    print('done calculating correspondence')
    inlier_evaluator.get_precision()
    inlier_evaluator.get_recall()
    inlier_evaluator.get_f1()
    print(f'{inlier_evaluator.precision = }')
    print(f'{inlier_evaluator.recall = }')
    print(f'{inlier_evaluator.f1 = }')
    f = set()
    for i, k in inlier_evaluator.correspondences.items():
        if k != None:
            f.add(k)
    print(f'found: {len(f)} / {len(ground_truth)}')
    input("press enter to continue with voxel evaluation")

    voxel_evaluator = Evaluator.create(points, ground_truth, test, voxel_grid)
    print('calculating correspondence')
    voxel_evaluator.correspondence()
    draw_voxel_correspondence(ground_truth, test, pointcloud)
    print('done calculating correspondence')
    voxel_evaluator.calc_voxels(pointcloud)
    voxel_evaluator.get_precision()
    voxel_evaluator.get_recall()
    voxel_evaluator.get_f1()
    print(f'{voxel_evaluator.precision = }')
    print(f'{voxel_evaluator.recall = }')
    print(f'{voxel_evaluator.f1 = }')
    f = set()
    for i, k in voxel_evaluator.correspondences.items():
        if k != None:
            f.add(k)
    print(f'found: {len(f)} / {len(ground_truth)}')
    input("press enter to continue with octree evaluation")

    octree_evaluator = Evaluator.create(points, ground_truth, test, octree)
    print('calculating correspondence')
    octree_evaluator.correspondence()
    print('calculating leaf nodes')
    octree_evaluator.calc_leafs(pointcloud)
    octree_evaluator.get_precision()
    octree_evaluator.get_recall()
    octree_evaluator.get_f1()

    print(f'{octree_evaluator.precision = }')
    print(f'{octree_evaluator.recall = }')
    print(f'{octree_evaluator.f1 = }')
    f = set()
    for i, k in octree_evaluator.correspondences.items():
        if k != None:
            f.add(k)
    print(f'found: {len(f)} / {len(ground_truth)}')
