from argparse import ArgumentParser
from operator import gt, index
from typing import Dict, List, Optional, Set, Tuple, Union
import open3d as o3d
import numpy as np
import sys
import os


class Plane:
    def __init__(self, points: np.array, index: int, name: str):
        self.inliers = points
        self.index = index
        self.name = name
        self.voxels = set()
        self.correspondence: float = -1  # -1: not found
        self.found = False
        self.gt_index: Union[int, None] = None


def read_planes(folder: str):
    planes: List[Plane] = []
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            # only use x,y,z values (ignore rgb)
            points = np.loadtxt(f'{folder}/{filename}', usecols=(0, 1, 2))
            name = filename.replace('.txt', '')
            planes.append(Plane(points, len(planes), name))
        else:
            points = []
            with open(f'{folder}/{filename}', 'r') as file:
                for line in file.readlines()[12:]:
                    x, y, z, _ = line.split(' ')
                    points.append([float(x), float(y), float(z)])
            points = np.array(points)
            name = filename.replace('.pcd', '')
            planes.append(Plane(points, len(planes), name))
    return planes


def calc_plane_voxels(planes: List[Plane], voxel_grid: o3d.geometry.VoxelGrid):
    plane_voxel: Dict[int, Set[Tuple[int, int, int]]] = dict()
    # calculate corresponding voxels for each plane
    for plane in planes:
        for point in plane.inliers:
            corr_voxel = tuple(voxel_grid.get_voxel(point))
            if plane.index not in plane_voxel.keys():
                plane_voxel[plane.index] = set()
            plane_voxel[plane.index].add(corr_voxel)
    return plane_voxel


if __name__ == '__main__':
    # if len(sys.argv) < 4:
    #     print('Too few arguments!')
    #     print(
    #         'Usage: python evaluate.py [cloud].pcd [found_planes_folder] [gt_folder]')
    #     sys.exit(1)
    # pcd_file = 'Stanford3dDataset_v1.2_Aligned_Version/Area_1/copyRoom_1/Annotations_out/wall_3.txt_out.pcd'
    # planes_folder = 'TESTGT'
    # gt_folder = 'TESTGT'
    parser = ArgumentParser(prog="Evaluation")
    # parser.add_argument('-r', '--recursive', action='store_true',
    #                     help="if set, all folders inside path are evaluated.")
    parser.add_argument('-s', '--scene', type=str,
                        help="path to complete [scene].pcd")
    parser.add_argument('-a', '--algo-planes', type=str,
                        help="path to directory containing [algo-planes].txt found by PDA.")
    parser.add_argument('-g', '--ground-truth', type=str,
                        help="path to directory containing [ground-truth].txt files")

    args = parser.parse_args()
    pcd_file = args.scene
    planes_folder = args.algo_planes
    gt_folder = args.ground_truth

    # uncomment for

    # read planes for evaluation
    ALGO_planes = read_planes(planes_folder)
    GT_planes = read_planes(gt_folder)

    # create Voxelization of complete Point cloud
    pcd = o3d.io.read_point_cloud(pcd_file)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd, voxel_size=0.1)

    # draw voxelgrid if needed
    # o3d.visualization.draw_geometries(
    #     [voxel_grid])

    # calculate corresponding voxels for all planes
    GT_plane_voxel = calc_plane_voxels(GT_planes, voxel_grid)
    ALGO_plane_voxel = calc_plane_voxels(ALGO_planes, voxel_grid)
    index_pairs: Dict[int, int] = dict()

    for gt_plane_index, gtp_voxels in GT_plane_voxel.items():
        for algo_plane_index, ap_voxels in ALGO_plane_voxel.items():
            correspondence = len(gtp_voxels & ap_voxels) / len(gtp_voxels)
            # if more than 50% of voxels match
            if correspondence > 0.5:
                # add or update best match
                GT_planes[gt_plane_index].found = True
                GT_planes[gt_plane_index].correspondence = correspondence
                GT_planes[gt_plane_index].ap_index = algo_plane_index
                ALGO_planes[algo_plane_index].gt_index = gt_plane_index
                ALGO_planes[algo_plane_index].correspondence = correspondence
                index_pairs[gt_plane_index] = algo_plane_index
    # count of voxels
    TP = 0
    FP = 0
    TN = 0  # not needed for calc, #total_voxels - (TP & FP & FN)
    FN = 0

    # voxel v einer ebene P ist TP, wenn v sowohl in P_a als auch P_gt enthalten ist
    # voxel v einer ebene P ist FP, wenn v in P_a, aber nicht in P_gt enthalten ist
    all_voxels = voxel_grid.get_voxels()

    for gti, api in index_pairs.items():
        for gtv in GT_plane_voxel[gti]:
            if gtv in ALGO_plane_voxel[api]:
                TP += 1
            else:
                FN += 1

    for a_i, a_v in ALGO_plane_voxel.items():
        if ALGO_planes[a_i].gt_index is None:
            # plane not in GT, add all voxels to FP
            FP += len(a_v)
        else:
            corr_gt_voxel = GT_plane_voxel[ALGO_planes[a_i].gt_index]
            for voxel in a_v:
                if voxel not in corr_gt_voxel:
                    # voxel in AP, not in GT -> FP
                    FP += 1

    print('found planes:')
    for p in GT_planes:
        if p.found:
            print(f'\t{p.name = } with {p.correspondence = }')
            best_match = ALGO_planes[p.ap_index].name
            print(f'\tbest match: {best_match}\n')
    print('-------')
    Precision = TP / (TP+FP)
    print(f'\t{Precision = :.4}')
    Recall = TP / (TP+FN)
    print(f'\t{Recall = :.4}')
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    print(f'\t{F1 = :.4}')

    # TODO:
    # Modify GT -> only plane inliers
    # implement saving of inlier points to file for algorithms
