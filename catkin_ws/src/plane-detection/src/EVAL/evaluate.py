from typing import Dict, List, Set, Tuple
import open3d as o3d
import numpy as np
import sys
import os


class Plane:
    def __init__(self, points: np.array, index: int):
        self.inliers = points
        self.index = index
        self.voxels = set()
        self.correspondence = -1  # -1: not found


def read_planes(folder: str):
    planes: List[Plane] = []
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            # only use x,y,z values (ignore rgb)
            points = np.loadtxt(filename, usecols=(0, 1, 2))
            planes.append(Plane(points, len(planes)))
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
    pcd_file = sys.argv[1]
    planes_folder = sys.argv[2]
    gt_folder = sys.argv[3]

    # read planes for evaluation
    ALGO_planes = read_planes(planes_folder)
    GT_planes = read_planes(gt_folder)

    # create Voxelization of complete Point cloud
    pcd = o3d.io.read_point_cloud(pcd_file)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd, voxel_size=0.1)

    # draw voxelgrid if needed
    # o3d.visualization.draw_geometries([voxel_grid])

    # calculate corresponding voxels for all planes
    GT_plane_voxel = calc_plane_voxels(GT_planes, voxel_grid)
    ALGO_plane_voxel = calc_plane_voxels(ALGO_planes, voxel_grid)

    for gtp in GT_planes:
        for ap in ALGO_planes:
            correspondence = len(gtp.voxels & ap.voxels)
            # if more than 50% of voxels match
            if correspondence > len(gtp.voxels)//2:
                # add or update best match
                if correspondence > gtp.correspondence:
                    gtp.ap_plane = ap.index
                gtp.found = True

    # TODO:
    # Modify GT -> only plane inliers
    # implement saving of inlier points to file for algorithms
