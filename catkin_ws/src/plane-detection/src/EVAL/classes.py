
from typing import List
import open3d as o3d
import numpy as np


class Plane:
    def __init__(self) -> None:
        self.indices: List[int] = []
        self.name: str = ""
        self.xyz_points = []
        self.kd = None
        self.set_indices = set()
        self.set_points = set()
        self.voxels = set()
        self.normal = []
        self.leafs = set()
    def calc_voxel(self, vg: o3d.geometry.VoxelGrid):
        for inlier in self.xyz_points:
            v = vg.get_voxel(inlier)
            self.voxels.add(tuple(v))

    def get_indices_from_kdtree(self, kdtree: o3d.geometry.KDTreeFlann):
        for point in self.xyz_points:
            [k, idx, _] = kdtree.search_knn_vector_3d(point, 1)
            self.indices.append(idx[0])
            self.set_indices.add(idx[0])

    def calc_xyz(self, pointcloud: o3d.geometry.PointCloud):
        for point in self.set_indices:
            self.xyz_points.append(pointcloud.points[point])

    def set_set(self, pc: o3d.geometry.PointCloud):
        # self.set_indices = set(self.indices)
        for i in self.set_indices:
            self.set_points.add(tuple(pc.points[i]))

    @staticmethod
    def xyzfrom_txt(file: str):
        points = np.loadtxt(file, usecols=(0, 1, 2)).tolist()
        p = Plane()
        p.xyz_points = points
        p.name = file.split('/')[-1]
        return p

    @staticmethod
    def i_from_txt(file: str):
        points = np.loadtxt(file, usecols=(0)).tolist()
        p = Plane()
        p.indices = points
        p.name = file.split('/')[-1]
        return p


class NULL(Plane):
    def __init__(self) -> None:
        self.name = "NULL"

    @staticmethod
    def create():
        return NULL()
