
from dataclasses import dataclass
import os
from typing import List
import open3d as o3d
import numpy as np


@dataclass
class Result():
    precision: float
    recall: float
    f1: float
    detected: int
    out_of: int
    dataset: str
    algorithm: str

    def to_file(self, path: str):
        print(f'Writing results to {path}')
        with open(path, 'w') as ofile:
            ofile.write(f'{self.algorithm} : {self.dataset} \n')
            ofile.write(f'precision: {self.precision}\n')
            ofile.write(f'recall: {self.recall}\n')
            ofile.write(f'f1-score: {self.f1}\n')
            ofile.write(f'found: {self.detected} / {self.out_of}\n')

    @staticmethod
    def from_file(path: str):
        dataset, algo = np.loadtxt(
            path, dtype=str, usecols=(0, 2), max_rows=1)
        prec, rec, f1 = np.loadtxt(
            path, dtype=float, skiprows=1, usecols=1, max_rows=3)
        detected, out_of = np.loadtxt(path, dtype=int, usecols=(1,3), skiprows=4)
        return Result(
            precision=prec,
            recall=rec,
            f1=f1,
            detected=detected,
            out_of=out_of,
            dataset=dataset,
            algorithm=algo
        )
    
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

    def calc_voxel(self, vg: o3d.geometry.VoxelGrid, pointcloud: o3d.geometry.PointCloud):
        if self.xyz_points == []:
            self.calc_xyz(pointcloud)
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
