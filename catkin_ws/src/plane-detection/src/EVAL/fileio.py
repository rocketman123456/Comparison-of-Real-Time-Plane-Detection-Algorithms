
from genericpath import isfile
import os
from pydoc import ispath
from struct import unpack
from typing import List
import numpy as np

from tqdm import tqdm
from classes import Plane


class IOHelper:
    def __init__(self, cloud_path: str, gt_path: str, algo_path: str) -> None:
        self._path_pcd = cloud_path
        self._path_gt = gt_path
        self._path_algo = algo_path

    def read_pcd(self)->np.ndarray:
        print('Reading Point cloud')
        if self._path_pcd.endswith('.pcl'):
            return self.read_pc_pcl()
        else:
            return self.read_pc_xyz()

    def read_gt(self):
        return self._read(self._path_gt)

    def read_algo(self):
        return self._read(self._path_algo)

    def _read(self, path: str)->List[Plane]:
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.endswith('.geo'):
                    return self.read_planes_geo(os.path.join(path, file))
                with open(os.path.join(path, file),'r') as f:
                    if len(f.readline().split(' ')) > 1:
                        return self.read_planes_xyz_from_folder(path)
                    else:
                        return self.read_planes_i_from_folder(path)
        elif path.endswith('.geo'):
            return self.read_planes_geo(path)
        elif path.endswith('asc'):
            return [Plane.xyzfrom_txt(path)]

        l: List[Plane] = [self.read_plane_i(path)]
        return l

    def read_planes_geo(self, filename: str) -> List[Plane]:
        planes: List[Plane] = []
        l = []
        with open(filename, "rb") as file:
            file.read(8)  # numcircles
            num_planes = int(file.read(8)[0])
            print(f'{num_planes = }')
            for i in range(num_planes):
                p: Plane
                color = [unpack('f', file.read(4)) for _ in range(3)]
                center = [unpack('f', file.read(4)) for _ in range(3)]
                normal = [unpack('f', file.read(4))[0] for _ in range(3)]
                basisu = [unpack('f', file.read(4)) for _ in range(3)]
                basisv = [unpack('f', file.read(4)) for _ in range(3)]
                num_in = unpack('N', file.read(8))[0]
                p = Plane()
                p.indices = []
                p.normal = normal
                for inl in range(num_in):
                    point = unpack('N', file.read(8))[0]
                    p.indices.append(point)
                p.name = filename.split('.')[0]
                p.set_indices = set(p.indices)
                planes.append(p)
                l.append(num_in)
        return planes

    def read_plane_xyz(self, filename: str):
        p = Plane.xyzfrom_txt(filename)
        return p

    def read_planes_xyz_from_folder(self, path: str)->List[Plane]:
        planes = []
        for file in os.listdir(path):
            planes.append(self.read_plane_xyz(os.path.join(path, file)))
        return planes
    def read_plane_i(self, filename: str)->Plane:
        p = Plane.i_from_txt(filename)
        return p

    def read_planes_i_from_folder(self, path: str)->List[Plane]:
        planes = []
        for file in os.listdir(path):
            planes.append(self.read_plane_i(os.path.join(path, file)))
        return planes

    def _xyzfrom_bytes(self, b):
        x = unpack('f', b[:4])[0]
        y = unpack('f', b[4:8])[0]
        z = unpack('f', b[8:12])[0]
        # p = Point(x,y,z)
        return [x, y, z]

    def read_pc_pcl(self)->np.ndarray:
        points: List[List[float]] = []
        with open(self._path_pcd, "rb") as file:
            size = unpack('N', file.read(8))[0]
            mode = unpack('N', file.read(8))[0]
            print(f'{size = }, {mode = }')
            for _ in tqdm(range(size)):
                c = self._xyzfrom_bytes(file.read(12))
                points.append(c)
                if mode & 1:
                    file.read(12)
                if mode & 2:
                    file.read(4)
                if mode & 4:
                    file.read(12)
                if mode & 8:
                    file.read(4)
                if mode & 16:
                    file.read(4)
        return np.array(points)

    def read_pc_xyz(self)->np.ndarray:
        points: np.ndarray = np.loadtxt(self._path_pcd, usecols=(0, 1, 2)).tolist()
        return points

    def save_results(self, p: float, r: float, f1: float, found_planes: int ,all_planes: int)->None:
        parent, method = self._path_algo.rsplit('/',1)
        output_file = os.path.join(parent, f'output_{method}.txt')
        print(f'Writing results to {output_file}')
        with open(output_file, 'w') as ofile:
            ofile.write(f'precision: {p}\n')
            ofile.write(f'recall: {r}\n')
            ofile.write(f'f1-score: {f1}\n')
            ofile.write(f'found: {found_planes}/{all_planes}\n')
