from typing import List
import open3d as o3d
from classes import Plane
import numpy as np

def draw_planes(planes: List[Plane]):
    points  = []
    for plane in planes:
        for pts in plane.xyz_points:
            points.append(pts)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([cloud])

def draw_compare(gt: List[Plane], test: List[Plane]):
    gtpoints  = []
    for plane in gt:
        for pts in plane.xyz_points:
            gtpoints.append(pts)
    gtcloud = o3d.geometry.PointCloud()
    gtcloud.points = o3d.utility.Vector3dVector(gtpoints)
    gtcloud.paint_uniform_color([0, 1,0])
    points  = []
    for plane in test:
        for pts in plane.xyz_points:
            points.append(pts)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.paint_uniform_color([0, 0,1])



    o3d.visualization.draw_geometries([gtcloud])
    o3d.visualization.draw_geometries([cloud])

def draw_bb_planes(planes: List[Plane]):
    to_draw = []
    for plane in planes:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(plane.xyz_points)
        cloud.paint_uniform_color([0.8,0.8,0.8])
        aabb = cloud.get_axis_aligned_bounding_box()
        aabb.color = (0, 1, 0)
        to_draw.append(aabb)
        to_draw.append(cloud)

    o3d.visualization.draw_geometries(to_draw)