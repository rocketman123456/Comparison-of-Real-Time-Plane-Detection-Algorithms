import sys
import os

from evaluate import evaluate
if __name__ == '__main__':
    # rootFolder = sys.argv[1]
    rootFolder = "/home/pedda/Documents/uni/BA/Thesis/catkin_ws/src/plane-detection/src/EVAL/Stanford3dDataset_v1.2_Aligned_Version/TEST"
    datasets = os.listdir(rootFolder)
    for dataset in datasets:
        if dataset.startswith('nope_'):
            # dataset has no GT, skipping
            continue
        dataset_path = os.path.join(rootFolder, dataset)
        gt_path = os.path.join(dataset_path, "GT")
        methods = []
        cloud_filename  = dataset + '.txt'
        cloud_path = os.path.join(dataset_path, cloud_filename)
        if 'RSPD' in os.listdir(dataset_path):
            methods.append(os.path.join(dataset_path, 'RSPD'))
        if 'OPS' in os.listdir(dataset_path):
            methods.append(os.path.join(dataset_path, 'OPS'))
        if '3DKHT' in os.listdir(dataset_path):
            methods.append(os.path.join(dataset_path, '3DKHT'))
        for algo_path in methods:
            evaluate(cloud_path, gt_path, algo_path)