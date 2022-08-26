import os
from typing import Dict, List
from classes import Result
from evaluate import evaluate

# TODO maybe implement some data analysis stuff using pandas and visualize it?

def collect_results(root_folder: str):
    results_per_scene: Dict[str, List[Result]] = dict() # scene -> [Result_RSPD, Result_OPS, ...] 
    scene_averages: List[Result] = []
    datasets = os.listdir(root_folder)
    for dataset in datasets:
        if os.path.isfile(os.path.join(root_folder,dataset)):
            continue
        if dataset.startswith('nope_'):
            # dataset has no GT, skipping
            continue
        results_per_scene[dataset] = []
        result_path = os.path.join(root_folder, dataset, 'results')

        for algorithm_resultfile in os.listdir(result_path):
            result = Result.from_file(os.path.join(result_path,algorithm_resultfile))
            results_per_scene[dataset].append(result)
    for scene, results in results_per_scene.items():
        scene_type = scene.split('_')[0]
        p = r = f1 = found = a = 0
        for result in results:
            p += result.precision
            r += result.recall
            f1 += result.f1
            found += result.detected
            a += result.out_of
        p /= len(results)
        r /= len(results)
        f1 /= len(results)
        scene_averages.append(Result(p, r, f1, found, a, scene_type, 'AVERAGE'))
    return scene_averages

if __name__ == '__main__':
    # rootFolder = sys.argv[1]
    rootFolder = "/home/pedda/Documents/uni/BA/Thesis/catkin_ws/src/plane-detection/src/EVAL/Stanford3dDataset_v1.2_Aligned_Version/TEST"
    datasets = os.listdir(rootFolder)

    for dataset in datasets:
        if os.path.isfile(os.path.join(rootFolder,dataset)):
            continue
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
    results = collect_results(rootFolder)
    for result in results:
        filepath = os.path.join(rootFolder, f'{result.algorithm}-{result.dataset}.out')
        result.to_file(filepath)