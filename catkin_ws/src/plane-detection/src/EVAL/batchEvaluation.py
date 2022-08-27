import os
from typing import Dict, List
from classes import Result
from evaluate import evaluate

# TODO maybe implement some data analysis stuff using pandas and visualize it?

ALGOS = {'RSPD': "/home/pedda/Documents/uni/BA/Thesis/catkin_ws/src/plane-detection/src/EVAL/AlgoBinaries/command_line"}


def collect_results(root_folder: str):
    # scene -> [Result_RSPD, Result_OPS, ...]
    results_per_scene: Dict[str, List[Result]] = dict()
    # algo -> [scene_type -> [scene_specific_result]]
    results_per_algo: Dict[str, Dict[str, List[Result]]] = {
        'RSPD': {}, 'OPS': {}, '3DKHT': {}}
    datasets = os.listdir(root_folder)
    for dataset in datasets:
        if os.path.isfile(os.path.join(root_folder, dataset)):
            continue
        if dataset.startswith('nope_') or dataset == 'results':
            # dataset has no GT, skipping
            continue
        results_per_scene[dataset] = []
        result_path = os.path.join(root_folder, dataset, 'results')
        scene_type = dataset.split('_')[0]

        for algorithm_resultfile in os.listdir(result_path):
            result = Result.from_file(os.path.join(
                result_path, algorithm_resultfile))
            results_per_scene[dataset].append(result)
            if scene_type not in results_per_algo[result.algorithm].keys():
                results_per_algo[result.algorithm][scene_type] = []
            results_per_algo[result.algorithm][scene_type].append(result)
    algo_results: List[Result] = []
    for algorithm, scenes_results in results_per_algo.items():
        for scene_type, scene in scenes_results.items():
            scene_p = scene_r = scene_f1 = 0.0
            scene_found = scene_all = 0
            for r in scene:
                scene_p += r.precision
                scene_r += r.recall
                scene_f1 += r.f1
                scene_found += r.detected
                scene_all += r.out_of
            scene_p /= len(scene)
            scene_r /= len(scene)
            scene_f1 /= len(scene)
            scene_type_average = Result(
                scene_p, scene_r, scene_f1, scene_found, scene_all, f'{scene_type}_avg', algorithm)
            algo_results.append(scene_type_average)
    return algo_results

def batch_evaluate(root_folder: str):
    datasets = os.listdir(root_folder)

    for dataset in datasets:
        if os.path.isfile(os.path.join(rootFolder, dataset)):
            continue
        if dataset.startswith('nope_') or dataset.startswith('results'):
            # dataset has no GT, skipping
            continue
        dataset_path = os.path.join(rootFolder, dataset)
        gt_path = os.path.join(dataset_path, "GT")
        methods = []
        cloud_filename = dataset + '.txt'
        cloud_path = os.path.join(dataset_path, cloud_filename)
        if 'RSPD' in os.listdir(dataset_path):
            methods.append(os.path.join(dataset_path, 'RSPD'))
        if 'OPS' in os.listdir(dataset_path):
            methods.append(os.path.join(dataset_path, 'OPS'))
        if '3DKHT' in os.listdir(dataset_path):
            methods.append(os.path.join(dataset_path, '3DKHT'))
        for algo_path in methods:
            evaluate(cloud_path, gt_path, algo_path)


def batch_detect(rootfolder: str, binaries_path: str) -> None:

    for dataset in os.listdir(rootfolder):
        dataset_path = os.path.join(rootfolder, dataset)
        if not os.path.isdir(dataset_path):
            continue
        if 'nope_' in dataset:
            continue
        cloud_file = os.path.join(dataset_path, f'{dataset}.txt')
        # if 'OPS' not in os.listdir(dataset_path):
        #     os.mkdir(os.path.join(dataset_path, 'OPS'))
        if 'RSPD' not in os.listdir(dataset_path):
            os.mkdir(os.path.join(dataset_path, 'RSPD'))
        # if '3DKHT' not in os.listdir(dataset_path):
        #     os.mkdir(os.path.join(dataset_path, '3DKHT'))
        result_file = os.path.join(dataset_path, 'RSPD', f'{dataset}.geo')
        print(f'Calling RSPD on {dataset}!')
        command = f'{ALGOS["RSPD"]} {cloud_file} {result_file}'
        os.system(command)


if __name__ == '__main__':
    # rootFolder = sys.argv[1]
    rootFolder = "/home/pedda/Documents/uni/BA/Thesis/catkin_ws/src/plane-detection/src/EVAL/Stanford3dDataset_v1.2_Aligned_Version/TEST"
    algorithm_binaries = "/home/pedda/Documents/uni/BA/Thesis/catkin_ws/src/plane-detection/src/EVAL/AlgoBinaries"

    # batch_detect(rootFolder, algorithm_binaries)
    batch_evaluate(rootFolder)
    results = collect_results(rootFolder)
    for result in results:
        if 'results' not in os.listdir(rootFolder):
            os.mkdir(os.path.join(rootFolder, 'results'))
        filepath = os.path.join(
            rootFolder, 'results', f'{result.algorithm}-{result.dataset}.out')
        result.to_file(filepath)
