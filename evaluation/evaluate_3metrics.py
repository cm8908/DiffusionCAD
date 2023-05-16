import h5py
from tqdm import tqdm
import numpy as np
import argparse
import pickle
import sys, os
sys.path.append("..")
from cadlib.visualize import vec2CADsolid, CADsolid2pc
from cadlib.macro import *
from joblib import Parallel, delayed
from glob import glob
from OCC.Core.BRepCheck import BRepCheck_Analyzer

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default=None, required=True)
parser.add_argument('--n_points', type=int, default=2000)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--novel', action='store_true')
parser.add_argument('--unique', action='store_true')
args = parser.parse_args()

result_dir = args.src
target_dir = '../data/cad_shape'

filenames = sorted(os.listdir(result_dir))

if args.novel:
    total_shape_list = []
    target_filenames_dirs = sorted(os.listdir(target_dir))
    # print(target_filenames_dirs); exit()
    for fnm_dir in tqdm(target_filenames_dirs, desc='Collecting shape file names...'):
        target_filenames = sorted(glob(os.path.join(target_dir, fnm_dir) + '/*.pkl'))
        for fnm in target_filenames:
            data_id = fnm.split('/')[-1].split('.')[0]
            with open(fnm, 'rb') as f:
                shape = pickle.load(f)
                if shape is not None:
                    total_shape_list.append((data_id, shape))

def is_valid(path) -> bool:
    with h5py.File(path, 'r') as fp:
        out_vec = fp['out_vec'][:].astype(int)

    data_id = path.split('/')[-1].split('.')[0]
    try:
        shape = vec2CADsolid(out_vec)
    except Exception as e:
        print("create_CAD failed", data_id)
        return False, None
    
    analyzer = BRepCheck_Analyzer(shape)
    if not analyzer.IsValid():
        print("validity check failed", data_id)
        return False, None
    
    try:
        out_pc = CADsolid2pc(shape, args.n_points, data_id)
    except Exception as e:
        print('convert pc failed', data_id)
        return False, None
    return True, shape

def is_novel(shape) -> bool:
    for i, (data_id, target_shape) in enumerate(total_shape_list):
        if shape.IsPartner(target_shape):
            return False
    return True

def is_unique(shape, idx) -> bool:
    for i, target_shape in enumerate(valid_shapes):
        if idx == i:
            continue
        if shape.IsPartner(target_shape):
            return False
    return True

novel_samples = []
unique_samples = []

if args.parallel:
    valid_samples, valid_shapes = zip(*Parallel(n_jobs=16, verbose=2)(delayed(is_valid)(os.path.join(result_dir, name)) for name in filenames))
    valid_shapes = list(filter(None, valid_shapes))  # Eliminate None i.e. invalid shapes
    if args.novel:
        novel_samples = Parallel(n_jobs=8, verbose=2, prefer='processes')(delayed(is_novel)(shape) for shape in valid_shapes)
    if args.unique:
        unique_samples = Parallel(n_jobs=8, verbose=2, prefer='processes')(delayed(is_unique)(shape, i) for i, shape in enumerate(valid_shapes))
else:
    valid_samples = []
    for name in tqdm(filenames):
        path = os.path.join(result_dir, name)
        valid_samples.append(is_valid(path))

validity = np.array(valid_samples).mean()
novelty = np.array(novel_samples).mean()
uniqueness = np.array(unique_samples).mean()
print(len(valid_shapes))
print(f'Validity: {validity}, Novelty: {novelty}, Uniqueness: {uniqueness}')
save_path = result_dir + '_3metrics.txt'
with open(save_path, 'w') as f:
    log_str = f'<Validity>\nN_TOTAL={len(valid_samples)}, N_VALID={len(valid_shapes)}, Validity={validity}\n\n' + \
              f'<Novelty>\nN_TOTAL={len(novel_samples)}, N_NOVEL={np.array(novel_samples).sum()}, Novelty={novelty}\n\n' + \
              f'<Uniqueness>\nN_TOTAL={len(unique_samples)}, N_UNIQUE={np.array(unique_samples).sum()}, Uniqueness={uniqueness}\n'
    f.write(log_str)

    
    # out_cmd = out_vec[:, 0]
    # out_param = out_vec[:, 1:]

    #TODO: Measure Validity->Novelty->Uniqueness