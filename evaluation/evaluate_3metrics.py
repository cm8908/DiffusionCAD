import h5py
from tqdm import tqdm
import numpy as np
import argparse
import sys, os
sys.path.append("..")
from cadlib.visualize import vec2CADsolid, CADsolid2pc
from cadlib.macro import *
from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default=None, required=True)
parser.add_argument('--n_points', type=int, default=2000)
parser.add_argument('--parallel', action='store_true')
args = parser.parse_args()

result_dir = args.src
target_dir = '../data/cad_vec'

filenames = sorted(os.listdir(result_dir))

def is_valid(path) -> bool:
    with h5py.File(path, 'r') as fp:
        out_vec = fp['out_vec'][:].astype(int)

    data_id = path.split('/')[-1].split('.')[0]
    try:
        shape = vec2CADsolid(out_vec)
    except Exception as e:
        print("create_CAD failed", data_id)
        return False, None
    
    try:
        out_pc = CADsolid2pc(shape, args.n_points, data_id)
    except Exception as e:
        print('convert pc failed', data_id)
        return False, None
    return True, shape

def is_novel(shape) -> bool:
    for target in target_shapes:
        if shape.IsPartner(target):
            return False
    return True

if args.parallel:
    valid_samples, valid_shapes = zip(*Parallel(n_jobs=8, verbose=2, prefer='threads')(delayed(is_valid)(os.path.join(result_dir, name)) for name in filenames))
else:
    valid_samples = []
    for name in tqdm(filenames):
        path = os.path.join(result_dir, name)
        valid_samples.append(is_valid(path))

validity = np.array(valid_samples).mean()
print(len(valid_shapes))
print(f'Validity: {validity}')


    
    # out_cmd = out_vec[:, 0]
    # out_param = out_vec[:, 1:]

    #TODO: Measure Validity->Novelty->Uniqueness