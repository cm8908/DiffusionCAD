import h5py
import os
import pickle
from glob import glob
from joblib import Parallel, delayed
from os.path import join, dirname, exists, basename
import sys
sys.path.append("..")
from cadlib.visualize import vec2CADsolid

VEC_DIR = '../data/cad_vec/*'
SAVE_DIR = '../data/cad_shape'
os.makedirs(SAVE_DIR, exist_ok=True)

all_path_list = []
data_id_list = []

directories = sorted(glob(VEC_DIR))
for directory in directories:
    vecs = sorted(glob(join(directory, '*.h5')))
    all_path_list += vecs
    
def process_one(path):
    dir_id, data_id_path = path.split('/')[-2:]
    data_id= data_id_path.split('.')[0]
    
    with h5py.File(path, 'r') as fp:
        cad_vec = fp['vec'][:].astype(int)

    try:
        shape = vec2CADsolid(cad_vec)
    except Exception as e:
        print("failed:", data_id)
        return
    
    save_path = join(SAVE_DIR, dir_id, data_id+'.pkl')
    truck_dir = dirname(save_path)
    if not exists(truck_dir):
        os.makedirs(truck_dir, exist_ok=True)
    with open(save_path, 'wb') as fp:
        pickle.dump(shape, fp)

Parallel(n_jobs=16, verbose=2, prefer='threads')(delayed(process_one)(path) for path in all_path_list)