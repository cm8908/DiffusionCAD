import os
import h5py
from utils import ensure_dir
from dataset.lgan_dataset import get_dataloader
from config.configDiffusion import ConfigDiffusion
from trainer.trainerDiffusion import TrainerGaussianDiffusion1D

cfg = ConfigDiffusion()
print('data path', cfg.data_root)
agent = TrainerGaussianDiffusion1D(cfg)

if not cfg.test:
    if cfg.cont:
        agent.load_ckpt(cfg.ckpt)
    
    train_loader = get_dataloader(cfg)

    agent.train(train_loader)
else:
    # load trained weights
    agent.load_ckpt(cfg.ckpt)

    # run generator
    generated_shape_codes = agent.generate(cfg.n_samples)

    # save generated z
    save_path = os.path.join(cfg.exp_dir, "results/fake_z_ckpt{}_num{}.h5".format(cfg.ckpt, cfg.n_samples))
    
    ensure_dir(os.path.dirname(save_path))
    with h5py.File(save_path, 'w') as fp:
        fp.create_dataset("zs", shape=generated_shape_codes.shape, data=generated_shape_codes)
