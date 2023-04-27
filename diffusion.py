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
    raise NotImplementedError()