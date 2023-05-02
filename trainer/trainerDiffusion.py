import os
import numpy as np
import torch
import torch.autograd as autograd
import torch.optim as optim
from tqdm import tqdm
from .base import BaseTrainer
from model.GaussianDiffusion1D import GaussianDiffusion1D
from model.noise_predictor import NoisePredictor, NoisePredictorMLP
from utils import cycle


class TrainerGaussianDiffusion1D(BaseTrainer):
    def __init__(self, cfg):
        super(TrainerGaussianDiffusion1D, self).__init__(cfg)
        self.batch_size = cfg.batch_size
        self.n_iters = cfg.n_iters
        self.save_frequency = cfg.save_frequency
        
        self.z_dim = cfg.z_dim

        # build diffusion
        self.build_net(cfg)

        # set optimizer
        self.set_optimizer(cfg)

    def build_net(self, cfg):
        if cfg.model_type == 'transformer_encoder':
            denoise_model = NoisePredictor(cfg)
        elif cfg.model_type == 'mlp':
            denoise_model = NoisePredictorMLP(cfg)
        else:
            raise NotImplementedError()
        
        self.net = GaussianDiffusion1D(denoise_model, cfg).cuda()

    def eval(self):
        self.net.eval()

    def set_optimizer(self, cfg):
        """set optimizer and lr scheduler used in training"""
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.9))

    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(self.model_dir, "ckpt_epoch{}.pth".format(self.clock.step))
            print("Saving checkpoint epoch {}...".format(self.clock.step))
        else:
            save_path = os.path.join(self.model_dir, "{}.pth".format(name))

        torch.save({
            'clock': self.clock.make_checkpoint(),
            'net_state_dict': self.net.cpu().state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, save_path)

        self.net.cuda()

    def load_ckpt(self, name=None):
        """load checkpoint from saved checkpoint"""
        name = name if name == 'latest' else "ckpt_epoch{}".format(name)
        load_path = os.path.join(self.model_dir, "{}.pth".format(name))
        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))

        checkpoint = torch.load(load_path)
        print("Loading checkpoint from {} ...".format(load_path))
        self.net.load_state_dict(checkpoint['net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.clock.restore_checkpoint(checkpoint['clock'])


    def train(self, dataloader):
        data = cycle(dataloader)
        
        pbar = tqdm(range(self.clock.step, self.n_iters))
        for iteration in pbar:
            real_data = next(data)
            
            real_data = real_data.cuda()
            real_data.requires_grad_(True)

            loss = self.net(real_data)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            pbar.set_postfix({'loss': loss.item()})
            self.train_tb.add_scalar('loss', loss.item(), global_step=self.clock.step)
            self.clock.tick()
            if self.clock.step % self.save_frequency == 0:
                self.save_ckpt()
            
    def generate(self, n_samples):
        """generate samples"""  # TODO:
        self.eval()

        chunk_num = n_samples // self.batch_size
        generated_z = []
        for i in range(chunk_num):
            fake = self.net.sample(self.batch_size)  # TODO:
            fake = fake.detach().cpu().numpy()
            generated_z.append(fake)
            print("chunk {} finished.".format(i))

        remains = n_samples - self.batch_size * chunk_num
        with torch.no_grad():
            fake = self.net.sample(self.batch_size)
            fake = fake.detach().cpu().numpy()
        generated_z.append(fake)

        generated_z = np.concatenate(generated_z, axis=0)
        return generated_z