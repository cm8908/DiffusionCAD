import os
import argparse
import json
import shutil
import torch.nn as nn
from cadlib.macro import *
from utils import ensure_dirs

class ConfigDiffusion:
    def __init__(self):
        # parse command line arguments
        parser, args = self.parse()
        self.model_type = args.model_type

        self.set_configuration()

        assert 'seq_len = saved z seq length'
        config_path = os.path.join(args.proj_dir, args.exp_name, 'config.txt')
        ae_configs = json.load(open(config_path))
        if 'keep_seq_len' in ae_configs.keys() and ae_configs['keep_seq_len'] == True:
            self.seq_len = MAX_TOTAL_LEN
        else:
            self.seq_len = 1
        
        # set as attributes
        print("----Experiment Configuration-----")
        for k, v in args.__dict__.items():
            print("{0:20}".format(k), v)
            self.__setattr__(k, v)

        # experiment paths
        self.data_root = os.path.join(args.proj_dir, args.exp_name, "results/all_zs_ckpt{}.h5".format(args.ae_ckpt))
        if args.exp_tag is None:
            if args.model_type == 'mlp':
                diffusion_dir = "mlp-diffusion"
            elif args.model_type == 'unet':
                diffusion_dir = 'unet-diffusion'
            else:
                diffusion_dir = "diffusion"
        else:
            diffusion_dir = args.exp_tag
        self.exp_dir = os.path.join(args.proj_dir, args.exp_name, diffusion_dir+"{}".format(args.ae_ckpt))
        self.log_dir = os.path.join(self.exp_dir, 'log')
        self.model_dir = os.path.join(self.exp_dir, 'model')

        if (not args.test) and args.cont is not True and os.path.exists(self.exp_dir):
            response = input(f'Experiment {self.exp_dir} already exists, overwrite? (y/n) ')
            if response != 'y':
                exit()
            shutil.rmtree(self.exp_dir)
        ensure_dirs([self.log_dir, self.model_dir])

        # GPU usage
        if args.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

        # save this configuration
        if not args.test:
            with open('{}/config.txt'.format(self.exp_dir), 'w') as f:
                json.dump(self.__dict__, f, indent=2)

    def set_configuration(self):
        # self.model_type = 'transformer_encoder'  # ['transformer_encoder', 'mlp']

        # network configuration
        self.h_dim = 512
        self.z_dim = 256
        self.ff_dim = 1024  # for Transformer encoder

        self.num_enc_layers = 6
        self.num_heads = 4
        if self.model_type != 'unet':
            self.num_heads *= 2
        self.norm_type = 'layernorm'
        self.dropout_rate = 0.1

        # Unet configuration
        self.num_res_blocks = 2
        self.attention_resolutions = '16,8'
        self.num_classes = None
        self.use_checkpoint = False
        self.num_heads_upsample = -1  # same as num_heads
        self.use_scale_shift_norm = True
        self.training_mode = 'emb'
        self.vocab_size = None
        
        self.beta1 = 0.5
        

    def parse(self):
        """initiaize argument parser. Define default hyperparameters and collect from command-line arguments."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--proj_dir', type=str, default="proj_log",
                            help="path to project folder where models and logs will be saved")
        parser.add_argument('--exp_name', type=str, required=True, help="name of this experiment")  
        parser.add_argument('--exp_tag', type=str, default=None, help="name tag for result directory")  
        parser.add_argument('--ae_ckpt', type=str, required=True, help="ckpt for autoencoder")
        parser.add_argument('--continue', dest='cont', action='store_true', help="continue training from checkpoint")
        parser.add_argument('--ckpt', type=str, default='latest', required=False, help="desired checkpoint to restore")
        parser.add_argument('--test', action='store_true', help="test mode")
        parser.add_argument('--n_samples', type=int, default=100, help="number of samples to generate when testing")
        parser.add_argument('-g', '--gpu_ids', type=str, default="0",
                            help="gpu to use, e.g. 0  0,1,2. CPU not supported.")

        parser.add_argument('--model_type', type=str, default='transformer_encoder', choices=['transformer_encoder', 'mlp', 'unet'])
        parser.add_argument('--batch_size', type=int, default=256, help="batch size")
        parser.add_argument('--num_workers', type=int, default=8, help="number of workers for data loading")

        parser.add_argument('--n_iters', type=int, default=200000, help="total number of iterations to train")
        parser.add_argument('--save_frequency', type=int, default=100000, help="save models every x iterations")
        parser.add_argument('--lr', type=float, default=2e-4, help="initial learning rate")

        parser.add_argument('--seq_len', type=int, default=1, help="length of z variable")
        parser.add_argument('--timesteps', type=int, default=2000, help='diffusion time steps')
        parser.add_argument('--loss_type', choices=['l1', 'l2'], default='l1', help='DDPM loss type')
        parser.add_argument('--beta_schedule', choices=['linear', 'cosine'], default='cosine', help='beta scheduling')

        args = parser.parse_args()
        return parser, args