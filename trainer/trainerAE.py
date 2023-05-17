import torch
import torch.optim as optim
from tqdm import tqdm
from model import CADTransformer
from .base import BaseTrainer
from .loss import CADLoss
from .scheduler import GradualWarmupScheduler
from cadlib.macro import *


class TrainerAE(BaseTrainer):
    def build_net(self, cfg):
        self.net = CADTransformer(cfg).cuda()

    def set_optimizer(self, cfg):
        """set optimizer and lr scheduler used in training"""
        self.optimizer = optim.Adam(self.net.parameters(), cfg.lr)
        self.scheduler = GradualWarmupScheduler(self.optimizer, 1.0, cfg.warmup_step)

    def set_loss_function(self):
        self.loss_func = CADLoss(self.cfg).cuda()

    def forward(self, data):
        commands = data['command'].cuda() # (N, S)
        args = data['args'].cuda()  # (N, S, N_ARGS)

        outputs = self.net(commands, args)
        loss_dict = self.loss_func(outputs)

        return outputs, loss_dict

    def encode(self, data, is_batch=False):
        """encode into latent vectors"""
        commands = data['command'].cuda()
        args = data['args'].cuda()
        if not is_batch:
            commands = commands.unsqueeze(0)
            args = args.unsqueeze(0)
        z = self.net(commands, args, encode_mode=True)
        return z

    def decode(self, z):
        """decode given latent vectors"""
        outputs = self.net(None, None, z=z, return_tgt=False)
        return outputs
    
    def set_idx_on_mask(self, mask, seq_idx, indices, batch_idx=None):
        # if unset=True, set only indices other than `indices` to 1
        if batch_idx is None:
            batch_idx = range(mask.size(0))
        for idx_to_set in indices:
            for i in range(len(ALL_COMMANDS)):
                if i == idx_to_set:
                    mask[batch_idx, seq_idx, i] += 1
        return mask
        

    def logits2vec(self, outputs, refill_pad=True, to_numpy=True):
        # TODO: repeatedly masking
        command_logits = outputs['command_logits']  # (N, S, N_CMD)
        args_logits = outputs['args_logits']  # (N, S, N_ARGS, ARGS_DIM)
        command_logits_mask = command_logits.new_zeros(*command_logits.size())
        prev_command_i = None
        for i in range(command_logits.size(1)):
            command_i = command_logits[:, i].softmax(-1).argmax(-1)  # (N,)
            if i == 0:
                command_logits_mask = self.set_idx_on_mask(command_logits_mask, i, [SOL_IDX])
            elif i == 1:
                command_logits_mask = self.set_idx_on_mask(command_logits_mask, i, [LINE_IDX, ARC_IDX, CIRCLE_IDX])
            elif i == 2:
                command_logits_mask = self.set_idx_on_mask(command_logits_mask, i, [LINE_IDX, ARC_IDX, CIRCLE_IDX, EXT_IDX])
            elif i > 2:
                batch_idx = torch.nonzero(prev_command_i == EXT_IDX)
                command_logits_mask = self.set_idx_on_mask(command_logits_mask, i, [LINE_IDX, ARC_IDX, CIRCLE_IDX, EOS_IDX], batch_idx)
                batch_idx = torch.nonzero(prev_command_i != EXT_IDX)
                command_logits_mask = self.set_idx_on_mask(command_logits_mask, i, [LINE_IDX, ARC_IDX, CIRCLE_IDX, EXT_IDX], batch_idx)
            prev_command_i = command_i  # (N,)
        
        command_logits *= command_logits_mask

        """network outputs (logits) to final CAD vector"""
        out_command = torch.argmax(torch.softmax(command_logits, dim=-1), dim=-1)  # (N, S)
        out_args = torch.argmax(torch.softmax(args_logits, dim=-1), dim=-1) - 1  # (N, S, N_ARGS)
        if refill_pad: # fill all unused command element to -1
            mask = ~torch.tensor(CMD_ARGS_MASK).bool().cuda()[out_command.long()]
            out_args[mask] = -1

        out_cad_vec = torch.cat([out_command.unsqueeze(-1), out_args], dim=-1)  # (N, S, N_ARGS+1)
        if to_numpy:
            out_cad_vec = out_cad_vec.detach().cpu().numpy()
        return out_cad_vec

    def evaluate(self, test_loader):
        """evaluatinon during training"""
        self.net.eval()
        pbar = tqdm(test_loader)
        pbar.set_description("EVALUATE[{}]".format(self.clock.epoch))

        all_ext_args_comp = []
        all_line_args_comp = []
        all_arc_args_comp = []
        all_circle_args_comp = []

        for i, data in enumerate(pbar):
            with torch.no_grad():
                commands = data['command'].cuda()
                args = data['args'].cuda()
                outputs = self.net(commands, args)
                out_args = torch.argmax(torch.softmax(outputs['args_logits'], dim=-1), dim=-1) - 1
                out_args = out_args.long().detach().cpu().numpy()  # (N, S, n_args)

            gt_commands = commands.squeeze(1).long().detach().cpu().numpy() # (N, S)
            gt_args = args.squeeze(1).long().detach().cpu().numpy() # (N, S, n_args)

            ext_pos = np.where(gt_commands == EXT_IDX)
            line_pos = np.where(gt_commands == LINE_IDX)
            arc_pos = np.where(gt_commands == ARC_IDX)
            circle_pos = np.where(gt_commands == CIRCLE_IDX)

            args_comp = (gt_args == out_args).astype(np.int)
            all_ext_args_comp.append(args_comp[ext_pos][:, -N_ARGS_EXT:])
            all_line_args_comp.append(args_comp[line_pos][:, :2])
            all_arc_args_comp.append(args_comp[arc_pos][:, :4])
            all_circle_args_comp.append(args_comp[circle_pos][:, [0, 1, 4]])

        all_ext_args_comp = np.concatenate(all_ext_args_comp, axis=0)
        sket_plane_acc = np.mean(all_ext_args_comp[:, :N_ARGS_PLANE])
        sket_trans_acc = np.mean(all_ext_args_comp[:, N_ARGS_PLANE:N_ARGS_PLANE+N_ARGS_TRANS])
        extent_one_acc = np.mean(all_ext_args_comp[:, -N_ARGS_EXT_PARAM])
        line_acc = np.mean(np.concatenate(all_line_args_comp, axis=0))
        arc_acc = np.mean(np.concatenate(all_arc_args_comp, axis=0))
        circle_acc = np.mean(np.concatenate(all_circle_args_comp, axis=0))

        self.val_tb.add_scalars("args_acc",
                                {"line": line_acc, "arc": arc_acc, "circle": circle_acc,
                                 "plane": sket_plane_acc, "trans": sket_trans_acc, "extent": extent_one_acc},
                                global_step=self.clock.epoch)
