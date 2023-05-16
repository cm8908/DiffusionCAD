import torch
import torch.nn as nn
import math

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class TransformerEncoder(nn.Module):
    def __init__(self, h_dim, ff_dim, num_layers, norm_type, dropout_rate, num_heads):
        super().__init__()
        
        self.num_layers = num_layers
        self.mhsa = nn.ModuleList([nn.MultiheadAttention(h_dim, num_heads, dropout_rate) for _ in range(num_layers)])
        self.norm1 = nn.ModuleList([Norm(norm_type, h_dim) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([Norm(norm_type, h_dim) for _ in range(num_layers)])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h_dim, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, h_dim)
            ) for _ in range(num_layers)
        ])
    def forward(self, h):
        for i in range(self.num_layers):
            h_rc = h
            h = self.norm1[i](self.mhsa[i](h, h, h)[0] + h_rc)
            h_rc = h
            h = self.norm2[i](self.ffn[i](h) + h_rc)
        return h

class Norm(nn.Module):
    def __init__(self, norm_type: str, h_dim):
        super().__init__()
        self.norm = {
            'layernorm': nn.LayerNorm,
            'batchnorm': nn.BatchNorm1d
        }[norm_type](h_dim)
    def forward(self, x):
        return self.norm(x)
        
class NoisePredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        z_dim = cfg.z_dim
        h_dim = cfg.h_dim
        ff_dim = cfg.ff_dim
        num_enc_layers = cfg.num_enc_layers
        num_heads = cfg.num_heads
        norm_type = cfg.norm_type
        dropout_rate = cfg.dropout_rate
        
        self.h_dim = h_dim
        
        self.input_up_proj = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, h_dim)
        )

        time_dim = h_dim * 4
        self.time_emb = nn.Sequential(
            nn.Linear(h_dim, time_dim),
            SiLU(),
            nn.Linear(time_dim, h_dim)
        )
        
        self.output_down_proj = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, z_dim)
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.layernorm = Norm(norm_type, h_dim)
        self.encoder = TransformerEncoder(h_dim, ff_dim, num_enc_layers, norm_type, dropout_rate, num_heads)
    
    def forward(self, seq, time):
        """_summary_

        Args:
            seq (Tensor): (N, L, Z)
            time (Tensor): (N,)

        Returns:
            noise (Tensor): (N, L, Z)
        """
        emb = self.time_emb(timestep_embedding(time, self.h_dim))  # (N, H)
        
        emb_seq = self.input_up_proj(seq)  # (N, L, H)

        emb_inputs = emb_seq + emb[:,None,:].expand(-1, seq.size(1), -1)

        emb_inputs = self.dropout(self.layernorm(emb_inputs))

        h = self.encoder(emb_inputs)

        noise = self.output_down_proj(h)
        
        return noise

class NoisePredictorMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.z_dim = cfg.z_dim
        self.h_dim = cfg.h_dim
        
        self.net = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            SiLU(),
            nn.Linear(self.h_dim, self.h_dim),
            SiLU(),
            nn.Linear(self.h_dim, self.h_dim),
            SiLU(),
            nn.Linear(self.h_dim, self.z_dim)
        )
        
        time_dim = self.h_dim * 4
        self.time_emb = nn.Sequential(
            nn.Linear(self.h_dim, time_dim),
            SiLU(),
            nn.Linear(time_dim, self.z_dim)
        )
    def forward(self, seq, time):
        time_emb = self.time_emb(timestep_embedding(time, self.h_dim))  # (N, H)
        emb = self.net(seq) + time_emb[:,None,:].expand(-1, seq.size(1), -1)
        output = torch.tanh(emb)
        return output
        
        
if __name__ == '__main__':
        # z_dim = cfg.z_dim
        # h_dim = cfg.h_dim
        # ff_dim = cfg.ff_dim
        # num_enc_layers = cfg.num_enc_layers
        # num_heads = cfg.num_heads
        # norm_type = cfg.norm_type
        # dropout_rate = cfg.dropout_rate
        
    from types import SimpleNamespace
    cfg = {
        'z_dim': 256,
        'h_dim': 512,
        'ff_dim': 1024,
        'num_enc_layers': 6,
        'num_heads': 8,
        'norm_type': 'layernorm',
        'dropout_rate': 0.1
    }
    model = NoisePredictor(SimpleNamespace(**cfg))
    seq = torch.rand(512, 256)
    time = torch.randint(0, 2000, (512,))
    noise = model.forward(seq, time)
    print(noise.shape)