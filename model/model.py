import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from atom_embedding import AtomFeaturizer

class InterpreMol(nn.Module):
    def __init__(self, encoder, head, config):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.config = config

    def forward(self, mol_batch):  # List[rdkit.Chem.Mol]
        x = self.encoder(mol_batch)
        return self.head(x)

    def save(self, path, epoch=None, metrics=None):
        torch.save({
            "encoder": self.encoder.state_dict(),
            "head": self.head.state_dict(),
            "config": self.config,
            "epoch": epoch,
            "metrics": metrics
        }, path)

    @classmethod
    def load(cls, path, device="cpu"):
        state = torch.load(path, map_location=device)
        config = state["config"]

        featurizer = AtomFeaturizer(d_model=config["d_model"])
        encoder_model = GraphTransformerEncoder(
            n_layers=config["n_layers"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            dim_ff=config["dim_ff"],
            dropout=config["dropout"]
        )
        encoder = GraphEncoder(featurizer, encoder_model, use_cls_token=config["use_cls_token"])
        encoder.load_state_dict(state["encoder"])

        head = MLPHead(
            input_dim=config["d_model"],
            hidden_dim=config["mlp_hidden_dim"],
            depth=config["mlp_head_depth"],
            out_dim=config.get("out_dim", 1)  # default = 1 (single-task)
        )

        head.load_state_dict(state["head"])

        return cls(encoder, head, config).eval()
    
    @classmethod
    def from_config(cls, config):
        featurizer = AtomFeaturizer(d_model=config["d_model"])
        encoder_model = GraphTransformerEncoder(
            n_layers=config["n_layers"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            dim_ff=config["dim_ff"],
            dropout=config["dropout"]
        )
        encoder = GraphEncoder(featurizer, encoder_model, use_cls_token=config["use_cls_token"])
        head = MLPHead(input_dim=config["d_model"], hidden_dim=config["mlp_hidden_dim"], depth=config["mlp_head_depth"])
        return cls(encoder, head, config)


class GraphEncoder(nn.Module):
    def __init__(self, featurizer, model, use_cls_token=True):
        super().__init__()
        self.featurizer = featurizer
        self.model = model
        self.use_cls_token = use_cls_token
        self.pooling = CLSPooling(model.d_model) if use_cls_token else None

    def forward(self, mols):  # List[rdkit.Chem.Mol]
        atom_embs = [self.featurizer(mol).to(next(self.parameters()).device) for mol in mols] # List of [n_i, d_model]
        padded = pad_sequence(atom_embs, batch_first=True)  # [batch, max_n, d_model]
        if self.use_cls_token:
            padded = self.pooling(padded)
        encoded = self.model(padded)
        return encoded[:, 0] if self.use_cls_token else encoded.mean(dim=1)
    
    def embed(self, mol):
        return self.featurizer(mol).unsqueeze(0)

    def encode_from_emb(self, atom_emb):
        if self.use_cls_token:
            atom_emb = self.pooling(atom_emb)
        return self.model(atom_emb)

class GraphTransformerBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, dim_ff=256, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x

class GraphTransformerEncoder(nn.Module):
    def __init__(self, n_layers=4, d_model=128, n_heads=4, dim_ff=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([
            GraphTransformerBlock(d_model, n_heads, dim_ff, dropout) for _ in range(n_layers)
        ])

    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x

class CLSPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, D]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1 + N, D]
        return x

class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth, out_dim=1):
        super().__init__()
        layers = []
        for _ in range(depth - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, out_dim))  # now supports multi-task
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

