import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from atom_embedding import AtomFeaturizer
from edge_bias import EdgeBiasEncoder

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
        encoder = GraphEncoder(
            featurizer, encoder_model,
            use_cls_token=config.get("use_cls_token", True),
            use_edge_bias=config.get("use_edge_bias", True),
            max_distance=config.get("max_distance", 6)
        )
        encoder.load_state_dict(state["encoder"], strict=False)

        head = MLPHead(
            input_dim=config["d_model"],
            hidden_dim=config["mlp_hidden_dim"],
            depth=config["mlp_head_depth"],
            out_dim=config.get("out_dim", 1)
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
        encoder = GraphEncoder(
            featurizer, encoder_model,
            use_cls_token=config.get("use_cls_token", True),
            use_edge_bias=config.get("use_edge_bias", True),
            max_distance=config.get("max_distance", 6)
        )
        head = MLPHead(
            input_dim=config["d_model"],
            hidden_dim=config["mlp_hidden_dim"],
            depth=config["mlp_head_depth"],
            out_dim=config.get("out_dim", 1)
        )
        return cls(encoder, head, config)


class GraphEncoder(nn.Module):
    def __init__(self, featurizer, model, use_cls_token=True, use_edge_bias=True, max_distance=6):
        super().__init__()
        self.featurizer = featurizer
        self.model = model
        self.use_cls_token = use_cls_token
        self.use_edge_bias = use_edge_bias
        self.pooling = CLSPooling(model.d_model) if use_cls_token else None

        if use_edge_bias:
            self.edge_bias_encoder = EdgeBiasEncoder(
                n_heads=model.n_heads,
                max_distance=max_distance
            )
        else:
            self.edge_bias_encoder = None

    def forward(self, mols):  # List[rdkit.Chem.Mol]
        device = next(self.parameters()).device
        batch_size = len(mols)

        # Compute atom embeddings
        atom_embs = [self.featurizer(mol).to(device) for mol in mols]
        lengths = [emb.shape[0] for emb in atom_embs]
        max_atoms = max(lengths)

        # Pad atom embeddings
        padded = pad_sequence(atom_embs, batch_first=True)  # [batch, max_n, d_model]

        # Compute edge biases
        edge_bias = None
        if self.use_edge_bias and self.edge_bias_encoder is not None:
            edge_biases, _ = self.edge_bias_encoder.forward_batch(mols, max_atoms=max_atoms)

            # Handle CLS token: prepend row/col of zeros
            if self.use_cls_token:
                # edge_biases: [batch, max_n, max_n, n_heads]
                # Need to expand to [batch, max_n+1, max_n+1, n_heads]
                n_heads = edge_biases.shape[-1]
                expanded = torch.zeros(batch_size, max_atoms + 1, max_atoms + 1, n_heads, device=device)
                expanded[:, 1:, 1:, :] = edge_biases
                edge_bias = expanded

        # Create padding mask (True for padding positions)
        key_padding_mask = torch.zeros(batch_size, max_atoms, dtype=torch.bool, device=device)
        for i, length in enumerate(lengths):
            key_padding_mask[i, length:] = True

        # Add CLS token
        if self.use_cls_token:
            padded = self.pooling(padded)
            # Extend padding mask for CLS token (never masked)
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
            key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)

        # Encode
        encoded = self.model(padded, edge_bias=edge_bias, key_padding_mask=key_padding_mask)

        return encoded[:, 0] if self.use_cls_token else encoded.mean(dim=1)

    def embed(self, mol):
        return self.featurizer(mol).unsqueeze(0)

    def encode_from_emb(self, atom_emb, edge_bias=None):
        """Encode from pre-computed embeddings (for interpretability)."""
        if self.use_cls_token:
            atom_emb = self.pooling(atom_emb)
            # If edge_bias provided, expand for CLS token
            if edge_bias is not None:
                batch_size = atom_emb.shape[0]
                seq_len = atom_emb.shape[1]
                n_heads = edge_bias.shape[-1]
                expanded = torch.zeros(batch_size, seq_len, seq_len, n_heads, device=atom_emb.device)
                expanded[:, 1:, 1:, :] = edge_bias
                edge_bias = expanded
        return self.model(atom_emb, edge_bias=edge_bias)

def scaled_dot_product_attention_with_bias(Q, K, V, edge_bias=None, key_padding_mask=None, dropout_p=0.0):
    """
    Scaled dot-product attention with edge bias.

    Args:
        Q: [batch, n_heads, seq_len, d_k]
        K: [batch, n_heads, seq_len, d_k]
        V: [batch, n_heads, seq_len, d_k]
        edge_bias: [batch, seq_len, seq_len, n_heads] bias to add to attention scores
        key_padding_mask: [batch, seq_len] True for positions to mask (padding)
        dropout_p: Dropout probability

    Returns:
        attn_output: [batch, n_heads, seq_len, d_k]
    """
    d_k = Q.shape[-1]
    orig_dtype = Q.dtype

    # Use contiguous FP32 tensors for attention matmuls to avoid intermittent CUBLAS
    # invalid-value failures with strided batched GEMM on some GPU/kernel combinations.
    Q = Q.float().contiguous()
    K = K.float().contiguous()
    V = V.float().contiguous()

    # Compute attention scores: [batch, n_heads, seq, seq]
    scores = torch.matmul(Q, K.transpose(-2, -1).contiguous()) / math.sqrt(d_k)

    # Add edge bias BEFORE softmax
    if edge_bias is not None:
        # edge_bias: [batch, seq, seq, n_heads] -> [batch, n_heads, seq, seq]
        bias = edge_bias.permute(0, 3, 1, 2)
        scores = scores + bias

    # Apply key padding mask
    if key_padding_mask is not None:
        # key_padding_mask: [batch, seq] -> [batch, 1, 1, seq]
        mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

    # Softmax
    attn_weights = F.softmax(scores, dim=-1)

    # Dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # Apply to values
    attn_output = torch.matmul(attn_weights, V)

    return attn_output.to(orig_dtype)


class GraphTransformerBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, dim_ff=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.attn_dropout = dropout

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, edge_bias=None, key_padding_mask=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            edge_bias: [batch, seq_len, seq_len, n_heads] (optional)
            key_padding_mask: [batch, seq_len] True for padding positions (optional)
        """
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Attention with edge bias
        dropout_p = self.attn_dropout if self.training else 0.0
        attn_output = scaled_dot_product_attention_with_bias(
            Q, K, V, edge_bias=edge_bias, key_padding_mask=key_padding_mask, dropout_p=dropout_p
        )

        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.W_o(attn_output)

        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x

class GraphTransformerEncoder(nn.Module):
    def __init__(self, n_layers=4, d_model=128, n_heads=4, dim_ff=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.layers = nn.ModuleList([
            GraphTransformerBlock(d_model, n_heads, dim_ff, dropout) for _ in range(n_layers)
        ])

    def forward(self, x, edge_bias=None, key_padding_mask=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            edge_bias: [batch, seq_len, seq_len, n_heads] (optional)
            key_padding_mask: [batch, seq_len] True for padding positions (optional)
        """
        for layer in self.layers:
            x = layer(x, edge_bias=edge_bias, key_padding_mask=key_padding_mask)
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
