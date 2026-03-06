import torch
from rdkit import Chem
from captum.attr import IntegratedGradients
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from atom_embedding import AtomFeaturizer
from model import GraphTransformerEncoder, GraphEncoder, MLPHead
from IPython.display import Image
import io
import numpy as np

# Initialize model components (example, should match your trained model config)
def load_model(config):
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

    return encoder.eval(), head.eval()

# Forward function for captum
class ForwardWrapper(torch.nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, atom_emb):
        encoded = self.encoder.encode_from_emb(atom_emb)
        pooled = encoded[:, 0] if self.encoder.use_cls_token else encoded.mean(dim=1)
        return self.head(pooled)

# Attribution + visualization

def calculate_aspect_ratio(molecule, base_size):
    conf = molecule.GetConformer()
    atom_positions = [conf.GetAtomPosition(i) for i in range(molecule.GetNumAtoms())]
    x_coords = [pos.x for pos in atom_positions]
    y_coords = [pos.y for pos in atom_positions]
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)
    aspect_ratio = width / height if height > 0 else 1

    canvas_width = max(base_size, int(base_size * aspect_ratio)) if aspect_ratio > 1 else base_size
    canvas_height = max(base_size, int(base_size / aspect_ratio)) if aspect_ratio < 1 else base_size

    return canvas_width, canvas_height

def interpret_smiles(smiles, model, target=0, bw=True, padding=0.05):
    mol = Chem.MolFromSmiles(smiles)
    Chem.rdDepictor.Compute2DCoords(mol)

    atom_emb = model.encoder.embed(mol).requires_grad_()

    wrapper = ForwardWrapper(model.encoder, model.head)
    wrapper.eval()

    ig = IntegratedGradients(wrapper)

    # Choose baseline
    baseline = torch.zeros_like(atom_emb)  # Zero baseline (default)
    # baseline = torch.randn_like(atom_emb)  # Random noise baseline
    # baseline = atom_emb * 0.5              # Scaled halfway baseline

    # Compute attributions
    attributions, delta = ig.attribute(
        atom_emb, baseline, target=target, return_convergence_delta=True
    )

    # Aggregate attributions per atom (signed!)
    scores = attributions.squeeze(0).sum(dim=1).detach().numpy()

    # ⬇️ Attribution summary logging
    print(f"Attributions for SMILES: {smiles}")
    print(f"  ➤ Min attribution: {scores.min():.4f}")
    print(f"  ➤ Max attribution: {scores.max():.4f}")
    print(f"  ➤ Mean attribution: {scores.mean():.4f}")
    print(f"  ➤ Attribution convergence delta: {delta.item():.6f}")

    # Visualization
    base_size = 400
    width, height = calculate_aspect_ratio(mol, base_size)
    drawer = Draw.MolDraw2DCairo(width, height)
    drawer.drawOptions().padding = padding
    if bw:
        drawer.drawOptions().useBWAtomPalette()

    SimilarityMaps.GetSimilarityMapFromWeights(mol, scores.tolist(), draw2d=drawer)
    drawer.FinishDrawing()
    return Image(data=drawer.GetDrawingText()), scores

