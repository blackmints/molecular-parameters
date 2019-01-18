import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from GCN.trainer import Trainer

if __name__ == "__main__":
    trainer = Trainer("delaney")

    trainer.fit("GCN", 150, batch=8, fold=10, normalize=True, units_conv=50, units_dense=128, num_layers=2,
                loss="mse", monitor="val_rmse", label="total",
                use_atom_symbol=False, use_atom_symbol_extended=True,
                use_atom_number=True, use_degree=True, use_hybridization=True, use_implicit_valence=True,
                use_partial_charge=True, use_formal_charge=True, use_ring_size=True, use_hydrogen_bonding=True,
                use_acid_base=True, use_aromaticity=True, use_chirality=True, use_num_hydrogen=True)
