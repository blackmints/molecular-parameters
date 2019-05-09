import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from GCN.trainer import Trainer

if __name__ == "__main__":
    trainer = Trainer("freesolv_no_common")

    hyperparameters = {"epoch": 150, "batch": 8, "fold": 30, "normalize": True, "units_conv": 128, "units_dense": 128,
                       "num_layers": 2, "loss": "mse", "monitor": "val_rmse", "label": "best_fs",
                       "use_multiprocessing": False}

    features = {"use_atom_symbol": True, "use_atom_symbol_extended": False,
                "use_atom_number": False, "use_degree": True, "use_hybridization": True, "use_implicit_valence": True,
                "use_partial_charge": True, "use_formal_charge": True, "use_ring_size": True,
                "use_hydrogen_bonding": True,
                "use_acid_base": False, "use_aromaticity": True, "use_chirality": False, "use_num_hydrogen": True}

    trainer.fit("GCN", **hyperparameters, **features)

    hyperparameters = {"epoch": 150, "batch": 8, "fold": 30, "normalize": True, "units_conv": 128, "units_dense": 128,
                       "num_layers": 2, "loss": "mse", "monitor": "val_rmse", "label": "total",
                       "use_multiprocessing": False}

    features = {"use_atom_symbol": True, "use_atom_symbol_extended": False,
                "use_atom_number": True, "use_degree": True, "use_hybridization": True, "use_implicit_valence": True,
                "use_partial_charge": True, "use_formal_charge": True, "use_ring_size": True,
                "use_hydrogen_bonding": True,
                "use_acid_base": True, "use_aromaticity": True, "use_chirality": True, "use_num_hydrogen": True}

    trainer.fit("GCN", **hyperparameters, **features)