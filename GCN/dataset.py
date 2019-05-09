from keras.utils import Sequence, to_categorical
from rdkit import Chem
from rdkit.Chem import rdmolops, AllChem
import numpy as np


def one_hot(x, allowable_set):
    # If x is not in allowed set, use last index
    if x not in allowable_set:
        x = allowable_set[-1]

    return list(map(lambda s: x == s, allowable_set))


class Dataset(object):
    def __init__(self, dataset, batch=128, normalize=False, use_atom_symbol=True, use_atom_symbol_extended=False,
                 use_atom_number=False, use_degree=False, use_hybridization=False, use_implicit_valence=False,
                 use_partial_charge=False, use_formal_charge=False, use_ring_size=False, use_hydrogen_bonding=False,
                 use_acid_base=False, use_aromaticity=False, use_chirality=False, use_num_hydrogen=False):

        self.dataset = dataset
        self.path = "../data/{}.sdf".format(dataset)
        self.task = "binary"
        self.target_name = "active"
        self.max_atoms = 0
        self.num_features = 0
        self.batch = batch
        self.normalize = normalize
        self.outputs = 1
        self.smiles = []
        self.target = []
        self.x, self.y = {}, {}

        self.use_atom_symbol = use_atom_symbol
        self.use_atom_symbol_extended = use_atom_symbol_extended
        self.use_atom_number = use_atom_number  # MPNN
        self.use_degree = use_degree  # NFP
        self.use_hybridization = use_hybridization  # Weave   MPNN
        self.use_implicit_valence = use_implicit_valence  # NFP
        self.use_partial_charge = use_partial_charge  # Weave
        self.use_formal_charge = use_formal_charge  # Weave
        self.use_ring_size = use_ring_size  # Weave
        self.use_hydrogen_bonding = use_hydrogen_bonding  # Weave
        self.use_acid_base = use_acid_base
        self.use_aromaticity = use_aromaticity  # NFP   Weave   MPNN
        self.use_chirality = use_chirality  # Weave
        self.use_num_hydrogen = use_num_hydrogen  # NFP           MPNN
        # self.use_electron_donor_acceptor = use_electron_donor_acceptor  # MPNN

        # Load data
        self.load_dataset()

        # Calculate number of features
        mp = MPGenerator([], [], 1,
                         use_atom_symbol=self.use_atom_symbol,
                         use_atom_symbol_extended=self.use_atom_symbol_extended,
                         use_atom_number=self.use_atom_number,
                         use_degree=self.use_degree,
                         use_hybridization=self.use_hybridization,
                         use_implicit_valence=self.use_implicit_valence,
                         use_partial_charge=self.use_partial_charge,
                         use_formal_charge=self.use_formal_charge,
                         use_ring_size=self.use_ring_size,
                         use_hydrogen_bonding=self.use_hydrogen_bonding,
                         use_acid_base=self.use_acid_base,
                         use_aromaticity=self.use_aromaticity,
                         use_chirality=self.use_chirality,
                         use_num_hydrogen=self.use_num_hydrogen)
        self.num_features = mp.get_num_features()

        # Normalize
        if self.task == "regression" and normalize:
            self.mean = np.mean(self.y["train"])
            self.std = np.std(self.y["train"])

            self.y["train"] = (self.y["train"] - self.mean) / self.std
            self.y["valid"] = (self.y["valid"] - self.mean) / self.std
            self.y["test"] = (self.y["test"] - self.mean) / self.std
        else:
            self.mean = 0
            self.std = 1

    def load_dataset(self):
        # Dataset parameters
        if self.dataset == "bace_reg" or "delaney" in self.dataset or "freesolv" in self.dataset:
            self.task = "regression"
            self.target_name = "target"

        elif self.dataset == "hiv":
            # Original max_atom = 222; sliced by num_atom < 130 >= 99.95 %; num_atom < 100 >= 99.82 %
            self.max_atoms = 130

        elif self.dataset == "tox21":
            self.target_name = ["NR-Aromatase", "NR-AR", "NR-AR-LBD", "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma", "NR-AhR",
                           "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]

        else:
            pass

        # Load file
        x, y = [], []
        mols = Chem.SDMolSupplier(self.path)

        for mol in mols:
            if mol is not None:
                if type(self.target_name) is list:
                    y.append([float(mol.GetProp(t)) if t in mol.GetPropNames() else -1 for t in self.target_name])
                    self.outputs = len(self.target_name)

                elif self.target_name in mol.GetPropNames():
                    _y = float(mol.GetProp(self.target_name))
                    if _y == -1:
                        continue
                    else:
                        y.append(_y)

                else:
                    continue

                x.append(mol)
        assert len(x) == len(y)

        # Filter and update maximum number of atoms
        new_smiles, new_target = [], []
        if self.max_atoms > 0:
            for mol, tar in zip(x, y):
                if mol.GetNumAtoms() <= self.max_atoms:
                    new_smiles.append(mol)
                    new_target.append(tar)

            x = new_smiles
            y = new_target

        else:
            for mol, tar in zip(x, y):
                self.max_atoms = max(self.max_atoms, mol.GetNumAtoms())

        if self.task != "regression":
            self.smiles, self.target = np.array(x), np.array(y, dtype=int)
        else:
            self.smiles, self.target = np.array(x), np.array(y)

        # Shuffle data
        idx = np.random.permutation(len(self.smiles))
        self.smiles, self.target = self.smiles[idx], self.target[idx]

        # Split data
        spl1 = int(len(self.smiles) * 0.2)
        spl2 = int(len(self.smiles) * 0.1)

        self.x = {"train": self.smiles[spl1:],
                  "valid": self.smiles[spl2:spl1],
                  "test": self.smiles[:spl2]}
        self.y = {"train": self.target[spl1:],
                  "valid": self.target[spl2:spl1],
                  "test": self.target[:spl2]}

    def save_dataset(self, pred, path, target="test", target_path=None):
        mols = []
        for idx, (mol, y, p) in enumerate(zip(self.x[target], self.y[target], pred)):
            mol.SetProp("true", str(y * self.std + self.mean))
            mol.SetProp("pred", str(p[0] * self.std + self.mean))
            mols.append(mol)

        w = Chem.SDWriter(path + target + "_results.sdf" if target_path is None else target_path)
        for mol in mols:
            if mol is not None:
                w.write(mol)

    def set_features(self, use_atom_symbol=True, use_atom_symbol_extended=False,
                     use_atom_number=False, use_degree=False, use_hybridization=False, use_implicit_valence=False,
                     use_partial_charge=False, use_formal_charge=False, use_ring_size=False, use_hydrogen_bonding=False,
                     use_acid_base=False, use_aromaticity=False, use_chirality=False, use_num_hydrogen=False):

        self.use_atom_symbol = use_atom_symbol
        self.use_atom_symbol_extended = use_atom_symbol_extended
        self.use_atom_number = use_atom_number  # MPNN
        self.use_degree = use_degree  # NFP
        self.use_hybridization = use_hybridization  # Weave   MPNN
        self.use_implicit_valence = use_implicit_valence  # NFP
        self.use_partial_charge = use_partial_charge  # Weave
        self.use_formal_charge = use_formal_charge  # Weave
        self.use_ring_size = use_ring_size  # Weave
        self.use_hydrogen_bonding = use_hydrogen_bonding  # Weave
        self.use_acid_base = use_acid_base
        self.use_aromaticity = use_aromaticity  # NFP   Weave   MPNN
        self.use_chirality = use_chirality  # Weave
        self.use_num_hydrogen = use_num_hydrogen  # NFP           MPNN
        # self.use_electron_donor_acceptor = use_electron_donor_acceptor  # MPNN

        # Calculate number of features
        mp = MPGenerator([], [], 1,
                         use_atom_symbol=self.use_atom_symbol,
                         use_atom_symbol_extended=self.use_atom_symbol_extended,
                         use_atom_number=self.use_atom_number,
                         use_degree=self.use_degree,
                         use_hybridization=self.use_hybridization,
                         use_implicit_valence=self.use_implicit_valence,
                         use_partial_charge=self.use_partial_charge,
                         use_formal_charge=self.use_formal_charge,
                         use_ring_size=self.use_ring_size,
                         use_hydrogen_bonding=self.use_hydrogen_bonding,
                         use_acid_base=self.use_acid_base,
                         use_aromaticity=self.use_aromaticity,
                         use_chirality=self.use_chirality,
                         use_num_hydrogen=self.use_num_hydrogen)
        self.num_features = mp.get_num_features()

    def generator(self, target, task=None):
        return MPGenerator(self.x[target], self.y[target], self.batch,
                           task=task if task is not None else self.task,
                           num_atoms=self.max_atoms,
                           use_atom_symbol=self.use_atom_symbol,
                           use_atom_symbol_extended=self.use_atom_symbol_extended,
                           use_atom_number=self.use_atom_number,
                           use_degree=self.use_degree,
                           use_hybridization=self.use_hybridization,
                           use_implicit_valence=self.use_implicit_valence,
                           use_partial_charge=self.use_partial_charge,
                           use_formal_charge=self.use_formal_charge,
                           use_ring_size=self.use_ring_size,
                           use_hydrogen_bonding=self.use_hydrogen_bonding,
                           use_acid_base=self.use_acid_base,
                           use_aromaticity=self.use_aromaticity,
                           use_chirality=self.use_chirality,
                           use_num_hydrogen=self.use_num_hydrogen)


class MPGenerator(Sequence):
    def __init__(self, x_set, y_set, batch, task="binary", num_atoms=0, use_atom_symbol=True,
                 use_atom_symbol_extended=False, use_atom_number=False, use_degree=False, use_hybridization=False,
                 use_implicit_valence=False, use_partial_charge=False, use_formal_charge=False, use_ring_size=False,
                 use_hydrogen_bonding=False, use_acid_base=False, use_aromaticity=False, use_chirality=False,
                 use_num_hydrogen=False):

        self.x, self.y = x_set, y_set

        self.batch = batch
        self.task = task
        self.num_atoms = num_atoms

        self.use_atom_symbol = use_atom_symbol
        self.use_atom_symbol_extended = use_atom_symbol_extended
        self.use_atom_number = use_atom_number  # MPNN
        self.use_degree = use_degree  # NFP
        self.use_hybridization = use_hybridization  # Weave   MPNN
        self.use_implicit_valence = use_implicit_valence  # NFP
        self.use_partial_charge = use_partial_charge  # Weave
        self.use_formal_charge = use_formal_charge  # Weave
        self.use_ring_size = use_ring_size  # Weave
        self.use_hydrogen_bonding = use_hydrogen_bonding  # Weave
        self.use_acid_base = use_acid_base
        self.use_aromaticity = use_aromaticity  # NFP   Weave   MPNN
        self.use_chirality = use_chirality  # Weave
        self.use_num_hydrogen = use_num_hydrogen  # NFP           MPNN
        # self.use_electron_donor_acceptor = use_electron_donor_acceptor  # MPNN

        self.hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
        self.hydrogen_acceptor = Chem.MolFromSmarts(
            "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")
        self.acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
        self.basic = Chem.MolFromSmarts(
            "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]")

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch:(idx + 1) * self.batch]
        batch_y = self.y[idx * self.batch:(idx + 1) * self.batch]

        if self.task == "category":
            return self._tensorize(batch_x), to_categorical(batch_y)
        elif self.task == "binary":
            return self._tensorize(batch_x), np.array(batch_y, dtype=int)
        elif self.task == "regression":
            return self._tensorize(batch_x), np.array(batch_y, dtype=float)
        elif self.task == "input_only":
            return self._tensorize(batch_x)

    def get_num_features(self):
        mol = Chem.MolFromSmiles("CC")
        return len(self.get_atom_features(mol)[0])

    def get_atom_features(self, mol):
        AllChem.ComputeGasteigerCharges(mol)
        Chem.AssignStereochemistry(mol)

        hydrogen_donor_match = sum(mol.GetSubstructMatches(self.hydrogen_donor), ())
        hydrogen_acceptor_match = sum(mol.GetSubstructMatches(self.hydrogen_acceptor), ())
        acidic_match = sum(mol.GetSubstructMatches(self.acidic), ())
        basic_match = sum(mol.GetSubstructMatches(self.basic), ())

        ring = mol.GetRingInfo()

        m = []
        for atom_idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(atom_idx)

            o = []
            o += one_hot(atom.GetSymbol(), ['C', 'N', 'O', 'F', 'H', 'S', 'P', 'Cl', 'Br', 'I', 'B', 'Unknown']) \
                if self.use_atom_symbol else []
            o += one_hot(atom.GetSymbol(),
                         ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I',
                          'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li',
                          'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) \
                if self.use_atom_symbol_extended else []
            o += [atom.GetAtomicNum()] if self.use_atom_number else []
            o += one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) if self.use_degree else []
            o += one_hot(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                                   Chem.rdchem.HybridizationType.SP2,
                                                   Chem.rdchem.HybridizationType.SP3,
                                                   Chem.rdchem.HybridizationType.SP3D,
                                                   Chem.rdchem.HybridizationType.SP3D2]) if self.use_hybridization else []
            o += one_hot(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) if self.use_implicit_valence else []
            o += one_hot(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3]) if self.use_degree else []
            o += [atom.GetProp("_GasteigerCharge")] if self.use_partial_charge else []
            o += [atom.GetIsAromatic()] if self.use_aromaticity else []
            o += [ring.IsAtomInRingOfSize(atom_idx, 3),
                  ring.IsAtomInRingOfSize(atom_idx, 4),
                  ring.IsAtomInRingOfSize(atom_idx, 5),
                  ring.IsAtomInRingOfSize(atom_idx, 6),
                  ring.IsAtomInRingOfSize(atom_idx, 7),
                  ring.IsAtomInRingOfSize(atom_idx, 8)] if self.use_ring_size else []
            o += one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) if self.use_num_hydrogen else []

            if self.use_chirality:
                try:
                    o += one_hot(atom.GetProp('_CIPCode'), ["R", "S"]) + [atom.HasProp("_ChiralityPossible")]
                except:
                    o += [False, False] + [atom.HasProp("_ChiralityPossible")]
            if self.use_hydrogen_bonding:
                o += [atom_idx in hydrogen_donor_match]
                o += [atom_idx in hydrogen_acceptor_match]
            if self.use_acid_base:
                o += [atom_idx in acidic_match]
                o += [atom_idx in basic_match]

            m.append(o)

        return np.array(m, dtype=float)

    def _tensorize(self, batch_x):
        atom_tensor = np.zeros((len(batch_x), self.num_atoms, self.get_num_features()))
        adjm_tensor = np.zeros((len(batch_x), self.num_atoms, self.num_atoms))

        for mol_idx, mol in enumerate(batch_x):
            mol_atoms = mol.GetNumAtoms()

            # Atom features
            atom_tensor[mol_idx, :mol_atoms, :] = self.get_atom_features(mol)

            # Adjacency matrix
            adjms = np.array(rdmolops.GetAdjacencyMatrix(mol), dtype="float")

            # Normalize adjacency matrix by D^(-1/2) * A_hat * D^(-1/2), Kipf et al. 2016
            adjms += np.eye(mol_atoms)
            degree = np.array(adjms.sum(1))
            deg_inv_sqrt = np.power(degree, -0.5)
            deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.
            deg_inv_sqrt = np.diag(deg_inv_sqrt)

            adjms = np.matmul(np.matmul(deg_inv_sqrt, adjms), deg_inv_sqrt)

            adjm_tensor[mol_idx, : mol_atoms, : mol_atoms] = adjms

        return [atom_tensor, adjm_tensor]
