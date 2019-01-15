import h5py
from keras.utils import to_categorical
from rdkit.Chem import rdmolops, rdchem
from GCN.features import *


class Dataset(object):
    def __init__(self, dataset, batch=128, normalize=False):
        self.dataset = dataset
        self.data_format = None
        self.batch = batch
        self.smiles = []
        self.coords = []
        self.target = []
        self.x, self.c, self.y = {}, {}, {}

        # Load data
        if dataset == "delaney":
            self.load_file("../../data/delaney/delaney.sdf")
            self.task = "regression"
        elif dataset == "bace_cla":
            self.load_file("../../data/bace/bace.sdf", target_name="Class")
            self.task = "binary"
        elif dataset == "bace_reg":
            self.load_file("../../data/bace/bace.sdf", target_name="pIC50")
            self.task = "regression"
        elif dataset == "freesolv":
            self.load_file("../../data/freesolv/freesolv.sdf", target_name="exp")
            self.task = "regression"
        elif dataset == "tox21_er":  # 7670 out of 7697
            self.load_file("../data/tox21/tox21_NR_ER.sdf", target_name="Active")
            self.task = "binary"
        elif "dude" in dataset:
            target = dataset.split("_")[-1]
            if "gpcr" in dataset:
                self.load_file("../../data/dude/gpcr/{}.sdf".format(target), target_name="active")
            elif "kinase" in dataset:
                self.load_file("../../data/dude/kinase/{}.sdf".format(target), target_name="active")
            elif "nuclear" in dataset:
                self.load_file("../../data/dude/nuclear/{}.sdf".format(target), target_name="active")
            elif "protease" in dataset:
                self.load_file("../../data/dude/protease/{}.sdf".format(target), target_name="active")
            self.task = "binary"
        else:
            assert dataset is not None, 'Unsupported dataset: {}'.format(dataset)

        # Get dataset parameters
        self.num_atoms, self.num_atom_features = self.get_parameters()
        self.num_degrees = 5

        if self.task == "regression" and normalize:
            self.mean = np.mean(self.y["train"])
            self.std = np.std(self.y["train"])

            self.y["train"] = (self.y["train"] - self.mean) / self.std
            self.y["valid"] = (self.y["valid"] - self.mean) / self.std
            self.y["test"] = (self.y["test"] - self.mean) / self.std
        else:
            self.mean = 0
            self.std = 1

        self.step = max(int(len(self.x["train"]) / self.batch), 1)
        self.val_step = max(int(len(self.x["valid"]) / self.batch), 1)

    def load_file(self, path, max_atoms=0, max_number=0, target_name="target", include_H=False):
        x, y = [], []

        if "sdf" in path:
            self.data_format = "mol"
            mols = Chem.SDMolSupplier(path)

            for mol in mols:
                if mol is not None and target_name in mol.GetPropNames():
                    if include_H:
                        mol = Chem.AddHs(mol)

                    x.append(mol)
                    y.append(float(mol.GetProp(target_name)))

        elif "h5" in path:
            self.data_format = "smiles"

            file = h5py.File(path, "r")
            x = file['smiles']
            y = file[target_name]

        else:
            raise ValueError("Unavailable input file type while loading dataset.")

        # Filter with maximum number of atoms
        new_smiles, new_coords, new_target = [], [], []
        if max_atoms > 0:
            for mol, tar in zip(x, y):
                num_atoms = mol.GetNumAtoms()

                if num_atoms <= max_atoms:
                    new_smiles.append(mol)
                    new_target.append(tar)

            x = new_smiles
            y = new_target

        self.smiles, self.target = np.array(x), np.array(y)

        # Cut with maximum number of data
        if max_number > 0:
            self.smiles, self.target = self.smiles[:max_number], self.target[:max_number]

        # Shuffle data
        idx = np.random.permutation(len(self.smiles))
        self.smiles, self.target = self.smiles[idx], self.target[idx]

        # Split data into train, validation, test set
        spl1 = int(len(self.smiles) * 0.2)
        spl2 = int(len(self.smiles) * 0.1)

        self.x = {"train": self.smiles[spl1:],
                  "valid": self.smiles[spl2:spl1],
                  "test": self.smiles[:spl2]}
        self.y = {"train": self.target[spl1:],
                  "valid": self.target[spl2:spl1],
                  "test": self.target[:spl2]}

    def save_file(self, pred, path, target="test"):
        mols = []
        for idx, (x, y, p) in enumerate(zip(self.x[target], self.y[target], pred)):
            if self.data_format == "smiles":
                mol = Chem.MolFromSmiles(x)
            else:
                mol = x

            mol.SetProp("true", str(y * self.std + self.mean))
            mol.SetProp("pred", str(p[0] * self.std + self.mean))
            mols.append(mol)

        w = Chem.SDWriter(path + target + "_results.sdf")
        for mol in mols:
            if mol is not None:
                w.write(mol)

    def tensorize(self, target="train", shuffle=False, task="regression"):
        x, y = self.x[target], self.y[target]

        idx_pointer = 0

        if shuffle:
            idx_range = np.random.permutation(len(x))
        else:
            idx_range = np.arange(len(x))

        while True:
            batch_x = []
            batch_y = []

            if idx_pointer + self.batch > len(idx_range):
                idx_pointer = 0
            idx = idx_range[idx_pointer:idx_pointer + self.batch]
            idx_pointer += self.batch

            for i in idx:
                if type(x[i]) is rdchem.Mol:
                    mol = x[i]
                else:
                    mol = Chem.MolFromSmiles(x[i])
                batch_x.append(mol)
                batch_y.append(y[i])

            atom_tensor = np.zeros((self.batch, self.num_atoms, num_atom_features()))
            adjm_tensor = np.zeros((self.batch, self.num_atoms, self.num_atoms))

            for mol_idx, mol in enumerate(batch_x):
                atoms = mol.GetAtoms()

                for atom in mol.GetAtoms():
                    atom_idx = atom.GetIdx()
                    atom_tensor[mol_idx, atom_idx, :] = atom_features(atom)

                # Adjacency matrix
                adjms = np.array(rdmolops.GetAdjacencyMatrix(mol), dtype="float")

                # Normalize adjacency matrix by D^(-1/2) * A_hat * D^(-1/2), Kipf et al.
                adjms += np.eye(len(atoms))
                degree = np.array(adjms.sum(1))
                deg_inv_sqrt = np.power(degree, -0.5)
                deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.
                deg_inv_sqrt = np.diag(deg_inv_sqrt)

                adjms = np.matmul(np.matmul(deg_inv_sqrt, adjms), deg_inv_sqrt)

                adjm_tensor[mol_idx, : len(atoms), : len(atoms)] = adjms

            if task == "category":
                yield [atom_tensor, adjm_tensor], to_categorical(batch_y)
            elif task == "binary":
                yield [atom_tensor, adjm_tensor], np.array(batch_y)
            elif task == "regression":
                yield [atom_tensor, adjm_tensor], np.array(batch_y)
            elif task == "input_only":
                yield [atom_tensor, adjm_tensor]

    def get_parameters(self, include_H=False):
        n_atom_features = num_atom_features()
        n_atoms = 0

        for mol_idx, mol in enumerate(self.smiles):
            if type(mol) is not rdchem.Mol:
                mol = Chem.MolFromSmiles(mol)

            if mol is not None:
                if include_H:
                    mol = Chem.AddHs(mol)
                n_atoms = max(n_atoms, mol.GetNumAtoms())

        return n_atoms, n_atom_features
