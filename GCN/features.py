import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from keras.utils import to_categorical


def one_hot(x, allowable_set):
    # If x is not in allowed set, use last index
    if x not in allowable_set:
        x = allowable_set[-1]

    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return np.array(one_hot(atom.GetSymbol(), ['C', 'N', 'O', 'F', 'H', 'S', 'P', 'Cl', 'Br', 'I', 'B', 'Unknown']) +
                    one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) +  # Does not include Hs
                    one_hot(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                                      Chem.rdchem.HybridizationType.SP2,
                                                      Chem.rdchem.HybridizationType.SP3,
                                                      Chem.rdchem.HybridizationType.SP3D,
                                                      Chem.rdchem.HybridizationType.SP3D2]) +
                    one_hot(atom.GetImplicitValence(), [0, 1, 2, 3, 4]) +  # Number of implicit Hs
                    [atom.GetIsAromatic()], dtype=int)


def bond_features(bond):
    """Generate array of bond features from given bond."""
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE,
                     bt == Chem.rdchem.BondType.DOUBLE,
                     bt == Chem.rdchem.BondType.TRIPLE,
                     bt == Chem.rdchem.BondType.AROMATIC,
                     bond.GetIsConjugated(),
                     bond.IsInRing()], dtype=int)


def pair_features(mol, idx1, idx2, max_distance=7):
    features = np.zeros((6 + max_distance + 1))

    # bond type
    bond = mol.GetBondBetweenAtoms(idx1, idx2)
    if bond is not None:
        bt = bond.GetBondType()
        features[:6] = np.array([bt == Chem.rdchem.BondType.SINGLE,
                                 bt == Chem.rdchem.BondType.DOUBLE,
                                 bt == Chem.rdchem.BondType.TRIPLE,
                                 bt == Chem.rdchem.BondType.AROMATIC,
                                 bond.GetIsConjugated(),
                                 bond.IsInRing()], dtype=int)

    # whether two atoms are in same ring
    rings = mol.GetRingInfo().AtomRings()
    for ring in rings:
        if idx1 in ring and idx2 in ring and idx1 != idx2:
            features[6] = 1

    # graph distance between two atoms
    distance = rdmolops.GetDistanceMatrix(mol)
    distance = np.where(distance < max_distance, distance, max_distance - 1)[idx1][idx2]
    distance = to_categorical(distance, num_classes=max_distance)
    features[7:] = distance

    return features


def num_atom_features():
    molecule = Chem.MolFromSmiles('CC')
    atom_list = molecule.GetAtoms()

    return len(atom_features(atom_list[0]))


def num_bond_features():
    molecule = Chem.MolFromSmiles('CC')
    # SanitizeMol() checks aromatic bonds and mark BondType.Aromatic
    Chem.SanitizeMol(molecule)
    bond_list = molecule.GetBonds()

    return len(bond_features(bond_list[0]))


def num_pair_features():
    molecule = Chem.MolFromSmiles('CC')
    # SanitizeMol() checks aromatic bonds and mark BondType.Aromatic
    Chem.SanitizeMol(molecule)

    return len(pair_features(molecule, 0, 1))
