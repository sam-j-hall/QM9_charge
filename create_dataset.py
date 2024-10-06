import random
from typing import List, Union
import pickle as pkl
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx
from rdkit import Chem

# --- Class variables
ATOM_FEATURES = {
    'atomic_num': [6.0, 7.0, 8.0, 9.0],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3
    ],
    'aromaticity': [True, False]
}

BOND_FEATURES = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ]
}

# --- Total number of atom features
ATOM_FDIM = sum(len(feature) for feature in ATOM_FEATURES.values()) + 1
# --- Number of bond features
BOND_FDIM = sum(len(feature) for feature in BOND_FEATURES.values())

def one_hot_encoding(value:int, features:List[int]) -> List[int]:
    '''
    Creates a one-hot encoding vector

    :params value: The 
    :params features: The dictionary 
    :return: A one-hot encoding list of chosen feature
    '''

    # ---Create a zero feature vector
    encoding = [0.0] * len(features)
    # --- Find the index value
    index = features.index(value)
    # --- Set value to 1
    encoding[index] = 1.0

    return encoding


def get_atom_features(atom) -> List[Union[bool, int, float]]:
    '''
    Builds a feature vector for an atom

    :param atom: An RDKit atom
    :return: A list containing the atom features
    '''

    if atom is None:
        atom_feat = [0] * ATOM_FDIM
    else:
        atom_feat = one_hot_encoding(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']) + \
            one_hot_encoding(atom.GetHybridization(), ATOM_FEATURES['hybridization']) + \
            [1.0 if atom.GetIsAromatic() else 0.0] + \
            [atom.GetTotalNumHs()] + \
            [atom.GetFormalCharge()]
     
    return atom_feat

def get_bond_features(bond) -> List[Union[bool, int, float]]:
    '''
    Builds a features vector for a bond

    :param bond: An RDKit bond
    :return: A list containing the bond features
    '''

    if bond is None:
        bond_feat = [0] * (BOND_FDIM)
    else:
        bond_feat = one_hot_encoding(bond.GetBondType(), BOND_FEATURES['bond_type'])

    return bond_feat
        

def mol_to_nx(mol, spec):
    '''
    Create a networkx graph of a molecule

    :params mol: An RDKit molecule
    :params spec: The X-ray absorption spectra
    :return: A networkx graph object
    '''

    # --- Create graph object
    G = nx.Graph()

    # --- For each atom
    for atom in mol.GetAtoms():
        # --- Add a node to graph and create one-hot encoding vector
        G.add_node(atom.GetIdx(), x=get_atom_features(atom))

    # --- For each bond
    for bond in mol.GetBonds():
        # --- Get start and end index
        begin = (bond.GetBeginAtom()).GetIdx()
        end = (bond.GetEndAtom()).GetIdx()
        # --- Add edge for bond with one-hot encoding vector
        G.add_edge(begin, end, edge_attr=get_bond_features(bond))

    # --- Normalize spectra
    max_int = np.max(spec)
    norm_spec = 1.0 * (spec / max_int)
    new = np.float32(norm_spec)
    # --- Set spectra to graph
    G.graph['spectrum'] = new

    return G

class XASDataset(InMemoryDataset):
    '''
    '''

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['qm9_xas_clean.pkl']
    
    @property
    def processed_file_names(self):
        return ['qm9_xas.pt']
    
    def process(self):
        '''
        '''

        # --- Load the data from pickle file
        with open(self.raw_paths[0], 'rb') as data:
            df = pkl.load(data)
        
        print(f'Total number of molecules: {len(df)}')

        # --- Empty data list and count index
        data_list = []
        idx = 0

        # --- Iterate through dataframe
        for index, row in df.iterrows():
            mol = Chem.MolFromSmiles(row['SMILES'])
            spec = row['Spectra']

            gx = mol_to_nx(mol, spec)
            pyg_graph = from_networkx(gx)
            pyg_graph.idx = idx
            pyg_graph.smiles = row['SMILES']

            data_list.append(pyg_graph)
            idx += 1

        random.Random(258).shuffle(data_list)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])

        