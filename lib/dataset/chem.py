from molvs import Standardizer
from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors

def remove_isotopes(mol):
    for atom in mol.GetAtoms():
        if atom.GetIsotope():
            atom.SetIsotope(0)
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    smiles = standardize_smiles(smiles)
    return smiles


def lipinski_rule_of_five(mol):
    vals = []
    vals.append( 300 <= Descriptors.MolWt(mol) <= 500 )
    vals.append( Crippen.MolLogP(mol) <= 5 )
    vals.append( Descriptors.NumHDonors(mol) <= 5 )
    vals.append( Descriptors.NumHAcceptors(mol) <= 10 )
    return all(vals)


def standardize_smiles(smiles: str) -> str:
    """Standardize SMILES for Mol2Mol

    This should only be used to validate and transform user input
    because the code will abort execution on any error it finds.

    param smiles: single SMILES string
    return: single SMILES string
    """

    mol = Chem.MolFromSmiles(smiles, sanitize=True)

    if not mol:
        return None

    standardizer = Standardizer()  # MolVS

    try:
        smol = standardizer(mol)  # runs SanitizeMol() first
        # largest fragment uncharged
        smol = standardizer.charge_parent(smol)
        smi = Chem.MolToSmiles(smol, isomericSmiles=True)
    except BaseException:
        return None

    # Sometimes when standardizing ChEMBL [H] are not removed so try a
    # second call
    if "[H]" in smi:
        return standardize_smiles(smi)
    else:
        return smi
