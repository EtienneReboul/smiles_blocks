from collections import defaultdict

from rdkit import Chem

from smiles_blocks.rbrics_patterns import RBRICSChemicalGroups, RBRICSRetrosynthesisBound


def prep_rbrics_data() -> dict:
    """Prepare the chemical patterns for the retrosynthetic analysis

    Returns:
        dict: dictionary with the chemical_group, set2tag and rbond:
            chemical_group (dict): contains the chemical tag of chemical group (e.g L1) as key
                                and the corresponding pattern as a mol object as value
            set2tag (dict): contains the mapping between
            the frozenset of chemical tag and string tag
            rbond (dict): contains the frozenset of chemical tag as key
                            and the bond information as value
    """
    # prepare the chemical group pattern
    chemical_group = {
        tag: Chem.MolFromSmarts(pattern)
        for tag, pattern in RBRICSChemicalGroups().patterns.items()
    }

    # prepare the retro-synthetic bond pattern
    rbond_pattern = {
        frozenset(tag.split("-")): Chem.MolFromSmarts(pattern)
        for tag, pattern in RBRICSRetrosynthesisBound().patterns.items()
    }

    # prepare the mapping between set of chemical group and string tag
    set2tag = {
        frozenset(tag.split("-")): tag for tag in RBRICSRetrosynthesisBound().patterns.keys()
    }

    return {"chemical_group": chemical_group, "rbond": rbond_pattern, "set2tag": set2tag}


def check_bondisinring(mol: Chem.Mol, bondidces: tuple[int, int]) -> bool:
    """Check if a bond is in a ring in the molecule

    Args:
        mol (object): an rdkit mol object
        bondidces (tuple[int]): tuple with the index of the first and second atom of the bond

    Returns:
        bool: True if the bond is in a ring, False otherwise
    """
    # get the ring info from the molecule
    bond = mol.GetBondBetweenAtoms(bondidces[0], bondidces[1])

    return bond.IsInRing() if bond else False


def check_chemical_group(mol: Chem.Mol, chemical_pattern: dict) -> set[str]:
    """Check if chemical groups are present in a molecule

    Args:
        mol (object): an rdkit mol object
        chemical_pattern (dict): contains the chemical tag of chemical group (e.g L1) as key
                                and the corresponding pattern as a mol object as value

    Returns:
        set[str]: a set of all the chemical tag of chemical group found in the mol
    """
    return {
        pattern for pattern, sma_mol in chemical_pattern.items() if mol.HasSubstructMatch(sma_mol)
    }


def package_bond_info(bonds: dict[dict]) -> list[dict]:
    """Package the bond information in a list of dictionary.
    This is needed to be able to write the bond information as a nested structure
    in a parquet file

    Args:
        bonds (dict[dict]): dictionary with the bond signature as key
        and the bond information as value

    Returns:
        list[dict]: a list of dictionary with the following key-value couple:
            "begin_idx" (int): the index of the first atom of the bond
            "end_idx" (int): the index of the second atom of the bond
            "begin_tag" (str): the chemical tag of the first atom
            "end_tag" (str): the chemical tag of the second atom
    """

    # declare local variable
    results_list = []

    for bond in bonds.values():
        temp = list(bond.keys())
        temp = {
            "begind_idx": temp[0],
            "end_idx": temp[1],
        }
        temp["begin_tag"] = "_".join(bond[temp["begind_idx"]])
        temp["end_tag"] = "_".join(bond[temp["end_idx"]])

        results_list.append(temp)

    return results_list


def annotate_rbond(
    mol: Chem.Mol, rbond_pattern: dict[frozenset, str], chemical_group: set[str], set2tag: dict
) -> dict[dict[set[str]]]:
    """Annotation of the retro-synthetic bonds in a molecule

    Args:
        mol (object): an rdkit mol object
        rbond_pattern (dict[frozenset, str]):   frozen set of string representing retro-syntetic
                                chemical group tag (e.g {"L1","L2"}) as keys
                                and the corresponding pattern as a mol object as value
        chemical_group (set[str]): The chemical groups detected in the mol

    Returns:
        dict[dict[set[str]]]:
            a dictionary with the bond signature as key and the bond information as value.
            The bond information is a dictionary with the index of the first and second atom
            of the bond as key and the chemical tag of the corresponding atom as value
    """
    # declare local variable
    bonds = defaultdict(lambda: defaultdict(set))

    # get map between current atoms index and canonical atoms idx
    idx2canonicalidx = Chem.CanonicalRankAtoms(mol)

    for rbond_tag, pattern in rbond_pattern.items():
        # Check if the rbond is possible with detected chemical group
        if not rbond_tag.issubset(chemical_group):
            continue

        # find the substruct match
        matches = mol.GetSubstructMatches(pattern)

        # if no match, continue to the next pattern
        if not matches:
            continue

        str_rbond_tag = set2tag[rbond_tag]

        for match in matches:
            # ring bond cannot be safely  cleaved in SMILES, so we skip the bond if it is in a ring
            if check_bondisinring(mol, match):
                continue

            # get canonical idx of match and create bond signature
            canonical_idx = [idx2canonicalidx[match[0]], idx2canonicalidx[match[1]]]
            bond_signature = frozenset(canonical_idx)

            # add the chemical tag of the bond to the corresponding atom in the bond signature
            bonds[bond_signature][canonical_idx[0]].add(str_rbond_tag.split("-")[0])
            bonds[bond_signature][canonical_idx[1]].add(str_rbond_tag.split("-")[1])

    return bonds


def retrosynthetic_analysis(
    smiles: str,
    chemical_dict: dict | None = None,
) -> dict:
    """This function find the retro-synthetic bonds in each molecule.

    Args:
        smiles (str): smile string from parquet files
        chemical_dict (dict): dictionary with the chemical_group, set2tag and rbond:
            chemical_group (dict): contains the chemical tag of chemical group (e.g L1) as key
                                and the corresponding pattern as a mol object as value
            set2tag (dict): contains the mapping between
            the frozenset of chemical tag and string tag
            rbond (dict): contains the frozenset of chemical tag as key
                            and the bond information as value

    Returns:
        result_dict (dictionary): Dictionary with the zinc_id, rbond_matches_set found,

    """
    # check the chemical dict is not empty
    chemical_dict = prep_rbrics_data() if not chemical_dict else chemical_dict

    # Generate the molecules from the smiles with RdKit
    mol = Chem.MolFromSmiles(smiles)

    # check for pattern in molecule and time the process
    chemical_matches_set = check_chemical_group(mol, chemical_dict["chemical_group"])

    # check for r-bond pattern in molecule and time the process
    result_dict = annotate_rbond(
        mol, chemical_dict["rbond"], chemical_matches_set, chemical_dict["set2tag"]
    )

    return result_dict
