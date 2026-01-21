"""
This module provides functions for generating randomized SMILES strings,
finding cleavable bonds in SMILES strings, and mapping these bonds to
canonical atomic indices. The primary goal is to facilitate the fragmentation
of SMILES strings while preserving chemical validity.

Functions:
    generate_rd_smiles(smiles: str, nb_random: int) -> set[str]:
        Generates a set of randomized SMILES strings from a given SMILES string.

    find_string_frag(smiles_array: np.ndarray) -> np.ndarray:
        Finds positions in a SMILES string where it can be safely split without
        breaking chemical grammar.

    find_cleavable_bond(smiles: str) -> np.array:
        Identifies cleavable bonds in a SMILES string with explicit bonds.

    map_bond2canonatomsidx(smiles: str,
                            bonds_array: np.ndarray
                            ) -> tuple[dict[int, int], dict[frozenset[int], int]]:
        Maps cleavable bond indices to pairs of canonical atomic indices.

    get_smiles_frag_mapping(smiles: str) -> dict[str, object]:
        Retrieves fragmentation information for a SMILES string, including
        mappings between canonical atomic indices and bond indices.

    filter_smiles_signature(smiles_mappings: list[dict]) -> tuple[dict]:
            Filters a list of SMILES mappings to remove redundant or contained signatures.
"""

from numba import njit
import numpy as np
from rdkit import Chem

from .baseparams import RandomizedSmilesThresholds


def generate_rd_smiles(
    smiles: str,
    nb_random: int | None = None,
    kekulesmiles: bool = True,
    isomericsmiles: bool = False,
    allbondsexplicit: bool = True
) -> set[str]:
    """    Generate a set of randomized SMILES strings from a given SMILES string.
    
    This function creates randomized SMILES representations of a molecule,
    which can be useful for data augmentation or generating alternative
    representations of the same molecular structure.
    
    Parameters
    ----------
    smiles : str
        The input SMILES string representing a molecule.
    thresholds : int or None, optional
        If an integer, specifies exactly how many randomized SMILES to generate.
        If None (default), uses increasing thresholds from RandomizedSmilesThresholds()
        and stops when no new unique SMILES are generated.
    kekuleSmiles : bool, optional
        Whether to return the kekule version of the SMILES (with explicit aromatic bonds).
        Default is True.
    isomericSmiles : bool, optional
        Whether to include stereochemistry information in the SMILES.
        Default is False.
    allBondsExplicit : bool, optional
        Whether to make all bonds explicit in the SMILES.
        Default is True.
    
    Returns
    -------
    set[str]
        A set of unique randomized SMILES strings.
        
    Notes
    -----
    The randomized SMILES are generated with properties based on the parameters provided.
    
    When thresholds=None, the function uses an adaptive approach that stops
    generating new SMILES once saturation is reached (no new unique SMILES
    are found in a generation batch).
    """

    # declare local variable
    rd_smiles = set()
    previous_size = 0

    mol = Chem.MolFromSmiles(smiles)

    if isinstance(nb_random, int):

        rd_smiles.update(
            Chem.MolToRandomSmilesVect(
                mol,
                nb_random,
                kekuleSmiles=kekulesmiles,
                isomericSmiles=isomericsmiles,
                allBondsExplicit=allbondsexplicit,
            )
        )
    elif nb_random is None:
        nb_random = RandomizedSmilesThresholds()

        for threshold in nb_random:

            # generate randomized smiles
            rd_smiles.update(
                Chem.MolToRandomSmilesVect(
                    mol,
                    threshold,
                    kekuleSmiles=kekulesmiles,
                    isomericSmiles=isomericsmiles,
                    allBondsExplicit=allbondsexplicit,
                )
            )

            # check if the size of the set has changed
            if len(rd_smiles) == previous_size:
                break

            previous_size = len(rd_smiles)
    else:
        raise ValueError(
            "thresholds should be either an int or None. "
            "If it is an int, it should be the number of random SMILES to generate."
        )

    return rd_smiles


@njit(cache=True)
def find_string_frag(smiles_array: np.ndarray) -> np.ndarray:
    """This function used to find where SMILES can be safely split
    without breaking grammar. The SMILES needs to have explicit bond
    for this function to work.

    Args:
        smiles_array (np.array):  one SMILES encoded in ASCII in a numpy array

    Returns:
        np.array:  return a 2D dimensionnal of dimension (m,2) m is number of solution,
        the first column in array is the bond index , the second column is the position
        in the string
    """
    # declare local variable
    cleavable_map = np.zeros(smiles_array.shape[0], dtype=np.int8)
    digit_array = np.zeros(9, dtype=np.bool_)

    # using ASCII corresponding caracter to detect bond
    # 45 : "-", 61 : "=", 35: "#", 47 : "/", 92 : "\", 58 : ":"
    bond_index = np.nonzero(
        (smiles_array == 45)
        | (smiles_array == 61)
        | (smiles_array == 35)
        | (smiles_array == 47)
        | (smiles_array == 92)
        | (smiles_array == 58)
    )[0]

    # check for overloading with charge character by checking i+1 character
    # 93 : "]"
    # Note: likelyhood of having [O-] is high but having [X--] is very low
    # checking for 45, i.e "-", after a bond character is most likely not needed
    charge_check = np.nonzero(
        (smiles_array[bond_index + 1] == 93)
        | (smiles_array[bond_index + 1] == 45)
        | ((smiles_array[bond_index + 1] > 48) & (smiles_array[bond_index + 1] <= 57))
    )[0]
    if len(charge_check) > 0:
        bond_index = np.delete(bond_index, charge_check)

    # iterate on SMILES
    for i in range(smiles_array.shape[0]):
        # hand branches in SMILES
        if smiles_array[i] == 40:  # ASCII for the ( token
            cleavable_map[i] = 1
        elif smiles_array[i] == 41:  # ASCII for the ) token
            cleavable_map[i] = -1

        # handle ring opening and closure
        # 48 to 57 correspond to digit 1 to 9 in ASCII
        elif np.logical_and(smiles_array[i] > 48, smiles_array[i] <= 57):
            digit_index = smiles_array[i] - 49
            # "is" not supported by numba
            if np.equal(digit_array[digit_index], True):
                cleavable_map[i] = -1
                digit_array[digit_index] = False
            else:
                cleavable_map[i] = 1
                digit_array[digit_index] = True

    cleavable_map = np.cumsum(cleavable_map)

    solution_idx = np.nonzero(cleavable_map[bond_index] == 0)[0]

    return np.column_stack((solution_idx, bond_index[solution_idx]))


def find_cleavable_bond(smiles: str) -> np.array:
    """This function find bond (-,=,#) that are cleavable in
    a smiles with EXPLICIT BOND.  Return bond idx bond  to string idx
    for cleavable bonds

    Args:
        smiles (str): smiles that is going to be sliced

    Returns:
        bonds_array (np.array):  return a 2D dimensionnal of dimension (m,2),
        m is number of solution,he first column in array is the bond index ,
        the second column is the position in the string
    """
    # encode smiles to ASCII array
    smiles_array = np.frombuffer(smiles.encode("ascii"), dtype=np.uint8)

    # find cleavable bond in smiles
    bonds_array = find_string_frag(smiles_array)

    return bonds_array


def map_bond2canonatomsidx(
    smiles: str, bonds_array: np.ndarray
) -> tuple[dict[int, int], dict[frozenset[int], int]]:
    """This function map the cleavable bond idx to the pair of canonical atomic
    indices. It gives a new map o

    Args:
        smiles (str): SMILES
        bond_array (np.array): _description_

    Returns:
        dict: contains frozens set of canonical atom idx idces
    """
    # declare local variable
    canonicalidces2bondstridx = {}

    # get mapping
    mol = Chem.MolFromSmiles(smiles)
    # convert to dict to be compatible with downstream code
    idx2canonicalidx = dict(enumerate(Chem.CanonicalRankAtoms(mol)))

    # map cleavable smiles bonds
    for bond_info in bonds_array:
        # retrieve bond atoms idx
        bond = mol.GetBondWithIdx(int(bond_info[0]))

        begin_idx = idx2canonicalidx[bond.GetBeginAtomIdx()]
        end_idx = idx2canonicalidx[bond.GetEndAtomIdx()]
        #
        canonicalidces2bondstridx[frozenset([begin_idx, end_idx])] = int(bond_info[1])
        # note, type casting to int is mandatory
        # python struggle to manipulate np.int64

    return idx2canonicalidx, canonicalidces2bondstridx

def get_smiles_frag_mapping(smiles: str) -> dict[str, object]:
    """Retrieves the essential fragmentation information of a SMILES string.

    This function identifies cleavable bonds in a SMILES string and maps them to 
    canonical atomic indices. 

    Args:
        smiles (str): A randomized kekule SMILES with explicit bonds

    Returns:
        dict[str,obj]: A dictionary with the following key-value pairs:
            "smiles" (str): The original SMILES string
            "canonicalidces2bondstridx" (dict): Mapping from pairs of canonical 
                atomic indices (as frozensets) to their position in the SMILES string
            "idx2canonicalidx" (dict): Mapping from atom indices to their 
                canonical atomic indices
    """
    # declare local variable
    smiles_fragment_results = {
        "smiles": smiles,
    }


    # get smiles cleavable bond mapping
    bonds_array = find_cleavable_bond(smiles)
    idx2canonicalidx, canonicalidces2bondstridx = map_bond2canonatomsidx(
        smiles, bonds_array
    )



    # package result as a dictionary
    smiles_fragment_results["canonicalidces2bondstridx"] = canonicalidces2bondstridx
    smiles_fragment_results["idx2canonicalidx"] = idx2canonicalidx


    return smiles_fragment_results

# def get_smiles_frag_mapping(smiles: str) -> dict[str, object]:
#     """This function get the fragmentation information of a smiles,
#     its primary goal is to be dask.bag.map compatible function as
#     it can work with unequally sized partition.


#     Args:
#         smiles (str): a randomized kekule smiles with EXPLICIT BOND

#     Returns:
#         dict[str,obj]: a dictionary with the following key-value couple:
#             "smiles" (str): the smiles
#             "signature" (frozenset): the signature of the block
#             "idx2canonicalidx" (dict): the mapping between
#             the canonical atomic indices of a bond and
#             its index in the smiles
#             "canonicalidx2idx" (dict): the mapping between
#             the canonical atomic indices and the atom idx
#             "bondidx2canonicalidces" (dict): the mapping between
#             the bond index and the canonical atomic indices
#     """
#     # declare local variable
#     smiles_fragment_results = {
#         "smiles": smiles,
#         "signature": frozenset(),
#         "idx2canonicalidx": {},
#         "canonicalidx2idx": {},
#         "bondidx2canonicalidces": {},
#     }


#     # get smiles cleavable bond mapping
#     bonds_array = find_cleavable_bond(smiles)
#     idx2canonicalidx, canonicalidces2bondstridx = map_bond2canonatomsidx(
#         smiles, bonds_array
#     )

#     # reverse dictionaries
#     canonicalidx2idx = reverse_dict(idx2canonicalidx)
#     bondidx2canonicalidces = reverse_dict(canonicalidces2bondstridx)

#     # package result as a dictionary
#     smiles_fragment_results["signature"] = frozenset(canonicalidces2bondstridx.keys())
#     smiles_fragment_results["canonicalidces2bondstridx"] = canonicalidces2bondstridx
#     smiles_fragment_results["idx2canonicalidx"] = idx2canonicalidx
#     smiles_fragment_results["canonicalidx2idx"] = canonicalidx2idx
#     smiles_fragment_results["bondidx2canonicalidces"] = bondidx2canonicalidces

#     return smiles_fragment_results


def filter_smiles_signature(smiles_mappings: list[dict]) -> tuple[dict]:
    """
    Filters a list of SMILES mappings to remove redundant or contained signatures.
        Args:
            smiles_mappings (list[dict]): A list of dictionaries, each containing
            a "signature" key with a set of SMILES strings.
        Returns:
            tuple[dict]: A tuple of dictionaries with unique and standalone SMILES signatures.
        The function processes each mapping dictionary in the input list
        and filters out signatures that are either:
        - Contained within another signature.
        - Supersets of already found signatures.
        The result is a collection of mappings with unique and standalone signatures.
    """
    # declare local variables
    signature2mem_score = {}
    filtered_mapping = {}

    for mapping_dict in smiles_mappings:
        # declare loop variables
        standalone_comb = True
        idx_signature = frozenset(mapping_dict["canonicalidces2bondstridx"].keys())

        # skip if no signature
        if not idx_signature:
            continue



        # skip comb if signature already found
        elif idx_signature in filtered_mapping:
            if signature2mem_score[idx_signature] < mapping_dict["mem_score"]:
                standalone_comb = False
        else:
            for ref_signature in filtered_mapping.copy():
                # if current signature is contained in another signature skip it
                if idx_signature.issubset(ref_signature):
                    standalone_comb = False
                    break
                # if current solution contains already found solution,
                # then delete old solution
                if idx_signature.issuperset(ref_signature):
                    filtered_mapping.pop(ref_signature)

        if standalone_comb:
            filtered_mapping[idx_signature] = mapping_dict
            signature2mem_score[idx_signature] = mapping_dict["mem_score"]

    return tuple(filtered_mapping.values())
