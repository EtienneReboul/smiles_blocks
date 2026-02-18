from dataclasses import dataclass, field
import re

from numba import njit
import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors


@dataclass
class ExtendedRoThreeThreshold:
    """Extended Ro3 thresholds for block quality assessment."""

    MolWt: float = 300
    nHDonors: int = 3
    nHAcceptors: int = 3
    nRotatableBonds: int = 3
    CrippenlogP: float = 3
    TPSA: float = 60


@dataclass
class BlockedSmilesResult:
    """Data class to store the results of blocked SMILES processing."""

    smiles: str = ""
    smiles_blocked: str = ""
    mem_score: float = -1.0
    unique_id_seq: str = ""
    retro_bond_ratio: float = -1.0
    nb_block_cq_ok: int = -1


@dataclass
class QualityDict:
    """Data class to store the quality assessment results of a block."""

    MolWt: float = 0.0
    nHDonors: int = -1
    nHAcceptors: int = -1
    nRotatableBonds: int = -1
    CrippenlogP: float = -1000.0
    TPSA: float = -1.0
    status: bool = False


@dataclass
class BlockResult:
    """Data class to store the results of block processing."""

    block: str = ""
    can_smiles: str = ""
    first_connected_can_idx: int = -1
    last_connected_can_idx: int = -1
    unique_id: str = ""
    begin_tag: str = ""
    end_tag: str = ""
    MolWt: float = 0.0
    nHDonors: int = -1
    nHAcceptors: int = -1
    nRotatableBonds: int = -1
    CrippenlogP: float = -1000.0
    TPSA: float = -1.0
    status: bool = False


@dataclass
class SmilesRegex:
    """Precompiled regex for tokenizing SMILES strings."""

    regex: re.Pattern = field(
        default=re.compile(
            r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        )
    )


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


def find_cleavable_bond(smiles: str) -> np.ndarray:
    """Find cleavable bonds in a SMILES string with explicit bonds.
    This function identifies bond characters (-,=,#) that are cleavable in a SMILES
    string  with all EXPLICIT BOND and returns their indices along with their positions
    in the string.

    Parameters
    ----------
    smiles : str
        SMILES string to be analyzed for cleavable bonds.

    Returns
    -------
    bonds_array : np.ndarray
        A 2D array of shape (m, 2) where m is the number of cleavable bonds found.
        The first column contains the bond index, and the second column contains
        the corresponding position in the SMILES string.
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
        bonds_array (np.ndarray): _description_

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


def get_smiles_frag_mapping(smiles: str) -> dict[str, str | dict]:
    """Retrieves the essential fragmentation information of a SMILES string.

    This function identifies cleavable bonds in a SMILES string and maps them to
    canonical atomic indices.

    Args:
        smiles (str): A randomized kekule SMILES with explicit bonds

    Returns:
        dict: A dictionary with the following key-value pairs:
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
    idx2canonicalidx, canonicalidces2bondstridx = map_bond2canonatomsidx(smiles, bonds_array)

    # package result as a dictionary
    smiles_fragment_results["canonicalidces2bondstridx"] = canonicalidces2bondstridx
    smiles_fragment_results["idx2canonicalidx"] = idx2canonicalidx
    smiles_fragment_results["bondidx2canonicalidces"] = {
        bond_stridx: bond_canonicalidces
        for bond_canonicalidces, bond_stridx in canonicalidces2bondstridx.items()
    }
    smiles_fragment_results["canonicalidx2idx"] = {
        canonical_idx: idx for idx, canonical_idx in idx2canonicalidx.items()
    }

    return smiles_fragment_results


def get_semantic_mem_score(
    smiles: str, smiles_regex: re.Pattern | str = SmilesRegex.regex
) -> np.float64:
    """
    This function will generate a semantic score from a SMILES,
    i.e the  average number of semantic feature open for every token.
    Semantic feature include branches and rings
        smiles (string) : a valid SMILES
    output :
        mem_map (numpy array): the semantic memory map of the input SMILES
    """
    tokens_list = smiles_regex.findall(smiles)
    mem_map = np.zeros(len(tokens_list), dtype=int)
    digit_set = set()

    for i, token in enumerate(tokens_list):
        if token == "(":
            mem_map[i] += 1
        elif token == ")":
            mem_map[i] -= 1
        elif token.isdigit():
            if token in digit_set:
                digit_set.remove(token)
                mem_map[i] -= 1
            else:
                digit_set.add(token)
                mem_map[i] += 1

    return np.mean(mem_map.cumsum())


def block_assement(block: str, thresholds: ExtendedRoThreeThreshold | None = None) -> dict:
    """
    This function is used to check the quality of the blocks generated from the fragmentation

    Args:
        blocks (str): blocks generated from the fragmentation
        threshold_dict (dict[float|int]): dictionary containing the
        threshold for the quality of the blocks

    Returns:
        dict[float|int]: dictionary containing the quality of the blocks
    """
    # declare local variable
    if thresholds is None:
        thresholds_dict = vars(ExtendedRoThreeThreshold())
    elif isinstance(thresholds, ExtendedRoThreeThreshold):
        thresholds_dict = vars(thresholds)
    elif isinstance(thresholds, dict):
        thresholds_dict = thresholds
    else:
        raise ValueError("thresholds should be either None, ExtendedRoThreeThreshold or dict")
    quality_dict = vars(QualityDict())

    # compute the quality of the block
    mol_block = Chem.MolFromSmiles(block)

    if mol_block is None:
        return quality_dict

    # compute the quality of the block
    quality_dict["MolWt"] = rdMolDescriptors.CalcExactMolWt(mol_block)
    quality_dict["nHDonors"] = rdMolDescriptors.CalcNumHBD(mol_block)
    quality_dict["nHAcceptors"] = rdMolDescriptors.CalcNumHBA(mol_block)
    quality_dict["nRotatableBonds"] = rdMolDescriptors.CalcNumRotatableBonds(mol_block)
    quality_dict["CrippenlogP"] = Crippen.MolLogP(mol_block)
    quality_dict["TPSA"] = rdMolDescriptors.CalcTPSA(mol_block)

    # check the quality of the block
    for key, value in quality_dict.items():
        if key not in thresholds_dict:
            continue
        if value >= thresholds_dict[key]:
            break
    else:
        quality_dict["status"] = True

    return quality_dict


def get_annotation(
    can_atom_idces: frozenset[int],
    canonicalidx2idx: dict[int, int],
    retrosynth: dict[frozenset[int], dict[int, set[str]]],
) -> dict[str, str]:
    """This function get the retro_synthetic annotation of bond in the
    context of randomized smiles
    Args:
        can_atom_idces (frozenset[int]): a frozenset of two canonical atom idx
        canonicalidx2idx (dict[int,int]): mapping between canonical atom idx and atom idx
        retrosynth (dict[frozenset[int],dict[int,set[str]]]): mapping between two canonical atom idx
        and dictionary mapping canonical atom idx to the set chemical tag


    Returns:
        dict[str,str]: a dictionary with the following key-value couple:
            "first_atom" (str): the chemical tag of the first atom
            "end_atom" (str): the chemical tag of the second atom
    """
    # declare local variable
    annotation_dict: dict[str, str] = {"first_atom": str, "end_atom": str}

    # get annotation
    annotation_info = retrosynth[can_atom_idces]

    # unpack the frozenset
    first_idx, second_idx = can_atom_idces

    if canonicalidx2idx[first_idx] < canonicalidx2idx[second_idx]:
        annotation_dict["first_atom"] = "_".join(annotation_info[first_idx])
        annotation_dict["end_atom"] = "_".join(annotation_info[second_idx])
    else:
        annotation_dict["first_atom"] = "_".join(annotation_info[second_idx])
        annotation_dict["end_atom"] = "_".join(annotation_info[first_idx])

    return annotation_dict


def find_last_connecting_atom_idx(
    block: str, smiles_regx: re.Pattern | str = SmilesRegex.regex
) -> int:
    """This function find the last connecting atom idx in a block of randomized SMILES"""
    # declare local variable
    last_connected_atom_idx = -1
    branch_depth = 0
    idx_tracker = -1

    # get smiles tokens
    tokens_list: list[str] = smiles_regx.findall(block)

    for token in tokens_list:
        if token == "(":
            branch_depth += 1
        elif token == ")":
            branch_depth -= 1
        elif token.isdigit():
            continue
        elif token.isalpha() or token.startswith("["):
            idx_tracker += 1
            if branch_depth == 0:
                last_connected_atom_idx = idx_tracker
        else:
            continue

    return last_connected_atom_idx


def get_block_unique_id(block: str) -> dict[str, str | int]:
    """Generate a unique identifier dictionary for a molecular block.

    This function takes a SMILES string representing a molecular block and computes
    its canonical SMILES, connected atom indices, and generates a unique identifier
    combining these properties.

    Parameters
    ----------
    block : str
        A SMILES string representing a molecular block.

    Returns
    -------
    dict[str, str | int]
        A dictionary containing:
        - "can_smiles" : str
            The canonical SMILES representation of the block.
        - "first_connected_can_idx" : int
            The canonical rank index of the first connected atom.
        - "last_connected_can_idx" : int
            The canonical rank index of the last connected atom.
        - "unique_id" : str
            A unique identifier string combining first connected index,
            canonical SMILES, and last connected index in the format:
            "{first_idx}_{can_smiles}_{last_idx}".

    Returns
    -------
    dict[str, str | int]
        An empty dictionary with type annotations if the SMILES string
        cannot be converted to a valid molecule.

    Notes
    -----
    If the input SMILES string is invalid and cannot be converted to a molecule,
    the function returns a dictionary with type hints but no computed values.
    """
    # declare local variable
    id_dict = {
        "can_smiles": str,
        "first_connected_can_idx": int,
        "last_connected_can_idx": int,
        "unique_id": str,
    }

    # get the can smiles of the block
    mol_block = Chem.MolFromSmiles(block)
    if mol_block is None:
        return id_dict
    id_dict["can_smiles"] = Chem.MolToSmiles(mol_block, canonical=True)
    can_mol = Chem.MolFromSmiles(id_dict["can_smiles"])

    # get caninical ranking of the block
    canonical_ranking_old = Chem.CanonicalRankAtoms(mol_block)

    # this may seems weird  and unecessary but there is
    # issue with symmetry and how the tie are breakoff
    # where the actual atoms ranking
    # of a SMILES doesn't reflect the canonical indexing
    canonical_ranks2canidx = {rank: i for i, rank in enumerate(Chem.CanonicalRankAtoms(can_mol))}

    # get first and last connected atom idx
    id_dict["first_connected_can_idx"] = canonical_ranks2canidx[canonical_ranking_old[0]]
    last_connected_can_idx = find_last_connecting_atom_idx(block)
    id_dict["last_connected_can_idx"] = canonical_ranks2canidx[
        canonical_ranking_old[last_connected_can_idx]
    ]

    # get unique id for the block
    id_dict["unique_id"] = (
        f"{id_dict['first_connected_can_idx']}_{id_dict['can_smiles']}_{id_dict['last_connected_can_idx']}"
    )

    return id_dict


def smilesfrag2block(
    block: str,
    threshold_dict: dict,
) -> dict:
    """This function is used to process one fragment of a SMILES and convert it to a block

    Args:
        smiles_frag_mapping (dict[str, str | dict]): a dictionary with the following key-value couple:
            "smiles" (str): the original SMILES string
            "canonicalidces2bondstridx" (dict): Mapping from pairs of canonical atomic indices (as frozensets) to their position in the SMILES string
            "idx2canonicalidx" (dict): Mapping from atom indices to their canonical atomic indices
        retrosynth_results (dict[frozenset[int], dict[int, set[str]]]): a dictionary with the bond signature as key and the value is a dictionary mapping canonical atom idx to the set chemical tag

    Returns:
        dict: a dictionary containing the results of the block fragmentation process
    """

    # get string fragment signature
    result = vars(BlockResult())
    result["block"] = block

    # get unique id of the block
    block_id_dict = get_block_unique_id(block)
    result.update(block_id_dict)

    # get quality of the block
    quality_dict = block_assement(block, threshold_dict)
    result.update(quality_dict)

    return result


def one_smiles2blocks(
    matched_bond: set[frozenset[int]],
    smiles_frag_mapping: dict[str, str | dict],
    retrosynth_results: dict[frozenset[int], dict[int, set[str]]],
    threshold_dict: dict,
) -> list[dict]:
    """This function is used to process one SMILES and convert it to blocks

    Args:
        matched_bond (set[frozenset[int]]): a set of matched bond signatures
        smiles_frag_mapping (dict[str, str | dict]): a dictionary with the following key-value couple:
            "smiles" (str): the original SMILES string
            "canonicalidces2bondstridx" (dict): Mapping from pairs of canonical atomic indices (as frozensets) to their position in the SMILES string
            "idx2canonicalidx" (dict): Mapping from atom indices to their canonical atomic indices
        retrosynth_results (dict[frozenset[int], dict[int, set[str]]]): a dictionary with the bond signature as key and the value is a dictionary mapping canonical atom idx to the set chemical tag

    Returns:
        dict: a dictionary containing the results of the block fragmentation process
    """

    # process the blocks of the current rd_smiles
    block_results = []
    block_start = 0
    start_annotation = "no_tag"
    smiles = smiles_frag_mapping["smiles"]

    for bond_stridx in matched_bond:
        block = smiles[block_start:bond_stridx]

        # get the block result
        temp_dict = smilesfrag2block(block, threshold_dict)

        annotation_dict = get_annotation(
            smiles_frag_mapping["bondidx2canonicalidces"][bond_stridx],
            smiles_frag_mapping["canonicalidx2idx"],
            retrosynth_results,
        )
        temp_dict["begin_tag"] = start_annotation
        temp_dict["end_tag"] = (
            annotation_dict["first_atom"] if annotation_dict["end_atom"] else "no_tag"
        )
        start_annotation = annotation_dict["end_atom"]

        quality_dict = block_assement(block, threshold_dict)
        temp_dict.update(quality_dict)

        block_results.append(temp_dict)

        block_start = bond_stridx + 1

    # get the last block
    temp_dict = smilesfrag2block(smiles[block_start:], threshold_dict)
    temp_dict["begin_tag"] = start_annotation
    temp_dict["end_tag"] = "no_tag"
    block_results.append(temp_dict)

    return block_results


def process_set_smiles(
    sorted_smiles: list[tuple[str, float]],
    retrosynth_results: dict,
) -> tuple[bool, dict, list[dict]]:
    """
    This function is used to process one set of SMILES and convert it to blocks

    Args:
        sorted_smiles (list[tuple[str, float]]): a list of unique SMILES sorted by mem score
        retrosynth_results (dict): a dictionary with the bond signature as key
    """

    # instantiate local variable
    max_matched_bonds = 0
    threshold_dict = vars(ExtendedRoThreeThreshold())
    blocked_smiles_results = vars(BlockedSmilesResult())

    # get the retrosynthetic bond signature
    retro_synt_signature = set(retrosynth_results.keys())
    max_breakdown = len(retro_synt_signature)

    for rd_smiles, mem_score in sorted_smiles:
        # clear blocks

        #  get string fragment signature mapping
        smiles_frag_mapping = get_smiles_frag_mapping(rd_smiles)

        # skip if there is no fragment mapping for the current rd_smiles
        if not smiles_frag_mapping["canonicalidces2bondstridx"]:
            continue

        # get string fragment signature
        string_frag_signature = set(smiles_frag_mapping["canonicalidces2bondstridx"].keys())

        matched_bond = retro_synt_signature.intersection(string_frag_signature)

        # skip the current rd_smiles if the recovery is less than the current max recovery
        if len(matched_bond) < max_matched_bonds:
            continue

        # update the max recovery
        max_matched_bonds = len(matched_bond)

        # get matched bond string idx
        matched_bond_stridx = sorted(
            [smiles_frag_mapping["canonicalidces2bondstridx"][bond] for bond in matched_bond]
        )
        # process the rd smiles to blocks
        block_results = one_smiles2blocks(
            matched_bond_stridx,
            smiles_frag_mapping,
            retrosynth_results,
            threshold_dict,
        )

        # check if maxium recovery is reached
        if len(matched_bond) == max_breakdown:
            break

    # check if a solution is found
    if not block_results:
        return False, blocked_smiles_results, []

    # package the result
    blocked_smiles_results["smiles"] = rd_smiles
    blocked_smiles_results["smiles_blocked"] = ".".join(
        [block_result["block"] for block_result in block_results]
    )
    blocked_smiles_results["unique_id_seq"] = ".".join(
        [block_result["unique_id"] for block_result in block_results]
    )
    blocked_smiles_results["nb_block_cq_ok"] = sum(
        block_result["status"] for block_result in block_results
    )
    blocked_smiles_results["retro_bond_ratio"] = max_matched_bonds / max_breakdown

    return True, blocked_smiles_results, block_results
