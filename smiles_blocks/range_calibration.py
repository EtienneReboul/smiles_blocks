"""This module is used to run  calibration range experiments. Those experiments
are used to determine the number of random SMILES needed to reach a the maximum
number of unique SMILES for a given molecule.
"""

from dataclasses import dataclass
from typing import Generator

# third party imports
import pandas as pd
from rdkit import Chem


@dataclass
class SmilesGenerationConfig:
    """Configuration for SMILES calibration experiments.
    Attributes
    ----------
    kekuleSmiles : bool
        Whether to generate Kekule SMILES. Default is True.
    isomericSmiles : bool
        Whether to generate isomeric SMILES. Default is False.
    allBondsExplicit : bool
        Whether to make all bonds explicit in the SMILES. Default is True.
    """

    kekuleSmiles: bool = True
    isomericSmiles: bool = False
    allBondsExplicit: bool = True


def make_calibrationrange(lower: int = 0, upper: int = 5) -> Generator[int, None, None]:
    """Create a list of integers representing a calibration range.
    This function generates a list of integers starting from 'lower' to 'upper'
    (inclusive) with a specified 'step' increment.
    Parameters
    ----------
    lower : int
        The starting integer of the range.
    upper : int
        The ending integer of the range (inclusive).
    step : int
        The increment between each integer in the range.
    Returns
    -------
    list
        A list of integers from 'lower' to 'upper' with the specified 'step'.
    """
    count = 0

    for i in range(lower, upper):
        for _ in range(10):
            count += 10**i

            yield 10**i, count


def make_one_replica(
    smiles: str,
    replica: int,
    max_power: int,
    patience: int,
    config: SmilesGenerationConfig | None = None,
) -> tuple[pd.DataFrame, set]:
    """Generate a DataFrame of randomized SMILES for a molecule without convergence monitoring.
    This function generates randomized SMILES using RDKit's MolToRandomSmilesVect
    for a range of nb_random_smiles values determined by make_calibrationrange.

    Parameters
    ----------
    smiles : str
        Input SMILES string for the molecule.
    replica : int
        Replica identifier to include in the output DataFrame.
    max_power : int
        Upper bound passed to the internal make_calibrationrange call. The internal
        generator yields nb_random_smiles values equal to 10**i for i in range(1, max_power)
        and repeats each such value 10 times; thus the internal iteration count is
        (max_power - 1) * 10 in practice.
    patience : int
        Number of consecutive iterations without discovering a new unique randomized
        SMILES before stopping early.
    config : SmilesGenerationConfig | None, optional
        Configuration for SMILES generation. If None, default configuration is used.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the following columns:
            - smiles: original input SMILES
            - nb_unique_smiles: number of distinct randomized SMILES observed
            - cumulative_count: cumulation count of random SMILES generated so far
            - nb_random_smiles: number of randomized SMILES generated in the current iteration
            - seed: seed used for RDKit's randomization
            - replica: the provided replica identifier
    set
        A set of all unique randomized SMILES generated during the process.
    """
    # check if custom config is provided
    if config is None:
        config = SmilesGenerationConfig()

    # initialize variables
    data_records = []
    smiles_set = set()
    nb_epochs_without_improvement = 0
    previous_nb_unique_smiles = 0

    # make molecule from smiles
    mol = Chem.MolFromSmiles(smiles)

    #
    for nb_smiles, count in make_calibrationrange(0, max_power):
        # create a random seed
        seed = hash(f"{count}_replica_{replica}") % 2**32

        # create randomized smiles batch
        rd_smiles = set(
            Chem.MolToRandomSmilesVect(
                mol,
                nb_smiles,
                kekuleSmiles=config.kekuleSmiles,
                isomericSmiles=config.isomericSmiles,
                allBondsExplicit=config.allBondsExplicit,
                randomSeed=seed,
            )
        )

        # update the set of smiles
        smiles_set.update(rd_smiles)

        # compute number of unique smiles
        nb_unique_smiles = len(smiles_set)

        # create a dictionary with the smiles and the number of unique smiles
        record = {
            "smiles": smiles,
            "nb_unique_smiles": nb_unique_smiles,
            "cumulative_count": count,
            "nb_random_smiles": nb_smiles,
            "seed": seed,
            "replica": replica,
        }
        # update patience counter
        if nb_unique_smiles == previous_nb_unique_smiles:
            nb_epochs_without_improvement += 1
        else:
            nb_epochs_without_improvement = 0

        # append the record to the list
        data_records.append(record)

        # check for early stopping
        if nb_epochs_without_improvement >= patience:
            break

        # update  previous nb of unique smiles
        previous_nb_unique_smiles = nb_unique_smiles

    return pd.DataFrame(data_records), smiles_set


def smiles_set_to_col(smiles_set: set, nb_row: int) -> list[str]:
    """Repackage a set of SMILES strings into a list of specified length.
    If the set has fewer elements than nb_row, the list is padded with empty strings.
    If the set has more elements than nb_row, the elements are grouped to fit into
    the specified length.

    Parameters
    ----------
    smiles_set : set
        A set of SMILES strings.
    nb_row : int
        Number of rows to generate.

    Returns
    -------
    list
        A list of SMILES strings of length nb_row.

    Raises
    ------
    ValueError
        If the smiles_set is empty or if nb_row is not a positive integer.
    """
    # get length of the set
    set_length = len(smiles_set)

    # check that the set is not empty
    if set_length == 0:
        raise ValueError("The smiles_set is empty")

    # check that nb_row is positive
    if nb_row <= 0:
        raise ValueError("The nb_row must be a positive integer")

    # initialize list
    smiles_col = list(smiles_set)

    # adjust length of the list
    if set_length < nb_row:
        # add padding
        smiles_col.extend([""] * (nb_row - set_length))
    else:
        # compute number of groups
        nb_smiles_per_group, remainder = divmod(set_length, nb_row)
        regrouped_smiles = []
        start_idx = 0

        for group_idx in range(nb_row):
            end_idx = start_idx + nb_smiles_per_group + (1 if group_idx < remainder else 0)
            regrouped_smiles.append("_".join(smiles_col[start_idx:end_idx]))
            start_idx = end_idx

        # overwrite smiles_col with regrouped smiles
        smiles_col = regrouped_smiles

    # safety check
    assert len(smiles_col) == nb_row, (
        f"The length of the smiles_col ({len(smiles_col)}) is not equal to nb_row ({nb_row})."
    )

    return smiles_col


def make_datapoints(
    smiles: str, nb_replica: int, max_power: int, patience: int = 5
) -> pd.DataFrame:
    """Generate a DataFrame of randomized SMILES for a molecule with convergence monitoring.
    This function repeatedly generates randomized SMILES using RDKit's MolToRandomSmilesVect,
    tracks the cumulative number of unique SMILES observed, and implements early stopping
    when no new unique SMILES are discovered for `patience` consecutive iterations.

    Parameters
    ----------
    smiles : str
        Input SMILES string for the molecule.
    replica : int
        Replica identifier to include in the output DataFrame.
    max_power : int
        Upper bound passed to the internal make_calibrationrange call. The internal
        generator yields nb_random_smiles values equal to 10**i for i in range(1, max_power)
        and repeats each such value 10 times; thus the internal iteration count is
        (max_power - 1) * 10 in practice.
    patience : int, optional
        Number of consecutive iterations without discovering a new unique randomized
        SMILES before stopping early. Per current validation, patience must be less than
        max_power * 10. Default is 5.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the following columns:
            - smiles: original input SMILES
            - nb_unique_smiles: cumulative number of distinct randomized SMILES observed
            - cumulative_count: running cumulative count used to derive the random seed
            - nb_random_smiles: number of randomized SMILES generated in the current iteration
            - seed: seed used for RDKit's randomization (hash(str(cumulative_count)) % 2**32)
            - replica: the provided replica identifier

    Raises
    ------
    ValueError
        If patience >= max_power * 10 (per the current validation check).

    Notes
    -----
    - Randomized SMILES are generated with kekuleSmiles=True, isomericSmiles=False,
      allBondsExplicit=True.
    - The function accumulates unique randomized SMILES across iterations so that
      nb_unique_smiles reflects the cumulative distinct SMILES discovered so far.
    """

    # initialize results dictionary
    replicas_results = []
    unique_smiles_set = set()

    # check that patience is less than nb of iterations
    if patience >= max_power * 10:
        raise ValueError(f"Patience must be less than number of iterations :  {max_power * 10}")

    # run replicas
    for replica_idx in range(nb_replica):
        # run one replica
        df_replica, smiles_set = make_one_replica(
            smiles=smiles,
            replica=replica_idx,
            max_power=max_power,
            patience=patience,
        )

        # update results dictionary
        unique_smiles_set.update(smiles_set)
        replicas_results.append(df_replica)

    # create a dataframe from the records
    df = pd.concat(replicas_results, ignore_index=True)

    # repackage the smiles set into a column
    smiles_col = smiles_set_to_col(unique_smiles_set, nb_row=len(df))
    df["unique_smiles"] = smiles_col

    return df
