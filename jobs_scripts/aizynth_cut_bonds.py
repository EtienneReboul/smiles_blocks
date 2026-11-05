"""Stage A: run retrosynthetic search over the reference SMILES of one MOSES part.

Why this is a separate stage from ``smiles2block.py``
-----------------------------------------------------
R-BRICS costs microseconds per molecule, so ``smiles2block.py`` calls it inline
inside its workers. AiZynthFinder costs ~11.5 s. Keeping it inline would mean
re-running a tree search on every job retry and welding a CPU-bound search to an
I/O-bound enumeration over 65 GB of randomised SMILES. So the search runs here,
once per reference molecule, and writes its cut bonds to parquet;
``smiles2block.py --retro-backend precomputed`` reads them back.

Work unit and cost
------------------
One task handles one ``smiles_data`` part. Each part holds ~57 M randomised SMILES
rows but only ~16 k *unique* reference SMILES, which is what gets searched:

    16 000 molecules x 11.5 s = ~51 core-hours per part
    119 parts                = ~6 100 core-hours total

On a 64-core node that is well under an hour of wall time per part, so the whole
of MOSES is one 119-task array.

Thread oversubscription
-----------------------
ONNX Runtime defaults to one thread per core. With one worker per core that is
64x oversubscription and the node thrashes. The limits are set below before
AiZynthFinder is imported, and should also be exported by the SLURM script.

Resume
------
A completed part is skipped on relaunch, matching ``smiles2block.py``. Results are
flushed in chunks, so a part killed by the time limit keeps everything it had
finished; rerun with ``--resume`` to search only the molecules still missing.

Example:
    python jobs_scripts/aizynth_cut_bonds.py \\
        --input data/processed/smiles_data/part-0.parquet \\
        --output data/interim/retro_cut_bonds/part-0.parquet \\
        --config models/aizynthfinder/config.yml \\
        --num-workers 64
"""

import argparse
import logging
import multiprocessing as mp
import os
from pathlib import Path
import time

# Must precede any import that pulls in ONNX Runtime / BLAS, or the thread pools
# are sized before we get a say.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "ORT_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from smiles_blocks.files import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, PROJ_ROOT  # noqa: E402
from smiles_blocks.retro_backends.aizynth import (  # noqa: E402
    DEFAULT_TAG_KEYS,
    SearchOutcome,
    load_finder,
    observed_metadata_keys,
    search_one,
)
from smiles_blocks.retro_backends.store import CutBondsParquet, bonds_to_records  # noqa: E402

logger = logging.getLogger(__name__)

#: Per-worker state, populated by :func:`init_worker` after the fork.
_FINDER = None
_SEARCH_KWARGS: dict = {}


def cut_bonds_argparse() -> dict:
    """Parse the command line.

    Returns:
        dict: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run retrosynthetic search over the reference SMILES of a MOSES part"
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(PROCESSED_DATA_DIR / "smiles_data/part-0.parquet"),
        help="Parquet file or directory holding a 'smiles' column of reference SMILES",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(INTERIM_DATA_DIR / "retro_cut_bonds/part-0.parquet"),
        help="Destination parquet file for the cut bonds",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the AiZynthFinder YAML config (expansion policy + stock)",
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default=None,
        help="Log file; logs to stderr when omitted",
    )

    # Search behaviour
    parser.add_argument(
        "--route-index",
        type=int,
        default=0,
        help="Which score-sorted route to use; 0 is the top route (default: 0)",
    )
    parser.add_argument(
        "--tag-keys",
        type=str,
        default=",".join(DEFAULT_TAG_KEYS),
        help=(
            "Comma-separated reaction metadata keys to try when labelling a "
            "disconnection, in order of preference. Use --inspect-metadata to see "
            f"what your model emits (default: {','.join(DEFAULT_TAG_KEYS)})"
        ),
    )
    parser.add_argument(
        "--allow-unsolved",
        action="store_true",
        help="Accept routes whose precursors are not all in stock (default: reject)",
    )
    parser.add_argument(
        "--expansion",
        type=str,
        default=None,
        help="Expansion policy key to select (default: all keys in the config)",
    )
    parser.add_argument(
        "--stock",
        type=str,
        default=None,
        help="Stock key to select (default: all keys in the config)",
    )

    # Compute
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Worker processes; one AiZynthFinder each (default: 4)",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=200,
        help="Flush results to parquet every N molecules (default: 200)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Search at most N molecules; 0 means all. For pilots and smoke tests",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep an existing partial output and search only the missing molecules",
    )

    # Diagnostics
    parser.add_argument(
        "--inspect-metadata",
        type=int,
        default=0,
        help=(
            "Diagnostic: search N molecules serially, report the reaction metadata "
            "keys observed, and exit without writing output"
        ),
    )

    return vars(parser.parse_args())


def read_reference_smiles(input_path: str) -> list[str]:
    """Collect the unique reference SMILES of a part, streaming to bound memory.

    A part holds ~57 M randomised rows against ~16 k unique reference SMILES, so
    the column is scanned in batches and deduplicated incrementally rather than
    materialised whole.

    Args:
        input_path: Parquet file or directory with a ``smiles`` column.

    Returns:
        list[str]: Unique reference SMILES, sorted for reproducible sharding and
        logging.
    """
    dataset = ds.dataset(input_path, format="parquet")

    unique: set[str] = set()
    for batch in dataset.to_batches(columns=["smiles"]):
        unique.update(pc.unique(batch["smiles"]).to_pylist())

    unique.discard(None)
    return sorted(unique)


def already_done(output_path: Path) -> set[str]:
    """Read back the reference SMILES a previous run already searched.

    Args:
        output_path: Partial output from an interrupted run.

    Returns:
        set[str]: Reference SMILES present in the file, empty if it is unreadable
        or absent. A corrupt tail is treated as "nothing done" rather than
        propagated, so a truncated file never blocks a rerun.
    """
    if not output_path.exists():
        return set()
    try:
        table = pq.read_table(output_path, columns=["ref_smiles"])
        return set(table["ref_smiles"].to_pylist())
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not read existing output %s (%s); restarting it", output_path, exc)
        return set()


def init_worker(
    config: str, expansion: str | None, stock: str | None, search_kwargs: dict
) -> None:
    """Load one AiZynthFinder per worker process.

    Runs once per worker, not once per molecule: loading the ONNX expansion model
    costs far more than a single search.

    Args:
        config: Path to the AiZynthFinder YAML config.
        expansion: Expansion policy key, or None for all.
        stock: Stock key, or None for all.
        search_kwargs: Forwarded to :func:`search_one`.
    """
    global _FINDER, _SEARCH_KWARGS
    _FINDER = load_finder(config, expansion=expansion, stock=stock)
    _SEARCH_KWARGS = search_kwargs


def run_one(smiles: str) -> dict:
    """Search one molecule and shape the result into a parquet row.

    Args:
        smiles: Reference SMILES.

    Returns:
        dict: A row matching :class:`CutBondsParquet`'s schema.
    """
    outcome = search_one(_FINDER, smiles, **_SEARCH_KWARGS)
    return outcome_to_row(outcome)


def outcome_to_row(outcome: SearchOutcome) -> dict:
    """Convert a :class:`SearchOutcome` into a parquet row.

    Args:
        outcome: Result of one search.

    Returns:
        dict: Row matching :class:`CutBondsParquet`'s schema.
    """
    return {
        "ref_smiles": outcome.ref_smiles,
        "retro_success": outcome.retro_success,
        "n_routes": min(outcome.n_routes, 32767),
        "n_steps": min(outcome.n_steps, 32767),
        "search_time_s": outcome.search_time_s,
        "backend": outcome.backend,
        "error": outcome.error,
        "bonds": bonds_to_records(outcome.bonds),
    }


def inspect_metadata(args: dict, smiles_list: list[str]) -> None:
    """Report which reaction metadata keys the configured model populates.

    Which keys exist is a property of the expansion policy, not of the library,
    so ``--tag-keys`` has to be chosen empirically. This searches a few molecules
    serially and prints what came back.

    Args:
        args: Parsed command line.
        smiles_list: Candidate molecules to search.
    """
    finder = load_finder(args["config"], expansion=args["expansion"], stock=args["stock"])
    sample = smiles_list[: args["inspect_metadata"]]

    keys: set[str] = set()
    mapped_steps = 0
    for smiles in sample:
        finder.target_smiles = smiles
        finder.tree_search()
        finder.build_routes()
        trees = list(finder.routes.reaction_trees)
        if not trees:
            logger.info("%s: no route", smiles)
            continue
        tree_keys = observed_metadata_keys(trees[0])
        keys.update(tree_keys)
        has_mapping = "mapped_reaction_smiles" in tree_keys
        mapped_steps += int(has_mapping)
        logger.info(
            "%s: solved=%s keys=%s", smiles, getattr(trees[0], "is_solved", "?"), sorted(tree_keys)
        )

    logger.info("metadata keys observed across %d molecules: %s", len(sample), sorted(keys))
    logger.info(
        "routes carrying mapped_reaction_smiles: %d/%d -- bond extraction needs this",
        mapped_steps,
        len(sample),
    )
    usable = [key for key in DEFAULT_TAG_KEYS if key in keys]
    logger.info(
        "of the default --tag-keys, present: %s%s",
        usable or "none",
        "" if usable else " (disconnections will fall back to the generic tag)",
    )


def main() -> None:
    os.chdir(PROJ_ROOT)
    args = cut_bonds_argparse()

    logging.basicConfig(
        filename=args["logfile"],
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    output_path = Path(args["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("reading reference SMILES from %s", args["input"])
    smiles_list = read_reference_smiles(args["input"])
    logger.info("%d unique reference SMILES", len(smiles_list))

    if args["inspect_metadata"]:
        inspect_metadata(args, smiles_list)
        return

    # Resume / skip handling.
    done: set[str] = set()
    if output_path.exists():
        if not args["resume"]:
            logger.info("%s already exists; skipping (pass --resume to continue it)", output_path)
            return
        done = already_done(output_path)
        logger.info("resuming: %d molecules already searched", len(done))
        smiles_list = [smiles for smiles in smiles_list if smiles not in done]

    if args["limit"]:
        smiles_list = smiles_list[: args["limit"]]
        logger.info("limited to %d molecules", len(smiles_list))

    if not smiles_list:
        logger.info("nothing left to search")
        return

    config = CutBondsParquet()
    search_kwargs = {
        "route_index": args["route_index"],
        "tag_keys": tuple(key.strip() for key in args["tag_keys"].split(",") if key.strip()),
        "require_solved": not args["allow_unsolved"],
    }

    # Write to a temporary file, then move into place, so a killed job never
    # leaves a file that looks complete. On resume the existing rows are copied
    # across first.
    temp_path = output_path.with_suffix(".partial.parquet")
    writer = pq.ParquetWriter(
        temp_path,
        config.schema,
        compression=config.compression,
        compression_level=config.compression_level,
    )

    processed = 0
    succeeded = 0
    buffer: list[dict] = []
    start = time.time()

    def flush() -> None:
        """Write the buffered rows as one row group and clear the buffer."""
        if not buffer:
            return
        writer.write_table(pa.Table.from_pylist(buffer, schema=config.schema))
        buffer.clear()

    try:
        if done:
            # Carry forward what the interrupted run had already written.
            writer.write_table(pq.read_table(output_path, schema=config.schema))

        context = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
        with context.Pool(
            processes=args["num_workers"],
            initializer=init_worker,
            initargs=(args["config"], args["expansion"], args["stock"], search_kwargs),
        ) as pool:
            # chunksize=1: each task is seconds long, so per-task dispatch
            # overhead is irrelevant and even load balancing matters more.
            for row in pool.imap_unordered(run_one, smiles_list, chunksize=1):
                buffer.append(row)
                processed += 1
                succeeded += int(row["retro_success"])

                if processed % args["flush_every"] == 0:
                    flush()
                    elapsed = time.time() - start
                    rate = processed / elapsed
                    remaining = (len(smiles_list) - processed) / rate if rate else 0
                    logger.info(
                        "%d/%d searched (%.1f%% solved) | %.2f mol/s | ETA %.1f min",
                        processed,
                        len(smiles_list),
                        100 * succeeded / processed,
                        rate,
                        remaining / 60,
                    )
        flush()
    finally:
        writer.close()

    temp_path.replace(output_path)

    elapsed = time.time() - start
    logger.info(
        "done: %d molecules, %d with usable cuts (%.1f%%), %.1f min wall, %.2f s/mol/core",
        processed,
        succeeded,
        100 * succeeded / processed if processed else 0.0,
        elapsed / 60,
        elapsed * args["num_workers"] / processed if processed else 0.0,
    )
    logger.info("wrote %s", output_path)


if __name__ == "__main__":
    main()
