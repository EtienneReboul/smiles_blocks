"""AiZynthFinder adapter: a found route becomes a set of cuts on the target.

This module is deliberately thin. All the chemistry lives in
:mod:`smiles_blocks.retro_backends.bond_extraction`, which works from atom-mapped
reaction SMILES alone; this file only drives the search and pulls those strings
out of the result. That keeps the part that needs AiZynthFinder installed small
and the part that needs testing independent of it.

Facts about the AiZynthFinder API this relies on (verified against the
MolecularAI/aizynthfinder sources):

- A route's reaction nodes are ``FixedRetroReaction``. Their molecules are
  ``UniqueMolecule`` and carry **no** atom maps, so the mapping survives in the
  route only as a string under ``metadata["mapped_reaction_smiles"]``, written by
  ``ReactionTreeLoader._unique_reaction`` and present only when the reaction was
  actually mapped.
- That string is in the **retro** direction, ``target>>precursors`` -- the product
  is on the *left*. ``RetroReaction.mapped_reaction_smiles`` names its locals
  "reactants"/"products" the other way round, which is misleading.
- ``ReactionTree.reactions()`` iterates the reaction nodes.
- ``finder.routes`` is score-sorted, so index 0 is the top route.

The import of ``aizynthfinder`` is deferred into :func:`load_finder` so that
importing this module costs nothing when the search is not being run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

from smiles_blocks.retro_backends.bond_extraction import (
    DEFAULT_TAG,
    RouteStep,
    route_cut_bonds,
)

__all__ = [
    "SearchOutcome",
    "DEFAULT_TAG_KEYS",
    "load_finder",
    "route_steps",
    "search_one",
]

#: Metadata keys tried in order when labelling a disconnection. Which of these
#: AiZynthFinder actually populates depends on the expansion policy, so the list
#: is a preference order with a fallback rather than a guarantee. Use
#: :func:`observed_metadata_keys` to find out what a given model emits.
DEFAULT_TAG_KEYS = ("classification", "template_hash", "template_code", "policy_name")


@dataclass
class SearchOutcome:
    """Result of running the search on one molecule.

    Failures are recorded rather than raised, so that a stage-A part file is a
    complete account of every molecule the job was given.
    """

    ref_smiles: str
    retro_success: bool = False
    n_routes: int = 0
    n_steps: int = 0
    search_time_s: float = -1.0
    backend: str = "aizynth"
    error: str = ""
    bonds: dict = field(default_factory=dict)


def load_finder(config_path: str, *, expansion: str | None = None, stock: str | None = None):
    """Build an AiZynthFinder instance and select its policy and stock.

    Expensive: it loads the ONNX expansion model. Call once per worker process,
    never per molecule.

    Args:
        config_path: Path to the AiZynthFinder YAML config.
        expansion: Expansion policy key to select. ``None`` selects every policy
            the config defines.
        stock: Stock key to select. ``None`` selects every stock the config
            defines.

    Returns:
        aizynthfinder.aizynthfinder.AiZynthFinder: A ready-to-use finder.

    Raises:
        ImportError: If aizynthfinder is not installed, with a pointer to the
            environment that is supposed to provide it.
    """
    try:
        from aizynthfinder.aizynthfinder import AiZynthFinder
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "aizynthfinder is not installed in this environment. The retrosynthesis "
            "stage needs it (tested with 4.4.1 on onnxruntime, Python 3.11); the "
            "fragmentation stage does not."
        ) from exc

    finder = AiZynthFinder(configfile=config_path)

    expansion_keys = [expansion] if expansion else list(finder.expansion_policy.items)
    finder.expansion_policy.select(expansion_keys)

    stock_keys = [stock] if stock else list(finder.stock.items)
    finder.stock.select(stock_keys)

    return finder


def _reaction_tag(metadata: dict, tag_keys: tuple[str, ...]) -> str:
    """Pick a short label for a disconnection from its reaction metadata.

    Args:
        metadata: A ``FixedRetroReaction.metadata`` dict.
        tag_keys: Keys to try, in order of preference.

    Returns:
        str: The first non-empty value found, else :data:`DEFAULT_TAG`.
    """
    for key in tag_keys:
        value = metadata.get(key)
        if value not in (None, "", []):
            return str(value)
    return DEFAULT_TAG


def route_steps(
    reaction_tree: Any, tag_keys: tuple[str, ...] = DEFAULT_TAG_KEYS
) -> list[RouteStep]:
    """Extract the mapped steps of one route.

    Args:
        reaction_tree: An AiZynthFinder ``ReactionTree``.
        tag_keys: Metadata keys to try when labelling each step.

    Returns:
        list[RouteStep]: One step per reaction node that carries a mapped
        reaction SMILES. Unmapped nodes are skipped -- without atom maps a step
        cannot be traced back to the target.
    """
    steps = []
    for reaction in reaction_tree.reactions():
        metadata = dict(getattr(reaction, "metadata", {}) or {})
        mapped = metadata.get("mapped_reaction_smiles")
        if not mapped:
            continue
        steps.append(
            RouteStep(mapped_reaction_smiles=mapped, tag=_reaction_tag(metadata, tag_keys))
        )
    return steps


def observed_metadata_keys(reaction_tree: Any) -> set[str]:
    """Report the metadata keys a route actually carries.

    Which keys an expansion policy populates is not guaranteed by the library, so
    this exists to discover the right value for ``--tag-keys`` empirically on the
    cluster before committing to a full run.

    Args:
        reaction_tree: An AiZynthFinder ``ReactionTree``.

    Returns:
        set[str]: Union of metadata keys over the route's reaction nodes.
    """
    keys: set[str] = set()
    for reaction in reaction_tree.reactions():
        keys.update((getattr(reaction, "metadata", {}) or {}).keys())
    return keys


def search_one(
    finder: Any,
    smiles: str,
    *,
    route_index: int = 0,
    tag_keys: tuple[str, ...] = DEFAULT_TAG_KEYS,
    require_solved: bool = True,
) -> SearchOutcome:
    """Search one molecule and convert the chosen route into cuts on the target.

    Args:
        finder: An instance from :func:`load_finder`, reused across molecules.
        smiles: Reference SMILES to search.
        route_index: Which route to use; ``finder.routes`` is score-sorted, so 0
            is the top route.
        tag_keys: Metadata keys to try when labelling disconnections.
        require_solved: Only accept a route whose precursors are all in stock.
            A route that is not solved is not a synthesis proposal.

    Returns:
        SearchOutcome: Always returned, with ``retro_success`` False and ``error``
        set on any failure, so one bad molecule never aborts a job of 16 000.
    """
    outcome = SearchOutcome(ref_smiles=smiles)
    start = time.perf_counter()

    try:
        finder.target_smiles = smiles
        finder.tree_search()
        finder.build_routes()

        trees = list(finder.routes.reaction_trees)
        outcome.n_routes = len(trees)

        if route_index >= len(trees):
            outcome.error = "no_route"
            return outcome

        tree = trees[route_index]

        if require_solved and not getattr(tree, "is_solved", False):
            outcome.error = "route_not_solved"
            return outcome

        steps = route_steps(tree, tag_keys)
        outcome.n_steps = len(steps)
        if not steps:
            # The route exists but carries no atom mapping, so its
            # disconnections cannot be located on the target.
            outcome.error = "no_mapped_steps"
            return outcome

        bonds = route_cut_bonds(smiles, steps)
        if not bonds:
            # Typically a ring-forming or ring-opening route: real chemistry, but
            # not expressible as a cut in a SMILES string.
            outcome.error = "no_acyclic_cut"
            return outcome

        outcome.bonds = bonds
        outcome.retro_success = True

    except Exception as exc:  # noqa: BLE001 - one molecule must not kill the job
        outcome.error = f"{type(exc).__name__}: {exc}"[:200]
    finally:
        outcome.search_time_s = time.perf_counter() - start

    return outcome
