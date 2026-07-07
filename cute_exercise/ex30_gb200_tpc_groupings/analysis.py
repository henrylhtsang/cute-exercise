"""Analysis helpers for the GB200 TPC grouping probe."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class ClusterObservation:
    cluster_size: int
    iteration: int
    cluster_id: int
    smids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.cluster_size < 1:
            raise ValueError("cluster_size must be positive")
        if len(self.smids) != self.cluster_size:
            raise ValueError(
                f"cluster_size={self.cluster_size} does not match "
                f"{len(self.smids)} SM ids"
            )
        if len(set(self.smids)) != len(self.smids):
            raise ValueError(f"duplicate SM ids in cluster observation: {self.smids}")


@dataclass(frozen=True)
class Topology:
    tpc_groups: tuple[tuple[int, ...], ...]
    cluster_groups: dict[int, tuple[tuple[int, ...], ...]]
    cluster_group_observation_counts: dict[int, dict[tuple[int, ...], int]]
    ambiguous_groups: dict[int, tuple[tuple[int, ...], ...]]
    smid_to_tpc: dict[int, int]
    notes: str


@dataclass(frozen=True)
class PhysicalGrouping:
    groups: tuple[tuple[int, ...], ...]
    group_counts: tuple[int, ...]
    merge_distances: tuple[float, ...]
    tpc_signatures: dict[int, tuple[float, ...]]


def _canonical_group(smids: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(smids))


def _select_non_overlapping(
    counts: Counter[tuple[int, ...]],
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    selected: list[tuple[int, ...]] = []
    ambiguous: list[tuple[int, ...]] = []
    used_smids: set[int] = set()

    for group, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        group_smids = set(group)
        if group_smids & used_smids:
            ambiguous.append(group)
            continue
        selected.append(group)
        used_smids.update(group_smids)

    return tuple(sorted(selected)), tuple(sorted(ambiguous))


def infer_topology(observations: Iterable[ClusterObservation]) -> Topology:
    """Infer stable TPC and larger cluster groups from raw cluster observations."""
    counts_by_size: dict[int, Counter[tuple[int, ...]]] = defaultdict(Counter)
    for obs in observations:
        counts_by_size[obs.cluster_size][_canonical_group(obs.smids)] += 1

    cluster_groups: dict[int, tuple[tuple[int, ...], ...]] = {}
    cluster_group_observation_counts: dict[int, dict[tuple[int, ...], int]] = {}
    ambiguous_groups: dict[int, tuple[tuple[int, ...], ...]] = {}
    for cluster_size, counts in sorted(counts_by_size.items()):
        selected, ambiguous = _select_non_overlapping(counts)
        cluster_groups[cluster_size] = selected
        cluster_group_observation_counts[cluster_size] = {
            group: counts[group] for group in selected
        }
        if ambiguous:
            ambiguous_groups[cluster_size] = ambiguous

    tpc_groups = cluster_groups.get(2, ())
    smid_to_tpc = {
        smid: tpc_id
        for tpc_id, group in enumerate(tpc_groups)
        for smid in group
    }

    note_parts: list[str] = []
    if 2 in ambiguous_groups:
        note_parts.append("conflicting cluster_size=2 groups were observed")
    missing_cluster2 = sorted(
        {
            smid
            for groups in cluster_groups.values()
            for group in groups
            for smid in group
            if smid not in smid_to_tpc
        }
    )
    if missing_cluster2:
        note_parts.append(
            "some SM ids were seen only in larger clusters: "
            + ",".join(str(smid) for smid in missing_cluster2)
        )

    return Topology(
        tpc_groups=tpc_groups,
        cluster_groups=cluster_groups,
        cluster_group_observation_counts=cluster_group_observation_counts,
        ambiguous_groups=ambiguous_groups,
        smid_to_tpc=smid_to_tpc,
        notes="; ".join(note_parts),
    )


def observations_from_rows(rows: Iterable[dict]) -> list[ClusterObservation]:
    """Build cluster observations from one-row-per-CTA probe output."""
    grouped: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for row in rows:
        key = (
            int(row["cluster_size"]),
            int(row["iteration"]),
            int(row["cluster_id"]),
        )
        grouped[key].append(int(row["smid"]))

    observations: list[ClusterObservation] = []
    for (cluster_size, iteration, cluster_id), smids in sorted(grouped.items()):
        if len(smids) != cluster_size or len(set(smids)) != cluster_size:
            continue
        observations.append(
            ClusterObservation(
                cluster_size=cluster_size,
                iteration=iteration,
                cluster_id=cluster_id,
                smids=_canonical_group(smids),
            )
        )
    return observations


def smid_inventory(rows: Iterable[dict]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    observed = sorted({int(row["smid"]) for row in rows})
    if not observed:
        return (), ()
    expected = set(range(observed[-1] + 1))
    missing = tuple(sorted(expected - set(observed)))
    return tuple(observed), missing


def _mean_signature(signatures: Iterable[tuple[float, ...]]) -> tuple[float, ...]:
    rows = list(signatures)
    if not rows:
        raise ValueError("cannot average an empty signature set")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("all signatures must have the same width")
    return tuple(sum(row[i] for row in rows) / len(rows) for i in range(width))


def _normalize_signatures(
    signatures: dict[int, tuple[float, ...]]
) -> dict[int, tuple[float, ...]]:
    if not signatures:
        return {}
    width = len(next(iter(signatures.values())))
    means = []
    stdevs = []
    for i in range(width):
        values = [signature[i] for signature in signatures.values()]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        means.append(mean)
        stdevs.append(math.sqrt(variance) or 1.0)
    return {
        key: tuple((signature[i] - means[i]) / stdevs[i] for i in range(width))
        for key, signature in signatures.items()
    }


def _rms_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if len(a) != len(b):
        raise ValueError("signature widths do not match")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def _average_linkage_distance(
    left: tuple[int, ...],
    right: tuple[int, ...],
    distances: dict[tuple[int, int], float],
) -> float:
    total = 0.0
    count = 0
    for a in left:
        for b in right:
            key = (a, b) if a < b else (b, a)
            total += distances[key]
            count += 1
    return total / count


def _auto_target_group_count(merge_distances: list[float], tpc_count: int) -> int:
    if len(merge_distances) < 2:
        return 1
    best_index = 0
    best_gap = -1.0
    for i in range(1, len(merge_distances)):
        gap = merge_distances[i] - merge_distances[i - 1]
        if gap > best_gap:
            best_gap = gap
            best_index = i
    # The largest jump at merge index i means use the grouping before that merge.
    return tpc_count - best_index


def cluster_tpc_signatures(
    smid_signatures: dict[int, tuple[float, ...]],
    smid_to_tpc: dict[int, int],
    *,
    target_groups: int | None = None,
) -> PhysicalGrouping:
    """Cluster TPCs by per-SM L2 latency signatures."""
    signatures_by_tpc: dict[int, list[tuple[float, ...]]] = defaultdict(list)
    for smid, signature in smid_signatures.items():
        if smid in smid_to_tpc:
            signatures_by_tpc[smid_to_tpc[smid]].append(signature)
    if not signatures_by_tpc:
        raise ValueError("no SM signatures matched the SM-to-TPC map")

    tpc_signatures = {
        tpc: _mean_signature(signatures)
        for tpc, signatures in sorted(signatures_by_tpc.items())
    }
    normalized = _normalize_signatures(tpc_signatures)
    tpcs = tuple(sorted(normalized))
    if target_groups is not None and not (1 <= target_groups <= len(tpcs)):
        raise ValueError("target_groups must be between 1 and the TPC count")

    distances = {
        (a, b): _rms_distance(normalized[a], normalized[b])
        for i, a in enumerate(tpcs)
        for b in tpcs[i + 1 :]
    }

    clusters: list[tuple[int, ...]] = [(tpc,) for tpc in tpcs]
    snapshots: dict[int, tuple[tuple[int, ...], ...]] = {len(clusters): tuple(clusters)}
    merge_distances: list[float] = []

    while len(clusters) > 1:
        best_pair = (0, 1)
        best_distance = math.inf
        for i, left in enumerate(clusters):
            for j in range(i + 1, len(clusters)):
                distance = _average_linkage_distance(left, clusters[j], distances)
                if distance < best_distance:
                    best_distance = distance
                    best_pair = (i, j)

        i, j = best_pair
        merged = tuple(sorted(clusters[i] + clusters[j]))
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in {i, j}
        ]
        clusters.append(merged)
        clusters.sort(key=lambda group: (group[0], len(group), group))
        merge_distances.append(best_distance)
        snapshots[len(clusters)] = tuple(clusters)

    if target_groups is None:
        target_groups = _auto_target_group_count(merge_distances, len(tpcs))
    groups = tuple(sorted(snapshots[target_groups], key=lambda group: (-len(group), group[0])))
    return PhysicalGrouping(
        groups=groups,
        group_counts=tuple(len(group) for group in groups),
        merge_distances=tuple(merge_distances),
        tpc_signatures=tpc_signatures,
    )
