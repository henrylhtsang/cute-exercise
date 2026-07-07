"""Render an SM-SM L2 latency-difference heatmap as SVG."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from cute_exercise.ex30_gb200_tpc_groupings.analysis import cluster_tpc_signatures


ROOT = Path(__file__).resolve().parent
DEFAULT_L2 = ROOT / "artifacts" / "gb200_l2_physical_probe_concurrent_256mb.json"
DEFAULT_TOPOLOGY = ROOT / "artifacts" / "gb200_tpc_probe_latest.json"


def viridis_like(value: float) -> str:
    stops = [
        (68, 1, 84),
        (59, 82, 139),
        (33, 145, 140),
        (94, 201, 98),
        (253, 231, 37),
    ]
    value = min(1.0, max(0.0, value))
    scaled = value * (len(stops) - 1)
    index = min(int(scaled), len(stops) - 2)
    frac = scaled - index
    a = stops[index]
    b = stops[index + 1]
    rgb = tuple(round(a[channel] + frac * (b[channel] - a[channel])) for channel in range(3))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def load_matrix(path: Path) -> tuple[list[int], np.ndarray]:
    payload = json.loads(path.read_text())
    rows = sorted(payload["rows"], key=lambda row: int(row["smid"]))
    smids = [int(row["smid"]) for row in rows]
    signatures = np.array([row["signature"] for row in rows], dtype=np.float32)
    matrix = np.mean(np.abs(signatures[:, None, :] - signatures[None, :, :]), axis=2)
    return smids, matrix


def clustered_order(smids: list[int], matrix: np.ndarray, topology_path: Path, groups: int) -> list[int]:
    topology = json.loads(topology_path.read_text())
    smid_to_tpc = {int(smid): int(tpc) for smid, tpc in topology["topology"]["smid_to_tpc"].items()}
    signatures = {smid: tuple(matrix[index]) for index, smid in enumerate(smids)}
    grouping = cluster_tpc_signatures(signatures, smid_to_tpc, target_groups=groups)
    ordered_tpcs = [tpc for group in grouping.groups for tpc in group]
    ordered_smids = []
    for tpc in ordered_tpcs:
        ordered_smids.extend(sorted(smid for smid, mapped in smid_to_tpc.items() if mapped == tpc))
    return ordered_smids


def tpc_stride_order(topology_path: Path, stride: int) -> list[int]:
    topology = json.loads(topology_path.read_text())
    smid_to_tpc = {int(smid): int(tpc) for smid, tpc in topology["topology"]["smid_to_tpc"].items()}
    tpc_to_smids: dict[int, list[int]] = {}
    for smid, tpc in smid_to_tpc.items():
        tpc_to_smids.setdefault(tpc, []).append(smid)

    ordered_smids: list[int] = []
    for offset in range(stride):
        for tpc in range(offset, max(tpc_to_smids) + 1, stride):
            ordered_smids.extend(sorted(tpc_to_smids.get(tpc, [])))
    return ordered_smids


def nearest_neighbor_order(smids: list[int], matrix: np.ndarray) -> list[int]:
    remaining = set(range(len(smids)))
    start = min(remaining, key=lambda index: float(np.mean(matrix[index])))
    order = [start]
    remaining.remove(start)

    while remaining:
        current = order[-1]
        next_index = min(
            remaining,
            key=lambda index: (float(matrix[current, index]), smids[index]),
        )
        order.append(next_index)
        remaining.remove(next_index)

    return [smids[index] for index in order]


def nearest_tpc_order(smids: list[int], matrix: np.ndarray, topology_path: Path) -> list[int]:
    topology = json.loads(topology_path.read_text())
    smid_to_tpc = {int(smid): int(tpc) for smid, tpc in topology["topology"]["smid_to_tpc"].items()}
    tpc_to_smids: dict[int, list[int]] = {}
    for smid, tpc in smid_to_tpc.items():
        tpc_to_smids.setdefault(tpc, []).append(smid)

    index_by_smid = {smid: index for index, smid in enumerate(smids)}
    tpcs = sorted(tpc_to_smids)
    tpc_matrix = np.zeros((len(tpcs), len(tpcs)), dtype=np.float32)
    for i, left in enumerate(tpcs):
        left_indices = [index_by_smid[smid] for smid in tpc_to_smids[left]]
        for j, right in enumerate(tpcs):
            right_indices = [index_by_smid[smid] for smid in tpc_to_smids[right]]
            tpc_matrix[i, j] = float(np.mean(matrix[np.ix_(left_indices, right_indices)]))

    ordered_tpcs = nearest_neighbor_order(tpcs, tpc_matrix)
    ordered_smids: list[int] = []
    for tpc in ordered_tpcs:
        ordered_smids.extend(sorted(tpc_to_smids[tpc]))
    return ordered_smids


def render_svg(
    *,
    smids: list[int],
    matrix: np.ndarray,
    output: Path,
    title: str,
    order: list[int],
) -> None:
    index_by_smid = {smid: index for index, smid in enumerate(smids)}
    ordered_indices = [index_by_smid[smid] for smid in order]
    ordered_matrix = matrix[np.ix_(ordered_indices, ordered_indices)]

    cell = 5
    left = 92
    top = 58
    heat = cell * len(order)
    bar_x = left + heat + 48
    width = bar_x + 92
    height = top + heat + 78
    vmax = float(np.percentile(ordered_matrix, 99.0))
    if vmax <= 0:
        vmax = float(np.max(ordered_matrix) or 1.0)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" font-family="Arial" font-size="18">{html.escape(title)}</text>',
        f'<text x="{left + heat / 2:.1f}" y="{height - 20}" text-anchor="middle" font-family="Arial" font-size="12">SM ID</text>',
        f'<text x="18" y="{top + heat / 2:.1f}" transform="rotate(-90 18 {top + heat / 2:.1f})" text-anchor="middle" font-family="Arial" font-size="12">SM ID</text>',
    ]

    for row, _ in enumerate(order):
        y = top + row * cell
        for col, _ in enumerate(order):
            value = float(ordered_matrix[row, col])
            color = viridis_like(value / vmax)
            x = left + col * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}"/>')

    parts.append(f'<rect x="{left}" y="{top}" width="{heat}" height="{heat}" fill="none" stroke="#333" stroke-width="1"/>')
    for tick in range(0, len(order), 16):
        x = left + tick * cell
        y = top + tick * cell
        label = order[tick]
        parts.append(f'<line x1="{x}" y1="{top + heat}" x2="{x}" y2="{top + heat + 5}" stroke="#333"/>')
        parts.append(f'<text x="{x}" y="{top + heat + 18}" text-anchor="middle" font-family="Arial" font-size="9">{label}</text>')
        parts.append(f'<line x1="{left - 5}" y1="{y}" x2="{left}" y2="{y}" stroke="#333"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 3}" text-anchor="end" font-family="Arial" font-size="9">{label}</text>')

    bar_height = heat
    for i in range(bar_height):
        color = viridis_like(1.0 - i / max(1, bar_height - 1))
        parts.append(f'<rect x="{bar_x}" y="{top + i}" width="22" height="1" fill="{color}"/>')
    parts.append(f'<rect x="{bar_x}" y="{top}" width="22" height="{bar_height}" fill="none" stroke="#333" stroke-width="1"/>')
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + bar_height * (1.0 - frac)
        value = vmax * frac
        parts.append(f'<line x1="{bar_x + 22}" y1="{y:.1f}" x2="{bar_x + 28}" y2="{y:.1f}" stroke="#333"/>')
        parts.append(f'<text x="{bar_x + 32}" y="{y + 4:.1f}" font-family="Arial" font-size="9">{value:.1f}</text>')
    parts.append(
        f'<text x="{bar_x + 62}" y="{top + bar_height / 2:.1f}" '
        f'transform="rotate(-90 {bar_x + 62} {top + bar_height / 2:.1f})" '
        'text-anchor="middle" font-family="Arial" font-size="11">mean |diff| per address (cycles)</text>'
    )

    parts.append("</svg>\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_L2)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--order",
        choices=("smid", "probe", "clustered", "tpc-stride8", "nearest", "nearest-tpc"),
        default="smid",
    )
    parser.add_argument("--cluster-groups", type=int, default=8)
    args = parser.parse_args()

    smids, matrix = load_matrix(args.input)
    payload = json.loads(args.input.read_text())
    if args.order == "smid":
        order = smids
    elif args.order == "probe":
        order = [int(row["smid"]) for row in payload["rows"]]
    elif args.order == "tpc-stride8":
        order = tpc_stride_order(args.topology, stride=8)
    elif args.order == "nearest":
        order = nearest_neighbor_order(smids, matrix)
    elif args.order == "nearest-tpc":
        order = nearest_tpc_order(smids, matrix, args.topology)
    else:
        order = clustered_order(smids, matrix, args.topology, args.cluster_groups)

    render_svg(
        smids=smids,
        matrix=matrix,
        output=args.output,
        title=f"GB200 SM-SM L2 Latency Difference ({args.order} order)",
        order=order,
    )


if __name__ == "__main__":
    main()
