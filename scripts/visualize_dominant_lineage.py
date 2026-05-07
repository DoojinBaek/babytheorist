#!/usr/bin/env python3
"""Render dominant-program lineage by replaying the real construction.

The dominant cache stores selected abstract programs, but not parent IDs.  This
script rebuilds the same segmented dominant library from cache metadata and
records max-score construction derivations as candidates are admitted.  The
official cache is not modified; the provenance is an audit/visualization layer.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from babytheorist.dominant import (
    AbstractPool,
    AbstractProgram,
    _binding_score,
    _binding_variants,
    _canonical_depth_one_pool,
    _count_primitive,
    _fill_initial_grid_shapes,
    _introduced_step_variants,
    _pick_top_or_sample,
    _primitive_sequence_score,
    _scope_offset,
    _shift_scopes,
    _with_fresh_scopes,
    abstract_pool_to_jsonable,
    abstract_pool_from_jsonable,
    build_pool,
    validate_canonical,
)
from babytheorist.program import build_bigram, sample_bigram


SEGMENT_COLORS = {
    "base": "#456990",
    "append:hollow": "#7B8C3A",
    "append:fill_interior": "#B66D3A",
    "append:add_border": "#A23E48",
}

EDGE_COLORS = {
    "base-left": "#456990",
    "base-right": "#456990",
    "append-prefix": "#2867A8",
    "append-suffix": "#4B8B3B",
}

EPS = 1e-10


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=EPS)


def _program_to_json(program: AbstractProgram) -> list[dict[str, Any]]:
    return abstract_pool_to_jsonable({len(program): {program: 1.0}})[str(len(program))][0]["program"]


def _ref_label(ref: dict[str, Any] | None) -> str:
    if ref is None:
        return "-"
    prefix = "G" if ref["kind"] == "group" else "S"
    return f"{prefix}{ref['scope']}:{ref['slot']}"


def _step_label(step: dict[str, Any]) -> str:
    target = step.get("target")
    if target is None:
        return f"{step['primitive']}({_ref_label(step['source'])})"
    return f"{step['primitive']}({_ref_label(step['source'])}->{_ref_label(target)})"


def _program_label(program: list[dict[str, Any]], max_steps: int = 4) -> str:
    parts = [_step_label(step) for step in program]
    if len(parts) > max_steps:
        return " -> ".join(parts[:max_steps]) + f" -> ... (+{len(parts) - max_steps})"
    return " -> ".join(parts)


def _node_id(segment: str, depth: int, index: int) -> str:
    safe = segment.replace(":", "_").replace("/", "_")
    return f"{safe}_d{depth}_{index}"


def _weights_from_metadata(raw: list[list[Any]] | None) -> dict[str, float] | None:
    if not raw:
        return None
    return {str(name): float(weight) for name, weight in raw}


def _bigram_weights_from_metadata(raw: list[list[Any]] | None) -> dict[tuple[str, str], float] | None:
    if not raw:
        return None
    return {(str(a), str(b)): float(v) for a, b, v in raw}


def _append_initial_grid_shapes(introduced: str, n_shapes: int) -> list[str] | None:
    if introduced == "fill_interior":
        return ["hollow"] * n_shapes
    return None


class LineageBuilder:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.metadata = payload["metadata"]
        self.max_depth = int(self.metadata["max_depth"])
        self.n_shapes = int(self.metadata["n_shapes"])
        self.selector_types = frozenset(str(s) for s in self.metadata["selectors"])
        self.palette = tuple(int(c) for c in self.metadata["palette"])
        self.primitive_weights = _weights_from_metadata(self.metadata.get("primitive_weights"))
        self.bigram_weights = _bigram_weights_from_metadata(self.metadata.get("bigram_weights"))
        pool_seed = self.metadata.get("pool_seed")
        self.pool_rng = np.random.default_rng(pool_seed) if pool_seed is not None else None
        self.segments: dict[str, AbstractPool] = {}
        self.nodes: list[dict[str, Any]] = []
        self.refs: dict[tuple[str, int, AbstractProgram], list[dict[str, Any]]] = {}

    def build(self) -> dict[str, Any]:
        base_primitives = [str(p) for p in self.metadata["base_primitives"]]
        base_bigram = build_bigram(base_primitives, bigram_weights=self.bigram_weights)
        self.segments["base"] = self._build_base_pool(
            base_bigram,
            base_primitives,
            pool_size=int(self.metadata["base_pool_size_per_depth"]),
        )

        unlocked_so_far: list[str] = []
        effective = set(str(p) for p in self.metadata["effective_primitives"])
        for introduced in self.metadata.get("withheld_primitives", []):
            introduced = str(introduced)
            if introduced not in effective:
                continue
            inherited_names = ("base",) + tuple(
                f"append:{primitive}" for primitive in unlocked_so_far
            )
            append_vocab = frozenset(base_primitives + unlocked_so_far + [introduced])
            self.segments[f"append:{introduced}"] = self._build_append_pool(
                inherited_names,
                introduced,
                vocab=append_vocab,
                pool_size=int(self.metadata["append_pool_size_per_depth"]),
                initial_grid_shapes=_append_initial_grid_shapes(introduced, self.n_shapes),
            )
            unlocked_so_far.append(introduced)

        rebuilt_segments = {
            name: abstract_pool_to_jsonable(pool)
            for name, pool in sorted(self.segments.items())
        }
        cache_match = rebuilt_segments == self.payload["segments"]
        return {
            "metadata": self.metadata,
            "cache_match": cache_match,
            "nodes": self.nodes,
            "edges": self._edges(),
            "segments": rebuilt_segments,
        }

    def _add_candidate(
        self,
        cands: dict[AbstractProgram, float],
        derivations: dict[AbstractProgram, list[dict[str, Any]]],
        program: AbstractProgram,
        score: float,
        derivation: dict[str, Any],
    ) -> None:
        previous = cands.get(program)
        if previous is None or score > previous + EPS:
            cands[program] = score
            derivations[program] = [derivation]
        elif _close(score, previous):
            derivations.setdefault(program, []).append(derivation)

    def _add_valid_binding_variants(
        self,
        cands: dict[AbstractProgram, float],
        derivations: dict[AbstractProgram, list[dict[str, Any]]],
        seq: tuple[str, ...],
        bigram: np.ndarray,
        primitives: list[str],
        *,
        source: str,
        attempt: int | None,
    ) -> None:
        seq_score = _primitive_sequence_score(
            seq,
            bigram,
            primitives,
            self.primitive_weights,
        )
        for abstract in _binding_variants(seq):
            if not validate_canonical(
                abstract,
                n_shapes=self.n_shapes,
                vocab=frozenset(primitives),
                selector_types=self.selector_types,
                initial_grid_shapes=None,
                palette=self.palette,
            ):
                continue
            self._add_candidate(
                cands,
                derivations,
                abstract,
                seq_score * _binding_score(abstract),
                {
                    "kind": "direct",
                    "source": source,
                    "attempt": attempt,
                    "primitive_sequence": list(seq),
                },
            )

    def _build_base_pool(
        self,
        bigram: np.ndarray,
        primitives: list[str],
        *,
        pool_size: int,
    ) -> AbstractPool:
        primitive_pool = build_pool(
            bigram,
            primitives,
            max_depth=self.max_depth,
            pool_size=max(pool_size * 16, 128),
            primitive_weights=self.primitive_weights,
            rng=self.pool_rng,
        )
        backfill_rng = self.pool_rng if self.pool_rng is not None else np.random.default_rng(0)
        pool: AbstractPool = {}
        for depth in range(1, self.max_depth + 1):
            if depth == 1:
                selected = _canonical_depth_one_pool(
                    primitives,
                    n_shapes=self.n_shapes,
                    selector_types=self.selector_types,
                    vocab=frozenset(primitives),
                    primitive_weights=self.primitive_weights,
                    initial_grid_shapes=None,
                    palette=self.palette,
                )
                derivations = {
                    program: [{
                        "kind": "direct",
                        "source": "canonical_depth_one",
                        "attempt": None,
                        "primitive_sequence": [program[0].primitive],
                    }]
                    for program in selected
                }
                pool[depth] = selected
                self._register_nodes("base", depth, selected, derivations)
                continue
            cands: dict[AbstractProgram, float] = {}
            derivations: dict[AbstractProgram, list[dict[str, Any]]] = {}
            for seq in primitive_pool.get(depth, {}):
                self._add_valid_binding_variants(
                    cands,
                    derivations,
                    seq,
                    bigram,
                    primitives,
                    source="build_pool",
                    attempt=None,
                )

            for split in range(1, depth):
                for left, left_score in pool.get(split, {}).items():
                    for right, right_score in pool.get(depth - split, {}).items():
                        composed = _with_fresh_scopes(left, right)
                        if not validate_canonical(
                            composed,
                            n_shapes=self.n_shapes,
                            vocab=frozenset(primitives),
                            selector_types=self.selector_types,
                            initial_grid_shapes=None,
                            palette=self.palette,
                        ):
                            continue
                        score = left_score * right_score * _binding_score(composed)
                        self._add_candidate(
                            cands,
                            derivations,
                            composed,
                            score,
                            {
                                "kind": "base-compose",
                                "split": split,
                                "left_parent_ids": self._parent_ids(("base",), split, left, left_score),
                                "right_parent_ids": self._parent_ids(("base",), depth - split, right, right_score),
                            },
                        )

            attempts = 0
            max_attempts = max(pool_size * 1000, 10_000)
            while len(cands) < pool_size and attempts < max_attempts:
                attempts += 1
                seq = tuple(
                    sample_bigram(
                        bigram,
                        primitives,
                        depth,
                        backfill_rng,
                        primitive_weights=self.primitive_weights,
                    )
                )
                self._add_valid_binding_variants(
                    cands,
                    derivations,
                    seq,
                    bigram,
                    primitives,
                    source="backfill_sample",
                    attempt=attempts,
                )

            selected = _pick_top_or_sample(cands, pool_size, self.pool_rng)
            pool[depth] = selected
            self._register_nodes("base", depth, selected, derivations)
        return pool

    def _build_append_pool(
        self,
        inherited_names: tuple[str, ...],
        introduced: str,
        *,
        vocab: frozenset[str],
        pool_size: int,
        initial_grid_shapes: list[str] | None,
    ) -> AbstractPool:
        pool: AbstractPool = {}
        introduced_variants = _introduced_step_variants(introduced)
        empty = {(): 1.0}
        inherited_pool = self._merge_segments(inherited_names)
        for depth in range(1, self.max_depth + 1):
            if depth == 1:
                selected = _canonical_depth_one_pool(
                    [introduced],
                    n_shapes=self.n_shapes,
                    selector_types=self.selector_types,
                    vocab=vocab,
                    primitive_weights=None,
                    initial_grid_shapes=initial_grid_shapes,
                    palette=self.palette,
                    fill_initial_grid_shapes=True,
                )
                derivations = {
                    program: [{
                        "kind": "append",
                        "introduced_primitive": introduced,
                        "left_depth": 0,
                        "right_depth": 0,
                        "left_parent_ids": [],
                        "right_parent_ids": [],
                        "introduced_variant": _program_to_json(program),
                    }]
                    for program in selected
                }
                pool[depth] = selected
                self._register_nodes(f"append:{introduced}", depth, selected, derivations)
                continue
            cands: dict[AbstractProgram, float] = {}
            derivations: dict[AbstractProgram, list[dict[str, Any]]] = {}
            for left_depth in range(depth):
                right_depth = depth - left_depth - 1
                left_items = empty if left_depth == 0 else inherited_pool.get(left_depth, {})
                right_items = empty if right_depth == 0 else inherited_pool.get(right_depth, {})
                if not left_items or not right_items:
                    continue
                for left, left_score in left_items.items():
                    for intro in introduced_variants:
                        intro_shifted = _shift_scopes(intro, _scope_offset(left))
                        prefix = left + intro_shifted
                        for right, right_score in right_items.items():
                            candidate = _with_fresh_scopes(prefix, right)
                            if _count_primitive(candidate, introduced) != 1:
                                continue
                            candidate_initial_grid_shapes = _fill_initial_grid_shapes(
                                candidate,
                                self.n_shapes,
                                initial_grid_shapes,
                            )
                            if (
                                candidate_initial_grid_shapes is not None
                                and len(candidate_initial_grid_shapes) != self.n_shapes
                            ):
                                continue
                            if not validate_canonical(
                                candidate,
                                n_shapes=self.n_shapes,
                                vocab=vocab,
                                selector_types=self.selector_types,
                                initial_grid_shapes=candidate_initial_grid_shapes,
                                palette=self.palette,
                            ):
                                continue
                            score = float(left_score) * float(right_score) * _binding_score(candidate)
                            self._add_candidate(
                                cands,
                                derivations,
                                candidate,
                                score,
                                {
                                    "kind": "append",
                                    "introduced_primitive": introduced,
                                    "left_depth": left_depth,
                                    "right_depth": right_depth,
                                    "left_parent_ids": self._parent_ids(
                                        inherited_names, left_depth, left, left_score,
                                    ),
                                    "right_parent_ids": self._parent_ids(
                                        inherited_names, right_depth, right, right_score,
                                    ),
                                    "introduced_variant": _program_to_json(intro),
                                },
                            )
            selected = _pick_top_or_sample(cands, pool_size, self.pool_rng)
            pool[depth] = selected
            self._register_nodes(f"append:{introduced}", depth, selected, derivations)
        return pool

    def _merge_segments(self, names: tuple[str, ...]) -> AbstractPool:
        merged: AbstractPool = {}
        for name in names:
            for depth, entries in self.segments.get(name, {}).items():
                bucket = merged.setdefault(depth, {})
                for program, score in entries.items():
                    bucket[program] = max(bucket.get(program, 0.0), score)
        return merged

    def _parent_ids(
        self,
        names: tuple[str, ...],
        depth: int,
        program: AbstractProgram,
        score: float,
    ) -> list[str]:
        if depth == 0 or not program:
            return []
        ids: list[str] = []
        for name in names:
            for ref in self.refs.get((name, depth, program), []):
                if _close(float(ref["score"]), float(score)):
                    ids.append(str(ref["id"]))
        return ids

    def _register_nodes(
        self,
        segment: str,
        depth: int,
        selected: dict[AbstractProgram, float],
        derivations: dict[AbstractProgram, list[dict[str, Any]]],
    ) -> None:
        for index, (program, score) in enumerate(selected.items(), start=1):
            node = {
                "id": _node_id(segment, depth, index),
                "segment": segment,
                "depth": depth,
                "index": index,
                "score": float(score),
                "program": _program_to_json(program),
                "derivations": derivations.get(program, []),
            }
            self.nodes.append(node)
            self.refs.setdefault((segment, depth, program), []).append(node)

    def _edges(self) -> list[dict[str, Any]]:
        edges: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for node in self.nodes:
            for derivation in node["derivations"]:
                kind = derivation["kind"]
                if kind == "base-compose":
                    roles = [
                        ("base-left", derivation.get("left_parent_ids", [])),
                        ("base-right", derivation.get("right_parent_ids", [])),
                    ]
                elif kind == "append":
                    roles = [
                        ("append-prefix", derivation.get("left_parent_ids", [])),
                        ("append-suffix", derivation.get("right_parent_ids", [])),
                    ]
                else:
                    roles = []
                for role, parent_ids in roles:
                    for parent_id in parent_ids:
                        key = (parent_id, node["id"], role)
                        if key in seen:
                            continue
                        seen.add(key)
                        edges.append({
                            "source": parent_id,
                            "target": node["id"],
                            "role": role,
                            "construction": kind,
                        })
        return edges


class CachedFormulaLineageBuilder:
    """Replay construction formulas over already-selected cached entries."""

    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.metadata = payload["metadata"]
        self.segments = {
            name: abstract_pool_from_jsonable(raw)
            for name, raw in payload["segments"].items()
        }
        self.nodes: list[dict[str, Any]] = []
        self.refs: dict[tuple[str, int, AbstractProgram], list[dict[str, Any]]] = {}

    def build(self) -> dict[str, Any]:
        for segment in self._segment_order():
            for depth in sorted(self.segments.get(segment, {})):
                self._register_nodes(segment, depth, self.segments[segment][depth])
        for node in self.nodes:
            program = self._program_for_node(node)
            if node["segment"] == "base":
                node["derivations"] = self._base_derivations(
                    int(node["depth"]), program, float(node["score"]),
                )
            elif str(node["segment"]).startswith("append:"):
                node["derivations"] = self._append_derivations(
                    str(node["segment"]), int(node["depth"]), program, float(node["score"]),
                )
            else:
                node["derivations"] = []
        return {
            "metadata": self.metadata,
            "cache_match": True,
            "provenance_mode": "cached_formula_replay",
            "nodes": self.nodes,
            "edges": self._edges(),
            "segments": self.payload["segments"],
        }

    def _segment_order(self) -> list[str]:
        order = ["base"]
        effective = set(str(p) for p in self.metadata["effective_primitives"])
        for primitive in self.metadata.get("withheld_primitives", []):
            primitive = str(primitive)
            if primitive in effective:
                order.append(f"append:{primitive}")
        order += sorted(set(self.segments) - set(order))
        return [segment for segment in order if segment in self.segments]

    def _register_nodes(
        self,
        segment: str,
        depth: int,
        selected: dict[AbstractProgram, float],
    ) -> None:
        for index, (program, score) in enumerate(selected.items(), start=1):
            node = {
                "id": _node_id(segment, depth, index),
                "segment": segment,
                "depth": depth,
                "index": index,
                "score": float(score),
                "program": _program_to_json(program),
                "derivations": [],
            }
            self.nodes.append(node)
            self.refs.setdefault((segment, depth, program), []).append(node)

    def _program_for_node(self, node: dict[str, Any]) -> AbstractProgram:
        segment = str(node["segment"])
        depth = int(node["depth"])
        index = int(node["index"])
        return list(self.segments[segment][depth])[index - 1]

    def _base_derivations(
        self,
        depth: int,
        program: AbstractProgram,
        score: float,
    ) -> list[dict[str, Any]]:
        derivations: list[dict[str, Any]] = []
        if depth <= 1:
            return derivations
        base = self.segments.get("base", {})
        for split in range(1, depth):
            for left, left_score in base.get(split, {}).items():
                for right, right_score in base.get(depth - split, {}).items():
                    candidate = _with_fresh_scopes(left, right)
                    if candidate != program:
                        continue
                    candidate_score = left_score * right_score * _binding_score(candidate)
                    if not _close(candidate_score, score):
                        continue
                    derivations.append({
                        "kind": "base-compose",
                        "split": split,
                        "left_parent_ids": self._parent_ids(("base",), split, left, left_score),
                        "right_parent_ids": self._parent_ids(("base",), depth - split, right, right_score),
                    })
        return derivations

    def _append_derivations(
        self,
        segment: str,
        depth: int,
        program: AbstractProgram,
        score: float,
    ) -> list[dict[str, Any]]:
        introduced = segment.split(":", 1)[1]
        inherited_names = tuple(
            candidate for candidate in self._segment_order()
            if candidate == "base" or (
                candidate.startswith("append:")
                and self._segment_order().index(candidate) < self._segment_order().index(segment)
            )
        )
        inherited_pool = self._merge_segments(inherited_names)
        introduced_variants = _introduced_step_variants(introduced)
        empty = {(): 1.0}
        derivations: list[dict[str, Any]] = []
        for left_depth in range(depth):
            right_depth = depth - left_depth - 1
            left_items = empty if left_depth == 0 else inherited_pool.get(left_depth, {})
            right_items = empty if right_depth == 0 else inherited_pool.get(right_depth, {})
            for left, left_score in left_items.items():
                for intro in introduced_variants:
                    intro_shifted = _shift_scopes(intro, _scope_offset(left))
                    prefix = left + intro_shifted
                    for right, right_score in right_items.items():
                        candidate = _with_fresh_scopes(prefix, right)
                        if candidate != program:
                            continue
                        candidate_score = float(left_score) * float(right_score) * _binding_score(candidate)
                        if not _close(candidate_score, score):
                            continue
                        derivations.append({
                            "kind": "append",
                            "introduced_primitive": introduced,
                            "left_depth": left_depth,
                            "right_depth": right_depth,
                            "left_parent_ids": self._parent_ids(
                                inherited_names, left_depth, left, left_score,
                            ),
                            "right_parent_ids": self._parent_ids(
                                inherited_names, right_depth, right, right_score,
                            ),
                            "introduced_variant": _program_to_json(intro),
                        })
        return derivations

    def _merge_segments(self, names: tuple[str, ...]) -> AbstractPool:
        merged: AbstractPool = {}
        for name in names:
            for depth, entries in self.segments.get(name, {}).items():
                bucket = merged.setdefault(depth, {})
                for program, score in entries.items():
                    bucket[program] = max(bucket.get(program, 0.0), score)
        return merged

    def _parent_ids(
        self,
        names: tuple[str, ...],
        depth: int,
        program: AbstractProgram,
        score: float,
    ) -> list[str]:
        if depth == 0 or not program:
            return []
        ids: list[str] = []
        for name in names:
            for ref in self.refs.get((name, depth, program), []):
                if _close(float(ref["score"]), float(score)):
                    ids.append(str(ref["id"]))
        return ids

    def _edges(self) -> list[dict[str, Any]]:
        return LineageBuilder._edges(self)  # type: ignore[arg-type]


def _render_html(lineage: dict[str, Any], out: Path) -> None:
    nodes = lineage["nodes"]
    edges = lineage["edges"]
    depths = sorted({int(n["depth"]) for n in nodes})
    segments: list[str] = []
    for node in nodes:
        if node["segment"] not in segments:
            segments.append(node["segment"])

    x_gap = 250
    row_gap = 42
    segment_gap = 52
    margin_x = 180
    margin_y = 96
    width = margin_x * 2 + x_gap * max(0, len(depths) - 1) + 260

    positions: dict[str, tuple[int, int]] = {}
    y = margin_y
    for segment in segments:
        max_rows = max(
            len([n for n in nodes if n["segment"] == segment and n["depth"] == depth])
            for depth in depths
        )
        for depth_i, depth in enumerate(depths):
            bucket = [n for n in nodes if n["segment"] == segment and n["depth"] == depth]
            for row, node in enumerate(bucket):
                positions[node["id"]] = (margin_x + depth_i * x_gap, y + row * row_gap)
        y += max_rows * row_gap + segment_gap
    height = y + margin_y

    edge_svg = []
    for edge in edges:
        if edge["source"] not in positions or edge["target"] not in positions:
            continue
        x1, y1 = positions[edge["source"]]
        x2, y2 = positions[edge["target"]]
        role = edge["role"]
        color = EDGE_COLORS.get(role, "#555")
        dash = ' stroke-dasharray="5 4"' if role.endswith("suffix") or role.endswith("right") else ""
        title = f"{role} via {edge['construction']}"
        edge_svg.append(
            f'<path d="M{x1 + 78},{y1} C{x1 + 140},{y1} {x2 - 140},{y2} {x2 - 78},{y2}" '
            f'stroke="{color}" class="edge"{dash}><title>{html.escape(title)}</title></path>'
        )

    node_svg = []
    for node in nodes:
        x, y0 = positions[node["id"]]
        color = SEGMENT_COLORS.get(node["segment"], "#666")
        short = f"{node['segment'].replace('append:', '+')} #{node['index']}"
        label = _program_label(node["program"], max_steps=2)
        derivation_summary = ", ".join(
            str(d["kind"]) for d in node.get("derivations", [])[:4]
        )
        if len(node.get("derivations", [])) > 4:
            derivation_summary += f", +{len(node['derivations']) - 4}"
        title = (
            f"{node['segment']} depth {node['depth']} #{node['index']}\n"
            f"score={node['score']:.6g}\n"
            f"derivations={derivation_summary or 'none'}\n"
            + " -> ".join(_step_label(step) for step in node["program"])
        )
        node_svg.append(
            f'<g class="node" transform="translate({x},{y0})">'
            f"<title>{html.escape(title)}</title>"
            f'<rect x="-78" y="-15" width="156" height="30" rx="6" fill="{color}"></rect>'
            f'<text y="-2">{html.escape(short)}</text>'
            f'<text y="11" class="tiny">{html.escape(label[:32])}</text>'
            "</g>"
        )

    depth_labels = [
        f'<text class="depth" x="{margin_x + i * x_gap}" y="42">depth {depth}</text>'
        for i, depth in enumerate(depths)
    ]
    segment_labels = []
    for segment in segments:
        ys = [positions[n["id"]][1] for n in nodes if n["segment"] == segment]
        if ys:
            segment_labels.append(
                f'<text class="segment" x="18" y="{min(ys) + 5}">{html.escape(segment)}</text>'
            )

    metadata = lineage.get("metadata", {})
    summary = (
        f"suite={metadata.get('suite')} level={metadata.get('level')} "
        f"introduced={metadata.get('introduced_primitive')} pool_seed={metadata.get('pool_seed')} "
        f"mode={lineage.get('provenance_mode', 'full_rebuild')} "
        f"cache_match={lineage.get('cache_match')}"
    )
    legend_items = "".join(
        f'<span><i style="background:{SEGMENT_COLORS.get(seg, "#666")}"></i>{html.escape(seg)}</span>'
        for seg in segments
    )
    edge_legend = "".join(
        f'<span><i style="background:{color}"></i>{html.escape(role)}</span>'
        for role, color in EDGE_COLORS.items()
    )
    html_text = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Dominant Program Lineage</title>
<style>
body {{ margin: 0; font-family: Inter, system-ui, sans-serif; color: #222; background: #f7f7f4; }}
header {{ position: sticky; top: 0; z-index: 2; background: #fff; border-bottom: 1px solid #ddd; padding: 12px 18px; }}
h1 {{ margin: 0 0 4px; font-size: 18px; }}
p {{ margin: 4px 0; font-size: 13px; }}
.legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 8px; font-size: 12px; }}
.legend i {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 5px; vertical-align: -2px; }}
.canvas {{ overflow: auto; height: calc(100vh - 116px); }}
svg {{ background: #fbfbf8; }}
.edge {{ fill: none; stroke-width: 1.2; opacity: 0.48; }}
.node text {{ fill: white; text-anchor: middle; font-size: 10px; pointer-events: none; font-weight: 650; }}
.node .tiny {{ font-size: 7px; font-weight: 500; opacity: 0.92; }}
.depth {{ font-size: 13px; font-weight: 700; text-anchor: middle; fill: #333; }}
.segment {{ font-size: 12px; font-weight: 700; fill: #333; }}
</style>
</head>
<body>
<header>
<h1>Dominant Program Lineage</h1>
<p>{html.escape(summary)}</p>
<p>Edges replay the actual dominant-library construction formula: base composition uses selected smaller base chunks; append nodes use inherited prefix/suffix chunks around the introduced primitive. Use --full-rebuild to regenerate candidates and record provenance during pool construction.</p>
<div class="legend">{legend_items}</div>
<div class="legend">{edge_legend}</div>
</header>
<div class="canvas">
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
{''.join(depth_labels)}
{''.join(segment_labels)}
{''.join(edge_svg)}
{''.join(node_svg)}
</svg>
</div>
</body>
</html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("library", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/dominant_lineage.html"))
    parser.add_argument("--lineage-out", type=Path)
    parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help="Regenerate dominant candidates and record max-score provenance. Slower.",
    )
    args = parser.parse_args()

    payload = json.loads(args.library.read_text())
    lineage = (
        LineageBuilder(payload).build()
        if args.full_rebuild
        else CachedFormulaLineageBuilder(payload).build()
    )
    _render_html(lineage, args.out)
    if args.lineage_out is not None:
        args.lineage_out.parent.mkdir(parents=True, exist_ok=True)
        args.lineage_out.write_text(json.dumps(lineage, indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    if args.lineage_out is not None:
        print(f"wrote {args.lineage_out}")
    print(
        f"nodes={len(lineage['nodes'])} edges={len(lineage['edges'])} "
        f"cache_match={lineage['cache_match']}"
    )


if __name__ == "__main__":
    main()
