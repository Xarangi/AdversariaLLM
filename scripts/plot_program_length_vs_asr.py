#!/usr/bin/env python3
"""Generate program-length vs ASR plots from April scoring summaries.

This script combines:
- Adversarial attack ASRs from outputs/*/scoring/asr_*_strong_reject.json (asr_by_model rows)
- Training/baseline rows from final-results.jsonl

It produces three plots:
1) OLMo-3-7B only (all methods)
2) Qwen3-4B vs OLMo-3-7B vs Llama-3-8B vs SmolLM3-Safe
3) SmolLM3-Safe vs SmolLM3-Unsafe (adversarial-only)

It also exports merged points and explicit bit assumptions.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D


ATTACKS = ("gcg", "autodan", "pair", "pgd")
DATE_GLOB = "2026-04-1*"

TITLE_FS = 24
LABEL_FS = 20
TICK_FS = 16
LEGEND_FS = 16
ANNOTATION_FS = 12

MODEL_QWEN = "Qwen3-4B"
MODEL_OLMO = "OLMo-3-7B-Instruct"
MODEL_LLAMA = "Llama-3-8B-Instruct"
MODEL_SMOL_SAFE = "SmolLM3-Safe"
MODEL_SMOL_UNSAFE = "SmolLM3-Unsafe"

MODEL_ORDER = [MODEL_QWEN, MODEL_OLMO, MODEL_LLAMA, MODEL_SMOL_SAFE, MODEL_SMOL_UNSAFE]

MODEL_COLORS = {
    MODEL_QWEN: "#F28E2B",
    MODEL_OLMO: "#8C564B",
    MODEL_LLAMA: "#9467BD",
    MODEL_SMOL_SAFE: "#1F77B4",
    MODEL_SMOL_UNSAFE: "#D62728",
}

METHOD_COLORS = {
    "Baseline": "#7F7F7F",
    "Roleplay": "#FF7F0E",
    "FT": "#1F77B4",
    "GCG": "#2CA02C",
    "AutoDAN": "#D62728",
    "PAIR": "#8C564B",
    "PGD": "#9467BD",
}

METHOD_LABEL = {
    "gcg": "GCG",
    "autodan": "AutoDAN",
    "pair": "PAIR",
    "pgd": "PGD",
}


@dataclass
class AttackBits:
    total_bits: int
    components: dict[str, int]
    assumptions: dict[str, Any]


def canonical_model(name: str) -> str | None:
    v = name.lower()
    if "qwen3-4b" in v:
        return MODEL_QWEN
    if "llama3_8b_instruct_local" in v or "meta-llama-3-8b-instruct" in v:
        return MODEL_LLAMA
    if "olmo3-7b" in v or "olmo-3-7b" in v:
        return MODEL_OLMO
    if "smollm3-safety-pair-safe" in v or "safety-pair-safe" in v:
        return MODEL_SMOL_SAFE
    if "smollm3-safety-pair-unsafe" in v or "safety-pair-unsafe" in v:
        return MODEL_SMOL_UNSAFE
    return None


def parse_autodan_initial_strings_bits(autodan_file: Path) -> tuple[int, int]:
    source = autodan_file.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "INITIAL_STRINGS":
                    if isinstance(node.value, ast.List):
                        strings = []
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                strings.append(elt.value)
                        if strings:
                            avg_bits = int(round(sum(len(s.encode("utf-8")) * 8 for s in strings) / len(strings)))
                            return avg_bits, len(strings)
    # Fallback if parsing fails
    return 0, 0


def class_code_bits(py_file: Path, class_name: str) -> int:
    source = py_file.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            seg = "".join(lines[node.lineno - 1:node.end_lineno])
            return len(seg.encode("utf-8")) * 8
    return 0


def compute_attack_bits(repo_root: Path) -> dict[str, AttackBits]:
    attack_cfg_path = repo_root / "conf" / "attacks" / "attacks.yaml"
    cfg = yaml.safe_load(attack_cfg_path.read_text(encoding="utf-8"))

    gcg_cfg = cfg["gcg"]
    autodan_cfg = cfg["autodan"]
    pair_cfg = cfg["pair"]
    pgd_cfg = cfg["pgd"]

    code_paths = {
        "gcg": repo_root / "adversariallm" / "attacks" / "gcg.py",
        "autodan": repo_root / "adversariallm" / "attacks" / "autodan.py",
        "pair": repo_root / "adversariallm" / "attacks" / "pair.py",
        "pgd": repo_root / "adversariallm" / "attacks" / "pgd.py",
    }

    class_by_attack = {
        "gcg": "GCGAttack",
        "autodan": "AutoDANAttack",
        "pair": "PAIRAttack",
        "pgd": "PGDAttack",
    }
    code_bits: dict[str, int] = {}
    for atk, p in code_paths.items():
        bits = class_code_bits(p, class_by_attack[atk])
        if bits <= 0:
            bits = int(p.stat().st_size * 8)
        code_bits[atk] = bits

    token_id_bits = 16
    embedding_dim_assumption = 4096
    embedding_precision_bits = 16
    helper_model_params_pair = 13_000_000_000
    helper_precision_bits = 16

    autodan_avg_prefix_bits, autodan_prefix_count = parse_autodan_initial_strings_bits(code_paths["autodan"])

    # GCG
    gcg_suffix_tokens = len(str(gcg_cfg["optim_str_init"]).split())
    gcg_search_bits = int(round(gcg_cfg["num_steps"] * gcg_suffix_tokens * math.log2(max(int(gcg_cfg["topk"]), 2))))
    gcg_payload_bits = gcg_suffix_tokens * token_id_bits
    gcg_meta_bits = int(gcg_cfg["num_steps"] * token_id_bits)
    gcg_total = code_bits["gcg"] + gcg_search_bits + gcg_payload_bits + gcg_meta_bits

    # AutoDAN
    autodan_steps = int(autodan_cfg["num_steps"])
    autodan_batch = int(autodan_cfg["batch_size"])
    autodan_evolution_bits = int(round(autodan_steps * autodan_batch * math.log2(max(autodan_batch, 2))))
    autodan_mutation_bits = int(round(autodan_steps * autodan_batch * float(autodan_cfg["mutation"]) * token_id_bits))
    autodan_total = (
        code_bits["autodan"]
        + autodan_avg_prefix_bits
        + autodan_evolution_bits
        + autodan_mutation_bits
    )

    # PAIR
    pair_steps = int(pair_cfg["num_steps"])
    pair_streams = int(pair_cfg["num_streams"])
    pair_max_new_tokens = int(pair_cfg["attack_model"]["max_new_tokens"])
    pair_max_attempts = int(pair_cfg["attack_model"]["max_attempts"])
    pair_payload_bits = pair_steps * pair_streams * pair_max_new_tokens * token_id_bits
    pair_search_bits = int(round(pair_steps * pair_streams * pair_max_attempts * math.log2(max(pair_max_new_tokens, 2))))
    pair_helper_bits = helper_model_params_pair * helper_precision_bits
    pair_total = code_bits["pair"] + pair_payload_bits + pair_search_bits + pair_helper_bits

    # PGD
    pgd_soft_tokens = len(str(pgd_cfg["optim_str_init"]).split())
    pgd_steps = int(pgd_cfg["num_steps"])
    pgd_soft_payload_bits = pgd_soft_tokens * embedding_dim_assumption * embedding_precision_bits
    pgd_state_bits = int(round(pgd_steps * pgd_soft_tokens * math.log2(max(embedding_dim_assumption, 2))))
    pgd_total = code_bits["pgd"] + pgd_soft_payload_bits + pgd_state_bits

    return {
        "gcg": AttackBits(
            total_bits=gcg_total,
            components={
                "code_bits": code_bits["gcg"],
                "payload_bits": gcg_payload_bits,
                "search_bits": gcg_search_bits,
                "meta_bits": gcg_meta_bits,
            },
            assumptions={
                "suffix_tokens": gcg_suffix_tokens,
                "token_id_bits": token_id_bits,
                "code_bits_policy": "attack_class_only",
                "search_formula": "num_steps * suffix_tokens * log2(topk)",
                "num_steps": int(gcg_cfg["num_steps"]),
                "topk": int(gcg_cfg["topk"]),
            },
        ),
        "autodan": AttackBits(
            total_bits=autodan_total,
            components={
                "code_bits": code_bits["autodan"],
                "prefix_bits": autodan_avg_prefix_bits,
                "evolution_bits": autodan_evolution_bits,
                "mutation_bits": autodan_mutation_bits,
            },
            assumptions={
                "avg_initial_prefix_bits": autodan_avg_prefix_bits,
                "initial_prefix_count": autodan_prefix_count,
                "code_bits_policy": "attack_class_only",
                "num_steps": autodan_steps,
                "batch_size": autodan_batch,
                "mutation": float(autodan_cfg["mutation"]),
                "token_id_bits": token_id_bits,
                "evolution_formula": "num_steps * batch_size * log2(batch_size)",
            },
        ),
        "pair": AttackBits(
            total_bits=pair_total,
            components={
                "code_bits": code_bits["pair"],
                "helper_model_bits": pair_helper_bits,
                "payload_bits": pair_payload_bits,
                "search_bits": pair_search_bits,
            },
            assumptions={
                "helper_model": "lmsys/vicuna-13b-v1.5",
                "helper_model_params": helper_model_params_pair,
                "helper_precision_bits": helper_precision_bits,
                "code_bits_policy": "attack_class_only",
                "num_steps": pair_steps,
                "num_streams": pair_streams,
                "max_new_tokens": pair_max_new_tokens,
                "max_attempts": pair_max_attempts,
                "token_id_bits": token_id_bits,
            },
        ),
        "pgd": AttackBits(
            total_bits=pgd_total,
            components={
                "code_bits": code_bits["pgd"],
                "soft_payload_bits": pgd_soft_payload_bits,
                "state_bits": pgd_state_bits,
            },
            assumptions={
                "soft_tokens": pgd_soft_tokens,
                "embedding_dim_assumption": embedding_dim_assumption,
                "embedding_precision_bits": embedding_precision_bits,
                "code_bits_policy": "attack_class_only",
                "num_steps": pgd_steps,
                "state_formula": "num_steps * soft_tokens * log2(embedding_dim)",
            },
        ),
    }


def load_adversarial_rows(repo_root: Path, attack_bits: dict[str, AttackBits]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    asr_files = sorted(repo_root.glob(f"outputs/{DATE_GLOB}/scoring/asr_*_strong_reject.json"))

    for path in asr_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        attack = str(data.get("attack", "")).lower()
        if attack not in ATTACKS:
            continue

        for row in data.get("asr_by_model", []):
            model_name = row.get("model") or row.get("model_id") or ""
            model_label = canonical_model(str(model_name))
            if model_label is None:
                continue
            completed = int(row.get("completed_instances") or 0)
            asr = row.get("asr")
            if completed <= 0 or not isinstance(asr, (int, float)):
                continue

            key = (model_label, attack)
            candidate = {
                "model": model_label,
                "method": METHOD_LABEL[attack],
                "attack": attack,
                "bits": attack_bits[attack].total_bits,
                "asr": float(asr),
                "completed_instances": completed,
                "source": "adversarial",
                "source_file": str(path.relative_to(repo_root)),
            }
            current = selected.get(key)
            if current is None:
                selected[key] = candidate
                continue

            if candidate["completed_instances"] > current["completed_instances"]:
                selected[key] = candidate
            elif candidate["completed_instances"] == current["completed_instances"] and candidate["asr"] > current["asr"]:
                selected[key] = candidate

    return list(selected.values())


def training_method_label(row: dict[str, Any]) -> str:
    exp = row.get("experiment_name")
    if exp == "BaselineStrategy":
        return "Baseline"
    if exp == "RoleplayStrategy":
        return "Roleplay"
    if exp == "DataStrategy":
        n = row.get("max_examples")
        if n is None:
            return "FT (n=all)"
        return f"FT (n={n})"
    return str(exp)


def row_sample_score(row: dict[str, Any]) -> int:
    n = row.get("max_examples")
    if isinstance(n, int):
        return n
    return 1_000_000_000


def load_training_rows(final_results_file: Path) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str, float], dict[str, Any]] = {}

    with final_results_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            if row.get("dataset_name") != "advbench_harmbench":
                continue
            asr = row.get("asr")
            bits = row.get("bits")
            if not isinstance(asr, (int, float)) or not isinstance(bits, (int, float)):
                continue

            model_label = canonical_model(str(row.get("model_name", "")))
            if model_label is None:
                continue

            method = training_method_label(row)
            key = (model_label, method, float(bits))
            candidate = {
                "model": model_label,
                "method": method,
                "attack": None,
                "bits": float(bits),
                "asr": float(asr),
                "completed_instances": row_sample_score(row),
                "source": "training",
                "source_file": str(final_results_file),
            }

            current = selected.get(key)
            if current is None:
                selected[key] = candidate
                continue

            if candidate["completed_instances"] > current["completed_instances"]:
                selected[key] = candidate
            elif candidate["completed_instances"] == current["completed_instances"] and candidate["asr"] > current["asr"]:
                selected[key] = candidate

    return list(selected.values())


def asr_percent(v: float) -> float:
    return v * 100.0


def bits_for_plot(bits: float) -> float:
    # Log-scale plots cannot represent zero; clamp baseline at 1 bit for display.
    return max(bits, 1.0)


def sorted_line_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(points, key=lambda p: (bits_for_plot(float(p["bits"])), asr_percent(float(p["asr"])), p["method"]))


def running_max_series(points: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    if not points:
        return np.array([]), np.array([])
    ordered = sorted_line_points(points)
    xs = np.array([bits_for_plot(float(p["bits"])) for p in ordered], dtype=float)
    ys = np.array([asr_percent(float(p["asr"])) for p in ordered], dtype=float)
    return xs, np.maximum.accumulate(ys)


def method_family(method: str) -> str:
    if method.startswith("FT "):
        return "FT"
    return method


def method_color(method: str) -> str:
    return METHOD_COLORS.get(method_family(method), "#444444")


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": TITLE_FS,
            "axes.labelsize": LABEL_FS,
            "xtick.labelsize": TICK_FS,
            "ytick.labelsize": TICK_FS,
            "legend.fontsize": LEGEND_FS,
        }
    )


def running_max_by_method(points: list[dict[str, Any]]) -> list[tuple[float, float, str]]:
    ordered = sorted_line_points(points)
    out: list[tuple[float, float, str]] = []
    best = -1.0
    dominant = "Baseline"
    for p in ordered:
        x = bits_for_plot(float(p["bits"]))
        y = asr_percent(float(p["asr"]))
        if y > best:
            best = y
            dominant = method_family(str(p["method"]))
        out.append((x, best, dominant))
    return out


def draw_colored_running_max(ax: Any, points: list[dict[str, Any]], line_width: float = 2.8) -> None:
    series = running_max_by_method(points)
    if len(series) < 2:
        return
    for i in range(len(series) - 1):
        x1, y1, m1 = series[i]
        x2, y2, m2 = series[i + 1]
        ax.plot([x1, x2], [y1, y1], color=method_color(m1), linewidth=line_width)
        if y2 > y1:
            ax.plot([x2, x2], [y1, y2], color=method_color(m2), linewidth=line_width)


def annotate_points(ax: Any, points: list[dict[str, Any]]) -> None:
    for p in points:
        x = bits_for_plot(float(p["bits"]))
        y = asr_percent(float(p["asr"]))
        ax.annotate(
            p["method"],
            (x, y),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=ANNOTATION_FS,
            alpha=0.85,
        )


def plot_olmo(points: list[dict[str, Any]], output_file: Path) -> None:
    rows = [p for p in points if p["model"] == MODEL_OLMO]
    if not rows:
        return

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale("log")

    for p in rows:
        x = bits_for_plot(float(p["bits"]))
        y = asr_percent(float(p["asr"]))
        color = method_color(str(p["method"]))
        marker = "s" if p["source"] == "adversarial" else "o"
        ax.scatter(x, y, s=70, marker=marker, color=color, edgecolors="black", linewidths=0.5, alpha=0.8)

    annotate_points(ax, rows)

    draw_colored_running_max(ax, rows, line_width=2.8)

    legend_order = ["Baseline", "Roleplay", "FT", "GCG", "AutoDAN", "PAIR", "PGD"]
    present = {method_family(str(p["method"])) for p in rows}
    handles = []
    handles.extend(
        Line2D([0], [0], color=METHOD_COLORS[m], lw=2.6, label=m)
        for m in legend_order
        if m in present
    )

    ax.set_title("OLMo-3-7B-Instruct: Program Length vs ASR")
    ax.set_xlabel("Program Length (bits)")
    ax.set_ylabel("ASR (%)")
    ax.grid(alpha=0.3)
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=LEGEND_FS)
    fig.tight_layout()
    fig.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_four_models(points: list[dict[str, Any]], output_file: Path) -> None:
    keep = {MODEL_QWEN, MODEL_OLMO, MODEL_LLAMA, MODEL_SMOL_SAFE}
    rows = [p for p in points if p["model"] in keep]
    if not rows:
        return

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale("log")

    for model in [MODEL_LLAMA, MODEL_OLMO, MODEL_QWEN, MODEL_SMOL_SAFE]:
        m_rows = [p for p in rows if p["model"] == model]
        if not m_rows:
            continue
        color = MODEL_COLORS[model]

        for p in m_rows:
            x = bits_for_plot(float(p["bits"]))
            y = asr_percent(float(p["asr"]))
            marker = "s" if p["source"] == "adversarial" else "o"
            ax.scatter(x, y, s=58, marker=marker, color=color, edgecolors="black", linewidths=0.4, alpha=0.45)

        tx, ty = running_max_series(m_rows)
        if len(tx) > 0:
            ax.step(tx, ty, where="post", color=color, linewidth=2.5, label=model)

    ax.set_title("Program Length vs ASR Across Models")
    ax.set_xlabel("Program Length (bits)")
    ax.set_ylabel("ASR (%)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=LEGEND_FS)
    fig.tight_layout()
    fig.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_smol_safe_unsafe_adversarial(points: list[dict[str, Any]], output_file: Path) -> None:
    rows = [p for p in points if p["model"] in {MODEL_SMOL_SAFE, MODEL_SMOL_UNSAFE}]
    if not rows:
        return

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale("log")

    for model in [MODEL_SMOL_SAFE, MODEL_SMOL_UNSAFE]:
        m_rows = [p for p in rows if p["model"] == model]
        if not m_rows:
            continue
        color = MODEL_COLORS[model]

        for p in m_rows:
            x = bits_for_plot(float(p["bits"]))
            y = asr_percent(float(p["asr"]))
            marker = "D" if p["source"] == "adversarial" else "o"
            alpha = 0.85 if p["source"] == "adversarial" else 0.5
            ax.scatter(x, y, s=82, marker=marker, color=color, edgecolors="black", linewidths=0.5, alpha=alpha)
            ax.annotate(p["method"], (x, y), textcoords="offset points", xytext=(4, 4), fontsize=ANNOTATION_FS)

        tx, ty = running_max_series(m_rows)
        if len(tx) > 0:
            ax.step(tx, ty, where="post", color=color, linewidth=2.8, label=model)

    ax.set_title("SmolLM3 Safe vs Unsafe (Adversarial Attacks)")
    ax.set_xlabel("Program Length (bits)")
    ax.set_ylabel("ASR (%)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_rows_json_csv(rows: list[dict[str, Any]], out_dir: Path) -> None:
    rows = sorted(rows, key=lambda r: (MODEL_ORDER.index(r["model"]), bits_for_plot(float(r["bits"])), r["method"]))
    json_path = out_dir / "program_length_asr_points.json"
    csv_path = out_dir / "program_length_asr_points.csv"

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model", "method", "attack", "source", "bits", "bits_plot", "asr", "asr_percent", "completed_instances", "source_file"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "model": r["model"],
                    "method": r["method"],
                    "attack": r.get("attack"),
                    "source": r["source"],
                    "bits": r["bits"],
                    "bits_plot": bits_for_plot(float(r["bits"])),
                    "asr": r["asr"],
                    "asr_percent": asr_percent(float(r["asr"])),
                    "completed_instances": r["completed_instances"],
                    "source_file": r["source_file"],
                }
            )


def write_assumptions(attack_bits: dict[str, AttackBits], out_dir: Path) -> None:
    payload = {
        "global": {
            "policy": "Include all modeled components; PAIR includes helper-model parameter bits; PGD includes soft-token embedding payload bits.",
            "baseline_bits_plot_clamp": "0-bit methods are clamped to 1 bit for log-scale plotting only.",
            "duplicate_selection": "Prefer larger completed_instances/sample coverage; ties broken by higher ASR.",
            "ignored_attacks": ["crescendo"],
        },
        "attacks": {
            atk: {
                "total_bits": b.total_bits,
                "components": b.components,
                "assumptions": b.assumptions,
            }
            for atk, b in attack_bits.items()
        },
    }
    (out_dir / "program_length_assumptions.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot program length vs ASR from scored outputs and final-results")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--final-results", default=None, help="Path to final-results.jsonl")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    final_results = Path(args.final_results).resolve() if args.final_results else repo_root / "final-results.jsonl"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else repo_root / "plots" / "program_length_vs_asr"
    output_dir.mkdir(parents=True, exist_ok=True)

    attack_bits = compute_attack_bits(repo_root)
    adversarial_rows = load_adversarial_rows(repo_root, attack_bits)
    training_rows = load_training_rows(final_results)

    all_rows = adversarial_rows + training_rows

    write_rows_json_csv(all_rows, output_dir)
    write_assumptions(attack_bits, output_dir)

    plot_olmo(all_rows, output_dir / "plot1_olmo_all_methods.png")
    plot_four_models(all_rows, output_dir / "plot2_qwen_olmo_llama_smolsafe.png")
    plot_smol_safe_unsafe_adversarial(all_rows, output_dir / "plot3_smolsafe_vs_smolunsafe_adversarial.png")

    print(f"Wrote plots and artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
