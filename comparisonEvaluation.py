#!/usr/bin/env python3
"""
comparisonEvaluation.py

Loads birdKnowledgeTests.csv and fills:
  commonNameErrorRate
  differenceToThinkingKey

Uses experiment_manifest.jsonl to know which birds were queried in each row.
Reference "thinking key" is the proprietary thinking model with:
  contextLevel = 1
  numSimultaneousQueries = 1
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

HERE = Path(__file__).resolve().parent
BIRD_ARTICLES = HERE / "birdArticles.csv"
ATTR_CONFIGS = HERE / "attributeConfigurations.csv"
TESTS = HERE / "birdKnowledgeTests.csv"
MANIFEST = HERE / "experiment_manifest.jsonl"


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def normalize_name(s: Any) -> str:
    s = normalize_space(s).lower()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s


def derive_common_name(source_url: str, species_name: str) -> str:
    if isinstance(source_url, str) and "/wiki/" in source_url:
        title = source_url.rsplit("/wiki/", 1)[-1].replace("_", " ")
        title = re.sub(r"\s*\(.*?\)\s*$", "", title).strip()
        if title and title.lower() != species_name.lower():
            return title
    return species_name


def load_ground_truth_map(article_csv: Path) -> Dict[str, str]:
    df = pd.read_csv(article_csv)
    gt = {}
    for _, row in df.iterrows():
        species = normalize_space(row["speciesName"])
        gt[species] = derive_common_name(str(row.get("sourceURL", "")), species)
    return gt


def parse_json_results(text: Any) -> List[Dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return []
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else []
    except Exception:
        return []


def load_manifest(path: Path) -> Dict[int, List[Dict[str, str]]]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[int(rec["rowNumber"])] = rec["birds"]
    return out


def build_reference_key(df: pd.DataFrame, manifest: Dict[int, List[Dict[str, str]]], thinking_model_name: str) -> Dict[tuple, Dict[str, Any]]:
    ref = {}
    subset = df[
        (df["modelName"] == thinking_model_name) &
        (df["contextLevel"] == 1) &
        (df["numSimultaneousQueries"] == 1)
    ]
    for rownum, row in subset.iterrows():
        birds = manifest.get(int(rownum), [])
        results = parse_json_results(row["jsonResults"])
        if birds and results:
            bird = birds[0]
            key = (normalize_space(bird["speciesName"]), normalize_space(bird["sex"]), int(row["attributeConfigurationNumber"]))
            ref[key] = results[0]
    return ref


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thinking-model-name", required=True)
    args = ap.parse_args()

    tests_df = pd.read_csv(TESTS)
    attr_df = pd.read_csv(ATTR_CONFIGS)
    gt_map = load_ground_truth_map(BIRD_ARTICLES)
    manifest = load_manifest(MANIFEST)
    ref_key = build_reference_key(tests_df, manifest, args.thinking_model_name)

    common_errs = []
    diff_errs = []

    for rownum, row in tests_df.iterrows():
        cfg_idx = int(row["attributeConfigurationNumber"])
        attr_dict = json.loads(attr_df.iloc[cfg_idx]["attributesDictionary"])
        attr_keys = list(attr_dict.keys())
        birds = manifest.get(int(rownum), [])
        results = parse_json_results(row["jsonResults"])

        if not birds or not results:
            common_errs.append(1.0)
            diff_errs.append(1.0)
            continue

        # common name error
        bird_cn_errs = []
        bird_attr_errs = []
        for i, bird in enumerate(birds):
            pred = results[i] if i < len(results) and isinstance(results[i], dict) else {}
            truth_common = gt_map.get(normalize_space(bird["speciesName"]), normalize_space(bird["speciesName"]))
            pred_common = pred.get("commonName", None)
            bird_cn_errs.append(0.0 if pred_common is not None and normalize_name(pred_common) == normalize_name(truth_common) else 1.0)

            ref = ref_key.get((normalize_space(bird["speciesName"]), normalize_space(bird["sex"]), cfg_idx), {})
            total = len(attr_keys) + 1
            wrong = 0
            if pred_common is None or normalize_name(pred_common) != normalize_name(ref.get("commonName", truth_common)):
                wrong += 1
            for k in attr_keys:
                pv = pred.get(k, None)
                rv = ref.get(k, None)
                if pv is None or normalize_name(pv) != normalize_name(rv):
                    wrong += 1
            bird_attr_errs.append(wrong / total if total else 1.0)

        common_errs.append(sum(bird_cn_errs) / len(bird_cn_errs))
        diff_errs.append(sum(bird_attr_errs) / len(bird_attr_errs))

    tests_df["commonNameErrorRate"] = common_errs
    tests_df["differenceToThinkingKey"] = diff_errs
    tests_df.to_csv(TESTS, index=False)
    print(f"Updated {TESTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
