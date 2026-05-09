#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
from openai import OpenAI


DEFAULT_NONREASONING_MODEL = "gpt-4.1-mini"
DEFAULT_REASONING_MODEL = "o4-mini"

CONTEXT_LEVELS = [0, 1, 4, 8]
WORKLOADS = [1, 8, 16, 32]

OUTPUT_COLUMNS = [
    "attributeConfigurationNumber",
    "modelName",
    "numSimultaneousQueries",
    "contextLevel",
    "jsonResults",
    "commonNameErrorRate",
    "differenceToThinkingKey",
    "avgInputTokens",
    "avgOutputTokens",
    "queriedSpecies",
    "status",
]

SYSTEM_PROMPT = (
    "You are a careful bird-identification assistant.\n"
    "Return ONLY valid JSON.\n"
    "No markdown.\n"
    "No explanations outside the JSON.\n"
)

USER_PROMPT_TEMPLATE = """
You will answer queries about bird species using a fixed categorical attribute dictionary.

ATTRIBUTE DICTIONARY:
{attribute_dictionary_json}

TASK:
For each queried bird below, return one JSON object with:
1. "scientificName"
2. "commonName"
3. every attribute key from the attribute dictionary

RULES:
- Output must be valid JSON only.
- Output must be a JSON list.
- The list length must exactly match the number of queried birds.
- For every attribute, choose exactly one value from the allowed values.
- If uncertain, choose the closest allowed value rather than inventing values.
- Do not add extra keys.
- Never output prose before or after the JSON list.

QUERIED BIRDS:
{query_block}

{context_block}
""".strip()


def normalize_space(s: str) -> str:
    s = str(s).replace("\ufeff", " ")
    s = s.replace("Ã—", " × ")
    s = s.replace("×", " × ")
    return re.sub(r"\s+", " ", s).strip()


def split_tokens(line: str) -> List[str]:
    parts = re.split(r"[,|\t;]+", line)
    return [normalize_space(p) for p in parts if normalize_space(p)]


def looks_like_scientific_name(text: str) -> bool:
    text = normalize_space(text)
    hybrid_pat = r"^[A-Z][a-zA-Z-]+ [a-z][a-zA-Z-]+ [×x] [a-z][a-zA-Z-]+$"
    regular_pat = r"^[A-Z][a-zA-Z-]+(?: [a-z][a-zA-Z-]+){1,2}$"
    return bool(re.match(hybrid_pat, text) or re.match(regular_pat, text))


def parse_bird_line(line: str) -> Optional[Dict[str, str]]:
    raw = normalize_space(line)
    if not raw or raw.startswith("#"):
        return None

    parts = split_tokens(raw)
    sex_words = {"male", "female", "unknown", "both", "sexes", "m", "f"}

    if len(parts) >= 2:
        if parts[0].lower() in sex_words:
            maybe_species = normalize_space(" ".join(parts[1:]))
            if looks_like_scientific_name(maybe_species):
                return {"sex": parts[0].lower(), "speciesName": maybe_species}

        if parts[-1].lower() in sex_words:
            maybe_species = normalize_space(" ".join(parts[:-1]))
            if looks_like_scientific_name(maybe_species):
                return {"sex": parts[-1].lower(), "speciesName": maybe_species}

    tokens = raw.split()
    if tokens and tokens[0].lower() in sex_words and len(tokens) >= 3:
        maybe_species = normalize_space(" ".join(tokens[1:]))
        if looks_like_scientific_name(maybe_species):
            return {"sex": tokens[0].lower(), "speciesName": maybe_species}

    if looks_like_scientific_name(raw):
        return {"sex": "unknown", "speciesName": raw}

    return None


def load_bird_entries(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    raw_bytes = path.read_bytes()
    text = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = raw_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        text = raw_bytes.decode("utf-8", errors="ignore")

    entries: List[Dict[str, str]] = []
    for line in text.splitlines():
        parsed = parse_bird_line(line)
        if parsed is not None:
            entries.append(parsed)

    if not entries:
        raise ValueError("No valid bird entries could be parsed from birdNames.txt")

    return entries


def safe_literal_or_json_parse(text: str) -> Any:
    if text is None:
        return None

    text = text.strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    text2 = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text2 = re.sub(r"```$", "", text2).strip()

    try:
        return json.loads(text2)
    except Exception:
        pass

    match = re.search(r"\[.*\]", text2, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    try:
        return ast.literal_eval(text2)
    except Exception:
        return None


def ensure_output_csv(path: Path) -> None:
    if path.exists():
        try:
            existing = pd.read_csv(path)
            missing = [c for c in OUTPUT_COLUMNS if c not in existing.columns]
            if missing:
                for c in missing:
                    existing[c] = None
                existing = existing[OUTPUT_COLUMNS]
                existing.to_csv(path, index=False)
        except Exception:
            pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(path, index=False)
    else:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(path, index=False)


def append_row(path: Path, row: Dict[str, Any]) -> None:
    ensure_output_csv(path)
    pd.DataFrame([row], columns=OUTPUT_COLUMNS).to_csv(
        path, mode="a", header=False, index=False
    )


def load_attribute_configurations(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    required = ["platformName", "thinkingLevel", "contextIncluded", "attributesDictionary"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"attributeConfigurations.csv missing columns: {missing}")
    return df


def load_bird_articles(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    required = ["speciesName", "sex", "sourceURL", "extractedTextDescription"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"birdArticles.csv missing columns: {missing}")

    df["speciesName"] = df["speciesName"].astype(str).map(normalize_space)
    df["sex"] = df["sex"].astype(str).map(normalize_space)
    df["sourceURL"] = df["sourceURL"].astype(str)
    df["extractedTextDescription"] = df["extractedTextDescription"].astype(str)
    return df


def clip_text(text: str, max_chars: int) -> str:
    text = normalize_space(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."


def context_chars_for_combo(workload: int, context_level: int) -> int:
    target_total = workload * context_level
    if target_total <= 1:
        return 350
    if target_total <= 8:
        return 140
    if target_total <= 32:
        return 70
    if target_total <= 64:
        return 40
    if target_total <= 128:
        return 24
    return 16


def max_output_tokens_for_workload(workload: int) -> int:
    if workload == 1:
        return 220
    if workload == 8:
        return 650
    if workload == 16:
        return 1100
    return 1600


def build_context_block(
    queried_entries: List[Dict[str, str]],
    bird_articles: pd.DataFrame,
    context_level: int,
    rng: random.Random,
    workload: int,
    desc_chars: int,
) -> str:
    if context_level == 0:
        return "CONTEXT:\nNone"

    queried_keys = {
        (normalize_space(x["speciesName"]), normalize_space(x["sex"]))
        for x in queried_entries
    }

    relevant_rows = []
    for q in queried_entries:
        species = normalize_space(q["speciesName"])
        sex = normalize_space(q["sex"])
        match = bird_articles[
            (bird_articles["speciesName"] == species) &
            (bird_articles["sex"] == sex)
        ]
        if match.empty:
            match = bird_articles[bird_articles["speciesName"] == species]
        if not match.empty:
            relevant_rows.append(match.iloc[0])

    target_total = max(workload * context_level, len(relevant_rows))

    irrelevant_pool = bird_articles[
        ~bird_articles.apply(
            lambda r: (normalize_space(r["speciesName"]), normalize_space(r["sex"])) in queried_keys,
            axis=1,
        )
    ]

    irrelevant_records = irrelevant_pool.to_dict("records")
    rng.shuffle(irrelevant_records)
    chosen_irrelevant = irrelevant_records[:max(0, target_total - len(relevant_rows))]

    all_context_rows = [r.to_dict() if hasattr(r, "to_dict") else r for r in relevant_rows] + chosen_irrelevant

    lines = ["CONTEXT:"]
    for i, row in enumerate(all_context_rows, start=1):
        species = normalize_space(row["speciesName"])
        sex = normalize_space(row["sex"])
        desc = clip_text(row["extractedTextDescription"], max_chars=desc_chars)
        lines.append(f"[{i}] sex={sex}; speciesName={species}; description={desc}")

    return "\n".join(lines)


def build_query_block(entries: List[Dict[str, str]]) -> str:
    lines = []
    for i, item in enumerate(entries, start=1):
        lines.append(
            f"[{i}] sex={normalize_space(item['sex'])}; speciesName={normalize_space(item['speciesName'])}"
        )
    return "\n".join(lines)


def normalize_model_output(
    parsed: Any,
    queried_entries: List[Dict[str, str]],
    attribute_keys: List[str],
) -> Optional[List[Dict[str, Any]]]:
    if parsed is None or not isinstance(parsed, list):
        return None

    out = []
    for i, q in enumerate(queried_entries):
        item = parsed[i] if i < len(parsed) and isinstance(parsed[i], dict) else {}
        norm_item = {
            "scientificName": str(item.get("scientificName", q["speciesName"])),
            "commonName": item.get("commonName", None),
        }
        for k in attribute_keys:
            norm_item[k] = item.get(k, None)
        out.append(norm_item)
    return out


def choose_entries(all_entries: List[Dict[str, str]], workload: int, rng: random.Random) -> List[Dict[str, str]]:
    return rng.sample(all_entries, workload)


def stringify_queried_species(entries: List[Dict[str, str]]) -> str:
    return json.dumps(
        [{"sex": e["sex"], "speciesName": e["speciesName"]} for e in entries],
        ensure_ascii=False,
    )


def make_prompt(queried_entries, attribute_dict, context_block):
    user_prompt = USER_PROMPT_TEMPLATE.format(
        attribute_dictionary_json=json.dumps(attribute_dict, ensure_ascii=False),
        query_block=build_query_block(queried_entries),
        context_block=context_block,
    )
    return user_prompt


def extract_output_text(resp) -> str:
    text = getattr(resp, "output_text", None)
    if text:
        return text.strip()

    pieces = []
    try:
        for item in resp.output:
            content = getattr(item, "content", []) or []
            for c in content:
                t = getattr(c, "text", None)
                if t:
                    pieces.append(t)
    except Exception:
        pass
    return "".join(pieces).strip()


def extract_usage(resp) -> Tuple[float, float]:
    in_tok = 0.0
    out_tok = 0.0
    usage = getattr(resp, "usage", None)
    if usage is not None:
        in_tok = float(getattr(usage, "input_tokens", 0) or 0)
        out_tok = float(getattr(usage, "output_tokens", 0) or 0)
    return in_tok, out_tok


def run_api_model(
    client: OpenAI,
    model_name: str,
    prompt: str,
    workload: int,
    max_output_tokens: int,
) -> Tuple[Optional[Any], float, float, str]:
    try:
        resp = client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=max_output_tokens,
        )

        text = extract_output_text(resp)
        parsed = safe_literal_or_json_parse(text)
        input_tokens, output_tokens = extract_usage(resp)

        avg_input_tokens = float(input_tokens) / max(workload, 1)
        avg_output_tokens = float(output_tokens) / max(workload, 1)

        return parsed, avg_input_tokens, avg_output_tokens, "ok"

    except Exception as e:
        msg = str(e)
        if "429" in msg or "rate_limit" in msg or "Request too large" in msg or "tokens per min" in msg:
            return None, 0.0, 0.0, f"retryable:{type(e).__name__}:{msg}"
        return None, 0.0, 0.0, f"error:{type(e).__name__}:{msg}"


def run_with_retries(
    client: OpenAI,
    model_name: str,
    queried_entries: List[Dict[str, str]],
    attribute_dict: Dict[str, List[str]],
    bird_articles: pd.DataFrame,
    context_level: int,
    rng: random.Random,
    workload: int,
) -> Tuple[Optional[Any], float, float, str]:
    desc_chars = context_chars_for_combo(workload, context_level)
    max_output_tokens = max_output_tokens_for_workload(workload)

    for _ in range(4):
        context_block = build_context_block(
            queried_entries=queried_entries,
            bird_articles=bird_articles,
            context_level=context_level,
            rng=rng,
            workload=workload,
            desc_chars=desc_chars,
        )
        prompt = make_prompt(queried_entries, attribute_dict, context_block)

        parsed, avg_in, avg_out, status = run_api_model(
            client=client,
            model_name=model_name,
            prompt=prompt,
            workload=workload,
            max_output_tokens=max_output_tokens,
        )

        if status == "ok":
            return parsed, avg_in, avg_out, "ok"
        if not status.startswith("retryable:"):
            return parsed, avg_in, avg_out, status

        desc_chars = max(8, desc_chars // 2)
        max_output_tokens = max(120, int(max_output_tokens * 0.75))
        time.sleep(1.5)

    return None, 0.0, 0.0, "error:retry_exhausted"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/content/drive/MyDrive/at3684-lab3")
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--nonreasoning-model", type=str, default=DEFAULT_NONREASONING_MODEL)
    parser.add_argument("--reasoning-model", type=str, default=DEFAULT_REASONING_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep_seconds", type=float, default=0.5)
    return parser.parse_args()


def resolve_models(args) -> List[str]:
    if args.models:
        return [m.strip() for m in args.models.split(",") if m.strip()]
    return [args.nonreasoning_model, args.reasoning_model]


def main():
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    client = OpenAI(api_key=api_key)

    base_dir = Path(args.base_dir)
    bird_names_path = base_dir / "birdNames.txt"
    bird_articles_path = base_dir / "birdArticles.csv"
    attr_cfg_path = base_dir / "attributeConfigurations.csv"
    out_csv_path = base_dir / "birdKnowledgeTests.csv"

    models = resolve_models(args)
    rng = random.Random(args.seed)

    ensure_output_csv(out_csv_path)

    all_entries = load_bird_entries(bird_names_path)
    bird_articles = load_bird_articles(bird_articles_path)
    attr_df = load_attribute_configurations(attr_cfg_path)

    print(f"Loaded {len(all_entries)} bird entries")
    print(f"Loaded {len(bird_articles)} bird article rows")
    print(f"Loaded {len(attr_df)} attribute configurations")

    for model_name in models:
        print(f"\n=== MODEL: {model_name} ===")

        for cfg_idx, row in attr_df.iterrows():
            attribute_dict = safe_literal_or_json_parse(str(row["attributesDictionary"]))
            if not isinstance(attribute_dict, dict):
                print(f"Bad attribute config row {cfg_idx}")
                continue

            attribute_keys = list(attribute_dict.keys())

            for work in WORKLOADS:
                for ctx in CONTEXT_LEVELS:
                    queried_entries = choose_entries(all_entries, work, rng)

                    parsed, avg_in, avg_out, status = run_with_retries(
                        client=client,
                        model_name=model_name,
                        queried_entries=queried_entries,
                        attribute_dict=attribute_dict,
                        bird_articles=bird_articles,
                        context_level=ctx,
                        rng=rng,
                        workload=work,
                    )

                    json_results = normalize_model_output(parsed, queried_entries, attribute_keys)

                    append_row(
                        out_csv_path,
                        {
                            "attributeConfigurationNumber": cfg_idx,
                            "modelName": model_name,
                            "numSimultaneousQueries": work,
                            "contextLevel": ctx,
                            "jsonResults": json.dumps(json_results, ensure_ascii=False) if json_results is not None else None,
                            "commonNameErrorRate": None,
                            "differenceToThinkingKey": None,
                            "avgInputTokens": avg_in,
                            "avgOutputTokens": avg_out,
                            "queriedSpecies": stringify_queried_species(queried_entries),
                            "status": status,
                        },
                    )

                    print(
                        f"done model={model_name} cfg={cfg_idx} work={work} ctx={ctx} "
                        f"in={avg_in:.1f} out={avg_out:.1f} status={status}"
                    )
                    time.sleep(args.sleep_seconds)

    print(f"\nFinished. Results appended to: {out_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())