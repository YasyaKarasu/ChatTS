from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import torch
import numpy as np
import pandas as pd
from datasets import DATASETS
import config as cfg
import pickle
from utils.standard_evaluator import StandardEvaluator
import json
import re
from difflib import SequenceMatcher
from utils.llm_wrapper import ImprovedLLMWrapper


def parse_json_array(response):
    try:
        data = json.loads(response)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def clean_numeric_series(series):
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.dropna().to_numpy()


def heuristic_id_col(df, metric_column):
    candidates = [c for c in df.columns if c != metric_column]
    if not candidates:
        return metric_column

    id_named = [c for c in candidates if "id" in str(c).lower()]
    if id_named:
        return id_named[0]

    # Prefer high-cardinality non-numeric columns as id-like columns.
    scored = []
    for c in candidates:
        nunique = df[c].nunique(dropna=True)
        non_null = max(df[c].notna().sum(), 1)
        ratio = nunique / non_null
        is_obj_like = df[c].dtype == "object"
        scored.append((is_obj_like, ratio, nunique, c))
    scored.sort(reverse=True)
    return scored[0][3]


def heuristic_name_cols(df, metric_column, id_col):
    name_keywords = [
        "name", "first", "last", "full", "player", "team", "club", "country", "city", "title"
    ]
    candidates = [c for c in df.columns if c not in {metric_column, id_col}]
    object_candidates = [c for c in candidates if df[c].dtype == "object"]

    keyword_cols = [
        c for c in object_candidates
        if any(k in str(c).lower() for k in name_keywords)
    ]
    if keyword_cols:
        return keyword_cols[:3]

    return object_candidates[:2]


def resolve_id_candidates(df, id_col, name_cols):
    id_to_label = {}
    grouped = df.groupby(id_col, dropna=False)
    for entity_id, group in grouped:
        parts = []
        for c in name_cols:
            vals = group[c].dropna().astype(str)
            if len(vals) > 0:
                parts.append(vals.iloc[0].strip())
        label = " ".join([p for p in parts if p])
        if not label:
            label = str(entity_id)
        id_to_label[entity_id] = label
    return id_to_label


def name_match_score(query_name, candidate_label):
    q = normalize_text(query_name)
    c = normalize_text(candidate_label)
    if not q or not c:
        return 0.0

    ratio = SequenceMatcher(None, q, c).ratio()
    q_tokens = set(q.split())
    c_tokens = set(c.split())
    overlap = len(q_tokens & c_tokens) / max(len(q_tokens), 1)

    return 0.7 * ratio + 0.3 * overlap


cfg.read_config()
method = cfg.config.get("method", "Jellyfish")
dataset = cfg.config.get("dataset", "wta")
record_start = cfg.config.get("record_start", 0)
record_end = cfg.config.get("record_end", 1)

dataset_path = f"{DATASETS[method]}/stage2_imputed_data_{method}_{dataset}_{record_start}-{record_end}.pkl"
with open(dataset_path, "rb") as f:
    data = pickle.load(f)

# Load the model, tokenizer and processor
MODEL_PATH = "/home/cz/ChatTS-8B"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map=0, torch_dtype='float16')
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, tokenizer=tokenizer)

client = ImprovedLLMWrapper(backend="vllm", model_name="Qwen3-8B")

for record in data:
    original_question = record['original_question']
    df = pd.DataFrame(record['filled_sub_table_df'])

    query = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": """Your task is to identify the single most likely **metric column name** from a table header, based on a natural-language question.

Definition:
- A “metric column” is the numeric/quantitative field that the question asks to analyze (e.g., trend, change, increase/decrease, value over time).

Follow this two-step thinking process:

Step 1 — Extract metric keywords from the question
- Read the question and identify the phrase(s) that describe the quantity being measured or analyzed.

Step 2 — Map the metric keywords to the table header
- Compare the extracted keywords with the column names in the header.
- Choose the column name that is the closest semantic match (including synonyms, abbreviations, and common schema variants).
- If multiple columns are plausible, pick the one that most directly represents the requested metric (prefer the core metric over derived or auxiliary metrics unless the question explicitly mentions them).

Output format:
Return ONLY the metric column name (exactly as it appears in the header). No extra text.

Example:
Question: "What was the overall trend direction of Destanee Aiava's rank position during 20151012 to 20160502?"
Header: ["ranking_date", "ranking", "player_id", "ranking_points", "tours", "first_name", "last_name", "hand", "birth_date", "country_code"]
Final answer: ranking

Question: {QUESTION}
Header: {HEADER}
Final answer:""".format(QUESTION=original_question, HEADER=list(df.columns))}
    ]
    metric_column = client.generate_response(query).strip()

    query = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Given the following question, please remove the part describing the time period.\nQuestion: {original_question}"}
    ]
    clean_question = client.generate_response(query)

    query = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": """A “single time series” question asks for a property of ONE time-indexed sequence (one entity × one metric) by itself, such as its overall trend direction, increase/decrease pattern, volatility, or turning points.

A question is NOT single time series if it requires comparing or relating TWO OR MORE time-indexed sequences, e.g., asking which of two players is more stable, higher, faster-changing, more volatile, or any other relationship between multiple entities/metrics.

Task: Decide whether the following question is about a single time series. Anwer with "Yes" or "No" only.

Rules:
- Output "Yes" if the question can be answered using ONLY ONE entity's ONE metric over time.
- Output "No" if the question involves TWO OR MORE entities and/or metrics and asks for a COMPARISON/RELATIONSHIP between them.
- The dataset may contain many entities, but ONLY what the question explicitly asks for matters.

Examples:
- "What was the overall trend direction of Destanee Aiava's rank position during 20151012 to 20160502?" -> Yes
- "Which player's points was more stable, Luksika Kumkhum or Valeria Nikolaev, during 20170102 to 20170703?" -> No

Question: {QUESTION}

Answer (Yes/No):""".format(QUESTION=original_question)}
    ]
    response = client.generate_response(query)

    if response.strip().lower() == "yes":
        print("Processing record with single time series question.")

        timeseries = clean_numeric_series(df[metric_column])
        print(timeseries)

        question = f"I have a time series length of {len(timeseries)}: <ts><ts/>. {clean_question}"

        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        inputs = processor(text=[prompt], timeseries=[timeseries], padding=True, return_tensors="pt")
        inputs = {k: v.to(0) for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=300)
        response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        print("Question:", original_question)
        print("Processed Question:", question)
        print("Response:", response)
    else:
        print("Processing record with multi time series question.")

        query = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": """Extract the entity name(s) that the question is asking about from a natural-language question. There may be zero, one, or multiple entity names.

Definition:
An "entity name" is a specific named object explicitly mentioned in the question, such as a person/player, team, organization, place, product, etc. Entity names are usually proper nouns and should be returned exactly as they appear (surface form).

Output format (strict):
Return ONLY a JSON array of strings. No extra text, no JSON object, no explanations.

Example:
Question: "Which player's rank points was more stable, Luksika Kumkhum or Valeria Nikolaev, during 20170102 to 20170703?"
Output:
["Luksika Kumkhum", "Valeria Nikolaev"]

Question: {QUESTION}
Output:""".format(QUESTION=original_question)}
        ]
        entity_names = parse_json_array(client.generate_response(query))
        print("Extracted entity names:", entity_names)

        query = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": """From the given header, choose:
1) id_col: the identifier column used to group rows belonging to the same entity (often an id/code, not the display name)
2) name_cols: one or more columns that can be combined to form the entity display name in the question (e.g., first_name + last_name)

Return ONLY JSON:
{"id_col": "...", "name_cols": ["...", "..."]}

Question: {QUESTION}
Header: {HEADER}
Entity names in question: {ENTITY_NAMES}
Answer:""".format(
                QUESTION=original_question,
                HEADER=list(df.columns),
                ENTITY_NAMES=entity_names,
            )}
        ]
        mapping_response = client.generate_response(query)

        id_col = None
        name_cols = []
        try:
            mapping_info = json.loads(mapping_response)
            if isinstance(mapping_info, dict):
                id_col = mapping_info.get("id_col")
                name_cols = mapping_info.get("name_cols", [])
        except json.JSONDecodeError:
            pass

        if id_col not in df.columns:
            id_col = heuristic_id_col(df, metric_column)

        name_cols = [c for c in name_cols if c in df.columns and c not in {id_col, metric_column}]
        if not name_cols:
            name_cols = heuristic_name_cols(df, metric_column, id_col)

        id_to_label = resolve_id_candidates(df, id_col, name_cols)

        selected_ids = []
        matched_entity_names = []
        for entity_name in entity_names:
            best_id, best_score = None, -1.0
            for candidate_id, candidate_label in id_to_label.items():
                score = name_match_score(entity_name, candidate_label)
                if score > best_score:
                    best_score = score
                    best_id = candidate_id
            if best_id is not None and best_score >= 0.55 and best_id not in selected_ids:
                selected_ids.append(best_id)
                matched_entity_names.append(entity_name)

        timeseries = []
        final_entity_labels = []
        for idx, entity_id in enumerate(selected_ids):
            ts = clean_numeric_series(df[df[id_col] == entity_id][metric_column])
            if len(ts) > 0:
                timeseries.append(ts)
                final_entity_labels.append(
                    matched_entity_names[idx] if idx < len(matched_entity_names) else id_to_label.get(entity_id, str(entity_id))
                )

        # Fallback: if matching is insufficient, use top groups from subtable.
        if len(timeseries) < 2:
            grouped = df.groupby(id_col, dropna=False)
            candidates = []
            for entity_id, group in grouped:
                ts = clean_numeric_series(group[metric_column])
                if len(ts) > 0:
                    label = id_to_label.get(entity_id, str(entity_id))
                    candidates.append((label, ts))

            candidates.sort(key=lambda x: len(x[1]), reverse=True)
            top_k = candidates[:2]
            final_entity_labels = [x[0] for x in top_k]
            timeseries = [x[1] for x in top_k]

        print(timeseries)

        question = f"I have {len(timeseries)} time series."
        for i, ts in enumerate(timeseries):
            entity_label = final_entity_labels[i] if i < len(final_entity_labels) else f"entity_{i+1}"
            question += f" Time series for {entity_label} has length of {len(ts)}: <ts><ts/>."
        question += " " + clean_question

        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        inputs = processor(text=[prompt], timeseries=timeseries, padding=True, return_tensors="pt")
        inputs = {k: v.to(0) for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=300)
        response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        print("Question:", original_question)
        print("Processed Question:", question)
        print("Response:", response)
