import pandas as pd


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
