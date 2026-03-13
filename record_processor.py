import json

import pandas as pd

from inference import extract_concise_answer
from prompts import (
    extract_entities_messages,
    id_name_mapping_messages,
    metric_column_messages,
    remove_time_period_messages,
    single_timeseries_messages,
)
from table_utils import (
    clean_numeric_series,
    heuristic_id_col,
    heuristic_name_cols,
    resolve_id_candidates,
)
from text_utils import name_match_score, parse_json_array


class RecordProcessor:
    def __init__(self, llm_client, ts_model):
        self.llm_client = llm_client
        self.ts_model = ts_model

    def process_record(self, record):
        original_question = record['original_question']
        original_metadata = record['original_metadata']
        original_answer = original_metadata['answer']
        df = pd.DataFrame(record['filled_sub_table_df'])

        metric_column = self.llm_client.generate_response(
            metric_column_messages(original_question, list(df.columns))
        ).strip()
        clean_question = self.llm_client.generate_response(remove_time_period_messages(original_question))

        response = self.llm_client.generate_response(single_timeseries_messages(original_question))
        if response.strip().lower() == "yes":
            self._process_single(df, metric_column, clean_question, original_question, original_answer)
        else:
            self._process_multi(df, metric_column, clean_question, original_question, original_answer)

    def _process_single(self, df, metric_column, clean_question, original_question, original_answer):
        print("Processing record with single time series question.")
        timeseries = clean_numeric_series(df[metric_column])
        print(timeseries)

        question = f"I have a time series length of {len(timeseries)}: <ts><ts/>. {clean_question}"
        response = self.ts_model.generate(question, [timeseries])
        concise_answer = extract_concise_answer(original_question, response, self.llm_client)

        print("Question:", original_question)
        print("Processed Question:", question)
        print("Response:", response)
        print("Concise Answer:", concise_answer)
        print("Original Answer:", original_answer)

    def _process_multi(self, df, metric_column, clean_question, original_question, original_answer):
        print("Processing record with multi time series question.")

        entity_names = parse_json_array(self.llm_client.generate_response(extract_entities_messages(original_question)))
        print("Extracted entity names:", entity_names)

        mapping_response = self.llm_client.generate_response(
            id_name_mapping_messages(original_question, list(df.columns), entity_names)
        )

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
            entity_label = final_entity_labels[i] if i < len(final_entity_labels) else f"entity_{i + 1}"
            question += f" Time series for {entity_label} has length of {len(ts)}: <ts><ts/>."
        question += " " + clean_question

        response = self.ts_model.generate(question, timeseries)
        concise_answer = extract_concise_answer(original_question, response, self.llm_client)

        print("Question:", original_question)
        print("Processed Question:", question)
        print("Response:", response)
        print("Concise Answer:", concise_answer)
        print("Original Answer:", original_answer)
