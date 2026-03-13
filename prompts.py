
def metric_column_messages(question, header):
    return [
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
Final answer:""".format(QUESTION=question, HEADER=header)}
    ]


def remove_time_period_messages(question):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Given the following question, please remove the part describing the time period.\nQuestion: {question}"}
    ]


def single_timeseries_messages(question):
    return [
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

Answer (Yes/No):""".format(QUESTION=question)}
    ]


def extract_entities_messages(question):
    return [
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
Output:""".format(QUESTION=question)}
    ]


def id_name_mapping_messages(question, header, entity_names):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": """From the given header, choose:
1) id_col: the identifier column used to group rows belonging to the same entity (often an id/code, not the display name)
2) name_cols: one or more columns that can be combined to form the entity display name in the question (e.g., first_name + last_name)

Return ONLY JSON:
{{"id_col": "...", "name_cols": ["...", "..."]}}

Question: {QUESTION}
Header: {HEADER}
Entity names in question: {ENTITY_NAMES}
Answer:""".format(QUESTION=question, HEADER=header, ENTITY_NAMES=entity_names)}
    ]


def concise_answer_messages(question, raw_response):
    return [
        {"role": "system", "content": "You are an answer extraction assistant."},
        {"role": "user", "content": """Given a user question and a model response, extract only the minimal final answer.

Rules:
- Remove explanations, analysis, reasoning, and extra words.
- For yes/no questions, output only "yes" or "no".
- For choice/selection questions, output only the selected entity or option text.
- For trend-direction questions (e.g., asks about trend direction / upward or downward movement), output exactly one label from: rise, fall, stable.
- Keep original language when possible.
- If multiple items are explicitly required by the question, output only those items as a concise comma-separated list.
- Output plain text only.

Question:
{QUESTION}

Model response:
{RESPONSE}

Concise answer:""".format(QUESTION=question, RESPONSE=raw_response)}
    ]
