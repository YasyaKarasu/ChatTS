from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import torch
import numpy as np
import pandas as pd
from datasets import DATASETS
import config as cfg
import pickle
from utils.standard_evaluator import StandardEvaluator
import json
from utils.llm_wrapper import ImprovedLLMWrapper

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
# Create time series and prompts
# timeseries = np.sin(np.arange(256) / 10) * 5.0
# timeseries[100:] -= 10.0
# prompt = f"I have a time series length of 256: <ts><ts/>. Please analyze the local changes in this time series."
# Apply Chat Template

client = ImprovedLLMWrapper(backend="vllm", model_name="Qwen3-8B")

for record in data:
    original_question = record['original_question']
    original_metadata = record['original_metadata']
    df = pd.DataFrame(record['filled_sub_table_df'])
    # metric_column = original_metadata['metric_col']
    
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
    # print(metric_column)
    
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
    # print(response.strip().lower())
    
    if response.strip().lower() == "yes":
        print("Processing record with single time series question.")
        
        
        timeseries = df[metric_column].to_numpy()
        # drop the nan values
        timeseries = timeseries[~np.isnan(timeseries)]
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
        id_col = original_metadata['id_col']
        # entity_ids = original_metadata['entity_ids']
        # entity_names = original_metadata['entity_names']
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
["Luksika Kumkhum", "Valeria Nikolaev"]""".format(QUESTION=original_question)}
        ]
        response = client.generate_response(query)
        entity_names = json.loads(response)
        print("Extracted entity names:", entity_names)
        timeseries = []
        for entity_id in entity_ids:
            ts = df[df[id_col] == entity_id][metric_column].to_numpy()
            ts = ts[~np.isnan(ts)]
            timeseries.append(ts)
        print(timeseries)
            
        question = f"I have {len(timeseries)} time series."
        for i, ts in enumerate(timeseries):
            question += f" Time series for {entity_names[i]} has length of {len(ts)}: <ts><ts/>."
        question += " " + clean_question
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        inputs = processor(text=[prompt], timeseries=timeseries, padding=True, return_tensors="pt")
        inputs = {k: v.to(0) for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=300)
        response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        print("Question:", original_question)
        print("Processed Question:", question)
        print("Response:", response)

# prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
# # Convert to tensor
# inputs = processor(text=[prompt], timeseries=[timeseries], padding=True, return_tensors="pt")
# # Move to GPU
# inputs = {k: v.to(0) for k, v in inputs.items()}
# # Model Generate
# outputs = model.generate(**inputs, max_new_tokens=300)
# print(tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True))
