import pickle

import config as cfg
from datasets import DATASETS
from inference import ChatTSInference
from record_processor import RecordProcessor
from utils.llm_wrapper import ImprovedLLMWrapper


def load_stage2_data(method, dataset, record_start, record_end):
    dataset_path = (
        f"{DATASETS[method]}/stage2_imputed_data_{method}_{dataset}_{record_start}-{record_end}.pkl"
    )
    with open(dataset_path, "rb") as f:
        return pickle.load(f)


def main():
    cfg.read_config()
    method = cfg.config.get("method", "Jellyfish")
    dataset = cfg.config.get("dataset", "wta")
    record_start = cfg.config.get("record_start", 0)
    record_end = cfg.config.get("record_end", 1)

    data = load_stage2_data(method, dataset, record_start, record_end)

    ts_model = ChatTSInference(model_path="/home/cz/ChatTS-8B", device=0)
    llm_client = ImprovedLLMWrapper(backend="vllm", model_name="Qwen3-8B")

    processor = RecordProcessor(llm_client=llm_client, ts_model=ts_model)
    total = 0
    consistent = 0
    for record in data:
        result = processor.process_record(record)
        if result is not None:
            total += 1
            consistent += int(result)

    accuracy = (consistent / total) if total else 0.0
    print(f"Final consistency accuracy: {consistent}/{total} = {accuracy:.2%}")


if __name__ == "__main__":
    main()
