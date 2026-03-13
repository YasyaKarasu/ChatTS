from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

from prompts import concise_answer_messages


class ChatTSInference:
    def __init__(self, model_path: str = "/home/cz/ChatTS-8B", device=0):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map=device,
            torch_dtype='float16'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, tokenizer=self.tokenizer)
        self.device = device

    def generate(self, question: str, timeseries, max_new_tokens: int = 300) -> str:
        prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        inputs = self.processor(text=[prompt], timeseries=timeseries, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)


def extract_concise_answer(question, raw_response, llm_client):
    return llm_client.generate_response(concise_answer_messages(question, raw_response)).strip()
