from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import huggingfaceLogin

huggingfaceLogin.login()

pipe = pipeline('text-generation', model="meta-llama/Llama-3.2-1B", device=-1)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")


def generate_response(prompt):
    try:
        response = pipe(prompt, max_length=100, truncation=True)[0]["generated_text"]
        return response
    except Exception as e:
        response = f"오류 발생: {e}"
        return response