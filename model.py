from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import huggingfaceLogin
import os
from dotenv import load_dotenv
import requests

huggingfaceLogin.login()

pipe = pipeline('text-generation', model="meta-llama/Llama-3.2-1B", device=-1)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

load_dotenv()
DID_API_URL = "https://api.d-id.com/talks"
source_url = os.getenv("PROF_IMAGE_URL")

def generate_response(prompt):
    try:
        response = pipe(prompt,pad_token_id=pipe.tokenizer.eos_token_id, max_length=300, truncation=True)[0]["generated_text"]
        return response
    except Exception as e:
        response = f"오류 발생: {e}"
        return response


def generate_DiD_id(answer):
    payload = {
        "source_url": source_url,
        "script": {
            "type": "text",
            "subtitles": "false",
            "provider": {
                "type": "google",
                "key": "ko-KR-Standard-D"
            },
            "input": answer
        },
        "config": {
            "fluent": "false",
            "pad_audio": "0.0"
        }
    }
    headers = {
    "Authorization": "Basic " + os.getenv("DID_KEY"),
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    print(headers)

    responses = requests.post(DID_API_URL, json=payload, headers=headers)
    response = responses.json()
    print(response)
    did_id = response["id"]

    return did_id

def generate_DiD_url(did_id):
    url = f"https://d-id.com/talks/{did_id}"
    headers = {
        "Authorization": "Basic " + os.getenv("DID_KEY"),
        "accept": "application/json"
    }
    response = requests.get(url, headers=headers)
    response = response.json()
    did_url = response["result_url"]
    return did_url



ans = generate_response("what is generativeAI?")
print(ans)
did_id = generate_DiD_id(ans)
print(did_id)
did_url = generate_DiD_url(did_id)