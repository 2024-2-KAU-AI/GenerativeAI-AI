import os
from dotenv import load_dotenv
import subprocess

def login():
    load_dotenv()

    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

    if not huggingface_token:
        raise ValueError("Huggingface token not found")

    try:
        subprocess.run(["huggingface-cli", "login", "--token", huggingface_token],
                       check=True,
                       text=True,
                       capture_output=True
                       )
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        raise ValueError("Huggingface login failed")
