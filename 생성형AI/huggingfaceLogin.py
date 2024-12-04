from google.colab import drive
import os
from dotenv import load_dotenv
import subprocess

# Google Drive 마운트
drive.mount('/content/drive')

# .env 파일 경로 설정 (Drive 내 .env 위치에 따라 수정)
dotenv_path = '/content/drive/My Drive/Colab Notebooks/생성형AI/.env'
load_dotenv(dotenv_path)

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