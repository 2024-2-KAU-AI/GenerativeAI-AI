#!/usr/bin/env python
# coding: utf-8

import gradio as gr
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from PIL import Image
import pytesseract
import spacy
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import requests
import huggingfaceLogin
from dotenv import load_dotenv

# Tesseract 실행 파일 경로 설정 (Windows용)
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 언어 데이터 경로 및 언어 설정 (Windows용)
#tessdata_dir_config = r'--tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata" -l kor+eng'
# Tesseract 한국어 지원 설정 (MacOS Homebrew 경로)
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"  # Tesseract 실행 파일 경로
tessdata_dir_config = '--tessdata-dir "/opt/homebrew/share/tessdata" -l kor+eng'  # 언어 설정

# Huggingface 로그인
huggingfaceLogin.login()

# spaCy 및 SentenceTransformer 초기화
nlp = spacy.load("en_core_web_sm")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# LLaMA 모델 초기화
pipe = pipeline('text-generation', model="meta-llama/Llama-3.2-1B", device=-1)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# 환경 변수 로드
load_dotenv()
# DID_API_URL = "https://api.d-id.com/talks"
# source_url = os.getenv("PROF_IMAGE_URL")

# 전역 변수
faiss_index = None
pdf_text_chunks = []
chunk_size = 200

# CSS 스타일 정의
custom_css = """
body {
    background-color: #001f3f;  /* 남색 계열 배경 */
    color: white;  /* 기본 텍스트 색상 */
}
.gradio-container {
    background-color: #001f3f;  /* Gradio 내부 배경 색상 */
    color: white;  /* Gradio 내부 텍스트 색상 */
}
h1, h2, h3, p {
    color: white;  /* 제목과 문단 텍스트 색상 */
}
"""

# PDF 청킹
def extract_text_from_pdf_with_spacy(file_path, chunk_size=200):
    pdf_text_chunks = []
    reader = PdfReader(file_path)
    for page in reader.pages:
        page_text = page.extract_text()
        doc = nlp(page_text)
        sentences = [sent.text.strip() for sent in doc.sents]
        current_chunk = []
        current_length = 0
        for sentence in sentences:
            current_chunk.append(sentence)
            current_length += len(sentence.split())
            if current_length >= chunk_size:
                pdf_text_chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
        if current_chunk:
            pdf_text_chunks.append(" ".join(current_chunk))
    return pdf_text_chunks

# PDF 임베딩 생성 및 인덱싱
def create_embeddings_and_index():
    global faiss_index
    embeddings = embedding_model.encode(pdf_text_chunks)
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)

# 모델 응답 생성
def generate_response(prompt):
    try:
        response = pipe(prompt, pad_token_id=pipe.tokenizer.eos_token_id, max_length=300, truncation=True)[0]["generated_text"]
        return response
    except Exception as e:
        return f"오류 발생: {e}"

# # DiD API 호출
# def generate_DiD_id(answer):
#     payload = {
#         "source_url": source_url,
#         "script": {
#             "type": "text",
#             "subtitles": "false",
#             "provider": {"type": "google", "key": "ko-KR-Standard-D"},
#             "input": answer
#         },
#         "config": {"fluent": "false", "pad_audio": "0.0"}
#     }
#     headers = {
#         "Authorization": "Basic " + os.getenv("DID_KEY"),
#         "accept": "application/json",
#         "Content-Type": "application/json"
#     }
#     responses = requests.post(DID_API_URL, json=payload, headers=headers)
#     response = responses.json()
#     return response["id"]

# def generate_DiD_id(answer):
#     try:
#         payload = {
#             "source_url": source_url,
#             "script": {
#                 "type": "text",
#                 "subtitles": "false",
#                 "provider": {"type": "google", "key": "ko-KR-Standard-D"},
#                 "input": answer
#             },
#             "config": {"fluent": "false", "pad_audio": "0.0"}
#         }
#         headers = {
#             "Authorization": "Basic " + os.getenv("DID_KEY"),
#             "accept": "application/json",
#             "Content-Type": "application/json"
#         }
#         responses = requests.post(DID_API_URL, json=payload, headers=headers)
#         response = responses.json()

#         if responses.status_code != 200 or "id" not in response:
#             raise Exception(f"Unexpected API response: {response}")

#         return response["id"]

#     except Exception as e:
#         return f"DiD API 호출 중 오류가 발생했습니다: {e}"


# Gradio 핸들러 함수
def handle_text_query(text):
    model_response = generate_response(text)
    # did_id = generate_DiD_id(model_response)
    # did_url = generate_DiD_url(did_id)
    return f"모델 응답:\n{model_response}\n\n"

def handle_image_query(image_path, question):
    # 이미지 분석 및 질문 처리
    extracted_text = pytesseract.image_to_string(Image.open(image_path), config=tessdata_dir_config)
    prompt = f"이미지에서 추출된 텍스트: {extracted_text}\n\n질문: {question}"
    model_response = generate_response(prompt)
    # did_id = generate_DiD_id(model_response)
    # did_url = generate_DiD_url(did_id)
    return f"추출된 텍스트:\n{extracted_text}\n\n모델 응답:\n{model_response}\n\n"

def handle_pdf_query(pdf, question):
    # PDF 질문 처리
    global pdf_text_chunks
    pdf_text_chunks = extract_text_from_pdf_with_spacy(pdf.name)
    create_embeddings_and_index()
    prompt = f"PDF에서 추출된 텍스트 기반 답변\n질문: {question}"
    model_response = generate_response(prompt)
    did_id = generate_DiD_id(model_response)
    did_url = generate_DiD_url(did_id)
    return f"모델 응답:\n{model_response}\n\nDiD 결과 URL: {did_url}"

# 플랫폼 페이지 정의
def tutor_platform(back_to_main):
    with gr.Column(visible=True) as platform:
        gr.HTML("<h2>AI 교수님 튜터 플랫폼</h2>")
        with gr.Tab("텍스트"):
            text_input = gr.Textbox(label="질문 입력")
            text_button = gr.Button("질문하기")
            text_output = gr.Textbox(label="답변")
            text_button.click(handle_text_query, inputs=text_input, outputs=text_output)

        with gr.Tab("이미지"):
            image_input = gr.Image(type="filepath", label="이미지 업로드")
            image_question = gr.Textbox(label="이미지 관련 질문")
            image_button = gr.Button("질문하기")
            image_output = gr.Textbox(label="답변")
            image_button.click(handle_image_query, inputs=[image_input, image_question], outputs=image_output)

        with gr.Tab("PDF"):
            pdf_input = gr.File(label="PDF 업로드")
            pdf_question = gr.Textbox(label="PDF 관련 질문")
            pdf_button = gr.Button("질문하기")
            pdf_output = gr.Textbox(label="답변")
            pdf_button.click(handle_pdf_query, inputs=[pdf_input, pdf_question], outputs=pdf_output)


        # 뒤로가기 버튼 추가
        back_button = gr.Button("메인화면으로 가기")
        back_button.click(
            lambda: (gr.update(visible=False), gr.update(visible=True)),
            None,
            [platform, back_to_main]
        )

    return platform

# 메인 페이지 정의
with gr.Blocks(css=custom_css) as main_page:
    with gr.Column(visible=True) as main_section:
        gr.HTML("<h1>AI 튜터 플랫폼에 오신 것을 환영합니다!</h1>")
        gr.HTML("<p>AI 기술을 활용해 질문에 답하고, 더욱 효과적으로 학습할 수 있도록 돕는 플랫폼입니다.</p>")
        gr.Image("logo.png", label="플랫폼 로고")  # 로고 이미지 추가 (로고 파일 필요)
        start_button = gr.Button("시작하기")

    with gr.Column(visible=False) as platform_section:
        platform = tutor_platform(main_section)

    # 시작하기 버튼 클릭 시 화면 전환
    start_button.click(
        lambda: (gr.update(visible=False), gr.update(visible=True)),
        None,
        [main_section, platform_section]
    )

main_page.launch(share=True)