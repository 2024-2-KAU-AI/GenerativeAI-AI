#!/usr/bin/env python
# coding: utf-8

import gradio as gr
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import pytesseract
from PIL import Image
import spacy
import mysql.connector
from mysql.connector import pooling
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import subprocess

# Huggingface 로그인 함수
def huggingface_login():
    load_dotenv()
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    if not huggingface_token:
        raise ValueError("Huggingface token not found")
    try:
        subprocess.run(["huggingface-cli", "login", "--token", huggingface_token],
                       check=True,
                       text=True,
                       capture_output=True)
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        raise ValueError("Huggingface login failed")

# Huggingface 모델 초기화
huggingface_login()
pipe = pipeline('text-generation', model="meta-llama/Llama-3.2-1B", device=-1)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

def generate_response(prompt):
    """
    Huggingface 모델을 사용하여 응답 생성
    """
    try:
        response = pipe(prompt, max_length=100, truncation=True)[0]["generated_text"]
        return response
    except Exception as e:
        return f"오류 발생: {e}"

# Gradio 핸들러: PDF 질문 처리
def handle_pdf_query_with_question(question):
    """
    PDF 업로드 후 질문 입력에 대한 답변 반환 (Huggingface 모델 통합)
    """
    if not pdf_text_chunks:
        return "PDF가 아직 업로드되지 않았습니다. 먼저 PDF를 업로드해주세요."
    if not question:
        return "질문을 입력해주세요."
    
    # JSON 데이터를 불러와 결과 가공
    results = json.loads(search_top_k(question, k=3))  # JSON 문자열 -> Python 객체
    formatted_results = "\n\n".join(
        [f"청크: {item['chunk']}\n점수: {item['score']}" for item in results["results"]]
    )
    
    # Huggingface 모델에 질문 전달
    response = generate_response(question)
    return f"질문: {question}\n\n검색 결과:\n{formatted_results}\n\n모델 응답:\n{response}"

# Gradio 실행
with gr.Blocks() as main_page:
    with gr.Column() as main_section:
        gr.HTML("<h1>AI 튜터 플랫폼</h1>")
        start_button = gr.Button("시작하기")
    with gr.Column(visible=False) as platform_section:
        pdf_input = gr.File(label="PDF 업로드")
        pdf_question = gr.Textbox(label="PDF 관련 질문")
        pdf_button_upload = gr.Button("PDF 업로드")
        pdf_button_question = gr.Button("질문하기")
        pdf_output = gr.Textbox(label="답변")
        
        # PDF 업로드 버튼
        pdf_button_upload.click(handle_pdf_upload, inputs=pdf_input, outputs=pdf_output)
        
        # 질문 버튼
        pdf_button_question.click(handle_pdf_query_with_question, inputs=pdf_question, outputs=pdf_output)
    start_button.click(
        lambda: (gr.update(visible=False), gr.update(visible=True)),
        None,
        [main_section, platform_section]
    )

main_page.launch(share=True)
