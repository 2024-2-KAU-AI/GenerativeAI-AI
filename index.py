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
import mysql.connector
import os  # 환경 변수 사용을 위한 모듈
import model

# Tesseract 한국어 지원 설정 (MacOS Homebrew 경로)
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"  # Tesseract 실행 파일 경로
tessdata_dir_config = '--tessdata-dir "/opt/homebrew/share/tessdata" -l kor+eng'  # 언어 설정

# spaCy 모델 로드
nlp = spacy.load("en_core_web_sm")

# SentenceTransformer 모델 초기화
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 전역 변수
faiss_index = None
pdf_text_chunks = []  # PDF 청크 저장
chunk_size = 200  # 청크 크기 (단어 단위)

# 환경 변수에서 MySQL 정보 로드
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")  # 기본값은 'password'
DB_NAME = os.getenv("DB_NAME", "search_logs")

# PDF 텍스트 추출 및 청크 분리 (spaCy 사용)
def extract_text_from_pdf_with_spacy(file_path, chunk_size=200):
    """
    PDF에서 텍스트를 추출하고 spaCy를 사용해 문장 단위로 청크 분리
    """
    global pdf_text_chunks
    pdf_text_chunks = []
    reader = PdfReader(file_path)
    full_text = ""

    # PDF 텍스트 전체 읽기
    for page in reader.pages:
        full_text += page.extract_text()

    # spaCy를 사용해 문장 분리
    doc = nlp(full_text)
    sentences = [sent.text.strip() for sent in doc.sents]

    # 문장을 청크 크기에 맞게 묶기
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        current_chunk.append(sentence)
        current_length += len(sentence.split())
        if current_length >= chunk_size:
            pdf_text_chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
    
    # 마지막 청크 추가
    if current_chunk:
        pdf_text_chunks.append(" ".join(current_chunk))

    return pdf_text_chunks

# 이미지에서 텍스트 추출
def extract_text_from_image(image_path):
    """
    이미지에서 OCR을 통해 텍스트 추출 (한국어 및 특수문자 지원)
    """
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, config=tessdata_dir_config)
    return text

# 환각 방지 프롬프트 설계
def build_safe_prompt(query, context_chunks):
    """
    모델이 환각하지 않도록 안전한 프롬프트 생성
    """
    prompt = (
        "아래는 PDF에서 추출한 관련 텍스트입니다. "
        "이 텍스트를 참고하여 사용자의 질문에 정확히 답변해주세요.\n\n"
        f"PDF 텍스트:\n{context_chunks}\n\n"
        f"사용자 질문: {query}\n\n"
        "답변:"
    )
    return prompt

# 임베딩 생성 및 Faiss 인덱스 생성
def create_embeddings_and_index():
    """
    PDF 청크 임베딩 생성 및 Faiss 인덱스 생성
    """
    global faiss_index
    embeddings = embedding_model.encode(pdf_text_chunks)
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)

# top-k 검색 기능
def search_top_k(query, k=5):
    """
    사용자 질문에 따라 top-k 검색 결과 반환
    """
    if faiss_index is None:
        return "PDF가 아직 업로드되지 않았습니다. 먼저 PDF를 업로드해주세요."
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(np.array(query_embedding), k)
    results = [pdf_text_chunks[idx] for idx in indices[0]]
    return results

# 데이터베이스 로그 관리
def log_search_query(user_id, query, result_indices):
    """
    검색 기록을 MySQL 데이터베이스에 저장
    """
    connection = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )
    cursor = connection.cursor()
    query_statement = "INSERT INTO logs (user_id, query, results) VALUES (%s, %s, %s)"
    cursor.execute(query_statement, (user_id, query, str(result_indices)))
    connection.commit()
    connection.close()

# Gradio 핸들러: 텍스트 질의
def handle_text_query(text):

    return text

# Gradio 핸들러: 이미지 질의
def handle_image_query(image, question):
    image_analysis = extract_text_from_image(image)
    return f"이미지 기반 답변:\n{image_analysis}"

# Gradio 핸들러: PDF 업로드 처리
def handle_pdf_upload(pdf):
    """
    PDF 업로드 후 처리
    """
    if pdf is not None:
        extract_text_from_pdf_with_spacy(pdf.name)
        create_embeddings_and_index()
        return "PDF가 성공적으로 업로드되었습니다. 질문을 입력해주세요."
    return "PDF 파일을 업로드하세요."

# Gradio 핸들러: PDF 질문 처리
def handle_pdf_query_with_question(question):
    """
    PDF 업로드 후 질문 입력에 대한 답변 반환
    """
    if not faiss_index:
        return "PDF가 아직 업로드되지 않았습니다. 먼저 PDF를 업로드해주세요."
    if not question:
        return "질문을 입력해주세요."
    
    results = search_top_k(question, k=3)
    context = "\n\n".join(results)
    safe_prompt = build_safe_prompt(question, context)
    # LLM에 프롬프트 전달 (여기서는 예시로 응답)
    response = safe_prompt
    return response

# CSS 스타일 정의
custom_css = """
body {
    background-color: #001f3f;
    color: white;
}
.gradio-container {
    background-color: #001f3f;
    color: white;
}
h1, h2, h3, p {
    color: white;
}
"""

# Gradio 플랫폼 정의
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
            image_button = gr.Button("텍스트 추출하기")
            image_output = gr.Textbox(label="추출된 텍스트")
            image_button.click(handle_image_query, inputs=[image_input, image_question], outputs=image_output)
        with gr.Tab("PDF"):
            pdf_input = gr.File(label="PDF 업로드")
            pdf_question = gr.Textbox(label="PDF 관련 질문")
            pdf_button_upload = gr.Button("PDF 업로드")
            pdf_button_question = gr.Button("질문하기")
            pdf_output = gr.Textbox(label="답변")
            
            # PDF 업로드 버튼
            pdf_button_upload.click(handle_pdf_upload, inputs=pdf_input, outputs=pdf_output)
            
            # 질문 버튼
            pdf_button_question.click(handle_pdf_query_with_question, inputs=pdf_question, outputs=pdf_output)
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
        gr.Image("logo.png", label="플랫폼 로고")
        start_button = gr.Button("시작하기")
    with gr.Column(visible=False) as platform_section:
        platform = tutor_platform(main_section)
    start_button.click(
        lambda: (gr.update(visible=False), gr.update(visible=True)),
        None,
        [main_section, platform_section]
    )

# Gradio 실행
main_page.launch(share=True)