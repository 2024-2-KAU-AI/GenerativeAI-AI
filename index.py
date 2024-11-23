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
from mysql.connector import pooling

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
'''
# 환경 변수에서 MySQL 정보 로드
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")  # 기본값은 'password'
DB_NAME = os.getenv("DB_NAME", "search_logs")
'''

# 환경 변수에서 연결 풀 크기 설정 (기본값: 5)
POOL_SIZE = int(os.getenv("POOL_SIZE", 5))

# PDF 텍스트 추출 및 청크 분리 (spaCy 사용)
def extract_text_from_pdf_with_spacy(file_path, chunk_size=200, adaptive_chunking=False):
    """
    PDF에서 텍스트를 추출하고 spaCy를 사용해 문장 단위로 청크 분리

    Args:
        file_path (str): PDF 파일 경로
        chunk_size (int): 기본 청크 크기 (단어 수 기준)
        adaptive_chunking (bool): 텍스트 길이에 따라 chunk_size를 유동적으로 설정할지 여부

    Returns:
        list: 청크 리스트
    """
    global pdf_text_chunks
    pdf_text_chunks = []
    reader = PdfReader(file_path)

    def process_text(text, chunk_size):
        """
        텍스트를 spaCy로 문장 분리 후 청크로 나누는 내부 함수
        """
        doc = nlp(text)
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

    # PDF 텍스트 추출 및 처리
    for page in reader.pages:
        try:
            page_text = page.extract_text()
            if not page_text.strip():
                raise ValueError("텍스트 추출 실패")
        except:
            # OCR로 대체
            images = convert_from_path(file_path)
            for img in images:
                page_text = pytesseract.image_to_string(img, lang='eng')
                process_text(page_text, chunk_size)
                return pdf_text_chunks
        
        # 청크 크기 조정
        
        if adaptive_chunking:
            avg_word_count = len(page_text.split()) // len(page_text.split("\n"))
            adjusted_chunk_size = max(chunk_size, avg_word_count * 3)
        else:
            adjusted_chunk_size = chunk_size

        # 페이지 텍스트 청크로 처리
        process_text(page_text, adjusted_chunk_size)

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

# MySQL 연결 풀 생성
connection_pool = pooling.MySQLConnectionPool(
    pool_name="mypool",
    pool_size=POOL_SIZE,
    pool_reset_session=True,
    host="localhost",
    user="root",
    password="030617",  # MySQL 비밀번호 입력
    database="search_logs"
)

def log_search_query(user_id, query, results):
    """
    검색 기록을 MySQL 데이터베이스에 저장 (연결 풀링 적용)
    """
    try:
        connection = connection_pool.get_connection()
        cursor = connection.cursor()
        query_statement = "INSERT INTO logs (user_id, query, results) VALUES (%s, %s, %s)"
        cursor.execute(query_statement, (user_id, query, str(results)))
        connection.commit()
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# Gradio 핸들러: 텍스트 질의
def handle_text_query(text):
    """
    사용자가 텍스트 질문을 입력했을 때의 처리
    """
    response = f"질문에 대한 답변: {text}"  # 간단한 응답 생성 예시
    log_search_query(user_id="test_user", query=text, results=[response])  # 로그 저장
    return response

# Gradio 핸들러: 이미지 질의
def handle_image_query(image, question):
    """
    사용자가 이미지 업로드 후 질문을 입력했을 때의 처리
    """
    image_analysis = extract_text_from_image(image)  # 이미지에서 텍스트 추출
    response = f"이미지 기반 답변:\n{image_analysis}"  # 간단한 응답 생성 예시
    log_search_query(user_id="test_user", query=question, results=[response])  # 로그 저장
    return response


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
    response = f"모델 응답:\n{safe_prompt}"
    
    # 로그 저장
    log_search_query(user_id="test_user", query=question, results=results)
    
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

