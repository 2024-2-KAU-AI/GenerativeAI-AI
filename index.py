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
import json
from datetime import datetime
from dotenv import load_dotenv

# Tesseract 실행 파일 경로 설정 (Windows용)
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 언어 데이터 경로 및 언어 설정 (Windows용)
#tessdata_dir_config = r'--tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata" -l kor+eng'
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
search_logos = []
load_dotenv()
chunk_size = 200  # 청크 크기 (단어 단위)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_NAME = os.getenv("DB_NAME", "search_logos")


# 환경 변수에서 연결 풀 크기 설정 (기본값: 5)
POOL_SIZE = int(os.getenv("POOL_SIZE", 5))

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
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text()
            if not page_text.strip():
                raise ValueError("텍스트 추출 실패")
        except:
            print(f"Page {page_number}: 텍스트 추출 실패, OCR로 전환 중...")
            try:
                image = convert_from_path(file_path, first_page=page_number, last_page=page_number)[0]
                page_text = pytesseract.image_to_string(image, lang='eng')
                if not page_text.strip():
                    print(f"Page {page_number}: OCR 결과가 비어 있습니다. 건너뜁니다.")
                    continue
            except Exception as e:
                print(f"Page {page_number}: OCR 실패: {e}")
                continue
        
        # 청크 크기 조정
        if adaptive_chunking:
            doc = nlp(page_text)
            avg_sentence_length = sum(len(sent.text.split()) for sent in doc.sents) / len(list(doc.sents))
            adjusted_chunk_size = max(chunk_size, int(avg_sentence_length * 3))
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

def create_embeddings_and_index(index_type="FlatL2"):
    """
    PDF 청크 임베딩 생성 및 Faiss 인덱스 생성

    Args:
        index_type (str): 사용할 인덱스 타입 ("FlatL2", "IVFFlat", "HNSWFlat")

    Returns:
        None
    """
    global faiss_index
    print(f"Creating embeddings for pdf_text_chunks: {pdf_text_chunks}")  # 디버깅용
    if not pdf_text_chunks:
        print("Error: pdf_text_chunks is empty. Cannot create embeddings.")
        return
    embeddings = embedding_model.encode(pdf_text_chunks)
    print(f"Generated embeddings: {embeddings}")  # 디버깅용
    if embeddings.shape[0] == 0:
        print("Error: Embeddings are empty. Cannot create FAISS index.")
        return
    dimension = embeddings.shape[1]
    
    if index_type == "FlatL2":
        # 완전탐색 방식
        faiss_index = faiss.IndexFlatL2(dimension)

    elif index_type == "IVFFlat":
        # Inverted File Index 방식
        nlist = 100  # 클러스터 수 (데이터 규모에 맞게 조정)
        quantizer = faiss.IndexFlatL2(dimension)
        faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)

        # 학습 단계 필요
        faiss_index.train(embeddings)

    elif index_type == "HNSWFlat":
        # Hierarchical Navigable Small World 방식
        hnsw_param = 32  # 그래프 내 이웃 수
        faiss_index = faiss.IndexHNSWFlat(dimension, hnsw_param)
        faiss_index.hnsw.efConstruction = 40  # 그래프 빌드 시 성능 조정
        faiss_index.hnsw.efSearch = 20  # 검색 시 성능 조정

    else:
        raise ValueError("지원하지 않는 index_type입니다. 'FlatL2', 'IVFFlat', 'HNSWFlat' 중 하나를 선택하세요.")

    # 임베딩 추가
    faiss_index.add(embeddings)

# top-k 검색 기능
def search_top_k(query, k=5):
    """
    사용자 질문에 따라 top-k 검색 결과 반환

    Args:
        query (str): 검색 질의
        k (int): 반환할 결과의 개수

    Returns:
        str: JSON 형식의 검색 결과
    """
    if faiss_index is None:
        return json.dumps({"error": "PDF가 아직 업로드되지 않았습니다. 먼저 PDF를 업로드해주세요."})

    # k 값 방어 로직
    if k <= 0:
        return json.dumps({"error": "k 값은 양수여야 합니다."})
    if k > len(pdf_text_chunks):
        k = len(pdf_text_chunks)  # 최대 청크 개수로 제한

    # 검색 수행
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(np.array(query_embedding), k)

    # 검색 결과 생성
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "chunk": pdf_text_chunks[idx],
            "score": float(dist)  # numpy 값 JSON 직렬화 처리
        })

    # 검색 로그 저장
    search_logos.append({
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "results": results
    })

    # 결과 반환
    return json.dumps({"query": query, "results": results}, ensure_ascii=False, indent=4)


# MySQL 연결 풀 생성
connection_pool = pooling.MySQLConnectionPool(
    pool_name="mypool",
    pool_size=POOL_SIZE,
    pool_reset_session=True,
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME")
)


def log_search_query(user_id, query, results):
    """
    검색 기록을 MySQL 데이터베이스에 저장 (연결 풀링 적용)

    Args:
        user_id (str): 사용자 ID
        query (str): 검색 질의
        results (list): 검색 결과 (JSON 직렬화 가능 객체)
    """
    try:
        # results를 JSON 형식으로 변환
        results_json = json.dumps(results, ensure_ascii=False, indent=4)

        # 데이터베이스 연결
        connection = connection_pool.get_connection()
        cursor = connection.cursor()

        # INSERT 쿼리 실행
        query_statement = "INSERT INTO logs (user_id, query, results) VALUES (%s, %s, %s)"
        cursor.execute(query_statement, (user_id, query, results_json))

        # 변경 사항 커밋
        connection.commit()

    except mysql.connector.Error as err:
        print(f"Database error: {err}")
    finally:
        # 리소스 정리
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
    global pdf_text_chunks
    if pdf is not None:
        pdf_text_chunks = extract_text_from_pdf_with_spacy(pdf.name)
        print(f"Extracted pdf_text_chunks: {pdf_text_chunks}")  # 디버깅용
        if not pdf_text_chunks:
            return "PDF에서 텍스트를 추출하지 못했습니다. 올바른 파일인지 확인해주세요."
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
    
    # JSON 데이터를 불러와 결과 가공
    results = json.loads(search_top_k(question, k=3))  # JSON 문자열 -> Python 객체
    formatted_results = "\n\n".join(
        [f"청크: {item['chunk']}\n점수: {item['score']}" for item in results["results"]]
    )
    
    # 안전한 프롬프트 생성
    safe_prompt = build_safe_prompt(question, formatted_results)
    response = f"질문: {question}\n\n검색 결과:\n{formatted_results}\n\n모델 응답:\n{safe_prompt}"
    
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

