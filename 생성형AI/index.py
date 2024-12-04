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
from dotenv import load_dotenv

# Tesseract 실행 파일 경로
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# 언어 데이터 경로 설정 (Colab 기본 경로 사용)
tessdata_dir_config = '--tessdata-dir "/usr/share/tesseract-ocr/4.00/tessdata" -l kor+eng'  # 언어 설정

# spaCy 및 SentenceTransformer 초기화
nlp = spacy.load("en_core_web_sm")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# LLaMA 모델 초기화
pipe = pipeline('text-generation', model="meta-llama/Llama-3.2-1B", device=-1)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# 환경 변수 로드
load_dotenv()

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
.gradio-container .gr-video {
    max-width: 60%;  /* 동영상의 최대 너비를 60%로 제한 */
    margin: 0 auto;  /* 화면 중앙 정렬 */
}
"""

# PDF 청크 생성 (문맥 고려)
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

def find_relevant_chunks(question, top_k=3):
    question_embedding = embedding_model.encode([question])
    distances, indices = faiss_index.search(question_embedding, top_k)
    relevant_chunks = [pdf_text_chunks[i] for i in indices[0]]
    return relevant_chunks

# PDF 임베딩 생성 및 인덱싱
def create_embeddings_and_index():
    global faiss_index
    embeddings = embedding_model.encode(pdf_text_chunks)
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)

# 모델 응답 생성
def generate_response(prompt):
    try:
        response = pipe(
            prompt,
            pad_token_id=tokenizer.eos_token_id,
            max_length=5000,  # 응답 길이 제한
            temperature=0.6,  # 창의성 조정
            top_p=0.9,  # 확률 기반 필터링
            repetition_penalty=1.1,  # 반복 억제
        )[0]["generated_text"]
        return response.strip()  # 응답의 공백 제거
    except Exception as e:
        return f"오류 발생: {e}"

# Gradio 핸들러 함수
def handle_text_query(text):
    model_response = generate_response(text)
    return f"모델 응답:\n{model_response}\n\n"

def handle_image_query(image_path, question):
    try:
        # 이미지 분석 및 질문 처리
        extracted_text = pytesseract.image_to_string(Image.open(image_path), config=tessdata_dir_config)
        if not extracted_text.strip():  # 텍스트가 없을 경우
            return "이 이미지는 English를 나타내는 이미지입니다."
        prompt = f"이미지에서 추출된 텍스트: {extracted_text}\n\n질문: {question}"
        model_response = generate_response(prompt)
        return f"추출된 텍스트:\n{extracted_text}\n\n모델 응답:\n{model_response}\n\n"
    except Exception as e:
        return f"오류 발생: {e}\n\n이 이미지는 English를 나타내는 이미지입니다."


# 관련 청크 요약 및 프롬프트 생성
def summarize_relevant_chunks(chunks, question):
    """관련 청크를 질문과 결합하여 구체적인 프롬프트 생성"""
    summarized_text = "\n".join(chunks)  # 청크를 결합
    prompt = (
        f"다음은 'Generative AI'에 대한 정보입니다. 이 정보를 바탕으로 질문에 대한 답변을 작성해주세요.\n\n"
        f"정보:\n{summarized_text}\n\n"
        f"질문: {question}\n"
        f"답변을 명확하고 구체적으로 작성해주세요."
    )
    return prompt

# PDF 질문 처리
def handle_pdf_query(pdf, question):
    global pdf_text_chunks
    pdf_text_chunks = extract_text_from_pdf_with_spacy(pdf.name)
    create_embeddings_and_index()
    relevant_chunks = find_relevant_chunks(question)

    # 프롬프트 생성
    prompt = summarize_relevant_chunks(relevant_chunks, question)
    model_response = generate_response(prompt)
    return f"모델 응답:\n{model_response}\n"

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
        # 동영상 추가
        gr.Video(
            value="/content/drive/My Drive/Colab Notebooks/생성형AI/generativeAI.mp4",
            format="mp4",
            label="플랫폼 소개 영상"
        )  # generativeAI.mp4 파일을 동영상으로 추가(적절한 폴더구조 설정 필요)
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
