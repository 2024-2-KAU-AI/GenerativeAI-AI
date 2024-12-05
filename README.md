# AI Tutor Platform

이 프로젝트는 **Gradio 기반 AI 튜터 플랫폼**으로, 텍스트, 이미지, PDF 입력에 대한 상호작용 Q&A 시스템을 제공합니다. 사용자는 플랫폼을 통해 AI 생성 응답을 받고, 직관적이고 사용자 친화적인 인터페이스를 활용할 수 있습니다.

---

## 주요 기능

- **텍스트 기반 Q&A**: 텍스트로 질문을 입력하고 AI 생성 응답을 받을 수 있습니다.
- **이미지 기반 Q&A**: 이미지를 업로드하고 이미지 내용과 관련된 질문을 할 수 있습니다.
- **PDF 기반 Q&A**: PDF 문서를 업로드하고 내용을 기반으로 질문에 답변을 받을 수 있습니다.
- **동적 화면 전환**: 메인 페이지와 Q&A 플랫폼 간의 전환이 간단합니다.
- **사용자 지정 스타일링**: 어두운 파란색 테마와 흰색 텍스트를 사용하는 시각적으로 매력적인 인터페이스.
- **DID 활용 소개 영상**: 메인 화면에서 DID 기술을 활용한 AI 튜터 소개 영상을 제공합니다.

---

## 실행 방법

Google Colab을 사용해 실행하려면 아래 단계를 따라 진행하세요.

### 1. 저장소 복제

Colab에서 새 노트를 생성하고 아래 코드를 실행합니다.

```python
# 저장소 클론
!git clone https://github.com/2024-2-KAU-AI/GenerativeAI-AI.git
%cd GenerativeAI-AI
```

```python
# 필요한 라이브러리 설치
# Gradio
!pip install gradio

# vLLM
!pip install vllm

# PyPDF2 (PDF 처리)
!pip install PyPDF2

# SentenceTransformers (문장 임베딩)
!pip install sentence-transformers

# spaCy 및 영어 모델 (자연어 처리)
!pip install spacy
!python -m spacy download en_core_web_sm

# Pillow (이미지 처리)
!pip install pillow

# Numpy (수치 계산)
!pip install numpy

# Faiss (효율적인 검색 및 임베딩 검색)
# CPU 버전 설치
!pip install faiss-cpu
# GPU 버전을 사용하려면:
# !pip install faiss-gpu

# Tesseract OCR (텍스트 추출)
# Colab에서는 기본 설치됨. 로컬 환경에서는 아래 명령어로 설치 필요:
# !sudo apt-get install tesseract-ocr

# Python-Dotenv (환경 변수 로드)
!pip install python-dotenv

# Transformers (Hugging Face 모델)
!pip install transformers

# Requests (HTTP 요청)
!pip install requests

# TensorFlow (머신러닝)
!pip install tensorflow

# Google Colab (Colab 환경에서만 필요)
# !pip install google-colab

# Jupyter Notebook 관련 (필요 시)
!pip install nbformat nbconvert

# pytesseract 설치
!pip install pytesseract

```

### 2. 환경 변수 설정

`.env` 파일을 생성하여 아래 내용을 추가합니다. Colab에서 `!echo` 명령어로 생성할 수 있습니다.

```bash
!echo """DB_HOST=database-1.cpygs8uky1d6.us-east-1.rds.amazonaws.com
DB_USER=admin
DB_PASSWORD=hwang0719
DB_NAME=search_logos
POOL_SIZE=5
HUGGINGFACE_TOKEN=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
""" > .env

```

### 3. 실행

아래 명령어를 실행하여 플랫폼을 시작합니다.

```python
!python gradio_ai_tutor.py
```

실행 후 제공되는 Gradio URL을 열어 플랫폼에 접속할 수 있습니다.

---

## 프로젝트 파일 구조

- **`gradio_ai_tutor.py`**: Gradio 애플리케이션 로직이 포함된 주요 Python 파일입니다.
- **`huggingfaceLogin.py`**: Hugging Face에 로그인하는 코드를 포함합니다.
- **`.env`**: 환경 변수 파일로 AWS RDS 연결 정보와 Hugging Face 토큰을 저장합니다.
- **`generativeAI.mp4`**: DID 기술을 활용한 AI 튜터 소개 영상 파일로, 메인 화면에서 재생됩니다.

---

## 요구 사항

아래와 같은 주요 라이브러리와 도구가 필요합니다:

- `gradio`
- `vllm`
- `PyPDF2`
- `sentence-transformers`
- `spacy`
- `pillow`
- `numpy`
- `faiss-cpu`
- `tesseract-ocr`
- `python-dotenv`
- `transformers`

---

## 앞으로의 계획

- **DID API 통합**: DID API를 활용해 가상 튜터 캐릭터가 AI 생성 응답을 제공하도록 확장 예정.
- **PDF 파싱 개선**: 더 정확한 답변을 위해 PDF 청크 추출 로직 개선.

---

## 문의

궁금한 점이나 피드백이 있다면 아래 이메일로 문의해주세요:
**hoeun0723@naver.com**