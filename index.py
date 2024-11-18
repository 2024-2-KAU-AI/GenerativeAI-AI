#!/usr/bin/env python
# coding: utf-8


import gradio as gr

def handle_text_query(text):
    response = f"질문에 대한 답변: {text}"  # 실제로는 LLM 응답이 여기에 들어감
    return response

def handle_image_query(image, question):
    image_analysis = "이미지 분석 결과"  # 이미지에서 필요한 정보를 추출하는 과정 추가
    response = f"이미지 기반 답변: {question}에 대한 설명입니다. {image_analysis}"
    return response

def handle_pdf_query(pdf, question):
    extracted_text = "PDF 청크 예시"  # PDF에서 청크를 추출하는 과정 추가
    response = f"PDF 기반 답변: {question}에 대한 설명입니다. 참고 정보: {extracted_text}"
    return response

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

# 기존 플랫폼 페이지 정의
def tutor_platform(back_to_main):
    with gr.Column(visible=True) as platform:
        gr.HTML("<h2>AI 교수님 튜터 플랫폼</h2>")  # 제목을 HTML로 생성

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

        gr.HTML("<h3>DID API 기반 영상 응답</h3>")
        gr.HTML("<p>AI가 생성한 답변을 교수님 캐릭터가 영상으로 설명합니다.</p>")

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





