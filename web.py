import gradio as gr
from qa_engine import run

notice_markdown = """
# 📈 대화형 공시정보 분석 시스템

#### 복잡하고 방대한 공시자료를 보지 않아도 한 번의 질문으로 정보를 얻을 수 있습니다. 기업에 대한 매출, 이익 등 다양한 정보 뿐 아니라 기업 간 비교도 손쉽게 가능합니다! (현재는 프로토타입으로 질문 난이도에 따라 약 1~3분 소요)  
"""


examples = ["롯데정보통신에서 하고 있는 사업을 알려줘", "삼성SDS의 최대 주주는?", "삼성SDS, 롯데정보통신, LG CNS 세 기업의 매출 구조를 비교해줘."]

def exc(question, progress=gr.Progress()):
    answer = run(question, progress)
    return answer

with gr.Blocks(theme="EveryPizza/Cartoony-Gradio-Theme", title="대화형 공시정보봇") as iface:
    gr.Markdown(notice_markdown, elem_id="notice_markdown")
    question = gr.Textbox(label="질문")
    gr.Examples(examples, inputs=question)
    greet_btn = gr.Button("물어보기")
    output = gr.Textbox(label="답변")
    greet_btn.click(fn=exc, inputs=question, outputs=output)

iface.launch(share=True)