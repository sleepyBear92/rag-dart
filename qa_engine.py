from data_management import load_data
from utils import read_config
from rag import divide_question, match_doc, generate_answer, summary_answer

config_file = "config.json"

def run(question, progress):
    progress(0, desc="Starting...")
    
    config_json = read_config(config_file)
    
    key = config_json["api_info"]["key"]
    model_name = config_json["api_info"]["model_name"]
    
    docsearch = load_data(config_json)
    progress(0.1, desc="질문 분석 중...")
    questions = divide_question(question, key, model_name)
    progress(0.3, desc="질문에 알맞은 문서 검색 중...")
    match_dic = match_doc(questions, docsearch, key, model_name, progress)
    progress(0.8, desc="찾은 문서에서 정보 추출 중...")
    answers = generate_answer(match_dic, key, model_name)
    progress(0.9, desc="추출된 정보로 답변 생성 중...")
    result = summary_answer(question, answers, key, model_name)
    progress(0.9, desc="완료!")
    
    return result