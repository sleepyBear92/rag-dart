from data_management import load_data, load_embedding
from utils import read_config
from rag import divide_question, match_doc, generate_answer, summary_answer, retriever_doc

config_file = "config.json"

def run(question, search_method, progress):
    # search_method = "both"
    
    progress(0, desc="Starting...")
    
    config_json = read_config(config_file)
    
    key = config_json["api_info"]["key"]
    model_name = config_json["api_info"]["model_name"]

    # path
    data_path = config_json["data_info"]["path"] 
    vector_path = data_path + config_json["data_info"]["vector_path"]
    dic_file = data_path + config_json["data_info"]["dic_file"]
    
    progress(0.1, desc="질문 분석 중...")
    questions = divide_question(question, key, model_name)
    
    if search_method == "summary":
        progress(0.3, desc="질문에 알맞은 문서 검색 중...")
        docsearch = load_data(config_json)
        match_dic = match_doc(questions, docsearch, key, model_name, progress=progress)
    elif search_method == "both" or search_method == "bm25" or search_method == "embedding":
        progress(0.3, desc="Vector 불러오는 중...")
        embedding_retriever_dict = load_embedding(config_json, search_method=search_method)
        match_dic = retriever_doc(questions=questions, retriever=embedding_retriever_dict, progress=progress)
    
    progress(0.8, desc="찾은 문서에서 정보 추출 중...")
    answers = generate_answer(match_dic, key, model_name)
    progress(0.9, desc="추출된 정보로 답변 생성 중...")
    result = summary_answer(question, answers, key, model_name)
    progress(0.9, desc="완료!")
    
    return result