from api import RequestChatGPT
from utils import read_config
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import pickle
import re
import os

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# pdf -> dictionary
def fetch_data(docs_path, chunk_size=512, chunk_overlap=128):
    filenames = get_file_list(docs_path, extension=".pdf")
    docs_dic = {}
    for filename in filenames:
        c = extract_company([filename])[0]
        print(f"{c} chunk 생성 중")
        loader = PyPDFLoader(f"{docs_path}/{filename}")
        docs = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size =chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(docs)
        docs_dic[c] = []
        for t in texts:
            t.page_content = f"[{c}]\n\n"+t.page_content
            docs_dic[c].append(t)

    return docs_dic

def get_file_list(directory, extension=".pdf"):
    return [f for f in os.listdir(directory) if f.endswith(extension)]

def extract_company(filenames:list):
    company_names = []
    for filename in filenames:
        matches = re.findall(r'\[(.*?)\]', filename)
        if matches:
            # 첫 번째 대괄호 안의 내용을 기업 이름으로 간주
            company_name = matches[0]  # '[정정]'이 있는 경우가 있기 때문에 주의 필요
            if company_name != "정정":
                company_names.append(company_name)
            else:
                # '[정정]' 다음에 나오는 기업 이름을 추출
                company_names.append(matches[1] if len(matches) > 1 else None)    
    return company_names

# 문서마다 문서 담긴 정보에 대한 요약문을 생성하여 dictionary형태로 라벨링
def parse_data(docs_dic:dict, key:str, model_name:str, docs_pages:dict={}):
    '''
    docs_dic = {'삼성SDS':[
                           '[삼성SDS]\n\n당사는 삼성...',
                           '[삼성SDS]\n\n당사는 삼성...'
                          ]
                '롯데정보통신':[
                                '롯데정보통신]\n\n당사는 롯데...'
                               ]
               }
    docsearch = {'삼성SDS':{
                            sen1: '삼성SDS의 사업 분야 중...',
                            sen2: '삼성SDS의 사업 분야 중...'
                           ]
                }
    '''
    docsearch = {}
    for c, docs in docs_dic.items():
        if c not in docs_pages.keys():
            # 페이지 번호를 사용해서 특정 페이지만 문서를 만들 수 있음
            docs_pages[c] = list(range(len(docs_dic[c])))
        page_num = 0
        docsearch[c] = {}
        for d in docs:
            if page_num in docs_pages[c]:
                sentence1 = "[기아]\n\n2. 주요 제품 및 서비스\n \n가. 매출 현황 \n  연결실체는 각종 차량의 제조 및 판매를 주요사업으로 영위하고 있으며, 당기 주요재무정\n보는 다음과 같습니다. \n* 연결실체는 보고서 작성 기준일 현재 완성차 및 부분품의 제조ㆍ판매, 렌트 및 정비용역을 사업으로 영위하고\n있으나, 렌트 및 정비부문이 연결실체 전체 매출에서 차지하는 비중이 미미하여 연결실체는 하 나의 사업부문\n으로 구성됨. (단일 사업부문이므로 사업부문별 비중은 작성하지 않음)\n \n나. 주요 제품 등의 가격변동추이 \n* 주요 품목의 가격은 전기차 포함한 단순평균 가격이며, 국내 및 해외에 따라 일부 차종 등이 상이함.\n* 해외의 경우 북미지역(미국), 유럽지역(독일), 아시아지역(호주) 대표시장의 판매가격에 해당 기간\n  말일기준 환율을 적용한  단순 평균가격임.\n (단위 : 백만원)\n구  분 주요제품 금  액\n총매출액\n승용, RV, 상용 등177,811,383\n내부매출액 (78,002,963)\n순매출액 99,808,420\n영업이익 11,607,873\n(단위 : 천원)\n구  분 품  목제80기\n('23.1.1~12.31)제79기\n('22.1.1~12.31)제78기\n('21.1.1~12.31)\n국 내승 용 34,010 34,341 33,659\nR V 47,998 43,559 41,308\n상 용 32,100 31,075 24,864\n해 외승 용 34,099 33,375 32,533\nR V 57,792 50,901 45,456\n전자공시시스템 dart.fss.or.kr Page 26"
                answer1 = "해당 페이지는 주요 제품 및 서비스에 대한 매출 현황과 가격 변동 추이에 대한 정보를 제공하고 있습니다.\n1. 매출 현황: 기아는 완성차 및 부품 제조와 판매, 렌트, 정비용역을 사업으로 영위하고 있으며, 이 중 렌트 및 정비 부문은 전체 매출에서 비중이 작아 단일 사업부문으로 보고되고 있습니다. 총매출액은 177,811,383백만원, 내부매출을 제외한 순매출액은 99,808,420백만원, 영업이익은 11,607,873백만원입니다.\n2. 제품 가격 변동 추이: 제품별로 보면, 국내 승용차, RV(레크리에이션 차량), 상용차의 가격이 각각 다르며 최근 3년간의 평균 가격을 제시합니다. 해외에서의 가격도 북미(미국), 유럽(독일), 아시아(호주) 지역별로 다르며, 전기차를 포함한 평균 가격으로 제공됩니다. 예를 들어, 최근 기준 국내 승용차의 평균 가격은 34,010천원, RV는 47,998천원, 상용차는 32,100천원입니다. 해외에서는 승용차와 RV의 가격이 더 높게 나타나며, 변동 추이를 연도별로 비교 가능합니다.\n\n이 정보를 통해 투자자 및 관련 이해관계자들은 기아의 제품별 매출과 가격 추이를 파악하며, 시장 동향과 경쟁력을 분석할 수 있습니다."
                sentence2 = "[하나은행]\n\n유형자산 2,415,210 2,426,014 2,004,441\n\u3000투자부동산 675,523 664,942 790,689\n\u3000무형자산 428,880 386,107 356,571\n\u3000순확정급여자산 \u3000 77,160 \u3000\n\u3000당기법인세자산 37,770 24,733 18,141\n\u3000이연법인세자산 133,742 316,018 182,851\n\u3000기타자산 18,634,009 13,505,955 13,864,730\n\u3000종합금융계정자산 4,637,824 4,631,361 4,741,939\n\u3000매각예정비유동자산 40,478 36,423 42,130\n\u3000자산총계 498,843,436 485,308,744 430,193,576\n자본과 부채 \u3000 \u3000 \u3000\n\u3000부채 \u3000 \u3000 \u3000\n\u3000\u3000예수부채 369,749,453 359,858,481 321,125,300\n\u3000\u3000당기손익-공정가치측정금융부채 6,729,210 11,228,307 4,188,107\n\u3000\u3000헤지목적파생상품부채 390,979 516,418 111,192\n\u3000\u3000차입부채 22,033,914 22,256,597 17,524,480\n\u3000\u3000사채 26,542,179 26,233,339 27,699,757\n\u3000\u3000순확정급여부채 173,575 9,106 177,077\n\u3000\u3000충당부채 760,573 564,978 518,964\n\u3000\u3000당기법인세부채 165,398 726,579 566,972\n\u3000\u3000이연법인세부채 253,200 2,259 151,143\n\u3000\u3000기타부채 35,805,942 30,280,039 26,465,833\n\u3000\u3000종합금융계정부채 4,154,697 3,667,273 2,908,280\n\u3000\u3000부채총계 466,759,120 455,343,376 401,437,105\n\u3000자본 \u3000 \u3000 \u3000\n\u3000\u3000지배기업의 소유주에게 귀속되는 자본 31,782,740 29,685,969 28,489,982\n\u3000\u3000\u3000자본금 5,359,578 5,359,578 5,359,578\n\u3000\u3000\u3000자본잉여금 6,161,303 6,159,820 9,653,868\n\u3000\u3000\u3000신종자본증권 353,738 533,475 533,475\n\u3000\u3000\u3000연결자본조정 (37,921) (37,686) (38,279)\n\u3000\u3000\u3000연결이익잉여금 21,050,087 19,236,315 13,897,317\n\u3000\u3000\u3000\u3000대손준비금 적립액 2,690,108 2,714,034 2,426,281\n\u3000\u3000\u3000\u3000대손준비금 전입(환입) 필요액 (112,706) (23,926) 287,753\n\u3000\u3000\u3000\u3000대손준비금 전입(환입) 예정액 (112,706) (23,926) 287,753\n\u3000\u3000\u3000연결기타포괄손익누계액 (1,104,045) (1,565,533) (915,977)\n\u3000\u3000비지배지분 301,576 279,399 266,489\n\u3000\u3000자본총계 32,084,316 29,965,368 28,756,471\n\u3000부채및자본총계 498,843,436 485,308,744 430,193,576\n전자공시시스템 dart.fss.or.kr Page 51"
                answer2 = "해당 페이지는 하나은행 하나은행이 지배력을 행사하는 자회들의 재무 정보로 보입니다. 주요 항목들을 통해 추측해보면, 연결재무제표 중에서 연결재무상태표의 일부분인 것 같습니다.  \n하나은행의 업태를 고려했을때, 각 항목에서 얻을 수 있는 주요 정보는 다음과 같습니다. \n\n1. 자산 : 은행의 자산은 주로 은행이 보유하고 있는 금융상품과 대출, 그리고 다른 금융기관에 대한 투자에서 발생합니다. 은행의 주요 자산은 대출 자산과 유동선 자산으로 나눌 수 있습니다. 대출 자산은 은행의 주요 자산 중 하나로, 개인, 기업, 또는 다른 금융기관에 대한 대출을 포함합니다. 대출은 은행 수익의 주된 원천이며, 이자율 및 대출 조건은 은행의 수익성에 직접적인 영향을 미칩니다. 그리고 유동성 자산은 현금, 중앙은행 예치금, 그리고 쉽게 현금화할 수 있는 증권 등을 포함합니다. 이러한 자산은 은행이 단기적인 금융 압박을 견딜 수 있게 해주며, 금융 위기 시에 은행이 즉시 사용할 수 있는 유동성을 제공합니다.\n2. 부채 : 은행의 부채는 주로 고객 예금과 차입금으로 구성되며, 은행의 자금 조달 방식을 나타냅니다. 은행의 주요 부채로는 예금 부채와 차입금이 있습니다. 예금 부채는  고객이 은행에 맡긴 예금으로, 은행 부채의 가장 큰 부분을 차지합니다. 이러한 예금은 체크 계좌, 저축 계좌, 정기 예금 등 다양한 형태가 있으며, 은행은 이 자금을 사용하여 대출을 실행하고 다른 투자를 진행합니다.차입금은 다른 금융기관으로부터의 차입, 국제 금융 시장에서의 자금 조달 등을 포함합니다. 은행은 이러한 자금을 통해 추가적인 유동성을 확보하고, 자산 운용의 규모를 확대할 수 있습니다.\n3. 자본 : 은행의 자본은 은행이 소유하는 자산과 부채의 차이로, 은행의 재무 건전성과 위기 대응 능력을 평가하는 기준이 됩니다. 은행의 주요 자본으로는 자본금, 자본잉여금(이익잉여금), 비지배 지분이 있습니다. 자본금은  은행이 주식을 발행하여 조달한 자본을 의미합니다. 이는 은행이 위험을 감당할 수 있는 기본적인 자금으로 사용됩니다. 자본잉여금(이익잉여금)은 은행이 영업 활동을 통해 얻은 순이익 중 일부를 재투자 목적으로 보유하고 있는 금액입니다. 이는 은행의 자본 안정성을 높이고, 미래의 확장이나 위기 관리에 사용됩니다. 비지배 지분은 자회사에서 지배기업이 소유하지 않은 지분을 나타냅니다. 이는 연결 재무제표에서 중요한 역할을 하며, 전체 기업 집단의 자본 구조를 이해하는 데 필요합니다.\n\n이 정보를 통해 하나은행의 재무 건전성, 수익성, 위험 관리 능력 및 전반적인 경영 효율성을 평가할 수 있습니다."
                sentence3 = "[크래프톤]\n\n라. 직원 등 현황 \n \n마. 미등기임원 보수 현황 \n (기준일 : 2023.12.31 ) (단위 : 백만원)\n직원소속 외\n근로자\n비고\n사업부문 성별직 원 수\n평 균\n근속연수연간급여\n총 액1인평균\n급여액남 여 계기간의 정함이\n없는 근로자기간제\n근로자\n합 계\n전체(단시간\n근로자)전체(단시간\n근로자)\n게임 남 1,005 0 56 11,061 2.9년 126,619 105\n26 93119-\n게임 여 494 3 24 0 518 2.6년 46,020 82 -\n합 계 1,499 3 80 11,579 2.8년 172,639 98 -\n※ 상기 '직원 등 현황'에는 미등기임원이 포함되어 있습니다. (등기임원 제외)\n※ 상기 '직원수'는 공시서류 작성기준일 현재 기준입니다.\n※상기 '연간급여총액'은 소득세법 제20조에 따라 관할 세무서에 제출하는 근로소득지급명세서 상 근로소득 기준입니\n다.\n※ 상기 '1인 평균 급여액'은 월별 평균 급여 합산액 기준입니다.\n(기준일 : 2023.12.31 ) (단위 : 백만원)\n구 분 인원수 연간급여 총액 1인평균 급여액 비고\n미등기임원 2 1,905 953 -\n※ 상기 '인원수'는 공시서류 작성기준일 현재 기준입니다.\n※상기 '연간급여총액'은 소득세법 제20조에 따라 관할 세무서에 제출하는 근로소득지급명세서 상 근로소득 기준입니\n다.\n※ 상기 '1인 평균 급여액'은 월별 평균 급여 합산액 기준입니다.\n전자공시시스템 dart.fss.or.kr Page 346"
                answer3 = "해당 페이지는 크래프톤의 직원과 수석 부사장이나 디렉터 혹은 최고기술책임자 등을 포함한 미등기임원의 보수(임금) 현황에 관한 데이터를 제공하고 있습니다.\n1. 직원 등 현황: 크래프톤의 게임 사업부문에서 근무하는 직원들의 수, 성별 분포, 근속 연수, 연간 급여 총액, 그리고 1인 평균 급여액에 대한 정보를 제공합니다. 직원들은 기간의 정함이 없는 근로자(정규직)와 기간제 근로자로 구분됩니다.\n2. 미등기임원 보수 현황: 크래프톤의 미등기임원(공식적으로 등기되지 않은 임원)에 대한 보수 정보를 포함합니다. \n\n이 정보를 통해 크래프톤의 재무 건전성과 직원 복지 수준을 평가할 수 있고 임금 구조와 관련된 전략적 결정을 내릴 때 기초 자료로 활용할 수 있습니다."
                messages = [{"role": "system", "content": "내가 기업 분기보고서를 각 페이지마다 입력할테니, 무엇에 대한 내용인지 요약해줘."},
                            {"role": "assistant", "content": "네 알겠습니다."},
                            {"role": "user", "content": sentence1},
                            {"role": "assistant", "content": answer1},
                            {"role": "user", "content": sentence2},
                            {"role": "assistant", "content": answer2},
                            {"role": "user", "content": sentence3},
                            {"role": "assistant", "content": answer3},
                            {"role": "user", "content": f"{d.page_content}"}]
        
                api = RequestChatGPT(key)
                result = api.run(model_name, messages)
                inform = result.choices[0].message.content.replace("\n\n", "\n")
                docsearch[c][inform] = d.page_content
            page_num += 1
    return docsearch

# 이미 완성된 문서정보를 불러오기
def load_data(config_json):
    data_path = config_json["data_info"]["path"]
    data_file = config_json["data_info"]["summary_file"]

    # pickle 파일로부터 딕셔너리 로드
    with open(data_path+data_file, "rb") as file:
        docsearch = pickle.load(file)

    return docsearch
    

def save_embedding(docs_dic:dict, embedding_model_path:str, auth_token:str, vector_path:str):
    
    embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model_path,
                model_kwargs={"device": "cuda", "use_auth_token": auth_token}
            )
    
    for company in docs_dic.keys():
        print(f"{company} 벡터화 중")
        faissIndex = FAISS.from_documents(docs_dic[company], embeddings)
        faiss_retriever = faissIndex.as_retriever(
                        search_type="mmr", search_kwargs={"k": 3}
                    )
    
        faissIndex.save_local(os.path.join(vector_path, company))
        

def load_embedding(config_json, search_method="both", weights=[0.5, 0.5]):

    # path
    data_path = config_json["data_info"]["path"]
    vector_path = data_path + config_json["data_info"]["vector_path"]
    dic_file = data_path + config_json["data_info"]["dic_file"]
    embedding_model_path = config_json["model_info"]["embedding_path"]
    
    embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model_path,
                model_kwargs={"device": "cuda", "use_auth_token": config_json["model_info"]["hf_token"]}
            )
    
    file_names = get_file_list(vector_path, extension="")
    
    # 폴더명과 기업명이 일치하므로 별도 전처리는 사용하지 않는다. (폴더명규칙 변경시, 수정 필요)
    # company_list = extract_company(file_names)
    company_list = file_names

    # retrieval 종류별로 보관하기 위한 용도
    embedding_retriever_dict = dict()

    if search_method == "both" or search_method == "bm25":
        # bm25를 위해 dic_file을 read
        with open(dic_file, "rb") as f:
            docs_dic = pickle.load(f)
    
    for company in company_list:
        folder_name = company
        retrievers = [] # retirevers 수집
        if search_method=="both" or search_method=="embedding":
            faiss_retriever = FAISS.load_local(
                os.path.join(vector_path, folder_name), embeddings, allow_dangerous_deserialization=True
            ).as_retriever(search_type="mmr", search_kwargs={"k": 3})
            retrievers.append(faiss_retriever)

        if search_method=="both" or search_method=="bm25":
            bm25_retriever = BM25Retriever.from_documents(docs_dic[company])
            bm25_retriever.k = 2
            retrievers.append(bm25_retriever)

        if search_method == "both":
            retriever = EnsembleRetriever(retrievers=retrievers, weights=weights)
        else:
            retriever = EnsembleRetriever(retrievers=retrievers)
            
        embedding_retriever_dict.update({company: retriever})

    return embedding_retriever_dict
    

if __name__ == "__main__":
    # 문서별 요약 데이터 작성 or 문서 embedding 생성
    db_type = "embedding"
    content_type = "full"
    
    config_file = "config.json"
    config_json = read_config(config_file)
    
    # 문서 저장 경로
    data_path = config_json["data_info"]["path"]
    dic_file = data_path + config_json["data_info"]["dic_file"]
    summary_file = config_json["data_info"]["summary_file"]
    vector_path = config_json["data_info"]["vector_path"]
    
    # 모델 경로
    embedding_model_path =  config_json["model_info"]["embedding_path"]
    auth_token = config_json["model_info"]["hf_token"]
    
    # 데이터 사전 적재
    # pdf -> dictionary
    docs_dic = fetch_data(config_json["docs_info"]["path"])
    
    with open(data_path+dic_file, 'wb') as f:
        pickle.dump(docs_dic, f)
    
    # docs_dic은 key를 회사명으로 하고 value는 list
    # list에는 회사문서와 chunk 내용을 string으로하는 원소를 
    # 문서에 대한 script 작성
    
    if db_type == "openai":
        docsearch = parse_data(docs_dic, config_file["api_info"]["key"], config_file["api_info"]["model_name"])
        with open(data_path+summary_file, 'wb') as f:
            pickle.dump(docsearch, f)
    elif db_type == "embedding":
        if content_type == "full": # 문서 전체 chunk를 기반으로 embedding
            save_embedding(docs_dic, embedding_model_path, auth_token, vector_path)
        elif content_type == "summary": # chunk별 문서 요약을 기반으로 embedding
            pass

        