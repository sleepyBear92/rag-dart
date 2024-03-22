from api import RequestChatGPT
from utils import read_config
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import pickle

# pdf -> dictionary
def fetch_data(docs_path):
    company = ["롯데정보통신", "삼성SDS", "LG CNS"]
    docs_dic = {}
    for c in company:
        loader = PyPDFLoader(f"{docs_path}/{c}.pdf")
        docs = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size = 512, chunk_overlap=20)
        texts = text_splitter.split_documents(docs)
        docs_dic[c] = []
        for t in texts:
            docs_dic[c].append(f"[{c}]\n\n"+t.page_content)    

    return docs_dic

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
                sentence1 = "[롯데정보통신]\n\nII. 사업의 내용 \n \n1. 사업의 개요\n \n당사와 연결종속회사가 영위하고 있는 사업은 SM(System Management)사업, SI (System\nIntegration)사업, 전기차 충전(EV-Charging)사업으로 구성되어 있습니다.\n \nSM사업은 고객의 요청에 따라 기업 정보시스템과 관련된 설비, 인력, 하드웨어 등을관리, 운\n영하는 것을 의미합니다. 이는 전문화되고 효율적인 IT 서비스 기업에 시스템 운영을 위임하\n는 경영전략입니다. 고객사는 IT 관련 업무를 당사에 아웃소싱 함으로써 고객사의 전략적 사\n업에 역량을 집중할 수 있습니다. 추가로 IT와 관련된 일체의 행위(기획, 운영, 유지보수 등\n)를 당사와 같은 IT서비스 기업에 위탁하여 운영효율을 높이고 IT 투자비를 절감할 수 있습\n니다.\n \nSI사업은 고객에게 필요한 정보시스템 구축 계획을 수립하고, 운영 전략을 기획하며,시스템\n설계 및 개발을 지원하는 서비스를 의미합니다. 즉 고객의 요구에 의하여 일정기간 시스템\n구축 전체를 책임지고 수행하는 서비스라 할 수 있습니다. 이러한 서비스에는 시스템의 설계,\n하드웨어 선정에서 발주, 조달, 애플리케이션 소프트웨어의개발 및 유지보수 등이 포함됩니\n다. 또한 SI 사업 내 신규 먹거리 사업 창출 등을 위하여 메타버스 사업 등에 진출하였습니다.\n메타버스 사업의 경우 VR 관련 레퍼런스 바탕으로 메타버스 플랫폼 제작을 위해 노력하고\n있습니다.\n \n전기차 충전사업은 전기차 충전기 판매사업과 전기차 충전 운영사업으로 나뉩니다. 판매 사\n업의 경우 충전기 full line up을 보유하고 있어 완속부터 급속 충전기까지 제조가 가능하며,\n이러한 역량을 바탕으로 공공 및 민간 전기차 충전기 시장에 진출하였습니다. 충전 운영 사\n업의 경우 전기차 충전기 제조 역량을 기반으로 관련 플랫폼인 EVSIS를 2022년 8월 런칭하\n여 운영중입니다.\n \n당사와 연결종속회사의 주요 제품 및 서비스인 SM사업의 매출은 137,817(백만원)이며 전체\n사업의 16%, SI 사업 매출은 663,035(백만원)으로 전체 사업의 78%, 그리고전기차 충전 사\n업의 매출은 49,501(백만원)으로 전체 사업의 6%를 구성하고 있습니다. 매출의 대부분이 고\n객의 IT 관련 필요에 따라서 제안, 입찰 등의 여러가지 방식을통해 사업을 수주하여 진행되\n고 있으며, 당기말 현재 연결회사의 주요 매출처는 롯데쇼핑㈜, ㈜호텔롯데, 롯데렌탈㈜ 등으\n로 전체 매출의 25%를 구성하고 있습니다. 사업은 고객사 맞춤으로 주문 제작되어 고객의\n요구 수준등에 따라 큰 차이를 보이고 있습니다. 따라서 기성품과 같은 정형화된 판매 가격\n산출이 어려우며 원재료 역시도고객의 요구수준에 따라 변경되기 때문에 가격 산출이 어렵\n습니다.\n \n당사와 연결 종속회사의 사업장은 국내 기준 서울, 청주 등에서 운영하고 있으며 주요 업무\n시설로는 업무동, 연구소, 데이터 센터 등이 있고, 대부분 별도의 생산실적이나 가동률을 기\n재하기는 어렵습니다. 해외에는 종속회사를 포함하여 베트남, 인도네시아 등 각 지역에서 사\n업을 진행하고 있습니다.\n \n전자공시시스템 dart.fss.or.kr Page 23"
                answer1 = "롯데정보통신이 영위 하고 있는 사업에 대한 설명으로 되어 있습니다. 각 사업 분야에 대한 설명과 분야별 수행 내역이 기재되어 있습니다. 그리고 전체 매출 중, 각 사업의 매출 비중을 보여주고 있습니다. 마지막으로 기재하기 어려운 정보들을 명시하고 있습니다."
                sentence2 = "[삼성SDS]\n\n당사는 삼성 관계사의 IT 경쟁력을 지속 강화하는 한편, 대외 시장 확대에도 역량을 집중하\n고 있습니다.  \n \n(2) 물류 부문\n \n물류사업은 자체 개발한 물류 통합관리 플랫폼인 Cello와 SCM 컨설팅 역량을 기반으로 글\n로벌 통합 물류를 실행하고 있습니다. 당사는 2011년 이후, 물류 계획과 실행\n간 연계를 통해 물류 가시성 제고, 원가절감, 효율증대 등의 가치창출을 목표로 하는 글로벌\n4PL 서비스를 지향하고 있으며, 해외법인을 중심으로 물류 서비스를 실행하고 있습니다. 당\n사의 주요 고객은 삼성전자 전세계 사업장이며, 관계사 확산 및 글로벌 고객 확대를 지속적\n으로 추진하고 있습니다.\n \n \n2. 주요 제품 및 서비스\n \n가. 주요 제품 등의 현황 \n \n(1)  IT서비스 사업 \n \n(가) SI\n \n① 솔루션서비스  \n당사는 다양한 산업의 고객을 대상으로 IT서비스를 제공하며 확보한 업종 전문성과 노하우\n를 활용하여 팩토리솔루션, 업무자동화 솔루션, 블록체인, AI/애널리틱스, 엔터프라이즈 모\n빌리티, IoT 등 자체 솔루션을 개발하였습니다. 이를 바탕으로 제조, 공공, 금융, 의료 분야\n고객에게 솔루션 및 서비스를 제공하고 있습니다. \n(단위 : 백만원)\n사업부문 품목매출액\n2023년 3분기\n(제39기)2022년 3분기\n(제38기)2022년\n(제38기)2021년\n(제37기)\nIT서비스SI851,705\n(8.6%)1,169,888\n(9.0%)1,483,889\n(8.6%)1,641,240\n(12.1%)\nITO2,313,073\n(23.4%)2,459,679\n(19,0%)3,321,629\n(19.3%)3,124,619\n(22.9%)\n클라우드1,325,081\n(13.4%)820,903\n(6.3%)1,162,676\n(6.7%)871,338\n(6.4%)\n소계4,489,859\n(45.4%)4,450,470\n(34.3%)5,968,194\n(34.6%)5,637,197\n(41.4%)\n물류5,409,889\n(54.6%)8,534,284\n(65.7%)11,266,556\n(65.4%)7,992,805\n(58.6%)\n합계9,899,748\n(100.0%)12,984,754\n(100.0%)17,234,750\n(100.0%)13,630,002\n(100.0%)\n전자공시시스템 dart.fss.or.kr Page 5"
                answer2 = "삼성SDS의 사업 분야 중 물류 관련하여 물류 부문의 사업 내용과 목표를 담고 있습니다. 그리고 주요 제품 중, IT 서비스 사업인 SI에 대한 내용을 담고 있습니다. 마지막으로 IT서비스SI, 클라우드, ITO, 클라우드, 물류 분야의 매출액 정보를 제공하고 있습니다."
                sentence3 = "[롯데정보통신]\n\n다. 최대주주의 변동 \n주1) 2017년 11월 당사의 설립이후 최대주주였던 롯데아이티테크㈜는 2018년 4월 1일(합병기일)로\n롯데지주㈜와 합병하였습니다. 이 과정에서 당사의 최대주주는 롯데아이티테크㈜에서 롯데지주㈜로\n변동되었습니다.\n \n주2) 당사는 2018년 7월 27일 유가증권시장에 신규상장하였습니다. 신규상장 공모에서\n4,286,000주를 유상증자 신주발행하였습니다. 이에 따라 롯데지주㈜ 지분율이 기존 100.00%에서\n70.00%로 변경되었습니다.\n \n주3) 당사는 2019년 7월 1일을 합병기일로 현대정보기술㈜를 소규모 합병하였으며 2019년 7월\n17일 신주가 상장되어 총 주식수가 14,286,000주에서 15,396,331주로 증가하였습니다. 이에 따라\n롯데지주㈜ 지분율이 기존 70.00%에서 64.95%로 변경되었습니다. \n \n라. 상호의 변경\n롯데정보통신은 해당사항 없습니다.\n \n[이브이시스㈜] \n주1) 2023년 3월 21일 개최된 이브이시스㈜ 주주총회에서 상호를 변경하는 안건이 가결되었으며,\n등기일은 2023년 3월 23일입니다.\n※ 영문으로는『EVSIS Co., Ltd.』로 표기합니다. \n \n마. 회사가 화의, 회사정리절차 그 밖에 이에 준하는 절차를 밟은 적이 있거나 현재 진행중인\n경우 그 내용과 결과\n해당사항 없습니다.2020.03.27 정기주총사외이사 홍철규\n사외이사 양홍석대표이사 마용득\n사내이사 오광우\n사외이사 이성규사외이사 박영석 임기만료 퇴임\n사외이사 임종인 임기만료 퇴임\n2020.12.01 - 대표이사 노준형 - 대표이사 마용득 사임\n2021.03.19 정기주총사내이사 고두영\n사내이사 김경엽대표이사 노준형기타비상무이사 추광식\n임기만료 퇴임('21.3.29)\n2021.12.01 - - - 사내이사 오광우 사임\n2022.03.18 정기주총사내이사 박종표\n사외이사 김상균사외이사 홍철규\n사외이사 양홍석사외이사 이성규\n임기만료 퇴임('22.3.27)\n2023.03.24 정기주총 - 사내이사 고두영 -\n변동년월 최대주주 지분율 비 고\n2018.04. 롯데지주㈜ 100.00% 주1)\n2018.07. 롯데지주㈜ 70.00% 주2)\n2019.07. 롯데지주㈜ 64.95% 주3)\n상호 변경일 변경 전 변경 후 비고\n2023.03.21 중앙제어 주식회사 이브이시스 주식회사 주1)\n전자공시시스템 dart.fss.or.kr Page 7"
                answer3 = "롯데정보통신의 최대주주 변동 내역과 지분율, 그리고 주식 발행 내역이 기재되어 있습니다. 그리고 사외이사, 대표이사, 사내이사 변경 내역이 있습니다."
                messages = [{"role": "system", "content": "내가 기업 분기보고서를 각 페이지마다 입력할테니, 무엇에 대한 내용인지 요약해줘."},
                            {"role": "assistant", "content": "네 알겠습니다."},
                            {"role": "user", "content": sentence1},
                            {"role": "assistant", "content": answer1},
                            {"role": "user", "content": sentence2},
                            {"role": "assistant", "content": answer2},
                            {"role": "user", "content": sentence3},
                            {"role": "assistant", "content": answer3},
                            {"role": "user", "content": f"{d}"}]
        
                api = RequestChatGPT(key)
                result = api.summary(model_name, messages)
                inform = result.choices[0].message.content.replace("\n\n", "\n")
                docsearch[c][inform] = d
            page_num += 1
    return docsearch

# 이미 완성된 문서정보를 불러오기
def load_data(config_json):
    data_path = config_json["data_info"]["path"]
    data_file = config_json["data_info"]["data_file"]

    # pickle 파일로부터 딕셔너리 로드
    with open(data_path+data_file, "rb") as file:
        docsearch = pickle.load(file)

    return docsearch
    

if __name__ == "__main__":
    # JSON 파일 열기
    config_file = "config.json"
    config_json = read_config(config_file)

    # 데이터 사전 적재
    # pdf -> dictionary
    docs_dic = fetch_data(config_json["docs_info"]["path"])
    
    # docs_dic은 key를 회사명으로 하고 value는 list
    # list에는 회사문서와 chunk 내용을 string으로하는 원소를 
    # 문서에 대한 script 작성
    docsearch = parse_data(docs_dic, config_file["api_info"]["key"], config_file["api_info"]["model_name"])
    
    # 문서를 저장
    data_path = config_json["data_info"]["path"]
    dic_file = config_json["data_info"]["dic_file"]
    data_file = config_json["data_info"]["data_file"]

    with open(data_path+dic_file, 'wb') as f:
        pickle.dump(docs_dic, f)

    # 문서 정보를 저장
    with open(data_path+data_file, 'wb') as f:
        pickle.dump(docsearch, f)