from api import RequestChatGPT
import re

def divide_question(question:str, key:str, model_name:str):
    # question = "롯데정보통신과 삼성SDS의 사업분야는?"
    messages = [{"role": "system", "content": "내가 기업 분기보고서에 대한 질문을 할테니, 질문을 '주제'와 '기업'으로 분리해서 dictionary 형태로 데이터를 줘."},
                {"role": "assistant", "content": "네 알겠습니다."},
                {"role": "user", "content": "[질문]\n삼성SDS와 롯데정보통신에서 진행하는 사업을 비교해줘"},
                {"role": "assistant", "content": "{'삼성SDS': ['삼성SDS에서 진행하는 사업은?'], '롯데정보통신': ['롯데정보통신에서 진행하는 사업은?']}"},
                {"role": "user", "content": "[질문]\n삼성SDS와 롯데정보통신의 매출액을 비교해줘."},
                {"role": "assistant", "content": "{'삼성SDS': ['삼성SDS의 매출액은?'], '롯데정보통신': ['롯데정보통신의 매출액은?']}"},
                {"role": "user", "content": "[질문]\n롯데정보통신의 최대 주주가 누구야?"},
                {"role": "assistant", "content": "{'롯데정보통신': ['롯데정보통신의 최대 주주가 누구야?']}"},
                {"role": "user", "content": "[질문]\n롯데정보통신에서 진행하고 있는 블록체인 사업과 향후 진행 계획을 알려줘."},
                {"role": "assistant", "content": "{'롯데정보통신': ['롯데정보통신에서 진행하고 있는 블록체인 사업을 알려줘', '롯데정보통신에서 진행하고 있는 블록체인 사업의 향후 진행 계획을 알려줘.']}"},
                {"role": "user", "content": f"[질문]\n{question}"}]
    
    api = RequestChatGPT(key)

    # regenerate dict 
    iter_num = 3 # Recreate until the dictionary is generated.

    for i in range(iter_num):
        result = api.summary(model_name, messages)
        result = eval(result.choices[0].message.content)
        if isinstance(result, dict):
            break

    # if not dict, assert error
    assert isinstance(result, dict), f"Expected value to be dictionary, got {type(result)} instead."    
    return result

def match_doc(questions:dict, docsearch:dict, key:str, model_name:str, n_chunk:int=3, intv=30):
    '''
    split_docsearch = {'삼성SDS':[
                                  ['이 문서는..',
                                   '이 페이지는..',]
                                  ['여기는..',
                                   '이 부분 부터는..']
                                 ]
                      }
    
    match_dic = {'삼성SDS': [
                             {'삼성은 사업분야는 뭐야?: [
                                              '이 문서는..',
                                              '알랄라..',
                                              '울룰루..
                                             ]
                             },
                
                             {'삼성 매출은 얼마야?: [
                                              '이 문서는..',
                                              '알랄라..',
                                              '울룰루..
                                             ]
                             }
                            ]
                }     
                             
    '''
    # prompt 예시
    question1 = "롯데정보통신의 블록체인 사업에 대해 알려줘"
    context1 = '''
[1] 롯데정보통신은 최근 몇 년간 사업 다각화를 위해 여러 새로운 분야에 진출하고 있습니다. 2021년과 2022년, 그리고 2023년에 걸쳐 추가된 주요 사업목적으로는 모빌리티 관련 사업(카커머스, 자율주행 및 전기차 관련 하드웨어 및 소프트웨어, 공유·커넥티드 서비스 등), 기계설비 성능점검업, 건강관리 서비스업, 디지털 자산 제작 판매 및 중개업이 포함됩니다. 이들 신규 사업 추진을 통해 롯데정보통신은 기존 IT 역량 강화와 함께 매출 다각화를 도모하며 미래 성장동력 확보에 긍정적인 영향을 기대하고 있습니다.\n\n한편 일부 사업목적에서는 수정과 삭제가 이루어져 핵심 역량 집중을 위한 조치가 취해진 것으로 보입니다. 예를 들어 선불카드 및 기타 선불 지급수단의 발행 등의 사업이 종료되거나 수정되면서 롯데정보통신은 주력사업에 더욱 집중하는 방식으로 전략을 조정하였습니다.
[2] 롯데정보통신은 롯데그룹의 디지털 변환을 선도하는 회사로서, 다양한 산업군 데이터를 활용해 고객에게 인사이트를 제공하고 비즈니스 변환을 지원하기 위해 데이터센터 구축 및 운영 사업 목적을 추가하였습니다. 국내 클라우드 시장과 데이터센터 시장의 성장성에 대한 전망, 신규 사업 관련 투자와 자금 소요액, 그리고 현재까지의 사업 추진 현황 등이 기술되어 있습니다. 또한 정관상 추가된 여러 사업목적들이 나열되어 있으며, 이는 정보통신공사부터 모빌리티 관련 서비스까지 다양합니다.
[3] 롯데정보통신은 메타버스 사업과 NFT(Non-Fungible Token)를 활용한 디지털 자산 제작, 판매 및 중개 사업으로 진출하였습니다. 이 회사는 블록체인 기술을 연구 개발하는 전담 팀을 운영하며, 이미 '이대호 NFT' 발행 등의 성과를 보여주었습니다. 롯데 계열사와 협력하여 추가적인 NFT 발행도 검토 중입니다.\n\n블록체인 및 디지털 자산 관련 사업은 롯데정보통신이 기존에 영위해오던 SI(Solution Integration), SM(System Management) 사업과 연관성이 깊으며, 최근 시장에서 부정적 전망에도 불구하고 신규 사업 진출로 인한 기술력 발전의 기회로 보고 있습니다. 향후 롯데 계열사 및 다양한 파트너와 협력하여 다양한 종류의 NFT를 판매할 계획입니다.\n\n별도로, IT/DT 서비스 컨설팅, 개발/구축 및 운영 분야에서 마이데이터 시스템 구축과 해당 플랫폼 구축/운영 사업을 추진 중입니다. 과학기술정보통신부 예상에 따르면 국내 데이터 산업 규모는 2023년 약 30조 원으로 성장할 것으로 보이며, 이러한 시장 성장성을 바탕으로 당사는 안정적인 자금 소요액 범위 내에서 해당 사업을 운영할 예정입니다. 고객 데이터 통합 관리 시스템 구축을 목적으로 하여 금융, 의료, 공공 및 에너지 분야까지 서비스 확대가 예상됩니다.
[4] 롯데정보통신은 IDC(인터넷 데이터 센터) 사업과 메타버스 관련 디지털 자산 제작 및 판매 사업을 진행하고 있습니다. \n\nIDC 사업에서는 4개의 데이터센터를 안정적으로 운영하며, 무중단 및 무장애 인프라 환경 구축을 통해 신뢰도를 향상시켜왔습니다. 이러한 경험과 노하우로 대내외 고객사에 서비스를 확장하여 전산 장비 추가 확장 및 노후화 설비 교체 등의 프로젝트를 수주하였으나, 고객사와의 비밀유지계약으로 인해 상세 내용은 기재되지 않았습니다.\n\n메타버스 분야에서는 2021년 자회사 칼리버스 인수를 통해 메타버스 플랫폼 구축에 착수하였으며, 가상세계에서 현실세계와 같이 경제활동이 가능하도록 하기 위해 블록체인 기반 디지털 자산 제작에 주력하고 있습니다. 글로벌 메타버스 시장은 크게 성장할 것으로 예상되며 NFT 시장 역시 발전 중입니다.\n\nIDC 사업은 수요 증가에도 불구하고 수도권 전력공급 포화와 입주민 반대 등 리스크가 존재하지만 롯데정보통신은 해당 분야에서 충분한 경험이 있다고 언급합니다. 회사는 신규 데이터센터 건설 계획을 검토 중이나 영업 비밀로 인해 세부 정보는 공개하지 않았습니다.\n\n메타버스 관련 서비스 개발과 디지털 자산 제작 및 판매/중개 역시 IT사업의 일환으로 추진되며, 추가적인 인력확보나 투자금 조달 계획은 아직 정해진 바 없음을 명시합니다.
'''

    # 예시 prompt 전처리
    context1 = context1.replace('\n\n', '\n')
    # 문서정보를 n개 단위로 입력해서 k개 페이지 찾도록 배치처리함
    match_dic = {}

    # split docs
    # {compnay: [[1~100], [101~200], ... }
    split_docsearch = {}    
    # 질문한 기업의 문서에서만 찾도록 설정 
    for c, qs in questions.items():  
        # 질문한 기업에 대한 문서정보가 있어야함
        # 있는 경우 문서정보를 n개씩 배치처리해서 하나의 list로 만듬
        if c in docsearch.keys():
            c_script = []
            # split scripts
            page_num = 0 # chunk num
            while True:
                # chunk 개수를 초과하면, 문서 정보 split을 중단
                # 이전 phase에서 page_num에 intv를 더한 페이지까지 이미 확보했으므로, 바로 중단.
                if page_num > len(docsearch[c]):
                    break
                # c_script = [[1page, 2page, 3page, ...], [101page, 102page, 103page, ...]]
                
                if page_num >= 5:
                    start_num = page_num-5
                else:
                    start_num = page_num
                end_num = page_num+intv
                
                c_script.append(list(docsearch[c].keys())[start_num:end_num])
                page_num += intv
            split_docsearch[c] = c_script

            # 기업별 문서 정보 수집
            match_dic[c] = {}
            
            # 질문 선택
            for q in qs:
                match_dic[c][q] = []
                selected_scripts = []

                # prompt 형태로 가공
                for scripts in split_docsearch[c]:
                    informs = ""
                    
                    # 이전 phase에서 찾은 문서정보를 신규 문서정보의 앞에 추가
                    scripts = selected_scripts + scripts
                    selected_scripts = []

                    # list에 담긴 문서정보를 하나의 문자열 str으로 변형
                    for i, script in enumerate(scripts):
                        informs += f"[{i}]. {script}\n"
                    informs = informs.replace('\n\n', '\n')
                    
                    messages = [{"role": "system", "content": f"내가 [질문]과 [문서 정보]를 줄테니, [질문]의 답변에 사용하기 가장 적절한 [문서 정보]를 알려줘. [문서정보]를 {n_chunk}가지 이하만 선정해서 list로 구성해줘."},
                                {"role": "assistant", "content": "네 알겠습니다."},
                                {"role": "user", "content": f"[질문]\n{question1}\n[문서 정보]\n{context1}"},
                                {"role": "assistant", "content": "[3, 4]"},
                                {"role": "user", "content": f"[질문]\n{q}\n[문서 정보]\n{informs}"}]
                    api = RequestChatGPT(key)
                    result = api.summary(model_name, messages)
                    
                    # 생성 결과 후처리
                    try:
                        selected_pages = eval(result.choices[0].message.content)
                        # 생성 결과가 예상을 벗어난 경우 대응
                        if not isinstance(selected_pages, list):
                            assert False, "selected_pages type error"
                    except:
                        # string이므로 정규식으로 정수만 추출
                        # 정수와 콤마로만 구성된 부분을 찾음
                        pattern = r'\b\d+(?:,\d+)*\b'
                        matches = re.findall(pattern, selected_scripts)
                        if matches:
                            integers = []
                            for match in matches:
                                # 콤마로 구분된 숫자들을 분리하여 정수로 변환
                                nums = match.split(',')
                                integers.extend([int(num) for num in nums])
                            selected_pages = integers
                        else:
                            selected_pages = []
                    for page_num in selected_pages:
                        selected_scripts.append(scripts[page_num])

                # 질문마다 매칭된 스크립트를 문서로 치환
                for script in selected_scripts:                
                    match_dic[c][q].append(docsearch[c][script])

    return match_dic  

def generate_answer(match_dic:dict, key:str, model_name:str):
    answer_dic = {}
    for c in match_dic.keys():
        for q, docs in match_dic[c].items():
            context = ""
            for d in docs:
                context += d + '\n'
            context = context.strip()
            answer_dic[q] = []
            messages = [{"role": "system", "content": "내가 [질문]을 줄테니 [문서]에서 해당 정보를 찾아줘."},
                        {"role": "assistant", "content": "네 알겠습니다."},
                        {"role": "user", "content": f"[질문]\n{q}\n[문서]\n{context}"}]
            api = RequestChatGPT(key)
            result = api.summary(model_name, messages)
            answer_sen = result.choices[0].message.content
            answer_dic[q].append(answer_sen)
    return answer_dic

def summary_answer(question, answer_dic, key, model_name):
    context = ""
    for q, a in answer_dic.items():
        context += a[0] + "\n"
        context = context.strip()
        
    messages = [{"role": "system", "content": "내가 [질문]과 [정보]를 줄테니, [정보]만 사용해서 [질문]에 대한 답변을 해줘."},
                {"role": "assistant", "content": "네 알겠습니다."},
                {"role": "user", "content": f"[질문]\n{question}\n[정보]\n{context}"}]
    api = RequestChatGPT(key)
    result = api.summary(model_name, messages)

    return result.choices[0].message.content