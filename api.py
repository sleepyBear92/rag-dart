from openai import OpenAI
import openai
import time
import os

class RequestChatGPT:
    def __init__(self, key: str):
        self.key = key
        os.environ['OPENAI_API_KEY'] = self.key
        self.client = OpenAI()
        self.model_info = {'gpt-4-turbo-preview':{'max_sequence': 128000},
                           'gpt-3.5-turbo-16k-0613':{'max_sequence':16385},
                           'gpt-3.5-turbo-1106':{'max_sequence':4097}, 
                           'text-davinci-003':{'max_sequence':4097}, 
                           'text-babbage-001':{'max_sequence':2049}}
        self.response_token_num = 300
        os.environ
    
    def summary(self, model_name: str, messages: list) -> str:
        # checking the length of a token       
        # Request API
        if model_name in self.model_info.keys():
            fail_num = 0
            while True:
                fail_num += 1
                try:
                    completion = self.client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=0.1,
                        frequency_penalty=1,
                        presence_penalty=1

                    )
                    break
                except openai.error.APIConnectionError as e:
                    print("[openai.error.APIConnectionError]")
                    print(e)
                    print("{}th Connection Failure".format(fail_num))
                    continue
                except openai.error.ServiceUnavailableError as e:
                    print("[openai.error.ServiceUnavailableError]")
                    print(e)
                    time.sleep(3)
                    continue
                except openai.error.APIError as e:
                    print("[openai.error.APIError]")
                    print(e)
                    time.sleep(10)
                    continue
                except openai.error.Timeout as e:
                    print("[openai.error.Timeout]")
                    print(e)
                    continue
                except Exception as e:
                    print("[InvalidError]")
                    print(e)
                    return None
            # return completion['choices'][0]['message']['content']
            return completion
        else:
            raise ValueError("There is no such model with {}.".format(model_name))
