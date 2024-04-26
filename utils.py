import json
import os

def read_config(config_file):
    # JSON 파일 열기
    with open(config_file, 'r') as file:
        data = json.load(file)

    return data