import requests
from tqdm import tqdm
import uuid
import time
import json
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='CLOVA OCR')
    parser.add_argument(
        '--data_path', type = 'str', default = 'data/raw_img', help = 'path of raw image')

    args = parser.parse_args()

    return args



def OCR(api_url, secret_key, data_path):
    for img in tqdm(os.listdir(data_path)):
        if 'jpg' in img:
            image_file  = 'available/{}'.format(img)
            file_path = 'data/json_result/json_{}.json'.format(img.split('_')[-1].split('.')[0])

            # Request
            request_json = {'images': [{'format': 'jpg','name': 'demo'}],
                            'requestId': str(uuid.uuid4()),
                            'version': 'V2',
                            'timestamp': int(round(time.time() * 1000))}
            payload = {'message': json.dumps(request_json).encode('UTF-8')}
            files = [('file', open(image_file,'rb'))]
            headers = {'X-OCR-SECRET': secret_key}
            response = requests.request("POST", api_url, headers=headers, data = payload, files = files)
            response = response.json()

            # Save response
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(response, file)
        else:
            pass   
    

if __name__=='__main__':
    args = parse_args()

    api_url = str(input('CLOVA api_url을 입력하세요 :'))
    secret_key = str(input('CLOVA secret key를 입력하세요 :'))
    
    OCR(api_url, secret_key, args.data_path)