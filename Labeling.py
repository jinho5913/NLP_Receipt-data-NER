
import requests
import uuid
import time
import json
import os
import re
import numpy as np
import pandas as pd
import datetime
import tqdm
import gc
import parser
from tqdm import tqdm

## 고정값 ##
api_url = 'https://51731638c7f54a7d8b86d2b890c3e4f3.apigw.ntruss.com/custom/v1/18026/623dbb6241510df8698fbf0e94208b7955ce61c5bc2c3ef76a11991718a6f755/general'
secret_key = 'R2poTk50QXZwWktzQVNXcHRtaXRMU0Fxd3hJeUxQVWQ='


## regex compilers ##
# 컬럼 유형에 따른 정규식 컴파일

# 단가 수량 금액
aqa = re.compile("( |^)(\d{0,3}[,.]\d{3}|0|\d{1,3}[,.]\d{1,3}[,.]\d{3}|\d{0,3}\d{3}원) (\d+\.\d{1,2}|\d{1,2}|!|I) (\d{0,3}[,.]\d{3}|0|\d{1,3}[,.]\d{1,3}[,.]\d{3}|\d{0,3}\d{3}원)($| )")
# 단가 수량
aq = re.compile("( |^)(\d{0,3}[,.]\d{3}|0|\d{1,3}[,.]\d{1,3}[,.]\d{3}|\d{0,3}\d{3}원) (\d+\.\d{1,2}|\d{1,2}|!|I)($| )")
# 수량 금액
qa = re.compile("( |^)(\d+\.\d{1,2}|\d{1,2}|!|I) (\d{0,3}[,.]\d{3}|0|\d{1,3}[,.]\d{1,3}[,.]\d{3}|\d{0,3}\d{3}원)($| )")
# 수량 할인 금액
qda = re.compile("( |^)(\d+\.\d{1,2}|\d{1,2}|!|I) (\d*[,.]\d{3}|0) (\d{0,3}[,.]\d{3}|0|\d{1,3}[,.]\d{1,3}[,.]\d{3}|\d{0,3}\d{3}원)($| )")

# 컬럼 유형별 : 컴파일러, 수량 추출 위치, 금액 추출 위치 저장 
do_aqa = {'compiler':aqa, 'quant': 3, 'amount': 4}
do_aq = {'compiler':aq, 'quant': 3, 'amount': 2}
do_qa = {'compiler':qa, 'quant': 2, 'amount': 3}
do_qda = {'compiler':qda, 'quant': 2, 'amount': 4}

dowhat = {'aqa':do_aqa, 'aq':do_aq, 'qa':do_qa, 'qda':do_qda}

# 제외 용도
cols = re.compile('수 ?량|소 ?계|금 ?액|내 ?역|기타 \(\d+|진료 \(\d+|용품 \(\d+|입원 \(\d+|백신 \(\d+|미용 \(\d+') # 컬럼 및 소목차
lessormore = re.compile('\(C:|\(P:|(?<!\S)[-\+](\d+[,.])?\d{3}(?!\S)|\(\d*[,.]\d{3}\)') # 할인/이벤트 line 구분
toolow = re.compile('비과세(?!\))|사업자|대 ?표 ?자|담 ?당 ?자?|수의사|고객|동물명|\d{4}-\d{2}-\d{2}|결 ?제|\[20\d{7}\]') # 무관한 line 구분
endhere = re.compile('청 ?구 ?금 ?액|총 ?액|총 ?금 ?액|^합 ?계|영 ?수 ?액|부 ?가 ?세[ 액]|과세 ?품목 ?합계|표시가 되어 있는') # 진단 추출 끝낼 지점 지정

# 동물명 
name_type = '(?<![가-힣])동물명(?!원) ?:? ?([가-힣]+)|동물이름 ?:? ?([가-힣]+)| ?\(?([가-힣]+)[a-zA-Z]* ?\[\d{8,9}]?'
name = re.compile('(?<![가-힣])동물명(?!원) ?:? ?([가-힣]+)|동물이름 ?:? ?([가-힣]+)| ?\(?([가-힣]+)[a-zA-Z]* ?\[\d{8,9}]?')
# name with date
name_d = re.compile(' ?\(?([가-힣]+)[a-zA-Z]*?\[\d{8,9}]?')

# 위의 컴파일러들 ("윗줄" 연결 여부 결정에 사용)
nope = [aqa, qa, aq, qda, cols, lessormore, toolow]





## Load Data

def data_load_into_df(json_path):

    # make the dataframe
    Satze = []  # 영수증별로 line 단위 문자열 저장
    Num = []    # 영수증 식별번호 저장
    Txt = []    # 영수증 내 텍스트 하나의 문자열로 저장
    # label all words
    Satze_wrd = [] # all receipts' words ---------- by sentence

    for i, image in enumerate(os.listdir( json_path )):  # <----- FOR this receipt
        if '.json' in image:
            receipt_no = image.split('_')[-1].split('.')[0]
            with open( json_path + '/' + image, 'r') as f:
                pet = json.load(f)

            # for Satze_wrd
            this_r_w = []   # this receipt's sentences' words 

            # for Satze
            this_pet_txts = []
            satz = ''

            # for Txt
            wholetext = ''

            idx = 0

            for i, j in enumerate(pet.get('images')[0].get('fields')):  # <--- FOR this sentence

                satz += j.get('inferText') + ' '  # 문장에 단어 추가            
                wholetext += j.get('inferText') + ' '

                if j.get('lineBreak'):    # <----- end of the sentence
                    # 한 문장 끝나면 모으기
                    this_pet_txts.append(satz)

                    # 그 문장 공백단위 쪼개서 단어+인덱스 리스트 만들기
                    this_s_w = satz.split()   # <------------------ this sentence's words
                    this_s_i = [ idx + i for i in  list(range(len( this_s_w )))  ]   # <---- this sentence's words' indices
                    idx = this_s_i[-1] + 1

                    this_r_w.append(this_s_w)

                    satz = ''

            Satze.append(this_pet_txts)
            Num.append(receipt_no)
            Txt.append(wholetext)
            Satze_wrd.append(this_r_w)

    # Satze_original : 영수증별(col) > 줄별(row) > 공백기준 단어들(sell) 
    Satze_original = pd.DataFrame(Satze_wrd)
    Satze_original.index = Num
    Satze_original = Satze_original.T  

    # Satze_wrd_dft : 공백기준 라벨링용
    Satze_wrd_dft = Satze_original.copy()

    ## 우선 전부 'O'로 라벨링 (추후 라벨링 추가)
    for receipt in Satze_wrd_dft.columns.tolist():
        for j,satz in enumerate(Satze_wrd_dft[receipt]):
            if satz is not None:
                Satze_wrd_dft[receipt][j] = ['O' for i in Satze_wrd_dft[receipt][j]]
                # Satze_df(t) : line break 기준으로 나눈 영수증 내용 => 진단 내용 추출에 사용

    # Satze_dft : line break 기준으로 나눈 영수증 내용 => 진단 내용 추출에 사용
    Satze_df = pd.DataFrame(Satze)
    Satze_df.index = Num
    Satze_dft = Satze_df.T

    # Txt_dft : 공백없이 붙인 영수증 내용 => 컬럼유형 파악에 사용
    txt_df = pd.DataFrame(Txt)
    txt_df.index = Num
    txt_dft = txt_df.T

    return Satze_original, Satze_wrd_dft, Satze_dft, txt_dft


## Desciption part : decide whether upper_line is included or not >> laebl them

# 윗줄 포함 여부 판단

def upper(compiler, receipt, row_no):
    if row_no > 0:
        upperrow = Satze_dft[receipt][row_no]
        if sum([True if (n.search( upperrow ) is None) else False for n in nope]) == len(nope):
            return True
        else:
            return False
    else:
        return False


# 진단 내용 라벨링

def desc_labeling(Satze_wrd_dft, receipt, j, desc):  # [receipt][j]에서 desc 찾기 => 라벨링
    
    desc_idx = [ Satze_dft[receipt][j].split().index( i ) for i in desc.split()]
    for m in range(len(desc_idx)):
        if m == desc_idx[0]:
            Satze_wrd_dft[ receipt ][ j ][ m ] = 'B_desc'
        else:
            Satze_wrd_dft[ receipt ][ j ][ m ] = 'I_desc'
    
    # 혼자 동떨어져 있는 문장부호/알파벳 등 진단 라벨링에서 제거
    for i in re.findall('(?:^| )([^가-힣\s])(?: |$)', desc):
        exclude_w = i.strip()
        Satze_wrd_dft[ receipt ][ j ][  Satze_dft[receipt][j].split().index( exclude_w )  ] = 'O'
        
    return Satze_wrd_dft


# 윗줄 포함 여부 결정 후 desc 내용 확정

def withupperline(Satze_wrd_dft, compiler, receipt, j):
    
    satz = Satze_dft[ receipt ][ j ]
    
    desc = satz[ : compiler.search(satz).start() ].strip()  # 지금 line j의 진단내용
    desc = re.sub('[( )]\d*,\d{3}[( ))]', ' ', desc)     # <-- 1) 진단에서 금액부분 제거
    
    Satze_wrd_dft = desc_labeling(Satze_wrd_dft, receipt, j, desc)
    
    # 윗줄의 진단내용 연결
    if upper(compiler, receipt, j-1):  
        upper_line = Satze_dft[receipt][j-1]
        upper_line = re.sub('[( )]\d*,\d{3}[( ))]', ' ', upper_line)    # <-- 1) 진단에서 금액부분 제거
        
        Satze_wrd_dft = desc_labeling(Satze_wrd_dft, receipt, j-1, upper_line)  # <-- 윗줄 추가시 라벨링 추가
        if Satze_wrd_dft[ receipt ][ j ][ 0 ] == 'B_desc':
            Satze_wrd_dft[ receipt ][ j ][ 0 ] = 'I_desc'  # <-- 기존 내용 시작도 I_desc로 변경
        
        desc = upper_line + ' ' + desc
        
    if upper(compiler, receipt, j-1) & upper(compiler, receipt, j-2):
        upper_line = Satze_dft[receipt][j-2]
        upper_line = re.sub('[( )]\d*,\d{3}[( ))]', ' ', upper_line)    # <-- 1) 진단에서 금액부분 제거

        Satze_wrd_dft = desc_labeling(Satze_wrd_dft, receipt, j-2, upper_line)  # <-- 윗줄 추가시 라벨링 추가
        if Satze_wrd_dft[ receipt ][ j-1 ][ 0 ] == 'B_desc':
            Satze_wrd_dft[ receipt ][ j-1 ][ 0 ] = 'I_desc'  # <-- 기존 내용 시작도 I_desc로 변경
        
        desc = upper_line + ' ' + desc
        
    desc = re.split('(^ |^|^\)) ?(\*|\-|\+)+', desc)[-1].strip()    # *으로 시작하는 비과세항목, 글머리기호 -/+ - */-/+ 부분 제거 
    desc = re.sub('\d{6,}', '', desc).strip()                       # 6개 이상 연속 숫자 제거
    
    return Satze_wrd_dft, desc



## get [which type of columns it is] >> get indicies of 'quant'&'amount' and label 'desc'

# get indicies of 'quant'&'amount' and label 'desc'

def get_NDQA(Satze_wrd_dft, compiler_name, receipt, j):
    
    satz = Satze_dft[ receipt ][ j ]
    
    # 수량 Quantity + labeling
    quant_matched = dowhat[ compiler_name ]['compiler'].search(satz).group( dowhat[ compiler_name ]['quant'] )
    quant_ = re.search('( |^)'+quant_matched+'($| )', satz).group().strip()
    
    # 금액 Amount + labeling
    amount_matched = dowhat[ compiler_name ]['compiler'].search(satz).group( dowhat[ compiler_name ]['amount'] )
    amount_ = re.search('( |^)'+amount_matched+'원?($| )', satz).group().strip()
    
    # 진단 Description + labeling
    Satze_wrd_dft, desc = withupperline(Satze_wrd_dft, dowhat[ compiler_name ]['compiler'] , receipt, j)
    
    return Satze_wrd_dft, quant_, amount_


# get [which type of columns it is] (a)
# 영수증의 "컬럼 형식" 파악하고 그에 따라 적합한 [컴파일러 이름] 할당

def typeofcols(receipt):
    coltype = {"단가수량금액" : 'aqa', "수량금액DC" : 'qa', "수량할인금액" : 'qda'}
    for typ in coltype:
        if typ in re.sub('\s', '', txt_dft[receipt][0]):
            return coltype[typ]
    return 'qa'

# get [which type of columns it is] (b)
# 수량, 금액이 실제로 나타난 형식에 따라 적합한 [컴파일러 이름] 할당
def realcolname(satz):
    if (lessormore.search(satz) is None) & (toolow.search(satz) is None) & (cols.search(satz) is None): 
        if aqa.search(satz) is not None : 
            return 'aqa'
        elif aq.search(satz) is not None : # 해당 줄의 금액 부분 안 읽힌 경우
            return 'aq'
        elif qda.search(satz) is not None:
            return 'qda'
        elif qa.search(satz) is not None : # 해당 줄의 단가 부분 안 읽힌 경우
            return 'qa'
        else:
            return False



## [after labeling] Make it as the right form
def right_form_for_NER(txt_df, label_df):
    df_seq = txt_df.copy().T.sort_index()
    df_tgt = label_df.copy().T.sort_index()

    seq_lst = []
    print('##### Making sequence text #####')
    for i in tqdm(range(df_seq.shape[0])):
        s_lst = []
        for j in range(len(df_seq.iloc[i,:])):
            if df_seq.iloc[i,:][j] is not None:
                s_lst += df_seq.iloc[i,:][j]
            else:
                pass
        seq_lst.append(' '.join(s_lst))

    tgt_lst = []
    print('##### Making label #####')
    for i in tqdm(range(df_tgt.shape[0])):
        t_lst = []
        for j in range(len(df_tgt.iloc[i,:])):
            if df_tgt.iloc[i,:][j] is not None:
                t_lst += df_tgt.iloc[i,:][j]
            else:
                pass
        tgt_lst.append(' '.join(t_lst))

    df = pd.DataFrame({'text' : seq_lst, 'labels' : tgt_lst})

    return df



## MAIN ##

def make_data( json_path ):
    print('##### Make Dataset #####')
    global Satze_original, Satze_wrd_dft, Satze_dft, txt_dft
    Satze_original, Satze_wrd_dft, Satze_dft, txt_dft = data_load_into_df(  json_path  )

    for receipt in tqdm(list(Satze_dft.columns)): # 영수증 i       # FOR on Santze_dft[ receipt ]
        
        compiler_name = typeofcols( receipt )
        
        for j, satz in enumerate( Satze_dft[ receipt ].dropna() ):   # FOR on Satze_dft[ receipt ][ j ]
            
            # 1) 동물 이름
            if re.search( name_type , satz):
                matched = name.search(satz).group()
                petname = re.sub(  '.*동물(명|이름) ?:? ?|^\(|\(|[a-zA-Z]* ?\[\d{8,9}\]?'  , '', matched  ).strip()                
                pet_ = re.search('\S*'+petname+'\S*', satz).group()  # <= petname 포함된 단어

                if name_d.search(matched): # 뒤에 날짜가 붙은 유형의 경우
                    Satze_wrd_dft[ receipt ][ j ][ satz.split().index( pet_ ) ] = 'B_nameD' # 그 위치에 라벨링
                else:
                    Satze_wrd_dft[ receipt ][ j ][ satz.split().index( pet_ ) ] = 'B_nameN' # 그 위치에 라벨링
             
            
            # 2) 항목 유형별 [진단/수량/금액] 처리
            # 무의미 line 아닌 경우에만
            elif realcolname(satz):
                compiler_name = realcolname(satz)
                # "desc 라벨링 된 df & quant, amount의 라벨링 위치" 받기
                Satze_wrd_dft, quant_, amount_ = get_NDQA(Satze_wrd_dft, compiler_name , receipt, j)
                # quant, amount 라벨링
                Satze_wrd_dft[ receipt ][ j ][ satz.split().index( quant_ ) ] = 'B_count'
                Satze_wrd_dft[ receipt ][ j ][ satz.split().index( amount_ ) ] = 'B_money'
                
            # 진단내역 지난 후의 line일 경우 => 해당 영수증 검수 종료
            if endhere.search(satz) is not None:
                break
        
        # 다음 영수증 처리 위해 name 초기화
        petname = ''
    
    labeled_df = right_form_for_NER(Satze_original, Satze_wrd_dft)
    
    labeled_df.to_csv('data/df.csv')
    print('========================')

    return labeled_df