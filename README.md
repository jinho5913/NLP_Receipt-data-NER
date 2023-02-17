# 영수증 거래내역 NER

1. Clova AI의 OCR을 통해 영수증 내 전체 text 추출
2. 정규표현식을 사용하여 거래내역 1차 추출 & NER을 위한 데이터셋 구축
3. `HuggingFace`의 `BERT`를 사용한 NER task 수행
___
```
# train
python train.py
```

epochs = 10

Train_Accuracy = 84%
