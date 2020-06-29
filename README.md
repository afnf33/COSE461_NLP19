

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->
---
### 0. 자연어처리(COSE461 201R) 팀프로젝트 19조 : 박건우, 오영운, 황이레

이하의 코드는 고려대학교 자연어처리(COSE461) 수업 내용을 기반으로  
1. 한국어 감성분석  
2. 영어 감성분석  
을 진행한 소스 코드입니다.  

대부분의 코드는 임희석 교수님의 [자연어처리 바이블 실습코드](https://github.com/nlpai-lab/nlp-bible-code)와  
[자연어처리 수업의 실습 예제 1~5](https://github.com/Parkchanjun/KU-NLP-2020-1)을 참고하였습니다. 


추가적으로 한국어 감성분석의 경우   
cyc1am3n님의 [KoNLPy를 이용한 한국어 영화 리뷰 감정 분석](https://cyc1am3n.github.io/2018/11/10/classifying_korean_movie_review.html)과   
SKTBrain에서 개발한 pretrained [KoBERT](https://github.com/SKTBrain/KoBERT)의 도움을 받았습니다.

#### Requirements

* Python >= 3.6
* konlpy >= 0.5.2
* nltk >= 3.4.5
* tensorflow >= 1.14.0
* matplotlib >= 3.1.3
* PyTorch >= 1.1.0
* MXNet >= 1.4.0
* gluonnlp >= 0.6.0
* sentencepiece >= 0.1.6
* onnxruntime >= 0.3.0
* transformers >= 2.1.1

### 1. 한국어 감성분석
한국어 감성분석은 네이버에서 공개한 [Naver Sentiment Movie Corpus](https://github.com/e9t/nsmc)를 기반으로  
긍정적인 반응(1)과 부정적인 반응(0)을 분류하는 모델을 학습하였습니다

#### 1.1 Keras를 이용한 감성분석

* Architecture

```python
>>> from konlpy.tag import Okt
>>> okt = Okt()
>>> selected_words = [f[0] for f in text.vocab().most_common(1000)] 

>>> from tensorflow.keras import models
>>> from tensorflow.keras import layers
>>> from tensorflow.keras import optimizers
>>> from tensorflow.keras import losses
>>> from tensorflow.keras import metrics

>>> model = models.Sequential()
>>> model.add(layers.Dense(64, activation='relu', input_shape=(1000,)))
>>> model.add(layers.Dense(64, activation='relu'))
>>> model.add(layers.Dense(1, activation='sigmoid'))

>>> model.compile(optimizer=optimizers.RMSprop(lr=0.001),
             loss=losses.binary_crossentropy,
             metrics=[metrics.binary_accuracy])

>>> history=model.fit(partial_x_train,partial_y_train,epochs=10,batch_size=512,validation_data=(x_val,y_val))
```

##### 환경 및 사용법

Jupyter Notebook 환경에서 코드를 실행했습니다  
해당 파일의 주석을 참고하여 그대로 돌리시되, 형태소 분석 결과를 JSON 파일로 저장하고 싶지 않다면 중간 부분의 주석 처리된 부분만 돌려주세요.  
새로운 데이터를 예측할 때에는 마지막 부분의 new_data 항목을 변경해서 Model.predict()를 실행해주시면 됩니다

#### 1.2 koBERT를 이용한 감성분석

* Architecture

SKTBrain의 [koBERT](https://github.com/SKTBrain/KoBERT)의 Tokenizer와 pretrained BERT 모델을 그대로 사용했습니다
```python
predefined_args = {
        'attention_cell': 'multi_head',
        'num_layers': 12,
        'units': 768,
        'hidden_size': 3072,
        'max_length': 512,
        'num_heads': 12,
        'scaled': True,
        'dropout': 0.1,
        'use_residual': True,
        'embed_size': 768,
        'embed_dropout': 0.1,
        'token_type_vocab_size': 2,
        'word_embed': None,
    }
```

##### 사용법
구글 Colab의 GPU 런타임 유형을 사용하였습니다.  
해당 파일의 주석을 참고하여 코드를 실행해주세요.  
Google Drive 안에 predict할 파일을 준비해주시고 마지막의 'dt' 변수에 할당해 주세요  
결과는 Kaggle 리더보드에 업로드할 형식으로 predict.csv 파일로 출력되도록 했습니다  

----
### 2. 영어 감성분석
영어 감성분석은  [Friends EmotionLines](http://doraemon.iis.sinica.edu.tw/emotionlines/download.html)데이터셋을 기반으로  
'neutral','joy','sadness','fear','anger','surprise','disgust', 'non-neutral'의 8가지 감정을 분류하는 모델을 학습하였습니다. 

#### 2.1 CNN 이용한 감성분석

* Architecture

```python
filters = [2,3,4,5]
conv_models = []
for filter in filters:
  conv_feat = layers.Conv1D(filters=100, 
                            kernel_size=filter, 
                            activation='relu',
                            padding='valid'  )(seq_embedded)
                          
  pooled_feat = layers.GlobalMaxPooling1D()(conv_feat) #MaxPooling
  conv_models.append(pooled_feat)

conv_merged = layers.concatenate(conv_models, axis=1) #filter size가 2,3,4,5인 결과들 Concat

model_output = layers.Dropout(0.6)(conv_merged)
logits = layers.Dense(8, activation='softmax')(model_output)

model = Model(seq_input, logits)
model.compile(optimizer='adam',
              loss= losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
```

<img width='1000' src='https://user-images.githubusercontent.com/29119738/85989559-a999a400-ba2b-11ea-93d1-d7591bddcee4.png'>


##### 환경 및 사용법
구글 Colab의 환경을 사용했습니다.   
파일들의 용량이 커서 google drive에 train, dev, test set들과 glove 파일을 두고  
google drive를 연결했습니다.
파일을 불러올 때 drive 내 파일의 경로를 수정하셔서 쓰시면 됩니다.


Google colab 런타임 유형 GPU 사용하였으며
자세한 사용법은 깃허브 ipython 파일 참조


#### 2.2 CNN-LSTM을 이용한 감성분석

* Architecture
```python
model_lstm = layers.LSTM(256, return_state=False)(seq_embedded)
filters = [2,3,4,5]
conv_models = [model_lstm]
for filter in filters:
  conv_feat = layers.Conv1D(filters=100, 
                            kernel_size=filter, 
                            activation='relu',
                            padding='valid'  )(seq_embedded) 
                          
  pooled_feat = layers.GlobalMaxPooling1D()(conv_feat) #MaxPooling
  
  conv_models.append(pooled_feat)

conv_merged = layers.concatenate(conv_models, axis=1) #filter size가 2,3,4,5인 결과들 Concatenation

model_dropout = layers.Dropout(0.5)(conv_merged)

logits = layers.Dense(8, activation='softmax')(model_dropout)



model = Model(seq_input, logits)
model.compile(optimizer='adam',
              loss= losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
```

##### 환경 및 사용법
구글 Colab의 환경을 사용했습니다.   
파일들의 용량이 커서 google drive에 train, dev, test set들과 glove 파일을 두고  
google drive를 연결했습니다.

파일을 불러올 때 drive 내 파일의 경로를 수정하셔서 쓰시면 됩니다.



#### 2.3 Transformer을 이용한 감성분석

* Architecture

```python
embed_dim = 32  # Embedding size for each token, 논문에서는 512차원
num_heads = 2  # Number of attention heads, 논문에서는 8개
ff_dim = 32  # Hidden layer size in feed forward network inside transformer, 논문에서는 2048차원
maxlen = MAX_SEQUENCE_LEN
inputs = layers.Input(shape=(maxlen,)) #처음 입력
embedding_layer = TokenAndPositionEmbedding(maxlen, VOCAB_SIZE, embed_dim) #객체 생성
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim) #객체 생성

x = embedding_layer(inputs)  #포지셔널 임베딩
x = transformer_block(x) #트랜스포머 
x = layers.GlobalAveragePooling1D()(x) #Average Pooling
x = layers.Dropout(0.5)(x) #드롯아웃
x = layers.Dense(20, activation="relu")(x) #FFNN
x = layers.Dropout(0.5)(x) #드롭아웃
outputs = layers.Dense(8, activation="softmax")(x) #Softmax

model = keras.Model(inputs=inputs, outputs=outputs) #모델 생성
model.compile("adam", "CategoricalCrossentropy", metrics=["accuracy"])
```


##### 사용법

Google colab 런타임 유형 GPU           

자세한 사용법은 깃허브 ipython 파일 참조


### 참고자료

[koBERT를 이용한 한국어 감성분석](https://github.com/SKTBrain/KoBERT/blob/master/scripts/NSMC/naver_review_classifications_gluon_kobert.ipynb),
[KU-NLP-2020-1](https://github.com/Parkchanjun/KU-NLP-2020-1)
