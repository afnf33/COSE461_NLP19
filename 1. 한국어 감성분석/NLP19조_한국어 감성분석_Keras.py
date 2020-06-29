#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1. 데이터 불러오기
get_ipython().system('git clone https://github.com/e9t/nsmc.git')
    
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import keras

train_data = pd.read_csv("nsmc/ratings_train.txt", sep='\t')
test_data = pd.read_csv("nsmc/ratings_test.txt", sep='\t')

print(train_data.shape)
print(test_data.shape)


# In[2]:


# Open Korea Text 로 분석

from konlpy.tag import Okt
okt = Okt()


# In[7]:


# tokenize 및 분석 결과 json으로 저장
#
import json
import os

def tokenize(doc):
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]

# # Json 저장 필요없을 시 이 부분만 돌려주세요
# train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
# test_docs = [(tokenize(row[1]), row[2]) for row in test_data]


if os.path.isfile('train_docs.json'):
    with open('train_docs.json', 'rb') as f:
        train_docs = json.load(f)
    with open('test_docs.json', 'rb') as f:
        test_docs = json.load(f)
else:
    train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
    test_docs = [(tokenize(row[1]), row[2]) for row in test_data]
    # JSON 파일로 저장
    with open('train_docs.json', 'w', encoding="utf-8") as make_file:
        json.dump(train_docs, make_file, ensure_ascii=False, indent="\t")
    with open('test_docs.json', 'w', encoding="utf-8") as make_file:
        json.dump(test_docs, make_file, ensure_ascii=False, indent="\t")

print(train_docs[0])


# In[8]:


tokens = [t for d in train_docs for t in d[0]]
print(len(tokens))


# In[10]:


# 토큰화, 개수, 자료 훑어보기
import nltk
text = nltk.Text(tokens, name='NMSC')

# 전체 토큰의 개수
print(len(text.tokens))

# 중복을 제외한 토큰의 개수
print(len(set(text.tokens)))            

# 출현 빈도가 높은 상위 토큰 10개
print(text.vocab().most_common(10))


# In[11]:


selected_words = [f[0] for f in text.vocab().most_common(1000)] 
# 기존 1만개였으나 overcommit_memory 문제로 1000개로 down.
# 1만개 기준 정확도 0.85, 1000개 정확도 0.82

def term_frequency(doc):
    return [doc.count(word) for word in selected_words]

train_x = [term_frequency(d) for d, _ in train_docs]
test_x = [term_frequency(d) for d, _ in test_docs]
train_y = [c for _, c in train_docs]
test_y = [c for _, c in test_docs]


# In[12]:


# 타입 변환. most_common 10000 설정시 이부분에서 overcommit_memory error 발생
import numpy as np

x_train = np.asarray(train_x).astype('float32')
x_test = np.asarray(test_x).astype('float32')

y_train = np.asarray(train_y).astype('float32')
y_test = np.asarray(test_y).astype('float32')


# In[13]:


# validation set 나누기
x_val=x_train[:10000]
partial_x_train=x_train[10000:]

y_val=y_train[:10000]
partial_y_train=y_train[10000:]


# In[14]:


from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(1000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
             loss=losses.binary_crossentropy,
             metrics=[metrics.binary_accuracy])

history=model.fit(partial_x_train,partial_y_train,epochs=10,batch_size=512,validation_data=(x_val,y_val))
results = model.evaluate(x_test, y_test)


# In[15]:


#시각화 하기 - 훈련과 검증 손실
import matplotlib.pyplot as plt

history_dict=history.history
loss=history_dict['loss']
val_loss=history_dict['val_loss']

epochs=range(1,len(loss)+1)

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validatin_loss')

plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.legend()

plt.show()


# In[16]:


print("정확도: ", results[1])


# In[44]:


# 새로운 파일에 대해서
new_data = pd.read_csv('ko_data.csv')
train_docs = [(tokenize(row)) for row in new_data['Sentence']]
train_x = [term_frequency(d) for d in train_docs]
new_datas = np.asarray(train_x).astype('float32')

results = model.predict(new_datas)
pd.DataFrame(np.around(results), columns=['Predicted']).to_csv('predict.csv')


# In[ ]:




