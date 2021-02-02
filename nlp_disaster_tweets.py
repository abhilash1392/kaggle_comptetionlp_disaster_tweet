# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Importing the libraries 
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer,ENGLISH_STOP_WORDS
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Cleaning data 
def cleanData(x):
    x=x.lower()
    x=re.sub(r'^https?:\/\/.*[\r\n]*','',x)
    x=re.sub(r'#','',x)
    x=re.sub(r'@','',x)
    x=re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '',x)
    x=re.sub(r'[^\w]', ' ', x)
    x=re.sub(r'[0-9]+', '', x)
    x=re.sub(r'_',' ',x)
    x=x.split()
    x=' '.join(x)
    return x
# Main function
def main():
    # Importing the dataset
    df_train=pd.read_csv('../input/nlp-getting-started/train.csv')
    df_test=pd.read_csv('../input/nlp-getting-started/test.csv')
    print('Files are loaded')
    df_train['text']=df_train['text'].apply(lambda x: cleanData(x))
    df_test['text']=df_test['text'].apply(lambda x:cleanData(x))
    print('Data is cleaned')
    id_train=df_train.id
    text_train=list(df_train.text)
    y_train=df_train.target.values
    id_test=df_test.id
    text_test=list(df_test.text)
    text=text_train+text_test
    print(len(text))
    lentraindata=len(text_train)
    tfv = TfidfVectorizer(  max_features=None,tokenizer=None,ngram_range=(1,1)
        ,analyzer='word', use_idf=1,smooth_idf=1,sublinear_tf=1)
    X = tfv.fit_transform(text)
    X_train = X[:lentraindata]
    X_test = X[lentraindata:]
    lr = LogisticRegression(C=1,max_iter=10000)
    print('Cross ValScore:{}'.format(np.mean(cross_val_score(lr,X_train,y_train,cv=10))))
    lr.fit(X_train,y_train)
    y_predict = lr.predict(X_test)
    print('Data Predicted')
    data = {'id':id_test,'target':y_predict}
    submission_df = pd.DataFrame(data)
    submission_df.to_csv('submission.csv',index=False)
    print('File Submitted')
    
    
if __name__=='__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    