#normal data

import pandas as pd
import numpy as np
import random
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
y=pd.read_csv(r"C:\Users\嵐平\Desktop\論文\NEH\data_neh\20j_train_y_old.csv",header=None)
X1=pd.read_csv(r"C:\Users\嵐平\Desktop\論文\NEH\data_neh\X_train.csv")
X2=pd.read_csv(r"C:\Users\嵐平\Desktop\論文\NEH\data_neh\X_train2.csv")
L=[]
for i in range(len(y)):
    for j in [1,3,4]:
        if y.iloc[i][j]==1:
            L.append(i)
new_y=y.iloc[L]
#new_y=new_y.reset_index(inplace=True, drop=True)
new_X1=X1.iloc[L]
new_X2=X2.iloc[L]   
X=new_X1.join(new_X2)
#X=X.reset_index(inplace=True, drop=True)
#special data
y_=pd.read_csv(r"C:\Users\嵐平\Desktop\論文\NEH\data_neh\20j_train_y.csv",header=None)
X1_=pd.read_csv(r"C:\Users\嵐平\Desktop\論文\NEH\data_neh\Features.csv")
X2_=pd.read_csv(r"C:\Users\嵐平\Desktop\論文\NEH\data_neh\Features2.csv")
L2=[]
for i in range(len(y_)):
    for j in [1,3,4]:
        if y_.iloc[i][j]==1:
            L2.append(i)
new_y_=y_.iloc[L2]
new_X1_=X1.iloc[L2]
new_X2_=X2.iloc[L2]
X_=new_X1_.join(new_X2_)
#data2
y_2=pd.read_csv(r"C:\Users\嵐平\Desktop\論文\NEH\data_neh\20j_train_y_new2.csv",header=None)
X1_2=pd.read_csv(r"C:\Users\嵐平\Desktop\論文\NEH\data_neh\Features_new2.csv")
X2_2=pd.read_csv(r"C:\Users\嵐平\Desktop\論文\NEH\data_neh\Features2_new2.csv")
L2=[]
for i in range(len(y_2)):
    for j in [1,3,4]:
        if y_2.iloc[i][j]==1:
            L2.append(i)
new_y_2=y_2.iloc[L2]
new_X1_2=X1_2.iloc[L2]
new_X2_2=X2_2.iloc[L2]
X_2=new_X1_2.join(new_X2_2)


#data3
y_3=pd.read_csv(r"C:\Users\嵐平\Desktop\論文\NEH\data_neh\20j_train_y_new3.csv",header=None)
X1_3=pd.read_csv(r"C:\Users\嵐平\Desktop\論文\NEH\data_neh\Features_new3.csv")
X2_3=pd.read_csv(r"C:\Users\嵐平\Desktop\論文\NEH\data_neh\Features2_new3.csv")
L2=[]
for i in range(len(y_3)):
    for j in [1,3,4]:
        if y_3.iloc[i][j]==1:
            L2.append(i)
new_y_3=y_3.iloc[L2]
new_X1_3=X1_3.iloc[L2]
new_X2_3=X2_3.iloc[L2]
X_3=new_X1_3.join(new_X2_3)

#data4
y_4=pd.read_csv(r"C:\Users\嵐平\Desktop\論文\NEH\data_neh\20j_train_y_new4.csv",header=None)
X1_4=pd.read_csv(r"C:\Users\嵐平\Desktop\論文\NEH\data_neh\Features_new4.csv")
X2_4=pd.read_csv(r"C:\Users\嵐平\Desktop\論文\NEH\data_neh\Features2_new4.csv")
L2=[]
for i in range(len(y_4)):
    for j in [1,3,4]:
        if y_4.iloc[i][j]==1:
            L2.append(i)
new_y_4=y_4.iloc[L2]
new_X1_4=X1_4.iloc[L2]
new_X2_4=X2_4.iloc[L2]
X_4=new_X1_4.join(new_X2_4)

#data5
y_5=pd.read_csv(r"C:\Users\嵐平\Desktop\論文\NEH\data_neh\20j_train_y_new5.csv",header=None)
X1_5=pd.read_csv(r"C:\Users\嵐平\Desktop\論文\NEH\data_neh\Features_new5.csv")
X2_5=pd.read_csv(r"C:\Users\嵐平\Desktop\論文\NEH\data_neh\Features2_new5.csv")
L2=[]
for i in range(len(y_5)):
    for j in [1,3,4]:
        if y_5.iloc[i][j]==1:
            L2.append(i)
new_y_5=y_5.iloc[L2]
new_X1_5=X1_5.iloc[L2]
new_X2_5=X2_5.iloc[L2]
X_5=new_X1_5.join(new_X2_5)

#data6
y_6=pd.read_csv(r"C:\Users\嵐平\Desktop\論文\NEH\data_neh\20j_train_y_new6.csv",header=None)
X1_6=pd.read_csv(r"C:\Users\嵐平\Desktop\論文\NEH\data_neh\Features_new6.csv")
X2_6=pd.read_csv(r"C:\Users\嵐平\Desktop\論文\NEH\data_neh\Features2_new6.csv")
L2=[]
for i in range(len(y_6)):
    for j in [1,3,4]:
        if y_6.iloc[i][j]==1:
            L2.append(i)
new_y_6=y_6.iloc[L2]
new_X1_6=X1_6.iloc[L2]
new_X2_6=X2_6.iloc[L2]
X_6=new_X1_6.join(new_X2_6)

#data7
y_7=pd.read_csv(r"C:\Users\嵐平\Desktop\論文\NEH\data_neh\20j_train_y_new7.csv",header=None)
X1_7=pd.read_csv(r"C:\Users\嵐平\Desktop\論文\NEH\data_neh\Features_new7.csv")
X2_7=pd.read_csv(r"C:\Users\嵐平\Desktop\論文\NEH\data_neh\Features2_new7.csv")
L2=[]
for i in range(len(y_7)):
    for j in [1,3,4]:
        if y_7.iloc[i][j]==1:
            L2.append(i)
new_y_7=y_7.iloc[L2]
new_X1_7=X1_7.iloc[L2]
new_X2_7=X2_7.iloc[L2]
X_7=new_X1_7.join(new_X2_7)

#合併
X_all=[]
for k in [X,X_,X_2,X_3,X_4,X_5,X_6,X_7]:
    for i in k.values:
        X_all.append(i)
X_all=pd.DataFrame(X_all)
y_all=[]
for k in [new_y,new_y_,new_y_2,new_y_3,new_y_4,new_y_5,new_y_6,new_y_7]:
    for i in k.values:
        y_all.append(i)
y_all=pd.DataFrame(y_all)
#合併
#X_all=X.append(X_)
#X_2.columns=X.columns
#X_all=X_all.append(X_2)
#X_3.columns=X.columns
#X_all=X_all.append(X_3)
#X_4.columns=X.columns
#X_all=X_all.append(X_4)
#X_5.columns=X.columns
#X_all=X_all.append(X_5)
#y_all=new_y.append(new_y_)
#y_all=y_all.append(new_y_2)
#y_all=y_all.append(new_y_3)
#y_all=y_all.append(new_y_4)
#y_all=y_all.append(new_y_5)
#X_all.reset_index(drop=True, inplace=True)
#y_all.reset_index(drop=True, inplace=True)
y_all=y_all[[1,3,4]]
L=[]
for i in range(len(y_all)):
    for j in y_all.columns:
        if y_all.iloc[i][j]==1:
            L.append(j)
ll=pd.DataFrame(L)
ll.columns=['label']
y_all=y_all.join(ll)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all['label'], random_state = 1)


from xgboost import XGBClassifier
params = { 'booster': 'gbtree', 
               'objective': 'multi:softmax', # 多分類的問題 
               'num_class': 10, # 類別數，與 multisoftmax 並用 
               'gamma': 0.1, # 用於控制是否後剪枝的引數,越大越保守，一般0.1、0.2這樣子。 
               'max_depth': 12, # 構建樹的深度，越大越容易過擬合 
               'reg_lambda': 2, # 控制模型複雜度的權重值的L2正則化項引數，引數越大，模型越不容易過擬合。 
               'subsample': 0.7, # 隨機取樣訓練樣本 
               'colsample_bytree': 0.7, # 生成樹時進行的列取樣 
               'min_child_weight': 3, 
               'silent': 1, # 設定成1則沒有執行資訊輸出，最好是設定為0. 
               'learning_rate': 0.007, # 如同學習率 
               'reg_alpha':0, # L1 正則項引數
               'seed': 1000, 
               'nthread': 4, # cpu 執行緒數 
              }
model=XGBClassifier(learning_rate=0.3,objective='multi:softmax',num_class=3)
clf = XGBClassifier(
        #樹的個數
        n_estimators=10,
        # 如同學習率
        learning_rate= 0.5, 
        # 構建樹的深度，越大越容易過擬合    
        max_depth=6, 
        # 隨機取樣訓練樣本 訓練例項的子取樣比
        subsample=1, 
        # 用於控制是否後剪枝的引數,越大越保守，一般0.1、0.2這樣子
        gamma=0, 
        # 控制模型複雜度的權重值的L2正則化項引數，引數越大，模型越不容易過擬合。
        reg_lambda=1,  
        
        #最大增量步長，我們允許每個樹的權重估計。
        max_delta_step=1,
        # 生成樹時進行的列取樣 
        colsample_bytree=1, 

        # 這個引數預設是 1，是每個葉子裡面 h 的和至少是多少，對正負樣本不均衡時的 0-1 分類而言
        # 假設 h 在 0.01 附近，min_child_weight 為 1 意味著葉子節點中最少需要包含 100 個樣本。
        #這個引數非常影響結果，控制葉子節點中二階導的和的最小值，該引數值越小，越容易 overfitting。
        min_child_weight=1, 

        #隨機種子
        seed=100 ,
        
        # L1 正則項引數
#        reg_alpha=0,
        
        #如果取值大於0的話，在類別樣本不平衡的情況下有助於快速收斂。平衡正負權重
        #scale_pos_weight=1,
        
        #多分類的問題 指定學習任務和相應的學習目標
        objective= 'multi:softmax', 
        
        # 類別數，多分類與 multisoftmax 並用
        num_class=3,
        
        # 設定成1則沒有執行資訊輸出，最好是設定為0.是否在執行升級時列印訊息。
#        silent=0 ,
        # cpu 執行緒數 預設最大
#        nthread=4,
    
        eval_metric= 'auc'
)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
c=0
for i in range(len(y_pred)):
   if y_pred[i]==y_test.values[i]:
       c=c+1
print(c/len(y_pred))
   




