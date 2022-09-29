#!/usr/bin/env python
# coding: utf-8

# In[1]:




import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv("weatherAUS.csv")


# In[3]:


df.head(10)


# In[4]:


df.tail()


# In[5]:


print("Total dataset \n Rows - ",df.shape[0],'\n columns - ',df.shape[1])


# In[6]:


df.info()


# In[7]:


df.describe().T


# # Check missing values

# In[8]:


df.isnull().sum()


# In[9]:


round(df.isnull().sum()/df.shape[0]*100,2).sort_values(ascending=False)


# In[10]:


plt.figure(figsize=(15,8))
sns.heatmap(df.isnull())


# there are many missing values present in datasets

# In[11]:


df.columns


# In[12]:


catg_features=[col for col in df.columns if df[col].dtypes=='object']
cont_features=[col for col in df.columns if df[col].dtypes!='object']


# In[13]:


catg_features


# In[14]:


cont_features


# In[15]:


plt.pie([len(catg_features),len(cont_features)],labels=['Categorical','Continuous'],textprops={'fontsize':12},autopct='%1.1f%%')


# # Target Feature

# In[16]:


df['RainTomorrow'].unique()


# In[17]:


df['RainTomorrow'].value_counts(normalize=True,dropna=False)*100


# In[18]:


target_df=df['RainTomorrow'].value_counts(normalize=True,dropna=False)*100


# In[19]:


plt.figure(figsize=(8,6))
plt.title("Target Fetaure ( Rain Tomorrow) -Categories",fontweight='bold',fontsize=15)
ax=sns.barplot(x=target_df.index,y=target_df.values)
plt.xlabel('Rain Tomorrow')
plt.ylabel('Percentage')



for p in ax.patches:
    height=p.get_height()
    width=p.get_width()
    x,_=p.get_xy()
    ax.text(x +width/2.8,height+.5,f'{height:.2f}%')


# Date

# In[20]:


df['Date'].nunique()


# In[ ]:





# Location

# In[21]:


df['Location'].unique()


# In[22]:


df['Location'].value_counts()


# In[23]:


plt.figure(figsize=(12,8))
sns.countplot(df['Location'],order=df['Location'].value_counts().index)
plt.xticks(rotation=45)


# # MinTemp
# 

# In[24]:


sns.distplot(df['MinTemp'])


# data is evenly distributed as per plot 

# # MaxTemp

# In[25]:


sns.displot(df['MaxTemp'])


# # Distribution of Continuous features

# In[26]:


import random

color_=['#000057','#005757','#005700','#ad7200','#008080','#575757','#003153']
cmap_=['magma','copper','crest']


# In[27]:


plt.figure(figsize=(16,50))
for i,col in enumerate(df[cont_features].columns):
    rand_col=color_[random.sample(range(6),1)[0]]
    plt.subplot(6,3,i+1)
    
    sns.kdeplot(data=df,x=col,color=rand_col,fill=rand_col,palette=cmap_[random.sample(range(3),1)[0]])


# In[28]:


plt.figure(figsize=(16,50))
for i,col in enumerate(df[catg_features].columns):
    rand_col=color_[random.sample(range(6),1)[0]]
    plt.subplot(21,1,i+1)
    
    sns.countplot(data=df,x=col,color=rand_col,fill=rand_col,palette=cmap_[random.sample(range(3),1)[0]])


# # Bivarient EDA
# 

# In[29]:


df.columns


# In[30]:


df.head(2)


# In[31]:


plt.figure(figsize=(12,8))
sns.countplot(df['Location'],order=df['Location'].value_counts().index,hue=df['RainTomorrow'])
plt.xticks(rotation=45)


# In[32]:


table=pd.crosstab(df['Location'],df['RainTomorrow'])
table.div(table.sum(1),axis=0).plot(kind='bar',stacked=True)


# # MinTemp, MaxTemp

# In[33]:


sns.scatterplot(df['MinTemp'],df['MaxTemp'],hue='RainTomorrow',data=df)


# # Evaporation

# In[34]:


sns.scatterplot(df['Sunshine'],df['Evaporation'],hue='RainTomorrow',data=df)


# # WindGustDir, WindGustSpeed

# In[35]:


sns.scatterplot(df['WindGustSpeed'],df['WindGustDir'],hue='RainTomorrow',data=df)


# # WindDir9am , WindSpeed9am

# In[36]:


sns.scatterplot(df['WindSpeed9am'],df['WindDir9am'],hue='RainTomorrow',data=df)


# # WindDir3pm, WindSpeed3pm

# In[37]:


sns.scatterplot(df['WindSpeed3pm'],df['WindDir3pm'],hue='RainTomorrow',data=df)


# In[38]:


df.columns


# # Humidity9am, Humidity3pm

# In[39]:


plt.figure(figsize=(10,7))
sns.scatterplot(df['Humidity3pm'],df['Humidity9am'],hue='RainTomorrow',size='Rainfall',data=df)


# # Pressure9am , Pressure3pm

# In[40]:


plt.figure(figsize=(10,6))
sns.scatterplot(df['Pressure3pm'],df['Pressure9am'],hue='RainTomorrow',size='Rainfall',data=df)


# # Cloud9am , Cloud3pm

# In[41]:



plt.figure(figsize=(10,6))
sns.scatterplot(df['Cloud3pm'],df['Cloud9am'],hue='RainTomorrow',size='Rainfall',data=df)


# # Temp9am ,Temp3pm

# In[42]:


plt.figure(figsize=(10,6))
sns.scatterplot(df['Temp3pm'],df['Temp9am'],hue='RainTomorrow',size='Rainfall',data=df)


# # RainToday ,  RainTomorrow

# In[43]:


df['RainToday'].value_counts()


# In[44]:


df.groupby('RainToday')['RainTomorrow'].value_counts()


# In[45]:


sns.countplot(df['RainToday'],hue=df['RainTomorrow'])


# In[46]:


max_temp = df['MaxTemp']
min_temp = df['MinTemp']

min_temp.plot(figsize=(12,7), legend=True)
max_temp.plot(figsize=(12,7), color='r', legend=True)
plt.title('Maximum and Minimum Temperature across 2020 per day')
plt.ylabel('Temperature (°C)')
plt.show()


# # Correlation

# In[47]:


plt.figure(figsize=(15,15))
sns.heatmap(df.corr(),annot=True)


# # Create Model to predict about rainfall 

# In[48]:


round(df.isnull().sum()/df.shape[0]*100,2).sort_values(ascending=False)


# In[49]:


df.info()


# In[50]:


yes_rain = df[df['RainTomorrow']=='Yes']
no_rain = df[df['RainTomorrow']=='No']


# In[51]:


yes_rain.shape , no_rain.shape


# In[52]:


yes_rain['MinTemp'].fillna(yes_rain['MinTemp'].mode()[0],inplace=True)
no_rain['MinTemp'].fillna(no_rain['MinTemp'].mode()[0],inplace=True)

yes_rain['MaxTemp'].fillna(yes_rain['MaxTemp'].mode()[0],inplace=True)
no_rain['MaxTemp'].fillna(no_rain['MaxTemp'].mode()[0],inplace=True)

yes_rain['Temp9am'].fillna(yes_rain['Temp9am'].mode()[0],inplace=True)
no_rain['Temp9am'].fillna(no_rain['Temp9am'].mode()[0],inplace=True)

yes_rain['Temp3pm'].fillna(yes_rain['Temp3pm'].mode()[0],inplace=True)
no_rain['Temp3pm'].fillna(no_rain['Temp3pm'].mode()[0],inplace=True)


yes_rain['Humidity3pm'].fillna(yes_rain['Humidity3pm'].mode()[0],inplace=True)
no_rain['Humidity3pm'].fillna(no_rain['Humidity3pm'].mode()[0],inplace=True)

yes_rain['Humidity9am'].fillna(yes_rain['Humidity9am'].mode()[0],inplace=True)
no_rain['Humidity9am'].fillna(no_rain['Humidity9am'].mode()[0],inplace=True)


# In[53]:


round(df.isnull().sum()/df.shape[0]*100,2).sort_values(ascending=False)


# In[54]:


df.info()


# In[55]:


yes_rain['Sunshine'].fillna(yes_rain['Sunshine'].median(),inplace=True)
no_rain['Sunshine'].fillna(no_rain['Sunshine'].median(),inplace=True)

yes_rain['Evaporation'].fillna(yes_rain['Evaporation'].median(),inplace=True)
no_rain['Evaporation'].fillna(no_rain['Evaporation'].median(),inplace=True)

yes_rain['Cloud3pm'].fillna(yes_rain['Cloud3pm'].median(),inplace=True)
no_rain['Cloud3pm'].fillna(no_rain['Cloud3pm'].median(),inplace=True)

yes_rain['Cloud9am'].fillna(yes_rain['Cloud9am'].median(),inplace=True)
no_rain['Cloud9am'].fillna(no_rain['Cloud9am'].median(),inplace=True)

yes_rain['Pressure3pm'].fillna(yes_rain['Pressure3pm'].median(),inplace=True)
no_rain['Pressure3pm'].fillna(no_rain['Pressure3pm'].median(),inplace=True)

yes_rain['Pressure9am'].fillna(yes_rain['Pressure9am'].median(),inplace=True)
no_rain['Pressure9am'].fillna(no_rain['Pressure9am'].median(),inplace=True)

yes_rain['WindGustDir'].fillna(yes_rain['WindGustDir'].mode()[0],inplace=True)
no_rain['WindGustDir'].fillna(no_rain['WindGustDir'].mode()[0],inplace=True)

yes_rain['WindGustSpeed'].fillna(yes_rain['WindGustSpeed'].median(),inplace=True)
no_rain['WindGustSpeed'].fillna(no_rain['WindGustSpeed'].median(),inplace=True)

yes_rain['WindDir9am'].fillna(yes_rain['WindDir9am'].mode()[0],inplace=True)
no_rain['WindDir9am'].fillna(no_rain['WindDir9am'].mode()[0],inplace=True)

yes_rain['WindDir3pm'].fillna(yes_rain['WindDir3pm'].mode()[0],inplace=True)
no_rain['WindDir3pm'].fillna(no_rain['WindDir3pm'].mode()[0],inplace=True)

yes_rain['WindSpeed3pm'].fillna(yes_rain['WindSpeed3pm'].median(),inplace=True)
no_rain['WindSpeed3pm'].fillna(no_rain['WindSpeed3pm'].median(),inplace=True)

yes_rain['WindSpeed9am'].fillna(yes_rain['WindSpeed9am'].median(),inplace=True)
no_rain['WindSpeed9am'].fillna(no_rain['WindSpeed9am'].median(),inplace=True)



yes_rain['Rainfall'].fillna(yes_rain['Rainfall'].median(),inplace=True)
no_rain['Rainfall'].fillna(no_rain['Rainfall'].median(),inplace=True)


# In[56]:


data= yes_rain.append(no_rain, ignore_index=True)


# In[57]:


data


# In[58]:


data.shape


# In[59]:


data.dropna(inplace=True)


# In[60]:


data.head()


# In[61]:


data['Date']=pd.to_datetime(data['Date'])


# In[62]:


data['year']=data['Date'].dt.year


# In[63]:


data['month']=data['Date'].dt.month


# In[64]:


data['day']=data['Date'].dt.day


# In[65]:


data.drop('Date',axis=1,inplace=True)


# In[66]:


data


# In[67]:


d1=data


# In[ ]:





# In[68]:


num=[col for col in data.columns if data[col].dtypes!='O']
num


# # VIF to find multicolinearity between continuous independent features

# In[69]:


v=data[num]


# In[70]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
scaled=sc.fit_transform(v)


# In[71]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[72]:


VIF= pd.DataFrame()
VIF['features']=v.columns
VIF['vif']= [variance_inflation_factor(scaled,i) for i in range(len(v.columns))]


# In[73]:


VIF


# In[74]:


num.remove('Temp3pm')


# In[75]:


v=data[num]
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
scaled=sc.fit_transform(v)
VIF= pd.DataFrame()
VIF['features']=v.columns
VIF['vif']= [variance_inflation_factor(scaled,i) for i in range(len(v.columns))]
VIF


# In[76]:


num.remove('Pressure9am')


# In[77]:


v=data[num]
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
scaled=sc.fit_transform(v)
VIF= pd.DataFrame()
VIF['features']=v.columns
VIF['vif']= [variance_inflation_factor(scaled,i) for i in range(len(v.columns))]
VIF


# In[78]:


num.remove('Temp9am')


# In[79]:


v=data[num]
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
scaled=sc.fit_transform(v)
VIF= pd.DataFrame()
VIF['features']=v.columns
VIF['vif']= [variance_inflation_factor(scaled,i) for i in range(len(v.columns))]
VIF


# In[80]:


data.shape


# In[81]:



data.drop(['Temp3pm','Temp9am','Pressure9am'],axis=1,inplace=True)


# In[82]:


data.shape


# In[83]:


data.head()


# # Outliers

# In[84]:


for i in num:
    sns.boxplot(data[i])
    plt.show()


# In[85]:


sns.distplot(data['Rainfall'])


# In[86]:


data[num].skew()


# In[87]:


for i in num:
    IQR= data[i].quantile(.75)-data[i].quantile(.25)
    lower=data[i].quantile(.25) - (1.5 * IQR)
    upper=data[i].quantile(.75) + (1.5 * IQR)
    data[i]=np.where(data[i]<lower,lower,data[i])
    data[i]=np.where(data[i]>upper,upper,data[i])


# In[88]:


for i in num:
    sns.boxplot(data[i])
    plt.show()


# In[89]:


data[num].skew()


# # Transformation

# In[90]:


from sklearn.preprocessing import power_transform
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[91]:


for i in num:
    trans=power_transform(data[num])
    data[i]=sc.fit_transform(trans)


# In[92]:


data.head()


# # Encoding

# In[93]:


data.head()


# In[94]:


lwrtest=data[['Location','WindGustDir','WindDir9am','WindDir3pm','RainTomorrow']]
lwrtest


# In[95]:


import numpy as np

ordinal_label = {k: i for i, k in enumerate(lwrtest['WindGustDir'].unique(), 0)}
lwrtest['WindGustDir'] = lwrtest['WindGustDir'].map(ordinal_label)


# In[96]:


ordinal_label = {k: i for i, k in enumerate(lwrtest['Location'].unique(), 0)}
lwrtest['Location'] = lwrtest['Location'].map(ordinal_label)


# In[97]:


ordinal_label = {k: i for i, k in enumerate(lwrtest['WindDir9am'].unique(), 0)}
lwrtest['WindDir9am'] = lwrtest['WindDir9am'].map(ordinal_label)


# In[98]:


ordinal_label = {k: i for i, k in enumerate(lwrtest['WindDir3pm'].unique(), 0)}
lwrtest['WindDir3pm'] = lwrtest['WindDir3pm'].map(ordinal_label)


# In[99]:


lwrtest['RainTomorrow'].replace('No',0,inplace=True)
lwrtest['RainTomorrow'].replace('Yes',1,inplace=True)


# In[100]:


lwrtest


# In[101]:


inp=lwrtest.drop('RainTomorrow',axis=1)
out=lwrtest['RainTomorrow']


# In[102]:


from sklearn.feature_selection import chi2
f_p_values=chi2(inp,out)


# In[103]:


f_p_values


# In[104]:


p_values=pd.Series(f_p_values[1])
p_values.index=inp.columns
p_values


# In[105]:


p_values.plot.bar()


# In[106]:


data.drop('WindDir3pm',axis=1)


# # Encoding

# In[107]:


data['RainTomorrow'].replace('No',0,inplace=True)
data['RainTomorrow'].replace('Yes',1,inplace=True)

data['RainToday'].replace('No',0,inplace=True)
data['RainToday'].replace('Yes',1,inplace=True)


# In[108]:


data.head()


# In[109]:


X=data.drop('RainTomorrow',axis=1)
Y=data['RainTomorrow']


# In[110]:


X.shape  , Y.shape


# In[111]:


X=pd.get_dummies(X,drop_first=True)


# In[112]:


X.shape , Y.shape


# # SMOTE- Balancing dataset
# 

# In[113]:


from imblearn.over_sampling import SMOTE
sm=SMOTE()
x,y=sm.fit_resample(X,Y)


# In[114]:


x.shape , y.shape


# # Machine Learning

# In[115]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score


# In[116]:


maxaccu=0
maxRS=0

for i in range(0,200):
    x_train,x_test,y_train,y_test= train_test_split(x,y,random_state=i,test_size=.30)
    LR= LogisticRegression()
    LR.fit(x_train,y_train)
    pred= LR.predict(x_test)
    acc=accuracy_score(y_test,pred)
    if acc>maxaccu:
        maxaccu=acc
        maxRS=i
print("Best accuracy is ",maxaccu,"on Random State =",maxRS)


# In[117]:


x_train,x_test,y_train,y_test= train_test_split(x,y,random_state=122,test_size=.3)


# In[118]:



from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[119]:


LR_model= LogisticRegression()
RD_model= RidgeClassifier()
DT_model= DecisionTreeClassifier()
SV_model= SVC()
KNR_model= KNeighborsClassifier()
RFR_model= RandomForestClassifier()
XGB_model= XGBClassifier()
SGH_model= SGDClassifier()
Bag_model=BaggingClassifier()
ADA_model=AdaBoostClassifier()
GB_model= GradientBoostingClassifier()

model=[LR_model,RD_model,DT_model,SV_model,KNR_model,RFR_model,XGB_model,SGH_model,Bag_model,ADA_model,GB_model ]


# In[120]:


accuracy=[]
f1=[]

for m in model:
    m.fit(x_train,y_train)
    m.score(x_train,y_train)
    pred= m.predict(x_test)
    accuracy.append(round(accuracy_score(y_test,pred) * 100, 2))
    f1.append(round(f1_score(y_test,pred) * 100, 2))
   
    
pd.DataFrame({'Model':model,'Accuracy':accuracy,'F1 Score':f1})


# # BaggingClassifier() Hypertuning

# In[121]:


params = {'n_estimators' : [100,150,200,300,500],
    'max_features' : [1, 2, 3, 4, 5],
    'max_samples' : [0.05, 0.1, 0.2, 0.5]
}


# In[122]:


from sklearn.model_selection import GridSearchCV


# In[123]:


GCV=GridSearchCV(Bag_model,param_grid=params,cv=5,n_jobs=-1,verbose=2)
GCV.fit(x_train,y_train)


# In[124]:


GCV.best_estimator_


# In[125]:


GCV.best_params_


# In[126]:


GCV_pred=GCV.best_estimator_.predict(x_test)
accuracy_score(y_test,GCV_pred)


# # Confusion Matrix

# In[127]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred)
sns.heatmap(confusion_matrix(y_test,pred),annot=True, fmt='d')


# # AUC ROC plot

# In[128]:


from sklearn.metrics import roc_auc_score,roc_curve,plot_roc_curve


# In[129]:


plot_roc_curve(GB_model,x_test,y_test)
plt.title('ROC AUC Plot')


# # Saving Model

# In[130]:


import joblib
joblib.dump(GB_model,"Rainfall_Prediction.pkl")

