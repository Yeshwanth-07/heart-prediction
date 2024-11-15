import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
data = pd.read_csv('heart.csv')

one_df = data[data['target']==1].sample(138)

zero_df = data[data['target']==0]

heart_df = pd.concat([one_df,zero_df],axis=0)

heart_df.drop(columns=['exang','oldpeak','ca','thal'],axis=1,inplace=True)

X = heart_df.drop(['target'],axis=1)

Y = heart_df['target']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scale = StandardScaler()
scale.fit(X)
x_std = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_std,Y,test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestClassifier
mrf = RandomForestClassifier()

mrf.fit(x_train,y_train)

y_pred = mrf.predict(x_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Save model
with open('heart_disease_model.pkl', 'wb') as f:
    pickle.dump(mrf, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scale, f)