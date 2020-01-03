import pandas as pd
import numpy as np
data = pd.read_csv("dataset/train.csv")
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#x = data.iloc[:, :-1].values
#y = data.iloc[:, -1].values
y=data['Result']
x=data.drop('Result',axis='columns')

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x, y)
dataa = pd.read_csv("dataset/test.csv")
datap=dataa.drop('ID',axis='columns')
datap['ID'] = np.arange(len(datap))
cols = datap.columns.tolist()
cols=[
'ID',
 'para_0',
 'para_1',
 'para_2',
 'para_3',
 'para_4',
 'para_5',
 'para_6',
 'para_7',
 'para_8',
 'para_9',
 'para_10',
 'para_11',
 'para_12',
 'para_13',
 'para_14',
 'para_15',
 'para_16',
 'para_17',
 'para_18',
 'para_19',
 'para_20',
 'para_21',
 'para_22',
 'para_23',
 'para_24',
 'para_25',
 'para_26',
 'para_27',
 'para_28',
 'para_29',
 'para_30',
 'para_31',
 'para_32',
 'para_33',
 'para_34',
 'para_35',
 'para_36',
 'para_37',
 'para_38',
 'para_39',
 'para_40',
 'para_41',
 'para_42',
 'para_43',
 'para_44',
 'para_45',
 'para_46',
 'para_47',
 'para_48',
 'para_49',
 'para_50']

datap = datap[cols]
y_pred = classifier.predict(datap)
nn=pd.DataFrame(y_pred, columns=['Result'])
nn['DD'] = np.arange(len(nn))
nn['ID'] = nn['DD']+1
nn=nn.drop('DD',axis='columns')
cols = nn.columns.tolist()
cols=['ID','Result']
nn = nn[cols]


nn.to_csv('submission.csv',index=False)
