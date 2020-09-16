import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
train=pd.read_csv("Training.csv")
test=pd.read_csv("Test.csv")

columns=['Red Std','Green Std','Blue Std','Red Mean','Green Mean','Blue Mean','Contrast','Dissimilarity']

#fitting the model
from sklearn import svm
X=train[columns]
y=train.Penyakit
clf=svm.SVC(kernel="rbf",C=50)
clf=clf.fit(X,y)
pred=clf.predict(test[columns])
#preparing the csv file
submission_df={"0":pred}
submission=pd.DataFrame(submission_df)
submission.to_csv("prediction.csv",index=False,header=False)
