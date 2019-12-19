import pandas as pd
import pickle

# Reading data from csv
Data = pd.read_csv('shuffled-full-set-hashed.csv',header=None)

# Changing Column names
Data.columns = ['Target','X']
# Removing null values, as they are very few
Data.dropna(inplace=True)

# We observe that the data is not balanced and it will give us poor recall as it won't correctly
# identify categories with less amount of data

# We use Oversampling technqiue to avoid this issue
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)

# CountVectorizer and TfidfTransformer to transform the document data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

CV = CountVectorizer()
X_data = CV.fit_transform(Data.X.values)

# Saving the fitted models for future use on test/new user input data
f = open('Count_Vectorizer.pkl','wb')
pickle.dump(CV,f)

tT = TfidfTransformer()
X_normalized_data = tT.fit_transform(X_data)

f = open('TfidfTransformer.pkl','wb')
pickle.dump(tT,f)

Y_value = Data['Target'].values
X_normalized_data,Y_value = ros.fit_resample(X_normalized_data,Y_value)

# Splitting the data into train and test ; default test split is 0.25
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X_normalized_data,Y_value)

from sklearn.ensemble import RandomForestClassifier

# Training the model on train set and chose random forest based on accuracy and recall score on the validation set

rf = RandomForestClassifier()
rf.fit(X_train,Y_train)

# Saving the model for future use on new user input data
f = open('model.pkl','wb')
pickle.dump(rf,f)
