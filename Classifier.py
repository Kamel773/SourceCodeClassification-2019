#This program will Train
"""
MultiNomial Naive Bayes
Random Forest
XGBoost
"""
import sqlite3
import re
import time
import csv
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.learning_curve import learning_curve
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_blobs
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import glob,os
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import pickle
from sklearn.decomposition import PCA
from sklearn.random_projection import sparse_random_matrix
from sklearn.manifold import TSNE
from sklearn.feature_extraction import DictVectorizer
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
import gensim
from nltk import word_tokenize
from multiprocessing import *
import pdb


def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted Label')
    plt.savefig("RandomForestClassifierUsingCode.png", dpi=300)


def MyMultiNomialNB(X_train, y_train):
	clf = MultinomialNB()
	param_grid = {'alpha': [0.000000001,0.00000001,0.0000001,0.01,0.1,1,10,100,1000] }
	#param_grid = {'alpha': [0.000000001] }
	# Ten fold Cross Validation
	classifier= GridSearchCV(estimator=clf, cv=5 ,param_grid=param_grid)
	#classifier.fit(X_train, y_train)
	return classifier.fit(X_train, y_train) #classifier.cv_results_

def MyExtraTreeClassifier(X_train, y_train):
	clf = ExtraTreesClassifier(min_samples_split=2, random_state=0,max_depth = 10)
	param_grid = {'n_estimators': [1000]}
	param_grid = {'max_depth': [1,5,10,25,50,75,100,500,1000,2000]}
	classifier= GridSearchCV(estimator=clf, cv=3 ,param_grid=param_grid,n_jobs=3,verbose=100)
	
	return classifier.best_estimator_ #classifier.cv_results_


def MyRandomForest(X_train, y_train):
	clf = RandomForestClassifier()
	#param_grid = {'n_estimators': [10,20,30,50,70,100]}
	param_grid = {'n_estimators': [700,1], 'max_depth':[4],'min_child_weight':[4]}#,100,150,200]} [50,75]}
	classifier= GridSearchCV(estimator=clf, cv=2 ,param_grid=param_grid,verbose=100)#refit=True
	#y_pred = classifier.fit(X_train, y_train).predict(X_test)
	#print(confusion_matrix(X_train,y_pred))
	return classifier.fit(X_train, y_train) #classifier.cv_results_

def rbf_svm(X_train, y_train):
	rbf_svc = svm.SVC(kernel='rbf',max_iter = 5000,cache_size =1024)#,max_iter = 10000,cache_size =1024,decision_function_shape ='ovo'/'ovo'
	param_grid = {'C': np.logspace(-3, 2, 6), 'gamma': np.logspace(-3, 2, 6)}
	classifier= GridSearchCV(estimator=rbf_svc, cv=5 ,param_grid=param_grid)
	y_train= np.array(y_train)
	classifier.fit(X_train, y_train)
	return classifier.fit(X_train, y_train) #classifier.cv_results_

def calculateDoc2Vec(data, _size, is_dm):
    print('Building Vocabulary...........')
    tagged_data = [gensim.models.doc2vec.TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
    print('finished reading from file.......')
    cores = 4
    #model = gensim.models.doc2vec.Doc2Vec(size=_size, dm=is_dm, min_count=0, alpha=0.025, min_alpha=0.025, negative=5, workers=cores, window=5, seed=2018)
    model = gensim.models.doc2vec.Doc2Vec(size=300, dm=is_dm, workers=cores)
    model.build_vocab(tagged_data)
    for epoch in range(30):
        model.train(tagged_data, total_examples=model.corpus_count, epochs=100)
        model.alpha -= 0.002
        model.min_alpha = model.alpha
    print('finished building vocabulary.......')

    vectors = []
    for i in range(len(data)):
        vectors.append(model.docvecs[ str(i)])
    return  np.array(vectors)
	

X=[]
y=[]
onlyJava=[]


code_loc='/Users/kamel/Desktop/explore/code25/'
name_file=['c', 'c#', 'c++','java', 'css', 'haskell', 'html', 'javascript', 'lua', 'objective-c', 'perl', 'php', 'python','ruby', 'r', 'scala', 'sql', 'swift', 'vb.net','markdown','bash']
#name_file=['c#-3.0', 'c#-4.0', 'c#-5.0']
print (name_file)
for item in name_file:
    #print (item)
    code_loc_current=code_loc+item+'/'
    file_list = glob.glob(os.path.join(code_loc_current, "*.txt"))
    #print (code_loc_current)
    i = 0
    count = 0
    length = 0 
    for file_path in file_list:
        f=open(file_path,'r')
        data=f.read()
        label=item
        i = i + 1

        X.append(data)
        y.append(label)
    print(item)

#Change Y to categorical labels.
labels= list(set(y))
labels.sort()

label_mapping ={}
print(labels)
for i in range(21):
    label_mapping[labels[i]] = i

print(label_mapping)
#Lets encode the training data with this labels

for i in range(len(y)):
	y[i] = label_mapping[y[i]]
#print(y)

a=time.time()
print ("Vectorization started")
cv = TfidfVectorizer(input ='X',stop_words = {'english'},lowercase=True,analyzer ='word',max_features =10000)#,non_negative=True)#,)max_features =10000,min_df=10

X = cv.fit_transform(X)


print (len(vocab))
print ("Time taken to vectorize is %s seconds" %(time.time()-a))#print vocab



#Lets split the data into train-test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=453456)


from collections import Counter
kk=list(set(y_train))
kk.sort()
print (kk)

# Data preprocessing step, e.g. Whitening with PCA
'''
pca = PCA(whiten=True, n_components=100)#100
pca.fit(X_train.toarray())
X_train = pca.transform(X_train.toarray())
'''


print (X_train.shape)
print ("Running code for Code")
X_train = X_train[0:10000]#[0:100000]
y_train  =y_train[0:10000]#[0:100000]
y_pred = MyRandomForest(X_train,y_train).predict(X_test)#(MyRandomForest(X_train,y_train).predict(pca.transform(X_test.toarray())))

cnf_matrix =  confusion_matrix(y_test,y_pred)
#print(results_)
print ('mean_squared_error = ',mean_squared_error(y_test, y_pred))
#list_pickle_path = 'Code_PCA_RandomClassifier_pickle.pkl'
# Create an variable to pickle and open it in write mode

print (np.array(cnf_matrix))
print(y_pred)

classification_report = classification_report(y_test, y_pred, target_names=name_file)
precision_recall_fscore_support = (precision_recall_fscore_support(y_test, y_pred, average='weighted'))
accuracy_score = (accuracy_score(y_test, y_pred))
print(accuracy_score)
joblib.dump((cnf_matrix,precision_recall_fscore_support,classification_report,accuracy_score),"c#.plk")
print (classification_report)
