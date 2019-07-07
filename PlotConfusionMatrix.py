import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import pickle
from sklearn.externals import joblib
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
'''
# pickle list object
 
numbers_list = [1, 2, 3, 4, 5]
list_pickle_path = 'list_pickle.pkl'
 
# Create an variable to pickle and open it in write mode
list_pickle = open(list_pickle_path, 'wb')
pickle.dump(numbers_list, list_pickle)
list_pickle.close()
'''

# unpickling the list object
 
# Need to open the pickled list object into read mode

def plot_confusion_matrix(cm, classes,
                          normalize=False,
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


    plt.imshow(cm, interpolation='nearest', cmap=cmap) #cmap=cmap
    plt.title(title,weight='bold',fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90,weight='bold')
    plt.yticks(tick_marks, classes,weight='bold')

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label',weight='bold')
    plt.xlabel('Predicted Label',weight='bold')
    plt.savefig("Code_RandomForest.png", dpi=300,bbox_inches='tight')


name_file=['javascript','sql','java','c#','python','php','c++','c','typescript','ruby','swift','objective-c','vb.net','assembly','r','perl','vba','matlab','go','scala','groovy','coffeescript','lua','haskell']


data = joblib.load('Code_RandomForest.plk')
cnf_matrix = data
red.tranpose()

