# -*- coding: utf-8 -*-
import numpy as np;
import re;
import math;
import string;
from scipy.stats import rankdata;
import pandas as pd;
import matplotlib.pyplot as plt;
from unicodedata import normalize;
from wordcloud import WordCloud;



def ksCurve(x, y, plot=False):
    arr = np.array((x, 1- np.array(y), y)).T;
    arr = np.array(sorted(arr, key=lambda a_entry: a_entry[0]));
    acum = np.cumsum(arr[:,1:],axis=0)
    acum = acum/np.max(acum,axis=0)
    ks = np.max(np.abs(acum[:,0]-acum[:,1]));
    if plot:
        plt.plot(arr[:,0],acum[:,0]);
        plt.plot(arr[:,0],acum[:,1], label='y %0.2f' % ks);
        plt.ylabel('Accumulated distribution')
        plt.legend(loc=0);
    return ks;

    
def rocCurve(ypred,y, plot=False):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y, ypred)
    roc_auc = auc(fpr, tpr)
    if plot:
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
    return roc_auc;



def ksDistanceCurve(x, y, plot=False):
    arr = np.array((x, 1- np.array(y), y)).T;
    arr = np.array(sorted(arr, key=lambda a_entry: a_entry[0]));
    acum = np.cumsum(arr[:,1:],axis=0)
    acum = acum/np.max(acum,axis=0)
    ks = np.abs(acum[:,0]-acum[:,1]);
    
    if plot:    
        plt.plot(arr[:,0],ks, label='distance KS');
        plt.ylabel('distance KS') 
    return np.max(ks);


def lda_print(model, feature_names, n_top_words, barchart=False):
    for topic_id, topic in enumerate(model.components_):
        print('\nTopic Nr.%d:' % int(topic_id + 1))
        pos = topic.argsort()[:-n_top_words-1:-1]
        
        topicsum =topic[pos].sum();
        print(''.join([feature_names[i] + ' ' + str(round(topic[i]/topicsum, 2))+' | ' for i in pos]));
        
        if barchart:
            y_pos = np.arange(n_top_words);
             
            plt.barh(y_pos, topic[pos]/topicsum, align='center', alpha=0.5)
            plt.yticks(y_pos, np.array(feature_names)[pos])
            plt.ylabel('weight')
            plt.title('\nTopic Nr.%d:' % int(topic_id + 1))
            plt.show();
            
        wd = dict(zip(np.array(feature_names)[pos],topic[pos]));
        # Open a plot of the generated image.
        plt.imshow(WordCloud(background_color="white").generate_from_frequencies(wd.items()))
        plt.axis("off");
        plt.show();
        
def odds(x, y, bins=10):
    df = pd.DataFrame({'data': x,'y':y});
    
    df['rank'] = rankdata(df['data'])/float(df['data'].count());
    df['rank'] = np.round(df['rank']*10,0);
    df['noty'] = 1.0-df['y'];
    
    odds = df.groupby('rank').agg({'y':np.sum,'noty':np.sum});
    odds['pery']=odds['y']/np.sum(odds['y']);
    odds['pernoty']=odds['noty']/np.sum(odds['noty']);
    odds['odds'] = odds['pery']/odds['pernoty'];
    odds['woe'] = np.log(odds['odds']);
    odds['iv'] = odds['woe'] * np.abs(odds['pery'] - odds['pernoty']);
    odds['rank'] = odds.index;
    return odds;

def desenhaLogistica(model, variaveis):
    coefs = model.coef_[0]
    positivo = ( [variaveis[i], coefs[i]]
                    for i in xrange(len(variaveis))
                        if coefs[i] >= 0
                );
 
    negativos = ( [variaveis[i], math.fabs(coefs[i])]
                    for i in xrange(len(variaveis))
                        if coefs[i] < 0
                );   
    # Open a plot of the generated image.
    plt.figure(figsize=(10,10))
    plt.imshow(WordCloud().generate_from_frequencies(positivo))
    plt.axis("off");
    plt.savefig('positivo.png')
    plt.show();       
 
    plt.figure(figsize=(10,10))
    plt.imshow(WordCloud().generate_from_frequencies(negativos))
    plt.axis("off");
    plt.savefig('negativo.png')
    plt.show(); 
    
    
def rocKS(predTrain,y_train, predTest, y_test, plot=True):
    if plot:
        plt.figure(figsize=(10,10))
        ax =plt.subplot(221)
        ax.set_title("KS desenvolvimento");
        ks1 = ksCurve(predTrain,y_train,plot);
       
        ax = plt.subplot(222)
        ax.set_title("KS validacao");
        ks2 = ksCurve(predTest,y_test,plot);
       
        ax = plt.subplot(223)
        ax.set_title("ROC desenvolvimento");
        roc1 =rocCurve(predTrain,y_train,plot);
       
        ax = plt.subplot(224)
        ax.set_title("ROC validacao");
        roc2 = rocCurve(predTest,y_test,plot);
    else:
        ks1  = ksCurve(predTrain,y_train,plot);
        ks2  = ksCurve(predTest,y_test,plot);
        roc1 = rocCurve(predTrain,y_train,plot);
        roc2 = rocCurve(predTest,y_test,plot);
 
    return ks1, roc1, ks2, roc2;
 