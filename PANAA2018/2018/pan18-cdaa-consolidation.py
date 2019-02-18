# -*- coding: utf-8 -*-

from __future__ import print_function
import os;
import re;
from os.path import join as pathjoin;
import glob;
import json;
import argparse;
import time;
import codecs;
import numpy as np;
import pandas as pd;
from collections import defaultdict
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing


def readFileAndMeta(evalFile):
    df = pd.read_csv(evalFile);
    config = evalFile[0:-4].split('/')[-1].split('_');
    try:
        df['classifier']= config[1];
        df['strategy']  = config[2];
        df['ngram_min'] = int(config[3]);
        df['ngram_max'] = int(config[4]);
        df['use_idf']   = config[5];
        df['df_min']    = float(config[6]);
        df['df_max']    = float(config[7]);
        df['norm']        = config[8];
        df['id']        = config[9];
    except:
        print (evalFile);
    
    return df;
  
def readEvaluationV2(evalFile):
    df = pd.read_csv(evalFile);
    config = re.sub(r'.*evaluation_(\w+)\_(\d+)_(\d+)\.csv',r'\1|\2|\3', evalFile).split("|");
    df['classifier']=config[0];
    df['ngrams']=config[1];
    df['corpusMinDf'] = config[2];
    return df;
    


def readProblem(fileName):
    df = pd.read_csv(fileName);
    iid = (fileName.split('/')[-1])[0:-4].split('_')[-1];
    df['id'] = iid;
    return df;


def test():
    baseDir = '/Users/joseeleandrocustodio/Dropbox/mestrado/02 - Pesquisa/code/out';
    
    
    #evaluation files
    files = glob.glob(pathjoin(baseDir,'*','evaluation*.csv'));
    
    dfEval = pd.concat([pd.read_csv(f) for f in files]);
    
    dfEval.rename(columns={
            'macro-f1':'macrof1',
            'macro-precision':'macroPrecision',
            'macro-recall':'macroRecall',
            'micro-accuracy':'microAccuracy',
            'problem-name':'problem'},
            inplace=True);
    dfEval.problem  = dfEval.problem.str.replace('problem', '').astype(int);
    
    dfEval.to_csv(pathjoin(baseDir,'totalEvaluation.csv'), index=False, sep=';');
    

if __name__ == '__main__':
    test()