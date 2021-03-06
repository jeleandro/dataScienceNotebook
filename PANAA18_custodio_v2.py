# -*- coding: utf-8 -*-
#python basic libs
from __future__ import print_function

from tempfile import mkdtemp
from shutil import rmtree
import os;
from os.path import join as pathjoin;

import re;
import glob;
import json;
import codecs;
import argparse;
from scipy.sparse import issparse

from time import time


#data analysis libs
import numpy as np;

#machine learning libs
#feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer

#preprocessing and transformation
from sklearn.preprocessing import MaxAbsScaler;
from sklearn.decomposition import PCA;

from sklearn.base import BaseEstimator

#classifiers
from sklearn.linear_model import LogisticRegression


from sklearn.pipeline import Pipeline


##############################################################################
class DenseTransformer(BaseEstimator):
    def __init__(self, return_copy=True):
        self.return_copy = return_copy
        self.is_fitted = False

    def transform(self, X, y=None):
        if issparse(X):
            return X.toarray()
        elif self.return_copy:
            return X.copy()
        else:
            return X

    def fit(self, X, y=None):
        self.is_fitted = True
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X=X, y=y)
    
    
##############################################################################
        
class ObfuscationTransformer(BaseEstimator):
    def __init__(self,re_from=r'(\b)(\w{0,2})\w+(\w{1,3})(\b)', re_to=r'\1\2XX\3\4', return_copy=True):
        self.re_from = re_from
        self.re_to = re_to

    def transform(self, X, y=None):
        X = np.array(X).copy();
        for i in range(len(X)):
            X[i] = re.sub(self.re_from,self.re_to, X[i])
        
        return X;

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X=X, y=y)
    
    
##############################################################################

def readCollectionsOfProblems(path):
    # Reading information about the collection
    infocollection = path+os.sep+'collection-info.json'
    with open(infocollection, 'r') as f:
        problems  = [
            {
                'problem': attrib['problem-name'],
                'language': attrib['language'],
                'encoding': attrib['encoding'],
            }
            for attrib in json.load(f)
            
        ]
    return problems;


def readProblem(path, problem):
    # Reading information about the problem
    infoproblem = path+os.sep+problem+os.sep+'problem-info.json'
    candidates = []
    with open(infoproblem, 'r') as f:
        fj = json.load(f)
        unk_folder = fj['unknown-folder']
        for attrib in fj['candidate-authors']:
            candidates.append(attrib['author-name'])
    return unk_folder, candidates;


def read_files(path,label):
    # Reads all text files located in the 'path' and assigns them to 'label' class
    files = glob.glob(pathjoin(path,label,'*.txt'))
    texts=[]
    for i,v in enumerate(files):
        f=codecs.open(v,'r',encoding='utf-8')
        texts.append((f.read(),label, os.path.basename(v)))
        f.close()
    return texts


def printSysInfo():
    import platform; print(platform.platform())
    print("NumPy", np.__version__)
    import scipy; print("SciPy", scipy.__version__)
    import sklearn; print("Scikit-Learn", sklearn.__version__)



def runML(problem, outputDir):
    print ("Problem: %s,  language: %s " %(problem['problem'],problem['language']))
    
    train_docs, train_labels, _   = zip(*problem['candidates'])
    problem['training_docs_size'] = len(train_docs);
    test_docs, _, test_filename   = zip(*problem['unknown'])
    
    pipeline1 = Pipeline([
        ('vect',   TfidfVectorizer(
                analyzer='char',
                min_df=0.05,
                max_df=1.0,
                ngram_range=(2,5),
                lowercase=False,
                norm='l2',
                sublinear_tf=True)),
        ('dense',  DenseTransformer()),
        ('scaler', MaxAbsScaler()),
        ('transf', PCA(0.999)),
        ('clf', LogisticRegression(random_state=0,multi_class='multinomial', solver='newton-cg')),
    ])
    
    pipeline2 = Pipeline([
        ('obs',ObfuscationTransformer(re_from=r'\w',re_to='x')),
        ('vect',   TfidfVectorizer(
                analyzer='char',
                min_df=0.05,
                max_df=1.0,
                ngram_range=(2,5),
                lowercase=False,
                norm='l2',
                sublinear_tf=True)),
        ('dense',  DenseTransformer()),
        ('scaler', MaxAbsScaler()),
        ('transf', PCA(0.999)),
        ('clf', LogisticRegression(random_state=0,multi_class='multinomial', solver='newton-cg')),
    ])
        
    
    pipeline3 = Pipeline([
        ('vect',   TfidfVectorizer(
                analyzer='word',
                min_df=0.05,
                max_df=1.0,
                ngram_range=(1,3),
                lowercase=True,
                norm='l2',
                sublinear_tf=True)),
        ('dense',  DenseTransformer()),
        ('scaler', MaxAbsScaler()),
        ('transf', PCA(0.999)),
        ('clf', LogisticRegression(random_state=0,multi_class='multinomial', solver='newton-cg')),
    ]);
                
                
    
    
    t0 = time()
    models = []
    if problem['language'] == 'en':
        models = [pipeline1, pipeline3];
    else:
        models = [pipeline1, pipeline2, pipeline3];
        
    for p in models:
        p.fit(train_docs, train_labels)

    xtrain_mix = np.hstack([p.predict_proba(train_docs) for p in models])
    xtest_mix  = np.hstack([p.predict_proba(test_docs) for p in models])

    clfFinal = Pipeline([
        ('clf',LogisticRegression(random_state=0,multi_class='multinomial', solver='newton-cg', C=10)
        )
    ]);
    clfFinal.fit(xtrain_mix, train_labels);
    test_pred =clfFinal.predict(xtest_mix);
    print("done in %0.3fs \n" % (time() - t0))
    # Writing output file
    out_data=[]
    for i,v in enumerate(test_pred):
        out_data.append(
                {'unknown-text': test_filename[i],
                 'predicted-author': v
                }
                )
    answerFile = pathjoin(outputDir,'answers-'+problem['problem']+'.json');
    with open(answerFile, 'w') as f:
        json.dump(out_data, f, indent=4)

    return;



def main(inputDir,outpath):
    t0 = time()
    printSysInfo();
    
    problems = readCollectionsOfProblems(inputDir);
    
    for index,problem in enumerate(problems):
        unk_folder, candidates_folder = readProblem(inputDir, problem['problem']); 
        problem['candidates_folder_count'] = len(candidates_folder);
        problem['candidates'] = [];
        for candidate in candidates_folder:
            problem['candidates'].extend(read_files(pathjoin(inputDir, problem['problem']),candidate));
        
        problem['unknown'] = read_files(pathjoin(inputDir, problem['problem']),unk_folder);
        
    for problem in problems:
        runML(problem, outpath);
    print("total in %0.3fs \n" % (time() - t0))
        


if __name__ == '__main__':
    #original main
    #parser = argparse.ArgumentParser()
    #parser = argparse.ArgumentParser(description='PAN-18 Authorship Attribution by Custodio')
    #parser.add_argument('-i', type=str, help='Path to the main folder of a collection of attribution problems')
    #parser.add_argument('-o', type=str, help='Path to an output folder')
    
    baseDir = '/Users/joseeleandrocustodio/Dropbox/mestrado/02 - Pesquisa/code';

    inputDir= pathjoin(baseDir,'pan18aa');
    outputDir= pathjoin(baseDir,'out');
    
    main(inputDir, outputDir)

