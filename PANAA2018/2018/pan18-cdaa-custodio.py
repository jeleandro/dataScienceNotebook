# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import os;
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
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import preprocessing
import random
import pickle


#*******************************************************************************************************
import warnings
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder


def eval_measures(gt, pred):
    """Compute macro-averaged F1-scores, macro-averaged precision, 
    macro-averaged recall, and micro-averaged accuracy according the ad hoc
    rules discussed at the top of this file.
    Parameters
    ----------
    gt : dict
        Ground truth, where keys indicate text file names
        (e.g. `unknown00002.txt`), and values represent
        author labels (e.g. `candidate00003`)
    pred : dict
        Predicted attribution, where keys indicate text file names
        (e.g. `unknown00002.txt`), and values represent
        author labels (e.g. `candidate00003`)
    Returns
    -------
    f1 : float
        Macro-averaged F1-score
    precision : float
        Macro-averaged precision
    recall : float
        Macro-averaged recall
    accuracy : float
        Micro-averaged F1-score
    """

    actual_authors = list(gt.values())
    encoder = LabelEncoder().fit(['<UNK>'] + actual_authors)

    text_ids, gold_authors, silver_authors = [], [], []
    for text_id in sorted(gt):
        text_ids.append(text_id)
        gold_authors.append(gt[text_id])
        try:
            silver_authors.append(pred[text_id])
        except KeyError:
            # missing attributions get <UNK>:
            silver_authors.append('<UNK>')

    assert len(text_ids) == len(gold_authors)
    assert len(text_ids) == len(silver_authors)

    # replace non-existent silver authors with '<UNK>':
    silver_authors = [a if a in encoder.classes_ else '<UNK>' 
                      for a in silver_authors]

    gold_author_ints   = encoder.transform(gold_authors)
    silver_author_ints = encoder.transform(silver_authors)

    # get F1 for individual classes (and suppress warnings):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f1 = f1_score(gold_author_ints,
                  silver_author_ints,
                  labels=list(set(gold_author_ints)),
                  average='macro')
        precision = precision_score(gold_author_ints,
                  silver_author_ints,
                  labels=list(set(gold_author_ints)),
                  average='macro')
        recall = recall_score(gold_author_ints,
                  silver_author_ints,
                  labels=list(set(gold_author_ints)),
                  average='macro')
        accuracy = accuracy_score(gold_author_ints,
                  silver_author_ints)

    return f1,precision,recall,accuracy

def evaluate(ground_truth_file,predictions_file):
    # Calculates evaluation measures for a single attribution problem
    gt = {}
    with open(ground_truth_file, 'r') as f:
        for attrib in json.load(f)['ground_truth']:
            gt[attrib['unknown-text']] = attrib['true-author']

    pred = {}
    with open(predictions_file, 'r') as f:
        for attrib in json.load(f):
            if attrib['unknown-text'] not in pred:
                pred[attrib['unknown-text']] = attrib['predicted-author']
    f1,precision,recall,accuracy =  eval_measures(gt,pred)
    return f1, precision, recall, accuracy

def evaluate_all(path_collection,path_answers,path_out, config, problemInfo):
    # Calculates evaluation measures for a PAN-18 collection of attribution problems
    infocollection = pathjoin(path_collection,'collection-info.json')
    problems = []
    data = []
    with open(infocollection, 'r') as f:
        for attrib in json.load(f):
            problems.append(attrib['problem-name'])
    scores=[];
    for problem in problems:
        f1,precision,recall,accuracy=evaluate(
                pathjoin(path_collection, problem, 'ground-truth.json'),
                pathjoin(path_answers,'answers-'+problem+'.json'))
        scores.append(f1);
        d = {
                'problem-name'   : problem,
                'macro-f1'       : round(f1,3),
                'macro-precision': round(precision,3),
                'macro-recall'   : round(recall,3),
                'micro-accuracy' : round(accuracy,3)
        };
        d.update(config);
        d.update(problemInfo[problem])
        data.append(d)
        print(str(problem),'Macro-F1:',round(f1,3))
        
    overall_score=sum(scores)/len(scores)
    # Saving data to output files (out.json and evaluation.prototext)
    with open(pathjoin(path_out,'out.json'), 'w') as f:
        json.dump({
                'problems': data,
                'overall_score': round(overall_score,3)
                }, f, indent=4, sort_keys=True)
    print('Overall score:', round(overall_score,3))
    prototext='measure {\n key: "mean macro-f1"\n value: "'+str(round(overall_score,3))+'"\n}\n'
    with open(pathjoin(path_out,'evaluation.prototext'), 'w') as f:
        f.write(prototext)
        
    pd.DataFrame(data).to_csv(pathjoin(path_out,'evaluation.csv'), index=False)
    
        
        
#*******************************************************************************************************

def read_files(path,label):
    # Reads all text files located in the 'path' and assigns them to 'label' class
    files = glob.glob(path+os.sep+label+os.sep+'*.txt')
    texts=[]
    for i,v in enumerate(files):
        f=codecs.open(v,'r',encoding='utf-8')
        texts.append((f.read(),label))
        f.close()
    return texts


def represent_text(text,n):
    # Extracts all character 'n'-grams from  a 'text'
    if n>0:
        tokens = [text[i:i+n] for i in range(len(text)-n+1)]
    frequency = defaultdict(int)
    for token in tokens:
        frequency[token] += 1
    return frequency



def extract_vocabulary(texts,n,ft):
    # Extracts all characer 'n'-grams occurring at least 'ft' times in a set of 'texts'
    occurrences=defaultdict(int)
    for (text,label) in texts:
        #extract vocabulary for one text
        text_occurrences=represent_text(text,n)
        #merges all vocabularies
        for ngram in text_occurrences:
            if ngram in occurrences:
                occurrences[ngram]+=text_occurrences[ngram]
            else:
                occurrences[ngram]=text_occurrences[ngram]
    vocabulary=[]
    for i in occurrences.keys():
        if occurrences[i]>=ft:
            vocabulary.append(i)
    return vocabulary



def readCollectionsOfProblems(path):
    # Reading information about the collection
    infocollection = path+os.sep+'collection-info.json'
    problems = []
    language = []
    encoding = []
    with open(infocollection, 'r') as f:
        for attrib in json.load(f):
            problems.append(attrib['problem-name'])
            language.append(attrib['language'])
            encoding.append(attrib['encoding'])
    return problems, language, encoding;

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


cache = {};
cacheProblems = {}

            
def improvements(path,outpath,testcase,case):
    ngram_range=(case['ngram_min'],case['ngram_max'])
    df=(case['df_min'],case['df_max'])
    classifier=case['classifier']
    multiclasses=case['multiclasses']
    use_idf = case['use_idf']
    norm=case['norm']
    
    start_time = time.time()
    problems, language, _ = readCollectionsOfProblems(path);
    
    
    allProblems = [];
    problemInfo = {};
    
    for index,problem in enumerate(problems):
        print(problem);
        
        if problem in cacheProblems:
            unk_folder = cacheProblems[problem]['unk_folder'];
            candidates_folder = cacheProblems[problem]['candidates_folder'];
        else:
            unk_folder, candidates_folder = readProblem(path, problem);
            cacheProblems[problem]= {
                    "unk_folder":unk_folder,
                    "candidates_folder":candidates_folder
                    };

        #adding an improvment
        if problem in cache:
            train_docs = cache[problem]['train'];
            test_docs  = cache[problem]['test'];
        else:
            train_docs=[]
            for candidate in candidates_folder:
                train_docs.extend(read_files(path+os.sep+problem,candidate));
                
            test_docs=read_files(path+os.sep+problem,unk_folder);
            cache[problem] = {
                    'train':train_docs,
                    'test': test_docs,
                    'data':{}
                    };
        
        
        
        # Building training set    
        train_texts , train_labels = zip(*train_docs);
        test_texts  , test_labels  = zip(*test_docs);
        

        
        if problem in cache:
            key = "%s_%s_%s_%s_%s_%s" % (ngram_range[0], ngram_range[1],use_idf, df[0], df[1], norm);
            if key in cache[problem]['data']:
                vectorizer = cache[problem]['data'][key]['vectorizer']
                train_data = cache[problem]['data'][key]['train']
                test_data  = cache[problem]['data'][key]['test']
            else:
                vectorizer = TfidfVectorizer(
                    analyzer='char',
                    ngram_range=ngram_range,
                    lowercase=False,
                    use_idf = use_idf,
                    min_df = df[0],
                    max_df = df[1],
                    norm=norm,
                    dtype=float)
                train_data = vectorizer.fit_transform(train_texts);
                test_data  = vectorizer.transform(test_texts).astype(float);
                
                cache[problem]['data'][key]= {
                        'vectorizer': vectorizer,
                        'train':train_data,
                        'test':test_data
                    };      
                
        
        # Building test set
        
        problemInfo[problem]={
                'language': language[index],
                'candidateAuthors': len(candidates_folder),
                'knownTexts':len(train_texts),
                'vocabularySize': len(vectorizer.vocabulary_)
                
                }
        
        #print('\t', 'language: ', language[index])
        #print('\t', len(candidates_folder), )
        #print('\t', len(train_texts), 'known texts')
        #print('\t', 'vocabulary size:', len(vocabulary))
        
        
        # Applying SVM
        max_abs_scaler = preprocessing.MaxAbsScaler()
        scaled_train_data = max_abs_scaler.fit_transform(train_data)
        scaled_test_data = max_abs_scaler.transform(test_data)
        
        if classifier == 'lr':
            if multiclasses == 'ovr':
                clf = LogisticRegression(multi_class='ovr', random_state=42);
            else:
                clf = LogisticRegression(multi_class='multinomial', solver='newton-cg',  random_state=42)
        else:
            if multiclasses=='ovr':
                clf=OneVsRestClassifier(LinearSVC(C=1))
            else:
                clf=OneVsOneClassifier(LinearSVC(C=1))
        clf = clf.fit(scaled_train_data, train_labels);
        predictions=clf.predict(scaled_test_data)
        
        
        # Writing output file
        out_data=[]
        unk_filelist = glob.glob(path+os.sep+problem+os.sep+unk_folder+os.sep+'*.txt')
        pathlen=len(path+os.sep+problem+os.sep+unk_folder+os.sep)
        for i,v in enumerate(predictions):
            out_data.append(
                    {'unknown-text': unk_filelist[i][pathlen:],
                     'predicted-author': v
                     }
                    )
        with open(outpath+os.sep+'answers-'+problem+'.json', 'w') as f:
            json.dump(out_data, f, indent=4)
        allProblems.extend(out_data)
        #print('\t', 'answers saved to file','answers-'+problem+'.json')
    
    print('elapsed time:', time.time() - start_time)
    return problemInfo;

def main():
    #original main
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='PAN-18 Baseline Authorship Attribution Method')
    parser.add_argument('-i', type=str, help='Path to the main folder of a collection of attribution problems')
    parser.add_argument('-o', type=str, help='Path to an output folder')
    parser.add_argument('-n', type=int, default=3, help='n-gram order (default=3)')
    parser.add_argument('-ft', type=int, default=5, help='frequency threshold (default=5)')
    parser.add_argument('-c', type=str, default='OneVsRest', help='OneVsRest or OneVsOne (default=OneVsRest)')
    args = parser.parse_args()
    if not args.i:
        print('ERROR: The input folder is required')
        parser.exit(1)
    if not args.o:
        print('ERROR: The output folder is required')
        parser.exit(1)
    
    baseline(args.i, args.o, args.n, args.ft, args.c)
    
    
def test():
    baseDir = '/Users/joseeleandrocustodio/Dropbox/mestrado/02 - Pesquisa/code';
    inputDir= pathjoin(baseDir,'pan18aa');
    
    for d in os.listdir(pathjoin(baseDir,'out')):
        try:
            os.rmdir(pathjoin(baseDir,'out',d));
        except:
            pass
    
#    if os.path.exists(pathjoin(baseDir,"cache.pkl")):
#        with open(pathjoin(baseDir,"cache.pkl"),"rb") as f:
#            cache = pickle.load(f);
    
    instanceId = 0 ;

    
    cases = [
            {
            'classifier': classifier,
            'multiclasses':multiclasses,
            'ngram_min':ngram_max,
            'ngram_max':ngram_min,
            'use_idf':use_idf,
            'df_min':df_min,
            'df_max':df_max,
            'norm':norm

            }
            for use_idf in [True, False]
            for norm in ['l2', 'l1', None]
            for ngram_min in [2,3,4,5]
            for ngram_max in [2,3,4,5]
            for df_min in [0.00, 0.25, 0.5, 0.75, 0.9, 0.95]
            for df_max in [0.01, 0.25, 0.5, 0.75,  0.95, 0.99 , 1.0]
            for classifier in ['lr','svm']
            for multiclasses in ['ovr','other']
                    
    ]
    
    cases = [ c for c in cases
             if c['df_max'] >= c['df_min']
                 and c['df_max'] - c['df_min'] > 0.2
                 and c['ngram_max'] > c['ngram_min']
                 and not (c['classifier']== 'svm' and c['multiclasses'] =='other')
                 ]

    random.shuffle(cases);
    
    lencases = len(cases)*1.0;
    
    while cases:
        if os.path.exists(pathjoin(baseDir,"stop")):
            break;
            
        print ("progress: %s" % (len(cases)/lencases))
            
        c = cases.pop();
        
        
        instanceId +=1;
        instanceName = '-'.join([ '_'.join([k,str(c[k])]) for k in c ]);
        print (instanceName);
        outputDir= pathjoin(baseDir,'out',instanceName);
        if not os.path.exists(outputDir):
            os.mkdir(outputDir);
        else:
            continue;

        try:
            problemInfo = improvements(inputDir,outputDir,instanceName,c);
            evaluate_all(inputDir,outputDir, outputDir, c, problemInfo);
        except ValueError as e:
            print("error : "+e.message);
#    
#    with open(pathjoin(baseDir,"cache.pkl"),"wb") as f:
#        pickle.dump(cache,f,pickle.HIGHEST_PROTOCOL);

if __name__ == '__main__':
    test()