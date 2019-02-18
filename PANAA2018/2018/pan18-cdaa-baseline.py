# -*- coding: utf-8 -*-

"""
 A baseline authorship attribution method
 based on a character n-gram representation
 and a linear SVM classifier
 for Python 2.7
 Questions/comments: stamatatos@aegean.gr

 It can be applied to datasets of PAN-18 cross-domain authorship attribution task
 See details here: http://pan.webis.de/clef18/pan18-web/author-identification.html
 Dependencies:
 - Python 2.7 or 3.6 (we recommend the Anaconda Python distribution)
 - scikit-learn

 Usage from command line: 
    > python pan18-cdaa-baseline.py -i EVALUATION-DIRECTORY -o OUTPUT-DIRECTORY [-n N-GRAM-ORDER] [-ft FREQUENCY-THRESHOLD] [-c CLASSIFIER]
 EVALUATION-DIRECTORY (str) is the main folder of a PAN-18 collection of attribution problems
 OUTPUT-DIRECTORY (str) is an existing folder where the predictions are saved in the PAN-18 format
 Optional parameters of the model:
   N-GRAM-ORDER (int) is the length of character n-grams (default=3)
   FREQUENCY-THRESHOLD (int) is the curoff threshold used to filter out rare n-grams (default=5)
   CLASSIFIER (str) is either 'OneVsOne' or 'OneVsRest' version of SVM (default=OneVsRest)
   
 Example:
     > python pan18-cdaa-baseline.py -i "mydata/pan18-cdaa-development-corpus" -o "mydata/pan18-answers"
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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing



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

def evaluate_all(path_collection,path_answers,path_out, instanceName):
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
        scores.append(f1)
        data.append({
                'problem-name'   : problem,
                'macro-f1'       : round(f1,3),
                'macro-precision': round(precision,3),
                'macro-recall'   : round(recall,3),
                'micro-accuracy' : round(accuracy,3)
        })
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
        
    pd.DataFrame(data).to_csv(pathjoin(path_out,'evaluation'+instanceName+'.csv'), index=False)
    
        
        
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
            
def baseline(path,outpath,n=3,ft=5,classifier='OneVsRest'):
    start_time = time.time()
    problems, language, _ = readCollectionsOfProblems(path);
    
    
    allProblems = [];
    
    for index,problem in enumerate(problems):
        print(problem)

        unk_folder, candidates_folder = readProblem(path, problem);       
                
        # Building training set
        train_docs=[]
        for candidate in candidates_folder:
            train_docs.extend(read_files(path+os.sep+problem,candidate));
            
        train_texts , train_labels = zip(*train_docs);
        
        
        vocabulary = extract_vocabulary(train_docs,n,ft)
        if len(vocabulary) < 10:
            continue;
        vectorizer = CountVectorizer(
                    analyzer='char',
                    ngram_range=(n,n),
                    lowercase=False,
                    vocabulary=vocabulary,
                    dtype=float)
        train_data = vectorizer.fit_transform(train_texts)
        for i,v in enumerate(train_texts):
            #the length of the text is equal to size of vocabulary
            train_data[i]=train_data[i]/len(train_texts[i]) #l1 normalizer.
            
            
            
        #print('\t', 'language: ', language[index])
        #print('\t', len(candidates_folder), 'candidate authors')
        #print('\t', len(train_texts), 'known texts')
        #print('\t', 'vocabulary size:', len(vocabulary))
        
        # Building test set
        test_docs=read_files(path+os.sep+problem,unk_folder)
        test_texts = [text for i,(text,label) in enumerate(test_docs)]
        test_data = vectorizer.transform(test_texts)
        test_data = test_data.astype(float)
        for i,v in enumerate(test_texts):
            test_data[i]=test_data[i]/len(test_texts[i])
        #print('\t', len(test_texts), 'unknown texts')
        
        
        # Applying SVM
        max_abs_scaler = preprocessing.MaxAbsScaler()
        scaled_train_data = max_abs_scaler.fit_transform(train_data)
        scaled_test_data = max_abs_scaler.transform(test_data)
        if classifier=='OneVsOne':
            clf=OneVsOneClassifier(LinearSVC(C=1)).fit(scaled_train_data, train_labels)
        else:
            clf=OneVsRestClassifier(LinearSVC(C=1)).fit(scaled_train_data, train_labels)
        predictions=clf.predict(scaled_test_data)
        
        
        # Writing output file
        out_data=[]
        unk_filelist = glob.glob(path+os.sep+problem+os.sep+unk_folder+os.sep+'*.txt')
        pathlen=len(path+os.sep+problem+os.sep+unk_folder+os.sep)
        for i,v in enumerate(predictions):
            out_data.append(
                    {'unknown-text': unk_filelist[i][pathlen:],
                     'predicted-author': v,
                     
                     #adicional information that is not necessary for PAN2018
                     'language':language[index],
                     'vocabularySize_global':len(vocabulary),
                     'vocabularySize_unknown':np.sum(scaled_test_data[i]>0),
                     'problem':problem,
                     'candicateAuthors':len(candidates_folder),
                     'knownAuthorsSetSize':len(train_texts),
                     'ngrams':n,
                     'corpusMinDf':ft,
                     'classifier' : clf.__class__.__name__
                     }
                    )
        with open(outpath+os.sep+'answers-'+problem+'.json', 'w') as f:
            json.dump(out_data, f, indent=4)
        allProblems.extend(out_data)
        #print('\t', 'answers saved to file','answers-'+problem+'.json')
    
    df = pd.DataFrame(allProblems);
    df.to_csv(pathjoin(outpath,'answers_'+classifier+'_'+str(n)+'_'+str(ft)+'.csv'),index=False)
    print('elapsed time:', time.time() - start_time)

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
    for c in ['OneVsRest','OneVsOne']:
        for ngrams in [1,2,3,4,5,6,7]:
            for ft in [1,5,10,50,100]:
                if ngrams == 7 and ft == 100:
                    continue;
                instanceName = c+'_'+str(ngrams)+'_'+str(ft);
                print (instanceName);
                outputDir= pathjoin(baseDir,'out',instanceName);
                if not os.path.exists(outputDir):
                    os.mkdir(outputDir);
                else:
                    continue;
                
                baseline(inputDir, outputDir, ngrams, ft, c);
                evaluate_all(inputDir,outputDir, outputDir, instanceName);

def test2():
    baseDir = '/Users/joseeleandrocustodio/Dropbox/mestrado/02 - Pesquisa/code';
    inputDir= pathjoin(baseDir,'pan18aa');
    outputDir= pathjoin(baseDir,'out','baseTest');
    baseline(inputDir, outputDir, 4, 5, 'OneVsRest');
    evaluate_all(inputDir,outputDir, outputDir, 'baseTest');

if __name__ == '__main__':
    test2()