#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 18:37:20 2018

@author: gykovacs
"""

import sys

# import SMOTE variants
import smote_variants as sv

# imbalanced databases
import imbalanced_databases as imbd

import numpy as np
import pandas as pd

def tokenize_bibtex(entry):
    """
    Tokenize bibtex entry string
    Args:
        entry(str): string of a bibtex entry
    Returns:
        dict: the bibtex entry
    """
    start= entry.find('{') + 1
    token_start= start
    quote_level= 0
    brace_level= 0
    tokens= []
    for i in range(start, len(entry)):
        if entry[i] == '"':
            quote_level= quote_level + 1
        if entry[i] == '{':
            brace_level= brace_level + 1
        if entry[i] == '}':
            brace_level= brace_level - 1
        if (entry[i] == ',' and brace_level == 0) or (entry[i] == '}' and brace_level < 0) and quote_level % 2 == 0:
            tokens.append(entry[token_start:(i)])
            token_start= i + 1
    
    result= {}
    result['key']= tokens[0].strip()
    for i in range(1, len(tokens)):
        splitted= tokens[i].strip().split('=', 1)
        if len(splitted) == 2:
            key= splitted[0].strip().lower()
            value= splitted[1].strip()[1:-1]
            result[key]= value
            
    if 'year' not in result:
        print("No year attribute in %s" % result['key'])
        
    return result
            
def extract_bibtex_entry(string, types= ['@article', '@inproceedings', '@book', '@unknown']):
    """
    Extract bibtex entry from string
    Args:
        string (str): string to process
        types (list(str)): types of bibtex entries to find
    Returns:
        str: the string of the bibtex entry
    """
    lowercase= string.lower()
    for t in types:
        t= t.lower()
        i= lowercase.find(t)
        if i >= 0:
            num_brace= 0
            for j in range(i+len(t), len(string)):
                if string[j] == '{':
                    num_brace+= 1
                if string[j] == '}':
                    num_brace-= 1
                if num_brace == 0:
                    be= tokenize_bibtex(string[i:(j+1)])
                    be['type']= t
                    return be
    return {}

def oversampler_summary_table():
    oversamplers= sv.get_all_oversamplers()

    all_categories= [sv.OverSampling.cat_noise_removal,
                        sv.OverSampling.cat_dim_reduction,
                        sv.OverSampling.cat_uses_classifier,
                        sv.OverSampling.cat_sample_componentwise,
                        sv.OverSampling.cat_sample_ordinary,
                        sv.OverSampling.cat_sample_copy,
                        sv.OverSampling.cat_memetic,
                        sv.OverSampling.cat_density_estimation,
                        sv.OverSampling.cat_density_based,
                        sv.OverSampling.cat_extensive,
                        sv.OverSampling.cat_changes_majority,
                        sv.OverSampling.cat_uses_clustering,
                        sv.OverSampling.cat_borderline,
                        sv.OverSampling.cat_application]
    
    for o in oversamplers:
        sys.stdout.write(o.__name__ + " ")
        sys.stdout.write("& ")
        for i in range(len(all_categories)):
            if all_categories[i] in o.categories:
                sys.stdout.write("$\\times$ ")
            else:
                sys.stdout.write(" ")
            if i != len(all_categories)-1:
                sys.stdout.write("& ")
            else:
                print("\\\\")
    
    oversampling_bibtex= {o.__name__: extract_bibtex_entry(o.__doc__) for o in oversamplers}
    oversampling_years= {o.__name__: oversampling_bibtex[o.__name__]['year'] for o in oversamplers}
    
    oversamplers= sorted(oversamplers, key= lambda x: oversampling_years[x.__name__])
    
    cat_summary= []
    for o in oversamplers:
        cat_summary.append({'method': o.__name__.replace('_', '-') + ' (' + oversampling_years[o.__name__] + ')' + 'cite(' + oversampling_bibtex[o.__name__]['key'] + '))'})
        for a in all_categories:
            cat_summary[-1][a]= str(a in o.categories)
    
    pd.set_option('max_colwidth', 100)
    cat_summary= pd.DataFrame(cat_summary)
    cat_summary= cat_summary[['method'] + all_categories]
    cat_summary.index= np.arange(1, len(cat_summary) + 1)
    cat_summary_first= cat_summary.iloc[:int(len(cat_summary)/2)].reset_index()
    cat_summary_second= cat_summary.iloc[int(len(cat_summary)/2):].reset_index()
    results= pd.concat([cat_summary_first, cat_summary_second], axis= 1)
    
    res= results.to_latex(index= False)
    res= res.replace('True', '$\\times$').replace('False', '')
    prefix= '\\begin{turn}{90}'
    postfix= '\\end{turn}'
    res= res.replace(' NR ', prefix + 'noise removal' + postfix)
    res= res.replace(' DR ', prefix + 'dimension reduction' + postfix)
    res= res.replace(' Clas ', prefix + 'uses classifier' + postfix)
    res= res.replace(' SCmp ', prefix + 'componentwise sampling' + postfix)
    res= res.replace(' SCpy ', prefix + 'sampling by cloning' + postfix)
    res= res.replace(' SO ', prefix + 'ordinary sampling' + postfix)
    res= res.replace(' M ', prefix + 'memetic' + postfix)
    res= res.replace(' DE ', prefix + 'density estimation' + postfix)
    res= res.replace(' DB ', prefix + 'density based' + postfix)
    res= res.replace(' Ex ', prefix + 'extensive' + postfix)
    res= res.replace(' CM ', prefix + 'changes majority' + postfix)
    res= res.replace(' Clus ', prefix + 'uses clustering' + postfix)
    res= res.replace(' BL ', prefix + 'borderline' + postfix)
    res= res.replace('index', '')
    res= res.replace('\\toprule', '')
    res= res.replace('cite(', '\\cite{')
    res= res.replace('))', '}')
    res= res.replace('\_', '_')

    print(res)

def dataset_summary_table():
    results= imbd.summary(include_citation= True)
    
    
    
    citation_keys= results['citation'].apply(lambda x: tokenize_bibtex(x)['key'])
    citation_keys= citation_keys.apply(lambda x: '((' + x + '))')
    results= results[['name', 'len', 'n_minority', 'encoded_n_attr', 'imbalance_ratio', 'imbalance_ratio_dist']]
    results['name']= results['name'] + citation_keys
    results.columns= ['name', 'n', 'n_min', 'n_attr', 'ir', 'idr']
    results= results.sort_values('ir')
    results.index= np.arange(1, len(results) + 1)
    results['ir']= results['ir'].round(2)
    results['idr']= results['idr'].round(2)
    res1= results.iloc[:int(len(results)/2)].reset_index()
    res2= results.iloc[int(len(results)/2):].reset_index()
    res_all= pd.concat([res1, res2], axis= 1)
    
    res= res_all.to_latex(index= False)
    res= res.replace('index', '')
    res= res.replace('\\toprule', '')
    res= res.replace('((', '\\cite{')
    res= res.replace('))', '}')
    
    print(res)