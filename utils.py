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

import pickle
import os

cache_path='/home/gykovacs/workspaces/smote_results/'

category_mapping= {'NR': 'noise removal',
                   'DR': 'dimension reduction',
                   'Clas': 'uses classifier',
                   'SCmp': 'componentwise sampling',
                   'SCpy': 'sampling by cloning',
                   'SO': 'ordinary sampling',
                   'M': 'memetic',
                   'DE': 'density estimation',
                   'DB': 'density based',
                   'Ex': 'extensive',
                   'CM': 'changes majority',
                   'Clus': 'uses clustering',
                   'BL': 'borderline',
                   'A': 'application'}

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

def oversampling_bib_lookup():
    oversamplers= sv.get_all_oversamplers()
    oversamplers.remove(sv.NoSMOTE)

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
    
    return oversampling_bibtex

def oversampler_summary_table():
    oversamplers= sv.get_all_oversamplers()
    oversamplers.remove(sv.NoSMOTE)

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
    cat_summary_first= cat_summary.iloc[:int(len(cat_summary)/2+0.5)].reset_index()
    cat_summary_second= cat_summary.iloc[int(len(cat_summary)/2+0.5):].reset_index()
    print(cat_summary_second.columns)
    cat_summary_second['index']= cat_summary_second['index'].astype(str)
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
    res= res.replace(' A ', prefix + 'application' + postfix)
    res= res.replace('index', '')
    res= res.replace('\\toprule', '')
    res= res.replace('cite(', '\\cite{')
    res= res.replace('))', '}')
    res= res.replace('\_', '_')
    res= res.replace('NaN', '')

    print(res)
    
    return oversampling_bibtex

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

oversampling_bibtex= oversampler_summary_table()

dataset_summary_table()

def top_results_classifier(classifier):
    results= pickle.load(open(os.path.join(cache_path, 'results.pickle'), 'rb'))
    
    results_cl= results[results['classifier'] == classifier]
    results_agg= results_cl.groupby(by='sampler').agg({'auc': np.mean, 'gacc': np.mean, 'brier': np.mean, 'p_top20': np.mean})
    results_agg= results_agg.reset_index()
    results_rank= results_agg.rank(numeric_only= True, ascending= False)
    results_rank['brier']= results_agg['brier'].rank(ascending= True)
    results_rank= pd.DataFrame(np.mean(results_rank, axis= 1))
    results_rank['sampler']= results_agg['sampler']
    results_rank.columns= ['rank', 'sampler']

    final_auc= results_agg[['sampler', 'auc']].sort_values(by= 'auc', ascending= False).iloc[:10].reset_index(drop= True)
    final_auc['sampler']= final_auc['sampler'].apply(lambda x: x.replace('_', '-') + ' cite(' + oversampling_bibtex[x]['key'] + '))')
    final_gacc= results_agg[['sampler', 'gacc']].sort_values(by= 'gacc', ascending= False).iloc[:10].reset_index(drop= True)
    final_gacc['sampler']= final_gacc['sampler'].apply(lambda x: x.replace('_', '-') + ' cite(' + oversampling_bibtex[x]['key'] + '))')
    final_brier= results_agg[['sampler', 'brier']].sort_values(by= 'brier', ascending= True).iloc[:10].reset_index(drop= True)
    final_brier['sampler']= final_brier['sampler'].apply(lambda x: x.replace('_', '-') + ' cite(' + oversampling_bibtex[x]['key'] + '))')
    final_ptop20= results_agg[['sampler', 'p_top20']].sort_values(by= 'p_top20', ascending= False).iloc[:10].reset_index(drop= True)
    final_ptop20['sampler']= final_ptop20['sampler'].apply(lambda x: x.replace('_', '-') + ' cite(' + oversampling_bibtex[x]['key'] + '))')
    
    final= pd.concat([final_auc, final_gacc, final_brier, final_ptop20], axis= 1)
    baseline= results_agg[results_agg['sampler'] == 'NoSMOTE']
    final.columns= ['sampler', 'score', 'sampler', 'score', 'sampler', 'score', 'sampler', 'score']

    iterables= [['AUC', 'GACC', 'Brier', '% top20'], ['sampler', 'score']]
    index= pd.MultiIndex.from_product(iterables, names= ['score', ''])

    final.columns= index

    final.index= final.index + 1

    final= final.append(pd.DataFrame({final.columns[0]: 'no sampling', final.columns[1]: baseline['auc'],
                           final.columns[2]: 'no sampling', final.columns[3]: baseline['gacc'],
                           final.columns[4]: 'no sampling', final.columns[5]: baseline['brier'],
                           final.columns[6]: 'no sampling', final.columns[7]: baseline['p_top20'],}), ignore_index= True)
    final.index= final.index + 1

    table= final.to_latex(float_format= lambda x: ('%.4f' % x).replace('0.', '.'))
    table= table.replace('cite(', '\\cite{')
    table= table.replace('))', '}')
    table= table.replace('\_', '_')
    print(table)

def top_results_score(score, databases= 'all'):
    ascending= False
    if score == 'brier':
        ascending= True
    
    results= pickle.load(open(os.path.join(cache_path, 'results.pickle'), 'rb'))
    
    if databases == 'extreme_ir':
        results= results[results['imbalanced_ratio'] > 9]
    
    results_score= results[['db_name', 'classifier', 'sampler', score]]
    results_agg= results_score.groupby(by=['classifier', 'sampler']).agg({score: np.mean})
    results_agg= results_agg.reset_index()
    
    results_svm= results_agg[results_agg['classifier'] == 'CalibratedClassifierCV']
    results_dt= results_agg[results_agg['classifier'] == 'DecisionTreeClassifier']
    results_knn= results_agg[results_agg['classifier'] == 'KNeighborsClassifier']
    results_mlp= results_agg[results_agg['classifier'] == 'MLPClassifierWrapper']
    
    final_svm= results_svm[['sampler', score]].sort_values(by= score, ascending= ascending).iloc[:8].reset_index(drop= True)
    final_dt= results_dt[['sampler', score]].sort_values(by= score, ascending= ascending).iloc[:8].reset_index(drop= True)
    final_knn= results_knn[['sampler', score]].sort_values(by= score, ascending= ascending).iloc[:8].reset_index(drop= True)
    final_mlp= results_mlp[['sampler', score]].sort_values(by= score, ascending= ascending).iloc[:8].reset_index(drop= True)
    
    final_svm['sampler']= final_svm['sampler'].apply(lambda x: x.replace('_', '-') + ' cite(' + oversampling_bibtex[x]['key'] + '))')
    final_dt['sampler']= final_dt['sampler'].apply(lambda x: x.replace('_', '-') + ' cite(' + oversampling_bibtex[x]['key'] + '))')
    final_knn['sampler']= final_knn['sampler'].apply(lambda x: x.replace('_', '-') + ' cite(' + oversampling_bibtex[x]['key'] + '))')
    final_mlp['sampler']= final_mlp['sampler'].apply(lambda x: x.replace('_', '-') + ' cite(' + oversampling_bibtex[x]['key'] + '))')
    
    final= pd.concat([final_svm, final_dt, final_knn, final_mlp], axis= 1)
    baseline= results_agg[results_agg['sampler'] == 'NoSMOTE']
    smote= results_agg[results_agg['sampler'] == 'SMOTE']
    
    iterables= [['SVM', 'DT', 'kNN', 'MLP'], ['sampler', score]]
    index= pd.MultiIndex.from_product(iterables, names= ['classifier', ''])
    
    final.columns= index
    
    final.index= final.index + 1
    
    final= final.append(pd.DataFrame({final.columns[0]: 'SMOTE', final.columns[1]: smote[smote['classifier'] == 'CalibratedClassifierCV'][score].iloc[0],
                                      final.columns[2]: 'SMOTE', final.columns[3]: smote[smote['classifier'] == 'DecisionTreeClassifier'][score].iloc[0],
                                      final.columns[4]: 'SMOTE', final.columns[5]: smote[smote['classifier'] == 'KNeighborsClassifier'][score].iloc[0],
                                      final.columns[6]: 'SMOTE', final.columns[7]: smote[smote['classifier'] == 'MLPClassifierWrapper'][score].iloc[0]}, index=['baseline']))
    
    final= final.append(pd.DataFrame({final.columns[0]: 'no sampling', final.columns[1]: baseline[baseline['classifier'] == 'CalibratedClassifierCV'][score].iloc[0],
                                      final.columns[2]: 'no sampling', final.columns[3]: baseline[baseline['classifier'] == 'DecisionTreeClassifier'][score].iloc[0],
                                      final.columns[4]: 'no sampling', final.columns[5]: baseline[baseline['classifier'] == 'KNeighborsClassifier'][score].iloc[0],
                                      final.columns[6]: 'no sampling', final.columns[7]: baseline[baseline['classifier'] == 'MLPClassifierWrapper'][score].iloc[0]}, index=['baseline']))
    
    table= final.to_latex(float_format= lambda x: ('%.4f' % x).replace('0.', '.'))
    table= table.replace('cite(', '\\cite{')
    table= table.replace('))', '}')
    table= table.replace('\_', '_')
    print(table)
    
    return final

def top_results_overall(databases= 'all'):
    results= pickle.load(open(os.path.join(cache_path, 'results.pickle'), 'rb'))
    
    if databases == 'high_ir':
        results= results[results['imbalanced_ratio'] > 9]
    elif databases == 'low_ir':
        results= results[results['imbalanced_ratio'] <= 9]
    elif databases == 'high_n_min':
        results= results[results['db_size']/(1.0 + results['imbalanced_ratio']) > 30]
    elif databases == 'low_n_min':
        results= results[results['db_size']/(1.0 + results['imbalanced_ratio']) <= 30]
    elif databases == 'high_n_attr':
        results= results[results['db_n_attr'] > 10]
    elif databases == 'low_n_attr':
        results= results[results['db_n_attr'] <= 10]
        
    results_agg= results.groupby(by='sampler').agg({'auc': np.mean, 'gacc': np.mean, 'f1': np.mean, 'p_top20': np.mean})
    results_agg= results_agg.reset_index()
    #results_rank= results_agg.rank(numeric_only= True, ascending= False)
    #results_rank['brier']= results_agg['brier'].rank(ascending= True)
    #results_rank= pd.DataFrame(np.mean(results_rank, axis= 1))
    #results_rank['sampler']= results_agg['sampler']
    #results_rank.columns= ['rank', 'sampler']

    final_auc= results_agg[['sampler', 'auc']].sort_values(by= 'auc', ascending= False).iloc[:8].reset_index(drop= True)
    final_auc['sampler']= final_auc['sampler'].apply(lambda x: x.replace('_', '-') + ' cite(' + oversampling_bibtex[x]['key'] + '))')
    final_gacc= results_agg[['sampler', 'gacc']].sort_values(by= 'gacc', ascending= False).iloc[:8].reset_index(drop= True)
    final_gacc['sampler']= final_gacc['sampler'].apply(lambda x: x.replace('_', '-') + ' cite(' + oversampling_bibtex[x]['key'] + '))')
    final_f1= results_agg[['sampler', 'f1']].sort_values(by= 'f1', ascending= False).iloc[:8].reset_index(drop= True)
    final_f1['sampler']= final_f1['sampler'].apply(lambda x: x.replace('_', '-') + ' cite(' + oversampling_bibtex[x]['key'] + '))')
    final_ptop20= results_agg[['sampler', 'p_top20']].sort_values(by= 'p_top20', ascending= False).iloc[:8].reset_index(drop= True)
    final_ptop20['sampler']= final_ptop20['sampler'].apply(lambda x: x.replace('_', '-') + ' cite(' + oversampling_bibtex[x]['key'] + '))')
    
    final= pd.concat([final_auc, final_gacc, final_f1, final_ptop20], axis= 1)
    smote= results_agg[results_agg['sampler'] == 'SMOTE']
    baseline= results_agg[results_agg['sampler'] == 'NoSMOTE']

    iterables= [['AUC', 'GACC', 'F1', '% top20'], ['sampler', 'score']]
    index= pd.MultiIndex.from_product(iterables, names= ['score', ''])

    final.columns= index

    final.index= final.index + 1

    final= final.append(pd.DataFrame({final.columns[0]: 'SMOTE', final.columns[1]: smote['auc'].iloc[0],
                                     final.columns[2]: 'SMOTE', final.columns[3]: smote['gacc'].iloc[0],
                                     final.columns[4]: 'SMOTE', final.columns[5]: smote['f1'].iloc[0],
                                     final.columns[6]: 'SMOTE', final.columns[7]: smote['p_top20'].iloc[0]}, index= ['baseline']))

    final= final.append(pd.DataFrame({final.columns[0]: 'no sampling', final.columns[1]: baseline['auc'].iloc[0],
                           final.columns[2]: 'no sampling', final.columns[3]: baseline['gacc'].iloc[0],
                           final.columns[4]: 'no sampling', final.columns[5]: baseline['f1'].iloc[0],
                           final.columns[6]: 'no sampling', final.columns[7]: baseline['p_top20'].iloc[0],}, index= ['baseline']))
    
    final.columns= ['sampler', 'AUC', 'sampler', 'GAcc', 'sampler', 'F1', 'sampler', 'P20']

    table= final.to_latex(float_format= lambda x: ('%.4f' % x).replace('0.', '.'))
    table= table.replace('cite(', '\\cite{')
    table= table.replace('))', '}')
    table= table.replace('\_', '_')
    print(table)




def top_results_overall2(databases= 'all'):
    results= pickle.load(open(os.path.join(cache_path, 'results.pickle'), 'rb'))
    
    if databases == 'high_ir':
        results= results[results['imbalanced_ratio'] > 9]
    elif databases == 'low_ir':
        results= results[results['imbalanced_ratio'] <= 9]
    elif databases == 'high_n_min':
        results= results[results['db_size']/(1.0 + results['imbalanced_ratio']) > 30]
    elif databases == 'low_n_min':
        results= results[results['db_size']/(1.0 + results['imbalanced_ratio']) <= 30]
    elif databases == 'high_n_attr':
        results= results[results['db_n_attr'] > 10]
    elif databases == 'low_n_attr':
        results= results[results['db_n_attr'] <= 10]
    
    
    results_agg= results.groupby(by='sampler').agg({'auc': np.mean, 'gacc': np.mean, 'f1': np.mean, 'p_top20': np.mean})
    results_agg= results_agg.reset_index()
    results_agg= results_agg[results_agg['sampler'] != 'NoSMOTE']
    
    results_rank= results_agg.rank(numeric_only= True, ascending= False)
    results_rank['sampler']= results_agg['sampler']
    results_rank['overall']= np.mean(results_rank[['auc', 'gacc', 'f1', 'p_top20']], axis= 1)
    results_rank.columns= ['rank_auc', 'rank_gacc', 'rank_f1', 'rank_ptop20', 'sampler', 'overall']
    results_agg['rank_auc']= results_rank['rank_auc']
    results_agg['rank_gacc']= results_rank['rank_gacc']
    results_agg['rank_f1']= results_rank['rank_f1']
    results_agg['rank_ptop20']= results_rank['rank_ptop20']
    results_agg['overall']= results_rank['overall']
    results_agg= results_agg.sort_values('overall')
    results_agg= results_agg[['sampler', 'overall', 'auc', 'rank_auc', 'gacc', 'rank_gacc', 'f1', 'rank_f1', 'p_top20', 'rank_ptop20']]
    results_agg= results_agg.reset_index(drop= True)
    results_agg.index= results_agg.index + 1
    final= results_agg.iloc[:10]
    
    final['rank_auc']= final['rank_auc'].astype(int)
    final['rank_gacc']= final['rank_gacc'].astype(int)
    final['rank_f1']= final['rank_f1'].astype(int)
    final['rank_ptop20']= final['rank_ptop20'].astype(int)
    
    final['sampler']= final['sampler'].apply(lambda x: x.replace('_', '-') + ' cite(' + oversampling_bibtex[x]['key'] + '))')

    table= final.to_latex(float_format= lambda x: ('%.4f' % x).replace(' 0.', '.'))
    table= table.replace('cite(', '\\cite{')
    table= table.replace('))', '}')
    table= table.replace('\_', '_')
    print(table)

def top_results_overall3():
    results= pickle.load(open(os.path.join(cache_path, 'results.pickle'), 'rb'))
    
    def ranking(results):
        results_agg= results.groupby(by='sampler').agg({'auc': np.mean, 'gacc': np.mean, 'f1': np.mean, 'p_top20': np.mean})
        results_agg= results_agg.reset_index()
        results_agg= results_agg[results_agg['sampler'] != 'NoSMOTE']
        
        results_rank= results_agg.rank(numeric_only= True, ascending= False)
        results_rank['sampler']= results_agg['sampler']
        results_rank['overall']= np.mean(results_rank[['auc', 'gacc', 'f1', 'p_top20']], axis= 1)
        results_rank.columns= ['rank_auc', 'rank_gacc', 'rank_f1', 'rank_ptop20', 'sampler', 'overall']
        results_agg['rank_auc']= results_rank['rank_auc']
        results_agg['rank_gacc']= results_rank['rank_gacc']
        results_agg['rank_f1']= results_rank['rank_f1']
        results_agg['rank_ptop20']= results_rank['rank_ptop20']
        results_agg['overall']= results_rank['overall']
        results_agg= results_agg.sort_values('overall')
        results_agg= results_agg[['sampler', 'overall', 'auc', 'rank_auc', 'gacc', 'rank_gacc', 'f1', 'rank_f1', 'p_top20', 'rank_ptop20']]
        results_agg= results_agg.reset_index(drop= True)
        results_agg.index= results_agg.index + 1
        final= results_agg
        
        final['rank_auc']= final['rank_auc'].astype(int)
        final['rank_gacc']= final['rank_gacc'].astype(int)
        final['rank_f1']= final['rank_f1'].astype(int)
        final['rank_ptop20']= final['rank_ptop20'].astype(int)
        
        return final[['sampler', 'overall']]
    
    results_all= ranking(results)
    results_all['overall']= results_all['overall'].rank()
    results_high_ir= ranking(results= results[results['imbalanced_ratio'] > 9])
    results_high_ir['overall']= results_high_ir['overall'].rank()
    results_high_ir.columns= ['sampler', 'overall_high_ir']
    results_low_ir= ranking(results= results[results['imbalanced_ratio'] <= 9])
    results_low_ir['overall']= results_low_ir['overall'].rank()
    results_low_ir.columns= ['sampler', 'overall_low_ir']
    results_high_n_min= ranking(results= results[results['db_size']/(1.0 + results['imbalanced_ratio']) > 30])
    results_high_n_min['overall']= results_high_n_min['overall'].rank()
    results_high_n_min.columns= ['sampler', 'overall_high_n_min']
    results_low_n_min= ranking(results= results[results['db_size']/(1.0 + results['imbalanced_ratio']) <= 30])
    results_low_n_min['overall']= results_low_n_min['overall'].rank()
    results_low_n_min.columns= ['sampler', 'overall_low_n_min']
    results_high_n_attr= ranking(results= results[results['db_n_attr'] > 10])
    results_high_n_attr['overall']= results_high_n_attr['overall'].rank()
    results_high_n_attr.columns= ['sampler', 'overall_high_n_attr']
    results_low_n_attr= ranking(results= results[results['db_n_attr'] <= 10])
    results_low_n_attr['overall']= results_low_n_attr['overall'].rank()
    results_low_n_attr.columns= ['sampler', 'overall_low_n_attr']
    
    final= results_all.merge(results_high_ir, on= 'sampler').merge(results_low_ir, on= 'sampler').merge(results_high_n_min, on= 'sampler').merge(results_low_n_min, on= 'sampler').merge(results_high_n_attr, on= 'sampler').merge(results_low_n_attr, on= 'sampler')
    final= final[(final['overall'] <= 10) | (final['overall_high_ir'] <= 10) | (final['overall_low_ir'] <= 10) | (final['overall_high_n_min'] <= 10) | (final['overall_low_n_min'] <= 10) | (final['overall_high_n_attr'] <= 10) | (final['overall_low_n_attr'] <= 10)]
    
    final['sampler']= final['sampler'].apply(lambda x: x.replace('_', '-') + ' cite(' + oversampling_bibtex[x]['key'] + '))')
    final= final.reset_index(drop= True)
    final.index= final.index + 1

    table= final.to_latex(float_format= lambda x: ('%.0f' % x).replace(' 0.', '.'))
    table= table.replace('cite(', '\\cite{')
    table= table.replace('))', '}')
    table= table.replace('\_', '_')
    
    
    
    print(table)

def top_results_by_categories(percentile= 75):
    results= pickle.load(open(os.path.join(cache_path, 'results.pickle'), 'rb'))
    categories= ['NR', 'DR', 'Clas', 'SCmp', 'SCpy', 'SO', 'M', 'DE', 'DB', 'Ex', 'CM', 'Clus', 'BL', 'A']

    for i in categories:
        results[i]= results['sampler_categories'].apply(lambda x: i in eval(x))
    
    res_cat= {}
    for i in categories:
        res_cat[i]= results[results[i] == True]
    
    for r in res_cat:
        res_cat[r]= res_cat[r].groupby(by=['sampler']).agg({'auc': np.mean,
                                                           'f1': np.mean,
                                                           'gacc': np.mean,
                                                           'acc': np.mean,
                                                           'brier': np.mean,
                                                           'p_top20': np.mean})
    
    res= {}
    for i in categories:
        res[category_mapping[i]]= {'auc': np.percentile(res_cat[i]['auc'], percentile),
                       'f1': np.percentile(res_cat[i]['f1'], percentile),
                       'gacc': np.percentile(res_cat[i]['gacc'], percentile),
                       'acc': np.percentile(res_cat[i]['acc'], percentile),
                       'brier': np.percentile(res_cat[i]['brier'], 100 - 75),
                       'p_top20': np.percentile(res_cat[i]['p_top20'], percentile)}
    
    r= pd.DataFrame.from_dict(res).T
    
    r_auc= r['auc'].sort_values(ascending= False)
    r_gacc= r['gacc'].sort_values(ascending= False)
    r_f1= r['f1'].sort_values(ascending= False)
    r_ptop20= r['p_top20'].sort_values(ascending= False)
    
    r_auc= r_auc.reset_index()
    r_gacc= r_gacc.reset_index()
    r_f1= r_f1.reset_index()
    r_ptop20= r_ptop20.reset_index()
    
    final= pd.concat([r_auc, r_gacc, r_f1, r_ptop20], axis= 1, ignore_index= True)
    
    iterables= [['AUC', 'GACC', 'F1', '% top20'], ['attribute', 'score']]
    index= pd.MultiIndex.from_product(iterables, names= ['score', ''])

    final.columns= index

    final.index= final.index + 1
    
    table= final.to_latex(float_format= lambda x: ('%.4f' % x).replace('0.', '.'))
    table= table.replace('cite(', '\\cite{')
    table= table.replace('))', '}')
    table= table.replace('\_', '_')
    print(table)
    
    return final

def runtimes():
    results= pickle.load(open(os.path.join(cache_path, 'results.pickle'), 'rb'))
    
    results= results[results['sampler'] != 'NoSMOTE']
    
    results_agg= results.groupby('sampler').aggregate({'runtime': np.mean})
    results_sorted= results_agg.sort_values('runtime')
    
    results_sorted= results_sorted.reset_index()
    
    n= int(len(results_sorted)/3 + 0.5) + 1
    
    results_sorted.index= results_sorted.index + 1
    
    results_sorted['sampler']= results_sorted['sampler'].apply(lambda x: x.replace('_', '-') + ' cite(' + oversampling_bibtex[x]['key'] + '))')
    
    final= pd.concat([results_sorted.iloc[:n].reset_index(), results_sorted.iloc[n:2*n].reset_index(), results_sorted.iloc[2*n:3*n].reset_index()], axis= 1)
    
    table= final.to_latex(float_format= lambda x: ('%.4f' % x).replace(' 0.', ' .'), index= False)
    table= table.replace('cite(', '\\cite{')
    table= table.replace('))', '}')
    table= table.replace('\_', '_')
    
    print(table)
    
    return final
    

def top_results_rank(score):
    ascending= False
    if score == 'brier':
        ascending= True
    
    results= pickle.load(open(os.path.join(cache_path, 'results.pickle'), 'rb'))
    
    results_svm= results[results['classifier'] == 'CalibratedClassifierCV'][['db_name', 'sampler', 'brier', 'acc', 'f1', 'auc', 'p_top20', 'gacc']]
    results_dt= results[results['classifier'] == 'DecisionTreeClassifier'][['db_name', 'sampler', 'brier', 'acc', 'f1', 'auc', 'p_top20', 'gacc']]
    results_knn= results[results['classifier'] == 'KNeighborsClassifier'][['db_name', 'sampler', 'brier', 'acc', 'f1', 'auc', 'p_top20', 'gacc']]
    results_mlp= results[results['classifier'] == 'MLPClassifierWrapper'][['db_name', 'sampler', 'brier', 'acc', 'f1', 'auc', 'p_top20', 'gacc']]
    
    results_svm_agg= results_svm.groupby(by=['sampler']).agg({'brier': np.mean, 'acc': np.mean, 'auc': np.mean, 'f1': np.mean, 'p_top20': np.mean, 'gacc': np.mean})
    results_svm_dt= results_svm.groupby(by=['sampler']).agg({'brier': np.mean, 'acc': np.mean, 'auc': np.mean, 'f1': np.mean, 'p_top20': np.mean, 'gacc': np.mean})
    results_svm_knn= results_svm.groupby(by=['sampler']).agg({'brier': np.mean, 'acc': np.mean, 'auc': np.mean, 'f1': np.mean, 'p_top20': np.mean, 'gacc': np.mean})
    results_svm_mlp= results_svm.groupby(by=['sampler']).agg({'brier': np.mean, 'acc': np.mean, 'auc': np.mean, 'f1': np.mean, 'p_top20': np.mean, 'gacc': np.mean})
    
    results_score= results[['db_name', 'classifier', 'sampler', score]]
    results_agg= results_score.groupby(by=['classifier', 'sampler']).agg({score: np.mean})
    results_agg= results_agg.reset_index()
    
    results_svm= results_agg[results_agg['classifier'] == 'CalibratedClassifierCV']
    results_dt= results_agg[results_agg['classifier'] == 'DecisionTreeClassifier']
    results_knn= results_agg[results_agg['classifier'] == 'KNeighborsClassifier']
    results_mlp= results_agg[results_agg['classifier'] == 'MLPClassifierWrapper']
    
    final_svm= results_svm[['sampler', score]].sort_values(by= score, ascending= ascending).iloc[:8].reset_index(drop= True)
    final_dt= results_dt[['sampler', score]].sort_values(by= score, ascending= ascending).iloc[:8].reset_index(drop= True)
    final_knn= results_knn[['sampler', score]].sort_values(by= score, ascending= ascending).iloc[:8].reset_index(drop= True)
    final_mlp= results_mlp[['sampler', score]].sort_values(by= score, ascending= ascending).iloc[:8].reset_index(drop= True)
    
    final_svm['sampler']= final_svm['sampler'].apply(lambda x: x.replace('_', '-') + ' cite(' + oversampling_bibtex[x]['key'] + '))')
    final_dt['sampler']= final_dt['sampler'].apply(lambda x: x.replace('_', '-') + ' cite(' + oversampling_bibtex[x]['key'] + '))')
    final_knn['sampler']= final_knn['sampler'].apply(lambda x: x.replace('_', '-') + ' cite(' + oversampling_bibtex[x]['key'] + '))')
    final_mlp['sampler']= final_mlp['sampler'].apply(lambda x: x.replace('_', '-') + ' cite(' + oversampling_bibtex[x]['key'] + '))')
    
    final= pd.concat([final_svm, final_dt, final_knn, final_mlp], axis= 1)
    baseline= results_agg[results_agg['sampler'] == 'NoSMOTE']
    smote= results_agg[results_agg['sampler'] == 'SMOTE']
    
    iterables= [['SVM', 'DT', 'kNN', 'MLP'], ['sampler', score]]
    index= pd.MultiIndex.from_product(iterables, names= ['classifier', ''])
    
    final.columns= index
    
    final.index= final.index + 1
    
    final= final.append(pd.DataFrame({final.columns[0]: 'SMOTE', final.columns[1]: smote[smote['classifier'] == 'CalibratedClassifierCV'][score].iloc[0],
                                      final.columns[2]: 'SMOTE', final.columns[3]: smote[smote['classifier'] == 'DecisionTreeClassifier'][score].iloc[0],
                                      final.columns[4]: 'SMOTE', final.columns[5]: smote[smote['classifier'] == 'KNeighborsClassifier'][score].iloc[0],
                                      final.columns[6]: 'SMOTE', final.columns[7]: smote[smote['classifier'] == 'MLPClassifierWrapper'][score].iloc[0]}, index=['baseline']), ignore_index= True)
    
    final= final.append(pd.DataFrame({final.columns[0]: 'no sampling', final.columns[1]: baseline[baseline['classifier'] == 'CalibratedClassifierCV'][score].iloc[0],
                                      final.columns[2]: 'no sampling', final.columns[3]: baseline[baseline['classifier'] == 'DecisionTreeClassifier'][score].iloc[0],
                                      final.columns[4]: 'no sampling', final.columns[5]: baseline[baseline['classifier'] == 'KNeighborsClassifier'][score].iloc[0],
                                      final.columns[6]: 'no sampling', final.columns[7]: baseline[baseline['classifier'] == 'MLPClassifierWrapper'][score].iloc[0]}, index=['baseline']), ignore_index= True)
    
    table= final.to_latex(float_format= lambda x: ('%.4f' % x).replace('0.', '.'))
    table= table.replace('cite(', '\\cite{')
    table= table.replace('))', '}')
    table= table.replace('\_', '_')
    print(table)
    
    return final

f= top_results_score('auc')
f= top_results_score('gacc')
f= top_results_score('p_top20')
f= top_results_score('f1')

top_results_overall()

top_results_overall2()

f= top_results_score('auc', 'extreme_ir')
f= top_results_score('gacc', 'extreme_ir')
f= top_results_score('brier', 'extreme_ir')
f= top_results_score('p_top20', 'extreme_ir')

top_results_classifier('CalibratedClassifierCV')
top_results_classifier('KNeighborsClassifier')
top_results_classifier('DecisionTreeClassifier')
top_results_classifier('MLPClassifierWrapper')
