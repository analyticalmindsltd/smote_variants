Ranking
*******

Based on a thorough evaluation using 104 imbalanced datasets, the following 10 techniques provide the highest performance in terms of the AUC, GAcc, F1 and P20 scores, in nearest neighbors, support vector machine, decision tree and multilayer perceptron based classification scenarios:


For more details on the evaluation methodology, see our paper on the comparative study.

=========================================================  =========  ========  ==========  ========  ===========  ========  =========  =========  =============
sampler                                                      overall       auc    rank_auc      gacc    rank_gacc        f1    rank_f1    p_top20    rank_ptop20
=========================================================  =========  ========  ==========  ========  ===========  ========  =========  =========  =============
polynom-fit-SMOTE cite(polynomial_fit_smote))                   2.25  0.902538           5  0.870753            1  0.695154          1   0.992496              2
ProWSyn cite(prowsyn))                                          3.5   0.904389           1  0.868449            2  0.690284          3   0.991112              8
Lee cite(lee))                                                  7     0.902318           6  0.868324            3  0.688082          8   0.991008             11
SMOBD cite(smobd))                                              8     0.902247           7  0.86766             4  0.688885          4   0.990583             17
G-SMOTE cite(g_smote))                                         10.75  0.901916           9  0.865103           11  0.686613         11   0.990846             12
CCR cite(ccr))                                                 11.75  0.902112           8  0.861994           23  0.687886          9   0.991254              7
LVQ-SMOTE cite(lvq_smote))                                     12     0.902799           3  0.862295           22  0.683646         20   0.992211              3
Random-SMOTE cite(random_smote))                               14     0.900703          16  0.864615           12  0.685186         15   0.990608             13
CE-SMOTE cite(ce_smote))                                       15.25  0.903073           2  0.862568           19  0.687163         10   0.990165             30
SMOTE-Cosine cite(smote_out_smote_cosine_selected_smote))      15.25  0.900688          17  0.864603           13  0.686279         12   0.990536             19
=========================================================  =========  ========  ==========  ========  ===========  ========  =========  =========  =============
