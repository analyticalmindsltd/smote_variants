Ranking
*******

Based on a thorough evaluation using 104 imbalanced datasets, the following 10 techniques provide the highest performance in terms of the AUC, GAcc, F1 and P20 scores, in nearest neighbors, support vector machine, decision tree and multilayer perceptron based classification scenarios.
For more details on the evaluation methodology, see our paper on the comparative study.

=================  =========  ========  ==========  ========  ===========  ========  =========  =========  =============
sampler              overall       auc    rank_auc      gacc    rank_gacc        f1    rank_f1    p_top20    rank_ptop20
=================  =========  ========  ==========  ========  ===========  ========  =========  =========  =============
polynom-fit-SMOTE       2.5   0.902538           6  0.870753            1  0.695154          1   0.992496              2
ProWSyn                 4.5   0.904389           1  0.868449            4  0.690284          3   0.991112             10
SMOTE-IPF               7.5   0.902565           5  0.868715            3  0.687935          9   0.990904             13
Lee                     8     0.902318           7  0.868324            5  0.688082          8   0.991008             12
SMOBD                   9.25  0.902247           8  0.86766             6  0.688885          4   0.990583             19
G-SMOTE                13.5   0.901916          10  0.865103           18  0.686613         12   0.990846             14
CCR                    14.25  0.902112           9  0.861994           30  0.687886         10   0.991254              8
LVQ-SMOTE              14.75  0.902799           3  0.862295           29  0.683646         24   0.992211              3
Assembled-SMOTE        15.5   0.902691           4  0.866914            7  0.688614          5   0.982685             46
SMOTE-TomekLinks       15.75  0.901016          14  0.866174            9  0.684708         20   0.990573             20
=================  =========  ========  ==========  ========  ===========  ========  =========  =========  =============

