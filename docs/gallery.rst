Gallery
********

In this page, we demonstrate the output of various oversampling                     and noise removal techniques, using default parameters.

For binary oversampling and nosie removal, an artificial database was used, available in the ``utils` directory of the github repository.

For multiclass oversampling we have used the 'wine' dataset from                     ``sklearn.datasets``, which has 3 classes and many features, out                     which the first two coordinates have been used for visualization.

Oversampling sample results
============================

In the captions of the images some abbreviations                     referring to the operating principles are placed. Namely:

    * NR: noise removal is involved
    * DR: dimension reduction is applied
    * Clas: some supervised classifier is used
    * SCmp: sampling is carried out componentwise (attributewise)
    * SCpy: sampling is carried out by copying instances
    * SO: ordinary sampling (just like in SMOTE)
    * M: memetic optimization is used
    * DE: density estimation is used
    * DB: density based - the sampling is based on a density of importance assigned to the instances
    * Ex: the sampling is extensive - samples are added successively, not optimizing the holistic distribution of a given number of samples
    * CM: changes majority - even majority samples can change
    * Clus: uses some clustering technique
    * BL: identifies and samples the neighborhoods of borderline samples
    * A: developed for a specific application

.. figure:: figures/base.png


.. image:: figures/SMOTE.png
.. image:: figures/SMOTE_TomekLinks.png
.. image:: figures/SMOTE_ENN.png
.. image:: figures/Borderline_SMOTE1.png

.. image:: figures/Borderline_SMOTE2.png
.. image:: figures/ADASYN.png
.. image:: figures/AHC.png
.. image:: figures/LLE_SMOTE.png

.. image:: figures/distance_SMOTE.png
.. image:: figures/SMMO.png
.. image:: figures/polynom_fit_SMOTE.png
.. image:: figures/Stefanowski.png

.. image:: figures/ADOMS.png
.. image:: figures/Safe_Level_SMOTE.png
.. image:: figures/MSMOTE.png
.. image:: figures/DE_oversampling.png

.. image:: figures/SMOBD.png
.. image:: figures/SUNDO.png
.. image:: figures/MSYN.png
.. image:: figures/SVM_balance.png

.. image:: figures/TRIM_SMOTE.png
.. image:: figures/SMOTE_RSB.png
.. image:: figures/ProWSyn.png
.. image:: figures/SL_graph_SMOTE.png

.. image:: figures/NRSBoundary_SMOTE.png
.. image:: figures/LVQ_SMOTE.png
.. image:: figures/SOI_CJ.png
.. image:: figures/ROSE.png

.. image:: figures/SMOTE_OUT.png
.. image:: figures/SMOTE_Cosine.png
.. image:: figures/Selected_SMOTE.png
.. image:: figures/LN_SMOTE.png

.. image:: figures/MWMOTE.png
.. image:: figures/PDFOS.png
.. image:: figures/IPADE_ID.png
.. image:: figures/RWO_sampling.png

.. image:: figures/NEATER.png
.. image:: figures/DEAGO.png
.. image:: figures/Gazzah.png
.. image:: figures/MCT.png

.. image:: figures/ADG.png
.. image:: figures/SMOTE_IPF.png
.. image:: figures/KernelADASYN.png
.. image:: figures/MOT2LD.png

.. image:: figures/V_SYNTH.png
.. image:: figures/OUPS.png
.. image:: figures/SMOTE_D.png
.. image:: figures/SMOTE_PSO.png

.. image:: figures/CURE_SMOTE.png
.. image:: figures/SOMO.png
.. image:: figures/ISOMAP_Hybrid.png
.. image:: figures/CE_SMOTE.png

.. image:: figures/Edge_Det_SMOTE.png
.. image:: figures/CBSO.png
.. image:: figures/E_SMOTE.png
.. image:: figures/DBSMOTE.png

.. image:: figures/ASMOBD.png
.. image:: figures/Assembled_SMOTE.png
.. image:: figures/SDSMOTE.png
.. image:: figures/DSMOTE.png

.. image:: figures/G_SMOTE.png
.. image:: figures/NT_SMOTE.png
.. image:: figures/Lee.png
.. image:: figures/SPY.png

.. image:: figures/SMOTE_PSOBAT.png
.. image:: figures/MDO.png
.. image:: figures/Random_SMOTE.png
.. image:: figures/ISMOTE.png

.. image:: figures/VIS_RST.png
.. image:: figures/GASMOTE.png
.. image:: figures/A_SUWO.png
.. image:: figures/SMOTE_FRST_2T.png

.. image:: figures/AND_SMOTE.png
.. image:: figures/NRAS.png
.. image:: figures/AMSCO.png
.. image:: figures/SSO.png

.. image:: figures/NDO_sampling.png
.. image:: figures/DSRBF.png
.. image:: figures/Gaussian_SMOTE.png
.. image:: figures/kmeans_SMOTE.png

.. image:: figures/Supervised_SMOTE.png
.. image:: figures/SN_SMOTE.png
.. image:: figures/CCR.png
.. image:: figures/ANS.png

.. image:: figures/cluster_SMOTE.png
.. image:: figures/NoSMOTE.png
Noise removal sample results
=============================

.. figure:: figures/base.png


.. image:: figures/TomekLinkRemoval.png
.. image:: figures/CondensedNearestNeighbors.png
.. image:: figures/OneSidedSelection.png
.. image:: figures/CNNTomekLinks.png

.. image:: figures/NeighborhoodCleaningRule.png
.. image:: figures/EditedNearestNeighbors.png
Multiclass sample results
==========================

.. figure:: figures/multiclass-base.png


.. image:: figures/multiclass-SMOTE.png
.. image:: figures/multiclass-Borderline_SMOTE1.png
.. image:: figures/multiclass-Borderline_SMOTE2.png
.. image:: figures/multiclass-LLE_SMOTE.png

.. image:: figures/multiclass-distance_SMOTE.png
.. image:: figures/multiclass-SMMO.png
.. image:: figures/multiclass-polynom_fit_SMOTE.png
.. image:: figures/multiclass-ADOMS.png

.. image:: figures/multiclass-Safe_Level_SMOTE.png
.. image:: figures/multiclass-MSMOTE.png
.. image:: figures/multiclass-SMOBD.png
.. image:: figures/multiclass-TRIM_SMOTE.png

.. image:: figures/multiclass-SMOTE_RSB.png
.. image:: figures/multiclass-ProWSyn.png
.. image:: figures/multiclass-SL_graph_SMOTE.png
.. image:: figures/multiclass-NRSBoundary_SMOTE.png

.. image:: figures/multiclass-LVQ_SMOTE.png
.. image:: figures/multiclass-SOI_CJ.png
.. image:: figures/multiclass-ROSE.png
.. image:: figures/multiclass-SMOTE_OUT.png

.. image:: figures/multiclass-SMOTE_Cosine.png
.. image:: figures/multiclass-Selected_SMOTE.png
.. image:: figures/multiclass-LN_SMOTE.png
.. image:: figures/multiclass-MWMOTE.png

.. image:: figures/multiclass-PDFOS.png
.. image:: figures/multiclass-RWO_sampling.png
.. image:: figures/multiclass-DEAGO.png
.. image:: figures/multiclass-MCT.png

.. image:: figures/multiclass-ADG.png
.. image:: figures/multiclass-KernelADASYN.png
.. image:: figures/multiclass-MOT2LD.png
.. image:: figures/multiclass-V_SYNTH.png

.. image:: figures/multiclass-OUPS.png
.. image:: figures/multiclass-SMOTE_D.png
.. image:: figures/multiclass-CURE_SMOTE.png
.. image:: figures/multiclass-SOMO.png

.. image:: figures/multiclass-CE_SMOTE.png
.. image:: figures/multiclass-Edge_Det_SMOTE.png
.. image:: figures/multiclass-CBSO.png
.. image:: figures/multiclass-DBSMOTE.png

.. image:: figures/multiclass-ASMOBD.png
.. image:: figures/multiclass-Assembled_SMOTE.png
.. image:: figures/multiclass-SDSMOTE.png
.. image:: figures/multiclass-G_SMOTE.png

.. image:: figures/multiclass-NT_SMOTE.png
.. image:: figures/multiclass-Lee.png
.. image:: figures/multiclass-MDO.png
.. image:: figures/multiclass-Random_SMOTE.png

.. image:: figures/multiclass-A_SUWO.png
.. image:: figures/multiclass-AND_SMOTE.png
.. image:: figures/multiclass-NRAS.png
.. image:: figures/multiclass-SSO.png

.. image:: figures/multiclass-NDO_sampling.png
.. image:: figures/multiclass-DSRBF.png
.. image:: figures/multiclass-Gaussian_SMOTE.png
.. image:: figures/multiclass-kmeans_SMOTE.png

.. image:: figures/multiclass-Supervised_SMOTE.png
.. image:: figures/multiclass-SN_SMOTE.png
.. image:: figures/multiclass-CCR.png
.. image:: figures/multiclass-ANS.png

.. image:: figures/multiclass-cluster_SMOTE.png
