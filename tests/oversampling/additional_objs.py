"""
This module lists some additional objects for testing (with special
parameterizations).
"""

import smote_variants as sv

additional_objs = [sv.Stefanowski(strategy='weak_amp', random_state=5),
                    sv.Stefanowski(strategy='weak_amp_relabel', random_state=5),
                    sv.Stefanowski(strategy='strong_amp', random_state=5),
                    sv.G_SMOTE(method='non-linear_2.0', random_state=5),
                    sv.SMOTE_PSOBAT(method='pso', random_state=5),
                    sv.AHC(strategy='maj', random_state=5),
                    sv.AHC(strategy='minmaj', random_state=5),
                    sv.SOI_CJ(method='jittering', random_state=5),
                    sv.ADG(kernel='rbf_1', random_state=5),
                    sv.SMOTE_IPF(voting='consensus', random_state=5),
                    sv.ASMOBD(smoothing='sigmoid', random_state=5)]
