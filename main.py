from typing import List, Union

import numpy
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auroc(ture_scores: Union[List[float], numpy.array], false_scores: Union[List[float], numpy.array]) -> float:
    scores = np.concatenate([np.array(ture_scores), np.array(false_scores)])
    labels = np.concatenate([np.ones_like(ture_scores), np.zeros_like(false_scores)])
    return roc_auc_score(labels, scores)


if __name__ == '__main__':
    normal_scores = [0.8, 0.9, 0.6, 0.4]
    abnormal_scores = [0.3, 0.6, 0.5, 0.7]
    auroc_score = compute_auroc(normal_scores, abnormal_scores)
    print(auroc_score)
