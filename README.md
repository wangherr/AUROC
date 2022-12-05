## Quickly Start:
```
def compute_auroc(ture_scores, false_scores):
    scores = np.concatenate([np.array(ture_scores), np.array(false_scores)])
    labels = np.concatenate([np.ones_like(ture_scores), np.zeros_like(false_scores)])
    return roc_auc_score(labels, scores)
```


## Input Format

Two list:

* one contains scores of all true samples, 

* one contains scores of all false samples.

For example:
```
normal_scores = [0.8, 0.9, 0.6, 0.4]
abnormal_scores = [0.3, 0.6, 0.5, 0.7]
```

## Compute
```
def compute_auroc(ture_scores, false_scores):
    scores = np.concatenate([np.array(ture_scores), np.array(false_scores)])
    labels = np.concatenate([np.ones_like(ture_scores), np.zeros_like(false_scores)])
    return roc_auc_score(labels, scores)

result = compute(normal_scores, abnormal_scores)
>>> 0.71875
```
