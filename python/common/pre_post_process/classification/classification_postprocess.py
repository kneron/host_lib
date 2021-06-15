import numpy as np

def softmax(A):
    e = np.exp(A)
    return e / np.sum(e, keepdims=True)

def postprocess(pre_output):
    score = softmax(pre_output)
    labels = list(range(len(pre_output)))
    score_labels = list(zip(score, labels))
    score_labels.sort(reverse=True)
    score, labels = list(zip(*score_labels))
    
    return score, labels
