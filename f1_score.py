from sklearn.metrics import f1_score

def f1_score(y_whole, y_pred, average = None):
    score = f1_score(y_whole, y_pred, average)
    return score