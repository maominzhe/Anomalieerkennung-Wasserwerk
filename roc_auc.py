from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_rocauc(y_test,y_pred):
    fpr,tpr,threshold = roc_curve(y_test, y_pred, drop_intermediate=False) #fpr tpr berechnen
    roc_auc = auc(fpr,tpr) #auc berechnen
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) #set fpr as x-axis and tpr as y-axis
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    return roc_auc