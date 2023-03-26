import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def showClassesGraph(predicted, actual, names_map):
    names_list = list(names_map.values())
    plt.figure(figsize=(10, 5))
    plt.hist([predicted, actual])
    plt.xticks(range(len(names_list)), names_list)
    plt.legend(['Predicted', 'Actual'])
    plt.show()

def showConfusionMatrixGraph(predicted, actual):
    print(len(predicted.tolist()))
    
    cm = confusion_matrix(actual, predicted.tolist())
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()