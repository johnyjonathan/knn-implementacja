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

def showBarClassesGraph(column_):
    class_count = column_.value_counts()
    plt.bar(class_count.index, class_count.values)
    plt.xlabel('Klasy')
    plt.ylabel('Liczba próbek')
    plt.title('Ilość poszczególnych klas w zbiorze danych "Car Evaluation Data Set"')
    plt.show()

def showModelGraph(model_graph, X, y):
    neighbors_array = model_graph.toarray()
    X_np = np.array(X)
    unique_labels = np.unique(y)
    for label in unique_labels:
        plt.scatter(X_np[y == label, 0], X_np[y == label, 1], label=f'Klasa {label}')

    # Rysowanie linii łączących sąsiednie punkty
    for i in range(neighbors_array.shape[0]):
        for j in range(neighbors_array.shape[1]):
            if neighbors_array[i, j] == 1:
                plt.plot([X_np[i, 0], X_np[j, 0]], [X_np[i, 1], X_np[j, 1]], 'r--', alpha=0.5)

    plt.title('Wykres sąsiedztwa dla k-NN')
    plt.legend()
    plt.show()