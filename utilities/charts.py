import matplotlib.pyplot as plt

def showClassesGraph(predicted, actual, names_map):
    names_list = list(names_map.values())
    plt.figure(figsize=(10, 5))
    plt.hist([predicted, actual])
    plt.legend(['Predicted', 'Actual'])
    plt.show()
