# imports
from utilities import functions, preprocessing, charts
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy
# Setttings
N_NEAREST_NEIGHBORS = 9



# Read data
file_path = 'data/car.data'
data = functions.readData(file_path)

# Split data, x - attributes, y - classes
x, y = preprocessing.getXandY(data)

# Train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# Set up model
model = KNeighborsClassifier(n_neighbors=N_NEAREST_NEIGHBORS, metric='jaccard')

# Train model
model.fit(x_train, y_train)

# Get graph of model
model_graph = model.kneighbors_graph(x_train, mode='distance', n_neighbors=N_NEAREST_NEIGHBORS)

# Test model / accuracy
accuracy = model.score(x_test, y_test)
print('Accuracy: ', accuracy)

# Predict
predictions = model.predict(x_test)
accuracy2 = accuracy_score(y_test, predictions)
print('Accuracy2: ', accuracy2)
report = classification_report(y_test, predictions)
print('Report: ', report)
# Get class and attrs names
values_map = preprocessing.preprocessData(data)[2]
attrs_values_map = preprocessing.preprocessData(data)[3]

# Print predictions
for idx, val in enumerate(predictions):
    attr = preprocessing.getAttributes(data)
    names = functions.AttrValueNames(attr, attrs_values_map, x_test[idx])
    print('Predicted: ', values_map[val], 'Data: ',  names, 'Actual: ', values_map[y_test[idx]])


# Show graphs
charts.showModelGraph(model_graph, x_train, y_train)
charts.showClassesGraph(predictions, y_test, values_map)
charts.showConfusionMatrixGraph(predictions, y_test)
charts.showBarClassesGraph(data['class'])