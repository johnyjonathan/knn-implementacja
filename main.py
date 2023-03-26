# imports
from utilities import functions, preprocessing, charts
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Setttings
N_NEAREST_NEIGHBORS = 9



# Read data
file_path = 'data/car.data'
data = functions.readData(file_path)

# Split data, x - attributes, y - class
x, y = preprocessing.getXandY(data)

# Train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# Set up model
model = KNeighborsClassifier(n_neighbors=N_NEAREST_NEIGHBORS)

# Train model
model.fit(x_train, y_train)

# Test model / accuracy
accuracy = model.score(x_test, y_test)

print('Accuracy: ', accuracy)

# Predict
predictions = model.predict(x_test)

# Get class and attrs names
values_map = preprocessing.preprocessData(data)[2]
attrs_values_map = preprocessing.preprocessData(data)[3]
print(attrs_values_map)

# Print predictions
for idx, val in enumerate(predictions):
    attr = preprocessing.getAttributes(data)
    names = functions.AttrValueNames(attr, attrs_values_map, x_test[idx])
    print('Predicted: ', values_map[val], 'Data: ',  names, 'Actual: ', values_map[y_test[idx]])

charts.showClassesGraph(predictions, y_test, values_map)