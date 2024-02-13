import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_text

# Load the CSV file
def load_data(filename):
    data = pd.read_csv(filename)
    return data

# Split data into features and target
def split_data(data):
    X = data.iloc[:, :-1] # Features
    y = data.iloc[:, -1]  # Target
    return X, y

# Encode categorical labels if present for features
def encode_labels(data):
    label_encoders = {}
    for column in data.columns[:-1]:
        if data[column].dtype == 'object':
            label_encoders[column] = LabelEncoder()
            data[column] = label_encoders[column].fit_transform(data[column])
    return data, label_encoders

# Load the dataset
filename = 'data.csv'  # Replace 'data.csv' with your file name
data = load_data(filename)

# Encode labels if present
data, label_encoders = encode_labels(data)

# Split data into features and target
X, y = split_data(data)

#class names
class_names = []

# If y is a string, encode it
map = {}
if y.dtype == 'object':
    output_set = set(y)
    for i, val in enumerate(output_set):
        map[val] = i
        class_names.append(val)
    y = y.map(map)
     


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# Change this variable in function of your problem, bigger the tree is, more you have chance to overfit
MAX_DEPTH = 20
MAX_LEAF = 10

# Initialize and train the decision tree classifier
clf = DecisionTreeClassifier(max_depth=MAX_DEPTH, max_leaf_nodes=MAX_LEAF)
# without constraint on depth and leafs, uncomment next line
# clf = DecisionTreeClassifier(max_depth=MAX_DEPTH, max_leaf_nodes=MAX_LEAF)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)


# print the tree
if len(class_names) == 0:
    class_names = clf.classes_
tree_rules = export_text(clf, feature_names=X.columns, class_names=class_names)
print(tree_rules)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
