import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os

CHECKPOINTS_DIR = "checkpoints/iris"

# Loading the dataset from Scikit Learn
iris = datasets.load_iris()

# Extracting the valuable information from it
labels = iris.target_names
features = iris.feature_names
df = pd.DataFrame(iris.data, index=None, columns=features)

y = iris.target
X = df.copy()
print(X.shape)
# Start of the Exploratory Data Analysis
print("EDA:")
all_data = df
all_data['CLASS'] = y

f, axes = plt.subplots(2, 2, figsize=(9,6))
f.suptitle("Features distribution")
sns.boxplot(ax=axes[0,0],x=[labels[i] for i in y], y=features[0], data=all_data)
sns.boxplot(ax=axes[0,1],x=[labels[i] for i in y], y=features[1], data=all_data)
sns.boxplot(ax=axes[1,0],x=[labels[i] for i in y], y=features[2], data=all_data)
sns.boxplot(ax=axes[1,1],x=[labels[i] for i in y], y=features[3], data=all_data)
#plt.show()

"""
It is easy to check that the features "petal_length" and "petal_width" are the most representative ones and, by using 
just those two, it is possible to classify "setosa" class. On the contrary, "sepal_width" is the least representative 
feature, as the distribution in each of the three classes is very similar.
"""

f, axes = plt.subplots(1, 3)
f.suptitle("Features covariance per class")
for idx, c in enumerate(np.unique(iris.target)):
    row_classes = all_data.loc[all_data['CLASS']==c]
    data = row_classes.drop(columns="CLASS").corr()
    sns.heatmap(ax=axes[idx],data=data, annot=True)
    axes[idx].title.set_text("Class {}".format(labels[c]))
    print("\n\nFor class {}...".format(labels[c]))
    print(data.unstack().sort_values())

#plt.show()
"""
With the different covariance matrices it is possible to assert the feeling from the last Figure: checking the 
sepal is enough for classifying "setosa" class, as the value among its width and height is close to 1 and higher
than other pairs of values.
"""

X_std = StandardScaler().fit_transform(X) # PCA assumes that the data is normally distributed
pca = PCA(n_components=2)
main_components = pca.fit_transform(X_std)
main_comp_df = pd.DataFrame(main_components, columns=['comp1','comp2'])
main_comp_df['CLASS'] = y
colors = ['r', 'g', 'b']
fig, ax = plt.subplots(1,1,figsize = (6,6))
for target, color in zip(np.unique(y),colors):
    indicesToKeep = main_comp_df['CLASS'] == target
    ax.scatter(main_comp_df.loc[indicesToKeep, 'comp1']
               , main_comp_df.loc[indicesToKeep, 'comp2']
               , c = color
               , s = 50,
               label = labels[target])
ax.legend()
ax.grid()
#plt.show()

"""
Again PCA shows that "setosa" is quite different from the other two classes.
"""
print("-"*100)
print("END EDA")
print("-"*100)
y_enc = OneHotEncoder(handle_unknown='ignore').fit_transform(y.reshape(-1,1))
print("X Shape {}".format(X.shape))
print("Y Encoded Shape {}".format(y_enc.shape))

X, y = shuffle(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15)
print("X_train shape: {}".format(X_train.shape))
print("X_val shape: {}".format(X_val.shape))
print("X_test shape: {}".format(X_test.shape))
print("Y_train shape: {}".format(y_train.shape))
print("Y_val shape: {}".format(y_val.shape))
print("Y_test shape: {}".format(y_test.shape))
print('-'*50)
print("X_train samples: {}".format(X_train[:5]))
print("X_test samples: {}".format(X_test[:5]))
print("Y_train samples: {}".format(y_train[:5]))
print("Y_test samples: {}".format(y_test[:5]))


models = []

# KNN Model
print('+'*50)
print('KNN MODEL:')

knn_best_acc = -1
knn_best_hyper = {'n_neighbors':-1, 'weights':None}
knn_best_model = None
for i in range(3,20):
    for j in ['uniform', 'distance']:
        knn = KNeighborsClassifier(n_neighbors=i, weights=j)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        acc = metrics.accuracy_score(y_val, y_pred)
        if acc > knn_best_acc:
            knn_best_hyper['n_neighbors'] = i
            knn_best_hyper['weights'] = j
            knn_best_acc = acc
            knn_best_model = knn

print("KNN Best Hyperparameters: {}".format(knn_best_hyper))
y_pred = knn_best_model.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred)
print("KNN Acc: {}".format(acc))
print("KNN Score: {}".format(knn_best_model.score(X_train, y_train)))
print("KNN Confusion matrix:\n {}".format(metrics.confusion_matrix(y_test, y_pred)))
models.append((knn_best_model,acc))
pickle.dump(knn_best_model, open(os.path.join(CHECKPOINTS_DIR, "iris_knn_best.pkl"), 'wb+'))
print('+'*50)

# RANDOM FOREST MODEL
print("RANDOM FOREST MODEL:")
rf_best_acc = -1
rf_best_hyper = {'n_estimators':-1}
rf_best_model = None

for i in range(20,160,20):
    rf = RandomForestClassifier(n_estimators=i)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    acc = metrics.accuracy_score(y_val, y_pred)
    if acc > rf_best_acc:
        rf_best_hyper['n_estimators'] = i
        rf_best_acc = acc
        rf_best_model = rf

print("RF Best Hyperparameters: {}".format(rf_best_hyper))
y_pred = rf_best_model.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred)
print("RF Acc: {}".format(rf_best_acc))
print("RF Score: {}".format(rf_best_model.score(X_train, y_train)))
print("RF Confusion matrix:\n {}".format(metrics.confusion_matrix(y_test, y_pred)))
models.append((rf_best_model,acc))

pickle.dump(rf_best_model, open(os.path.join(CHECKPOINTS_DIR, "iris_rf_best.pkl"), 'wb+'))

models.sort(key=lambda tup: tup[1])
pickle.dump(models[0][0], open(os.path.join(CHECKPOINTS_DIR, "iris_best.pkl"), 'wb+'))



