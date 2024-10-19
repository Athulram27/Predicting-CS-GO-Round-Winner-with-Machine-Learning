#!/usr/bin/env python
# coding: utf-8

# import requests
# 
# url = "https://www.openml.org/data/download/22102255/dataset"
# r = requests.get(url, allow_redirects = True)

# with open("dataset.txt", "wb") as f:
#     f.write(r.content)

# import pandas as pd

# data = []
# 
# with open("dataset.txt",'r') as f:
#     for line in f.read().split("\n"):
#         if line.startswith("@") or line.startswith("%") or line =="":
#             continue
#         data.append(line)

# columns =[]
# 
# with open("dataset.txt","r") as f:
#     for line in f.read().split("\n"):
#         if line.startswith("@ATTRIBUTE"):
#             columns.append(line.split(" ")[1])
#             

# with open("df.csv","w") as f:
#     f.write(",".join(columns))
#     f.write("\n")
#     f.write("\n".join(data))

# df = pd.read_csv("df.csv")
# df.columns = columns

# df

# df['t_win'] = df.round_winner.astype("category").cat.codes

# import matplotlib.pyplot as plt

# import seaborn as sns

# correlations = df[columns+["t_win"]].corr()
# print(correlations['t_win'].apply(abs).sort_values(ascending = False).iloc[:25])

# selected_columns = []
# 
# for col in columns+["t_win"]:
#     try:
#         if abs(correlations[col]['t_win'])>0.15:
#             selected_columns.append(col)
#     except KeyError:
#             pass
# df_selected = df[selected_columns]

# df_selected

# plt.figure(figsize =(18,12))
# sns.heatmap(df_selected.corr().sort_values(by="t_win"), annot =True, cmap = "YlGnBu")

# df_selected.hist(figsize = (18,12))

# df_selected.info()

# from sklearn.model_selection import train_test_split
# 
# X,y = df_selected.drop(["t_win"],axis =1), df_selected["t_win"]
# 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
# 
# scaler = StandardScaler()
# 
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# 
# knn = KNeighborsClassifier(n_neighbors=5) 
# knn.fit(X_train_scaled, y_train) 

# knn.score(X_test_scaled, y_test)

# from sklearn.model_selection import RandomizedSearchCV
# 
# param_grid = {
#     "n_neighbors": list(range(5,17,2)),
#     "weights":["unifrom","distance"]
# }
# 
# knn = KNeighborsClassifier(n_jobs = 4)
# 
# clf = RandomizedSearchCV(knn, param_grid, n_jobs= 4, n_iter = 3, verbose = 2, cv = 3)
# clf.fit(X_train_scaled, y_train)

# knn = clf.best_estimator_

# knn.score(X_test_scaled, y_test)

# from sklearn.ensemble import RandomForestClassifier
# 
# forest = RandomForestClassifier(n_jobs = 4)
# forest.fit(X_train_scaled, y_train)

# forest.score(X_test_scaled, y_test)

# from tensorflow import keras
# 
# model = keras.models.Sequential()
# model.add(keras.layers.Input(shape =(20,)))
# model.add(keras.layers.Dense(200, activation= "relu"))
# model.add(keras.layers.Dense(100, activation= "relu"))
# model.add(keras.layers.Dense(100, activation= "relu"))
# model.add(keras.layers.Dense(1, activation= "sigmoid"))
# 
# 

# model.compile(loss="binary_crossentropy", optimizer = "adam", metrics=["accuracy"])

# early_stopping_cb = keras.callbacks.EarlyStopping(patience = 5)
# 
# X_train_scaled_train, X_valid, y_train_train, y_valid = train_test_split(X_train_scaled, y_train, test_size = 0.15)
# 
# model.fit(X_train_scaled_train,y_train_train, epochs = 30, callbacks = [early_stopping_cb], validation_data = (X_valid, y_valid) )

# !pip install tensorflow

# In[94]:


model.evaluate(X_test_scaled, y_test)


# 
