import pandas as pd
import numpy as np
import pdb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBRegressor

# df = pd.read_csv("imdb-tv-ratings/top-250-movie-ratings.csv")
# df['Label'] = np.nan
# df.to_csv("imdb-tv-ratings/top-250-movie-ratings-label.csv")
use_val_set = True
model_option = ["XGB", "NN", "SVR", "RF", "RULE_BASED", "LR", "PR"][0]
df = pd.read_csv("data/imdb-tv-ratings/top-250-movie-ratings-label.csv")

X_train, y_train, X_val, y_val, X_test = [], [], [], [], []
cols = ["Title", "Year", "Rating", "Rating Count"]
cv = CountVectorizer()

for i in range(len(df)):
    print(df.at[i, "Label"], df.at[i, "Label"] == np.nan)

    datapoint = [df.at[i, col] for col in cols]
    datapoint[1] = float(datapoint[1])
    datapoint[2] = float(datapoint[2])
    no_comma_number = datapoint[3].replace(",", "")
    datapoint[3] = float(no_comma_number)

    if pd.isna(df.at[i, "Label"]):
        X_test.append(datapoint)
    else:
        X_train.append(datapoint)
        y_train.append(df.at[i, "Label"])


train_texts = [elem[0].lower() for elem in X_train]
test_texts = [elem[0].lower() for elem in X_test]

print(len(train_texts), len(test_texts))
train_texts = cv.fit_transform(train_texts)
test_texts = cv.transform(test_texts)

X_train = [X[1:] for X in X_train]
X_test = [X[1:] for X in X_test]

scaler = StandardScaler()

print(len(X_test))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

label_scaler = StandardScaler()

y_train = [y_train]
y_train = label_scaler.fit_transform(y_train)[0]

classifier = XGBRegressor()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_test = np.zeros(y_pred.shape)
for i in range(0, len(y_pred) - len(y_pred) % len(y_train), len(y_train)):
    y_test[i: min(len(y_pred), i + len(y_train))] = label_scaler.inverse_transform(y_pred[i: min(len(y_pred), i + len(y_train))])

