import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB

def fill_missing_and_encode(df: pd.DataFrame):
    df["Age"] = df["Age"].fillna(df["Age"].mean()).ffill()
    df["Fare"] = df["Fare"].fillna(df["Fare"].mean()).ffill()
    
    df["Pclass"] = df["Pclass"].astype("category")
    df["Pclass_numeric"] = df["Pclass"].cat.codes
    
    df["Sex"] = df["Sex"].astype("category")
    df["Sex_numeric"] = df["Sex"].cat.codes
    
    return df

def load_and_prepare_data(train_path="data/titanic_train.csv", test_path="data/titanic_test.csv"):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_df = fill_missing_and_encode(train_df)
    test_df = fill_missing_and_encode(test_df)
    
    return train_df, test_df

def build_features(df: pd.DataFrame):
    continuous_cols = ["Age", "SibSp", "Parch", "Fare"]
    discrete_cols = ["Pclass_numeric", "Sex_numeric"]
    
    continuous_data = df[continuous_cols].to_numpy()
    discrete_data = df[discrete_cols].to_numpy()
    
    return continuous_data, discrete_data

def train_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_cont, train_disc = build_features(train_df)
    test_cont, test_disc = build_features(test_df)
    
    y_train = train_df["Survived"].to_numpy()
    
    gaussian_clf = GaussianNB()
    gaussian_clf.fit(train_cont, y_train)
    multinomial_clf = MultinomialNB()
    multinomial_clf.fit(train_disc, y_train)
    
    log_prob_g = gaussian_clf.predict_log_proba(test_cont) 
    log_prob_m = multinomial_clf.predict_log_proba(test_disc) 
    combined_log_prob = log_prob_g + log_prob_m
    y_pred = np.argmax(combined_log_prob, axis=1)
    
    return y_pred

def main():
    train_df, test_df = load_and_prepare_data()
    predictions = train_and_predict(train_df, test_df)

    passenger_ids = [pid for pid in range(892, 1310)]
    output = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": predictions
    })

    output.to_csv("output.csv", index=False)

if __name__ == "__main__":
    main()