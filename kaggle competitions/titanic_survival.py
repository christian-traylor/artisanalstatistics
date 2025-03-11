import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

# 73% accuracy

def transform(df: pd.DataFrame):
    df["Age"] = df["Age"].fillna(df["Age"].median()).ffill()  # Age is positively skewed, populating na with the mean value would be an overestimation of the value

    df["Fare"] = df["Fare"].fillna(df["Fare"].median()).ffill()  # Same reasoning as above

    df["SibSp"] = df["SibSp"].fillna(0).ffill()

    df["Parch"] = df["Parch"].fillna(0).ffill()

    df["Pclass"] = df["Pclass"].astype("category")
    df["Pclass_numeric"] = df["Pclass"].cat.codes

    df["Sex"] = df["Sex"].astype("category")
    df["Sex_numeric"] = df["Sex"].cat.codes # 0 = male, 1 = female

    # df["Has_Family"] = df["SibSp"] > 0 or df['Parch'] > 0
    df["Has_Family"] = df.apply(lambda row: row["SibSp"] > 0 or row['Parch'] > 0, axis=1)
    df["Has_Family_numeric"] = df["Has_Family"].astype("category").cat.codes 

    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()).ffill()
    df["Embarked_numeric"] = df["Embarked"].astype("category").cat.codes

    df["Has_Cabin"] = df["Cabin"].notna()
    df["Has_Cabin_numeric"] = df["Has_Cabin"].astype("category").cat.codes 

    df["Is_Elder"] = df.apply(lambda row: ((row["Age"] > 50 and row["Sex"] == 1) or (row["Age"] > 60 and row["Sex"] == 0) ), axis=1)
    df["Is_Elder_numeric"] = df["Is_Elder"].astype("category").cat.codes 
    
    return df

def extract(train_path="data/titanic_train.csv", test_path="data/titanic_test.csv"):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_df = transform(train_df)
    test_df = transform(test_df)

    
    return train_df, test_df

def build(df: pd.DataFrame):
    continuous_cols = ["Age", "SibSp", "Parch", "Pclass_numeric", "Fare", "Sex_numeric", "Has_Family_numeric", "Embarked_numeric", "Has_Cabin_numeric", "Is_Elder_numeric"]
    discrete_cols = ["Pclass_numeric", "Sex_numeric", "Has_Family_numeric", "Embarked_numeric", "Has_Cabin_numeric", "Is_Elder_numeric"]
    
    continuous_data = df[continuous_cols].to_numpy()
    discrete_data = df[discrete_cols].to_numpy()
    
    return continuous_data, discrete_data

def train_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_cont, train_disc = build(train_df)
    test_cont, test_disc = build(test_df)
    
    y_train = train_df["Survived"].to_numpy()
    
    gaussian_clf = GaussianNB()
    gaussian_clf.fit(train_cont, y_train)
    # multinomial_clf = MultinomialNB()
    # multinomial_clf.fit(train_disc, y_train)
    
    log_prob_g = gaussian_clf.predict_log_proba(test_cont) 
    # log_prob_m = multinomial_clf.predict_log_proba(test_disc) 
    combined_log_prob = log_prob_g # + log_prob_m
    y_pred = np.argmax(combined_log_prob, axis=1)
    
    return y_pred

def visualize(dataset, enabled):
    if enabled:
        plt.figure(figsize=(8, 5))
        sns.histplot(dataset["Fare"], bins=30, kde=True)
        skewness = skew(dataset["Fare"])
        distribution_skewness = "negative" if skewness < 0 else ("uniform" if skewness == 0 else "positive")
        plt.title(f"Distribution skewness: {distribution_skewness}, {skewness:.2f}")
        plt.xlabel("Age")
        plt.ylabel("Frequency")
        plt.show()

def main():
    train_df, test_df = extract()
    visualize(train_df, enabled=False)
    predictions = train_and_predict(train_df, test_df)

    passenger_ids = [pid for pid in range(892, 1310)]
    output = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": predictions
    })

    output.to_csv("output.csv", index=False)

if __name__ == "__main__":
    # notes to self:
    # we could have used only gaussian Naive Bayes
    # correlation matrix?
    # why does combining gaussian and multinomial NB lead to worse results?
    # how do I actually use MNB?
    main()