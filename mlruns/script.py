import warnings
from statistics import mean


import streamlit as st
import plotly.express as px
from mlflow.tracking.fluent import log_param
from numpy import std

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler


import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score

import mlflow
import mlflow.sklearn

import logging

st.title("A/B Testing")

choice = st.sidebar.selectbox("Select Analysis", (
    "A/B Testing", "A/B Testing with Machine learning "))


@st.cache
def loadData():
    warnings.filterwarnings('ignore')

    v_browser = pd.read_csv('../data/v_browser.csv')
    v_platform = pd.read_csv('../data/v_platform.csv')
    return v_browser, v_platform


@st.cache
def encode_labels_browser(df):
    date_encoder = preprocessing.LabelEncoder()
    device_encoder = preprocessing.LabelEncoder()
    browser_encoder = preprocessing.LabelEncoder()
    experiment_encoder = preprocessing.LabelEncoder()
    target_encoder = preprocessing.LabelEncoder()

    df['date'] = date_encoder.fit_transform(df['date'])
    df['device_make'] = device_encoder.fit_transform(df['device_make'])
    df['browser'] = browser_encoder.fit_transform(df['browser'])
    df['experiment'] = experiment_encoder.fit_transform(df['experiment'])
    df['target'] = target_encoder.fit_transform(df['target'])

    return df, date_encoder, device_encoder, browser_encoder, experiment_encoder, target_encoder


@st.cache
def train_test_val_split(X, Y, split=(0.2, 0.1), shuffle=True):
    """Split dataset into train/val/test subsets by 70:20:10(default).

    Args:
      X: List of data.
      Y: List of labels corresponding to data.
      split: Tuple of split ratio in `test:val` order.
      shuffle: Bool of shuffle or not.

    Returns:
      Three dataset in `train:test:val` order.
    """
    from sklearn.model_selection import train_test_split
    assert len(X) == len(Y), 'The length of X and Y must be consistent.'
    X_train, X_test_val, y_train, Y_test_val = train_test_split(X, Y,
                                                                test_size=(split[0] + split[1]), shuffle=shuffle)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, Y_test_val,
                                                    test_size=split[1], shuffle=False)
    return (X_train, y_train), (X_test, y_test), (X_val, y_val)




def test(self, x_test, y_test):

    y_pred = self.clf.predict(x_test)
    accuracy = self.calculate_score(y_test, y_pred)
    self.__printAccuracy(accuracy, label="Test")
    report = self.report(y_pred, y_test)
    matrix = self.confusion_matrix(y_pred, y_test)
    return accuracy, report, matrix

@st.cache
def boxPlot(df, column):
    bx = df.boxplot(column=column, return_type='axes');
    return bx


@st.cache
def checkSkew(df):
    skewValue = df.skew(axis=1)
    return skewValue


@st.cache
def scatterPlot(df, colum1, column2):
    fig = px.scatter(
        x=df[colum1],
        y=df[column2],
    )
    fig.update_layout(
        xaxis_title=colum1,
        yaxis_title=column2,
    )
    return fig


def scatterPlot(colum1, column2):
    fig = px.scatter(
        x=colum1,
        y=column2,
    )
    fig.update_layout(
        xaxis_title=colum1,
        yaxis_title=column2,
    )
    return fig





def main():
    with mlflow.start_run():
        mlflow.set_experiment(experiment_name='ML')
        if choice == "A/B Testing":
            st.subheader("A/B Testing")
            sel = st.selectbox("Select choice", (
                "Classical A/B Testing", "Sequential A/B testing"))
            if sel == "Classical A/B Testing":
                st.subheader("Classical A/B Testing")

            elif sel == "Sequential A/B testing":
                st.subheader("Sequential A/B testing")

        elif choice == "A/B Testing with Machine learning ":
            st.subheader("A/B Testing with Machine learning ")
            mlchoice = st.selectbox("Select choice", (
                "Data Versions", "target and features","Models"))
            if mlchoice == "Data Versions":
                v_browser, v_platform = loadData()
                rd = st.radio("Select feature", ('v_browser', 'v_platform'))
                if rd == "v_browser":
                    st.write(v_browser)
                elif rd == "v_platform":
                    st.write(v_platform)

            elif mlchoice == "target and features":
                v_browser, v_platform = loadData()
                encoded_df, date_encoder, device_encoder, browser_encoder, experiment_encoder, target_encoder = encode_labels_browser(
                    v_browser)
                feature_col = ["experiment", "hour", "date", "device_make", "browser"]
                features_X = encoded_df[feature_col]
                target_y = encoded_df["target"]

                rd2 = st.radio("Select feature", ('target', 'features'))
                if rd2 == "target":
                    st.write(target_y)
                elif rd2 == "features":
                    st.write(features_X)
            elif mlchoice == "Models":
                rd3 = st.radio("Select Model", ('Logistic', 'Decission Tree'))
                if rd3 == "Logistic":
                    v_browser, v_platform = loadData()
                    encoded_df, date_encoder, device_encoder, browser_encoder, experiment_encoder, target_encoder = encode_labels_browser(
                        v_browser)
                    feature_col = ["experiment", "hour", "date", "device_make", "browser"]
                    features_X = encoded_df[feature_col]
                    target_y = encoded_df["target"]
                    (X_train, y_train), (X_test, y_test), (X_val, y_val) = train_test_val_split(features_X, target_y)
                    cv = KFold(n_splits=5, random_state=1, shuffle=True)
                    # create model
                    model = LogisticRegression()

                    #Train model

                    model.fit(X_train, y_train)
                    threshold = 0.5
                    y_pred=model.predict(X_test)
                    mlflow.log_param("Predictions", y_pred)


                    # evaluate model using cross validation
                    scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
                    st.warning(scores)
                    st.write("""The average accuracy is""")

                    st.write((mean(scores), std(scores)))
                    log_param("Accuracy", mean(scores))

                    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
                    st.write(cnf_matrix)
                    log_param("Confusion matrics", cnf_matrix)

                    #Visualize confusion matrix

                    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
                    plt.title('Confusion matrix')
                    plt.ylabel('Actual')
                    plt.xlabel('Predicted')
                    st.plotly_chart(plt)




if __name__ == '__main__':

    main()
