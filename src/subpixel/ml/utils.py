import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import json


# Make function to find if a column has classification type values that are not numeric, if True get_dummies. If False, do nothing.


def accuracy(y_true, y_pred):
    """
    Function that finds the accuracy of a model based on the true and predicted values.
    :param y_true: True values
    :param y_pred: Predicted values

    :return: Accuracy of the model
    """

    return sklearn.metrics.accuracy_score(y_true, y_pred)


def correlation_matrix(df, cols=False):
    """
    Gets the correlation matrix of the dataframe.
    :param df: Dataframe

    :return: Correlation matrix
    """

    if cols:
        df = df[cols]

    numeric_cols = df._get_numeric_data().columns.tolist()

    return df[numeric_cols].corr()


def find_outliers(df, cols=False):
    """
    Finds outliers in each column of the dataframe.
    :param df: Dataframe
    :param cols: Columns to check for outliers
    :param remove: If True, removes outliers from the dataframe

    :return: list of outliers and dataframe without outliers if remove is True.
    """

    if cols:
        numeric_cols = df[cols]._get_numeric_data().columns.tolist()
    else:
        numeric_cols = df._get_numeric_data().columns.tolist()

    outlier_idx = []

    for col in numeric_cols:

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        low_bound = q1 - (iqr * 1.5)
        high_bound = q3 + (iqr * 1.5)

        for i, val in enumerate(df[col]):
            if val < low_bound or val > high_bound:
                outlier_idx.append(i)

    outlier_idx = list(set(outlier_idx))
    df = df.drop(index=outlier_idx)

    return outlier_idx, df


def boxplot(df, cols=False):
    """
    Shows a boxplot of the dataframe.
    :param df: Dataframe

    :return: None
    """

    if cols:
        df = df[cols]
        numeric_cols = df._get_numeric_data().columns.tolist()
    else:
        numeric_cols = df._get_numeric_data().columns.tolist()

    i = 1
    plt.figure(figsize=(15, 25))
    for col in numeric_cols:
        plt.subplot(6, 3, i)
        sns.boxplot(y=df[col], color="green")
        i += 1

    plt.show()


def get_combinations(list_of_values):
    """
    Gets the combinations of the list of values.
    :param list_of_values: List of values

    :return: List of combinations
    """
    return list(itertools.combinations(list_of_values, 2))


# NEEDS TO BE CHANGED TO DISPLAY IN SAME PAGE
def feature_correlation(df, cols=False, kind="reg"):
    """
    Gets the correlation matrix of the dataframe.
    :param df: Dataframe
    :param cols: Columns to check for outliers
    :param kind: Type of plot to show

    :return: None
    """

    if cols:
        lst = get_combinations(cols)

    else:
        lst = get_combinations(df.columns)

    for i, j in lst:

        sns.jointplot(x=i, y=j, data=df, kind=kind, truncate=False, color="m", height=7)

    plt.show()


def fill_nan_with_mean(df):
    """
    Fills the NaN values with the mean of the column.
    :param df: Dataframe

    :return: Dataframe with NaN values filled with mean.
    """

    for col in df.columns:
        df[col] = df[col].fillna(get_mean(df, col))
    return df


def delete_row_with_nan(df):
    """
    Delete rows with NaN values.
    :param df: Dataframe

    :return: Dataframe without rows with NaN values.
    """

    df.dropna(inplace=True)
    return df


def pie_chart(df, col):
    """
    Pie chart of the dataframe.
    :param df: Dataframe
    :param col: Column to show in the pie chart

    :return: None
    """

    df[col].value_counts().plot(kind="pie", autopct="%1.1f%%")
    plt.show()


def count_plot(df, col):
    """
    Count chart of the dataframe.
    :param df: Dataframe
    :param col: Column to show in the count chart

    :return: None
    """

    df[col].value_counts().plot(kind="bar")
    plt.show()


def feature_importance(x, y, show_plot=False):
    """
    Gets the important features for the given target column.
    :param x: Dataframe
    :param y: Target column
    :param show_plot: If True, shows the plot of the feature importance.

    :return: List of important features
    """

    model = ExtraTreesClassifier()
    model.fit(x, y)

    feat_importances = pd.Series(model.feature_importances_, index=x.columns)

    if show_plot:
        feat_importances.nlargest(len(x.columns) // 2).plot(kind="barh")
        plt.show()

    return feat_importances


def histogram(df, cols, bins=10):
    """
    Shows a histogram of the dataframe.
    :param df: Dataframe
    :param cols: Columns to show in the histogram
    :param bins: Number of bins in the histogram

    :return: None
    """

    n = len(cols)

    plt.figure(figsize=(10, 10))

    for i, col in enumerate(cols):
        plt.subplot(n, 1, i + 1)
        sns.histplot(
            df[col],
            bins=bins,
            color="Red",
            kde_kws={"color": "y", "lw": 3, "label": "KDE"},
        )

    plt.show()


def get_median(df, col):
    """
    Gets the median of the column.
    :param df: Dataframe
    :param col: Column to get the median of

    :return: Median of the column
    """

    return df[col].median()


def get_mean(df, col):
    """
    Gets the mean of the column.
    :param df: Dataframe
    :param col: Column to get the mean of

    :return: Mean of the column
    """

    return df[col].mean()


def check_for_outliers(df, cols=False, threshold=10):
    """
    Finds columns that might have outliers in the dataframe.
    :param df: Dataframe
    :param cols: Columns to check for outliers
    :param threshold: Threshold for deviation of mean from median

    :return: List of columns with outliers
    """

    cols = df.columns if cols is False else cols

    cols_with_outliers = []

    for col in cols:
        mean = get_mean(df, col)
        median = get_median(df, col)

        if abs(mean - median) > (threshold / 100) * max(mean, median):
            cols_with_outliers.append(col)

    return cols_with_outliers


def get_correlation_with_target(df, target, cols=False):
    """
    Gets the correlation between the target column and the other columns.
    :param df: Dataframe
    :param target: Target column

    :return: List of correlations
    """

    if cols:
        df = df[cols]

    return df.corrwith(df[target]).sort_values(ascending=False)[1:]


def get_kurtosis(df, col):
    """
    Gets the kurtosis of the column.
    :param df: Dataframe
    :param col: Column to get the kurtosis of

    :return: Kurtosis of the column
    """

    return df[col].kurtosis()


def get_skewness(df, col):
    """
    Gets the skewness of the column.
    :param df: Dataframe
    :param col: Column to get the skewness of

    :return: skewness of the column
    """

    return df[col].skew()


def get_variance(df, col):
    """
    Gets the variance of the column.
    :param df: Dataframe
    :param col: Column to get the variance of

    :return: variance of the column
    """

    return df[col].var()


def get_count_of_unique_values(df, col):
    """
    Gets the count of unique values in the column.
    :param df: Dataframe
    :param col: Column to get the count of unique values of

    :return: count of unique values in the column
    """

    return df[col].nunique()


def get_statistics(df, cols=False, save=False):
    """
    Gets the statistics of the dataframe.
    :param df: Dataframe
    :param cols: Columns to get the statistics of

    :return: Dictionary with the statistics
    """

    if not cols:
        cols = df.columns

    stats = {}

    for col in cols:
        stats[col] = {}
        stats[col]["unique_count"] = get_count_of_unique_values(df, col)
        stats[col]["mean"] = get_mean(df, col)
        stats[col]["median"] = get_median(df, col)
        stats[col]["variance"] = get_variance(df, col)
        stats[col]["skewness"] = get_skewness(df, col)
        stats[col]["kurtosis"] = get_kurtosis(df, col)

    if save:
        with open("stats.json", "w") as f:
            json.dump(stats, f)

    return stats
