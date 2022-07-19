from utils import *


class EDA:
    def __init__(self, df, target_col=None):
        self.df = df
        self.target_col = target_col

    def show_corrMatrix(self):
        return correlation_matrix(self.df)

    def get_importantFeatures(self):

        if self.target_col:
            imp_features = feature_importance(self.df, self.target_col)
            return imp_features
        else:
            raise Exception("Target column not specified.")

    def deal_withNaN(self, method="mean"):
        if method == "mean":
            return fill_nan_with_mean(self.df)
        elif method == "delete":
            return delete_row_with_nan(self.df)
        else:
            raise Exception("Method not supported.")

    def check_and_deal_wtihOutliers(self):

        cols_with_outliers = check_for_outliers(self.df)
        if cols_with_outliers:
            _, df = find_outliers(self.df, cols=cols_with_outliers, remove=True)
            return df
        else:
            return None

    def data_stats(self):
        return get_statistics(self.df)

    def show_chart(self, df, col, chart="pie"):

        if chart == "pie":
            pie_chart(self.df, col)
        elif chart == "count":
            count_plot(self.df, col)
        elif chart == "hist":
            histogram(self.df, col)
        elif chart == "box":
            boxplot(self.df, col)
        else:
            raise Exception("Chart not supported.")
