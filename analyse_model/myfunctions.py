import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from scipy.stats import kurtosis, skew
import seaborn as sns

import matplotlib.pyplot as plt




from matplotlib.cbook import boxplot_stats
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import shapiro 
from scipy.stats import kstest
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import probplot
from sklearn.model_selection import train_test_split,GridSearchCV,learning_curve, RandomizedSearchCV, cross_val_score, KFold
from sklearn.metrics import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,PolynomialFeatures,RobustScaler
from sklearn.linear_model import Ridge,LinearRegression,Lasso, ElasticNet
from statistics import mean
from statistics import stdev
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
import statsmodels.api as sm

def extended_describe_all_columns(df):
    # Initialize an empty DataFrame to store the results
    result_df = pd.DataFrame()

    # Iterate through each column in the DataFrame
    for column in df.columns:
        # Calculate kurtosis and skewness for the column
        kurt = kurtosis(df[column])
        skewness = skew(df[column])

        # Describe the column and add kurtosis and skewness as new rows
        column_stats = df[column].describe().to_frame()
        column_stats.loc['kurtosis'] = kurt
        column_stats.loc['skewness'] = skewness

        # Add the column's stats to the result DataFrame
        result_df[column] = column_stats[column]

    return result_df



def missing_values_summary(df):
    # Calculate the count of missing values per column
    missing_values_count = df.isna().sum()

    # Calculate the percentage of missing values per column
    total_rows = len(df)
    percentage_missing = (missing_values_count / total_rows) * 100

    # Create a DataFrame with missing values count and percentage
    result = pd.concat([missing_values_count, percentage_missing], axis=1, keys=['Missing Count', 'Percentage Missing %'])

    # Sort the values in descending order
    result_sorted = result.sort_values(by='Percentage Missing %', ascending=False)

    return result_sorted




def calculate_category_occurrence(df):
    # Initialize an empty dictionary to store results
    result_dict = {}
    
    # Loop through each categorical column
    for col in df.columns:
        value_counts = df[col].value_counts()
        total_samples = df.shape[0]
        
        # Calculate the percentage occurrence of each category
        percentage_occurrence = (value_counts / total_samples) * 100
        
        # Store the percentage occurrence in the result dictionary
        result_dict[col] = percentage_occurrence
    
    # Convert the result dictionary to a DataFrame
    result_df = pd.DataFrame(result_dict)
    
    return result_df


def calculate_category_variables(df, threshold_min=95, threshold_max=100):
    category_variables_list = []

    for col in df.columns:
        if df[col].dtype == 'object':  # Assuming only categorical variables are of type 'object'
            total_count = df[col].count()
            category_counts = df[col].value_counts()
            category_percentages = (category_counts / total_count) * 100

            # Check if any category has a percentage within the specified range
            if any((category_percentages >= threshold_min) & (category_percentages <= threshold_max)):
                category_variables_list.append(col)

    return category_variables_list












def plot_scatter_with_saleprice(df):
    # Calculate the number of rows and columns required for subplots
    num_rows = int(len(df.columns) / 4) + (len(df.columns) % 4 > 0)
    num_cols = 4

    # Creating subplots for numeric variables against 'SalePrice'
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5*num_rows))
    axes = axes.flatten()

    for i, col in enumerate(df.columns):
        # Skip the scatter plot when y='SalePrice' and x='SalePrice'
        
            sns.scatterplot(x=col, y='SalePrice', data=df, ax=axes[i])
            axes[i].set_ylabel("SalePrice")

            # # Hide any empty subplots if the number of columns is not a multiple of 4
            # if i >= len(df.columns) - (len(df.columns) % 4):
            #     axes[i].axis('off')

    plt.tight_layout()
    plt.show()







def get_extreme_values(data, variable, threshold=10):
    """
    Obtient les valeurs extrêmes d'une variable numérique.

    Parameters:
        data (pandas.DataFrame): Le DataFrame contenant les données.
        variable (str): Le nom de la variable numérique dont on veut obtenir les valeurs extrêmes.
        n (int, optional): Le nombre de valeurs extrêmes à retourner pour chaque extrémité. Par défaut, n=10.

    Returns:
        low_range (list): Une liste contenant les n valeurs les plus basses arrondies à 2 décimales.
        high_range (list): Une liste contenant les n valeurs les plus élevées arrondies à 2 décimales.
    """
    # Vérifier que la variable est numérique
    if data[variable].dtype != np.number:
        raise ValueError("La variable doit être numérique.")

    # Standardiser les données
    scaled_data = StandardScaler().fit_transform(data[variable].values[:, np.newaxis])

    # Obtenir les n valeurs les plus basses (low_range) et les n valeurs les plus élevées (high_range)
    low_range = scaled_data[scaled_data[:, 0].argsort()][:threshold].flatten()
    high_range = scaled_data[scaled_data[:, 0].argsort()][-threshold:].flatten()

    # Arrondir chaque élément des listes à 2 décimales
    low_range = np.round(low_range, 2)
    high_range = np.round(high_range, 2)

    return low_range, high_range




def plot_box_and_scatter(df, column_name):
    # Create a 1x2 subplot grid
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the box plot on the first subplot (ax1)
    sns.boxplot(x=df[column_name], orient='h', ax=ax1)
    ax1.set_xlabel(column_name)
    ax1.set_title(f'Box Plot of {column_name}')

    # Plot the scatter plot on the second subplot (ax2)
    sns.scatterplot(x=column_name, y='SalePrice', data=df, ax=ax2)
    ax2.set_xlabel(column_name)
    ax2.set_ylabel('SalePrice')
    ax2.set_title(f'Scatter Plot of {column_name} vs. SalePrice')

    # Adjust the layout to avoid overlapping labels
    plt.tight_layout()

    # Display the combined plot
    plt.show()





import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dython.nominal import associations

def plot_corr_heatmaps(df, top = 15):
    # Compute complete correlation
    complete_correlation = associations(df, compute_only=True)

    # Get the top 15 correlated variables with 'SalePrice'
    top_n_corr_var = complete_correlation['corr']['SalePrice'].abs().sort_values(ascending=False).to_frame().T.iloc[:, :top]

    # Plot the heatmap for the top 15 correlated variables
    plt.figure(figsize=(15, 6))
    sns.heatmap(top_n_corr_var, annot=True, fmt=".2f")
    plt.title("Heatmap des 15 variables les plus corrélées avec 'SalePrice'")
    plt.xlabel("Variables")
    plt.ylabel("SalePrice")
    plt.show()

    # Get the names of the top correlated variables
    top_n_variables = top_n_corr_var.columns[1:]
    df_sub_selection = df.loc[:, top_n_variables]

    # Compute the correlation for the selected variables
    sub_correlation = associations(df_sub_selection, compute_only=True)
    mask = np.triu(np.ones_like(sub_correlation['corr'], dtype=bool))

    # Plot the heatmap for the selected variables
    f, ax = plt.subplots(figsize=(15, 15))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(sub_correlation['corr'], mask=mask, cmap=cmap, center=0,
                square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title("Heatmap des variables les plus corrélées entre elles")
    plt.show()

    # Return the list of top correlated variables
    top_correlated_variables_list = df_sub_selection.columns.tolist()
    return top_correlated_variables_list







def plot_regression_results(model_name, model, y_train, X_train, y_pred, y_test, R2, MAE, RMSE, include_learning_curve=True):
    if include_learning_curve == True:
        ncols = 3
    else :
        ncols = 2
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(25, 8))
    axs = axs.ravel()

    fig.suptitle(f'{model_name}')

    axs[0].scatter(y_pred, y_test, alpha=0.5)
    axs[0].plot(np.arange(max(y_test.values)), np.arange(max(y_test.values)), '-', color='r')
    axs[0].set_xlabel('Prediction')
    axs[0].set_ylabel('Real')
    axs[0].set_title("")
    axs[0].legend([f'R2 : {round(R2,4)} \nMAE : {round(MAE,4)} \nRMSE : {round(RMSE,4)}'], loc='upper left')

    y_test_array = y_test.values
    residuals = y_test_array - y_pred
    residuals = list(residuals.flatten())

    parplot = probplot(residuals, dist='norm', plot=axs[1])
    axs[1].set_title("Probility plot of residuals")

    if include_learning_curve:
        # Generate the learning curve data
        train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=10, train_sizes=np.linspace(0.01, 1.0, 100))

        # Extract the mean training and test scores
        mean_train_scores = np.mean(train_scores, axis=1)
        mean_test_scores = np.mean(test_scores, axis=1)
        std_train = np.std(train_scores, axis=1)
        std_test = np.std(test_scores, axis=1)

        # Plot the mean training and test scores
        axs[2].plot(train_sizes, mean_train_scores, label='Training', color = 'blue')
        axs[2].fill_between(train_sizes, mean_train_scores + std_train, mean_train_scores - std_train, alpha=0.15, color='blue')
        axs[2].plot(train_sizes, mean_test_scores, label='Validation', color = 'green')
        axs[2].fill_between(train_sizes, mean_test_scores + std_test, mean_test_scores - std_test, alpha=0.15, color='green')
        axs[2].set_xlabel('Number of Training Samples')
        axs[2].set_ylabel('Model Score')
        axs[2].legend()

    plt.show()






def get_metrics(model, y_test, X_test):
    y_pred = model.predict(X_test)
    R2 = r2_score(y_pred , y_test).round(4)
    MAE = mean_absolute_error(y_pred , y_test)
    RMSE = np.sqrt(mean_squared_error(y_pred , y_test))
    return R2, MAE, RMSE, y_pred


def Lasso_with_CV(PolynomialFeatures_degree, best_alpha, X_train, y_train , X_test , y_test, preprocessor , shuffle=True, random_state=42, isplot= False, isinfo = False, include_learning_curve = False):

    PolynomialFeatures_degree = PolynomialFeatures_degree
    model = make_pipeline(preprocessor, PolynomialFeatures(degree=PolynomialFeatures_degree), Lasso(best_alpha)    )


    kfold = KFold(n_splits=5, shuffle=shuffle, random_state=random_state)
    scores = cross_val_score(model, X_train, y_train, cv=kfold)


    model.fit(X_train, y_train)
    Model_score_test = model.score(X_test, y_test)
    Model_score_training = model.score(X_train, y_train)

    R2, MAE, RMSE, y_pred = get_metrics(model, y_test = y_test, X_test = X_test)


    scores_mean = round(mean(scores),4)
    scores_std = round(stdev(scores),4)
    if isplot == True:
        plot_regression_results(f'LR with Kfold CV (Polynomial degree={PolynomialFeatures})', model, y_train, X_train, y_pred, y_test, R2, MAE, RMSE, include_learning_curve)
    if isinfo == True:
            print(f"LR with Kfold CV (Polynomial degree={PolynomialFeatures_degree})")
            print("="*50)
            print()
            print(scores)
            print(mean(scores))
            print("CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
            print(f"R2: {round(R2,4)}")
            print(f"MAE: {round(MAE,4)}")
            print(f"RMSE: {round(RMSE,4)}")
            print(f"Model_score_test: {round(RMSE,4)}")
            print(f"Model_score_training: {round(RMSE,4)}")
    return R2, MAE, RMSE, Model_score_test, Model_score_training, scores_mean, scores_std, model




def get_best_params_Lasso(param_grid, preprocessor, X_train, y_train , y_test, X_test):
    model = make_pipeline(preprocessor, Lasso())
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=kfold, n_jobs=-1,  scoring='neg_mean_absolute_error', verbose=0)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_['lasso__alpha']
    best_estimator = grid_search.best_estimator_

    # Train the best estimator on the entire training set
    best_estimator.fit(X_train, y_train)

    # Evaluate the performance of the best estimator on the test set using MAE
    test_mae = mean_absolute_error(y_test, best_estimator.predict(X_test))

    print("Best alpha:", best_params)
    print("Test MAE:", test_mae)
    # model = grid_search.best_estimator_






from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.impute import KNNImputer

# https://www.kaggle.com/discussions/questions-and-answers/153147
def KNNImputer_imputation(df, cols):
    df = df.copy()

    # Separate categorical and numerical features
    numerical_features = df.select_dtypes(include=[np.number]).columns
    categorical_features = [col for col in cols if col not in numerical_features]

    # MinMax scaling of numerical features
    mm = MinMaxScaler()
    df[numerical_features] = mm.fit_transform(df[numerical_features])

    # KNN imputation on numerical features
    knn_imputer = KNNImputer()
    df[numerical_features] = knn_imputer.fit_transform(df[numerical_features])

    # Inverse transform numerical features back to their original scale
    df[numerical_features] = mm.inverse_transform(df[numerical_features])

    # Round numerical features to integers
    df[numerical_features] = df[numerical_features].round().astype(int)

    # Handle categorical variables with NaN values by filling with a placeholder
    df[categorical_features] = df[categorical_features].fillna("NaN")

    # Encode categorical variables
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df[categorical_features] = ordinal_encoder.fit_transform(df[categorical_features])

    # Inverse transform categorical variables back to their original categories
    df[categorical_features] = ordinal_encoder.inverse_transform(df[categorical_features])

    return df
