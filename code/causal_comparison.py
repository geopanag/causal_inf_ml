# %%

import pandas as pd 
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier, XGBRegressor
encoder = OneHotEncoder()
seed=42

from sklearn.metrics import mean_squared_error, log_loss
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from call_models import sklift_dml, causalml_dml, econml_dml
from causalml.inference.meta import BaseXClassifier, BaseTClassifier, BaseSClassifier

import os


def main(test_size,result_name):
    
    #%% Load datasets
    # Criteo from https://ailab.criteo.com/criteo-uplift-prediction-dataset/
    # RetailHero from https://ods.ai/competitions/x5-retailhero-uplift-modeling/data

    criteo = pd.read_csv('criteo-uplift-v2.1.csv')

    results = open(result_name, "w")
    results.write("Package,Dataset,Model,Method,Score\n")


    confounders = criteo.drop(columns=['conversion'])
    outcome = criteo['conversion']
    criteo_X_train, criteo_X_test, criteo_y_train, criteo_y_test = train_test_split(confounders, outcome, test_size=test_size)

    df_features = pd.read_csv("retailhero-uplift/data/clients.csv").set_index("client_id")
    train = pd.read_csv("retailhero-uplift/data/uplift_train.csv").set_index("client_id")

    # Fix times
    df_features['first_issue_time'] = \
        (pd.to_datetime(df_features['first_issue_date'])
        - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')
    df_features['first_redeem_time'] = \
        (pd.to_datetime(df_features['first_redeem_date'])
        - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')
    df_features['issue_redeem_delay'] = df_features['first_redeem_time'] \
        - df_features['first_issue_time']
    df_features = df_features.drop(['first_issue_date', 'first_redeem_date'], axis=1)

    retail = train.join(df_features).reset_index(drop=True) 

    retail.rename(columns={'treatment_flg':'treatment'}, inplace=True)

    retail = retail.dropna().reset_index(drop=True)

    # Fit and transform the data to one-hot encoding
    one_hot_encoded = encoder.fit_transform(retail[["gender"]])
    one_hot_encoded_array = one_hot_encoded.toarray()
    encoded_categories = encoder.categories_

    df_encoded = pd.DataFrame(one_hot_encoded_array, columns=encoded_categories[0])
    retail = retail.drop("gender",axis=1)
    columns = list(retail.columns)+list(encoded_categories[0])
    retail = pd.concat([retail, df_encoded], axis=1,ignore_index=True)

    retail.columns = columns
    confounders = retail.drop(columns=['target'])
    outcome = retail['target']
    retail_X_train, retail_X_test, retail_y_train, retail_y_test = train_test_split(confounders, outcome, test_size=test_size)


    dic = {"XGB":XGBClassifier, "LR":LogisticRegression}

    # %% Sklift DML
    print("Sklift DML")
    for mod in ['XGB','LR']:
        model = dic[mod]
        for label_l in ['S','T','X/DDR']:
            print(label_l)
            m_score = sklift_dml(retail_X_train, retail_y_train.values , retail_X_test, retail_y_test.values, label_l, model)
            res = f"SKlift,RetailHero,{mod},{label_l}, {round(m_score, 5)}\n" #, {round(m_auc_score, 5)}, {round(m_qini_score, 5)}
            results.write(res)
            
            m_score = sklift_dml(criteo_X_train, criteo_y_train.values, criteo_X_test, criteo_y_test.values, label_l, model)
            res = f"SKlift,Criteo,{mod},{label_l}, {round(m_score, 5)}\n" #, {round(m_auc_score, 5)}, {round(m_qini_score, 5)}
            results.write(res)
    results.flush()


    # %% CausalML
    print("CausalML DML")
    for mod in ['XGB','LR']:
        model = dic[mod]
        for base_learner,label_l in zip([BaseSClassifier, BaseTClassifier, BaseXClassifier],['S','T','X/DDR']):
            m_score = causalml_dml(retail_X_train, retail_y_train.values , retail_X_test, retail_y_test.values, base_learner,label_l, model)
            res = f"CausalML,RetailHero,{mod},{label_l}, {round(m_score, 5)}\n" # , {round(m_auc_score, 5)}, {round(m_qini_score, 5)}\n"

            results.write(res)
            m_score  = causalml_dml(criteo_X_train, criteo_y_train.values, criteo_X_test, criteo_y_test.values, base_learner,label_l, model)
            res = f"CausalML,Criteo,{mod},{label_l}, {round(m_score, 5)}\n" #, {round(m_auc_score, 5)}, {round(m_qini_score, 5)}\n"
            results.write(res)
    results.flush()



    # %% EconML
    print("EconML DML")
    for mod in ['XGB','LR']:
        model = dic[mod]
        for label_l in ['S','T','X/DDR']:
            print(label_l)

            m_score = econml_dml(retail_X_train, retail_y_train.values , retail_X_test, retail_y_test.values, label_l, model)
            res = f"EconML,RetailHero,{mod},{label_l}, {round(m_score, 5)}\n" 

            results.write(res)


            m_score = econml_dml(criteo_X_train, criteo_y_train.values, criteo_X_test, criteo_y_test.values,label_l, model)

            res = f"EconML,Criteo,{mod},{label_l}, {round(m_score, 5)}\n"

            results.write(res)
    
    results.close()



if __name__ == '__main__':
    
    main(test_size = 0.5, result_name = "../results/results_new.csv")
    
    df = pd.read_csv('../results/results_new.csv')
    df['Score'] = df['Score'] * 100
    df['Method'] = pd.Categorical(df['Method'], categories=["S", "T", "X/DDR"])

    # Loop through datasets
    for dataset in ["RetailHero", "Criteo"]:
        df1 = df[df['Dataset'] == dataset]
        
        g = sns.catplot(data=df1, x='Method', y='Score', hue='Package', kind='bar', dodge=True, col='Model')
        g.set_titles(f"{dataset}-Uplift@40% - {g.col_names[0]}")
        g.set_ylabels("Uplift*100")

        for i, ax in enumerate(g.axes.flat):
            model = g.col_names[i]
            ax.set_title(f"{dataset}-Uplift@40% -{model}")
        # Save the plot
        g.savefig(f"figures/{dataset}_uplift40.png")
