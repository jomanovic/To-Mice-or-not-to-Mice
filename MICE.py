# -*- coding: utf-8 -*-
"""
Multiple Imputation by Chained Equations
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3074241/
"""

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeClassifier
from sklearn.model_selection import train_test_split

class MiceImputer(object):

    def __init__(self, seed_values = True, seed_strategy="mean", copy=True):
        self.strategy = seed_strategy # seed_strategy in ['mean','median','most_frequent', 'constant']
        self.seed_values = seed_values # seed_values = False initializes missing_values using not_null columns
        self.copy = copy
        self.imp = SimpleImputer(strategy=self.strategy, copy=self.copy)

    def fit_transform(self, X, method = 'Linear', corr_thrs = 0, iter = 5, verbose = True):
        
        # Why use Pandas?
        # http://gouthamanbalaraman.com/blog/numpy-vs-pandas-comparison.html
        # Pandas < Numpy if X.shape[0] < 50K
        # Pandas > Numpy if X.shape[0] > 500K
        
        # Data necessary for masking missing-values after imputation
        null_cols = X.columns[X.isna().any()].tolist()
        null_X = X.isna()[null_cols]
      
        ### Initialize missing_values
        
        if self.seed_values:
            
            # Impute all missing values using SimpleImputer 
            if verbose:
                print('Initilization of missing-values using SimpleImputer')
            new_X = pd.DataFrame(self.imp.fit_transform(X))
            new_X.columns = X.columns
            new_X.index = X.index
            
        else:
   
            # Initialize a copy based on value of self.copy
            if self.copy:
                new_X = X.copy()
            else:
                new_X = X
                
            not_null_cols = X.columns[X.notna().any()].tolist()
            
            if verbose:
                print('Initilization of missing-values using regression on non-null columns')
               
            for column in null_cols:
                
                null_rows = null_X[column]
                train_x = new_X.loc[~null_rows, not_null_cols]
                test_x = new_X.loc[null_rows, not_null_cols]
                train_y = new_X.loc[~null_rows, column]
                
                if X[column].nunique() > 2:
                    m = LinearRegression(n_jobs = -1)
                    m.fit(train_x, train_y)
                    new_X.loc[null_rows,column] = pd.Series(m.predict(test_x))
                    not_null_cols.append(column)
                    
                elif X[column].nunique() == 2:
                    m = LogisticRegression(n_jobs = -1, solver = 'lbfgs')
                    m.fit(train_x, train_y)
                    new_X.loc[null_rows,column] = pd.Series(m.predict(test_x))
                    not_null_cols.append(column)
                
       
        ### Create a dictionary { cols : [ other_cols if corr(cols, other_cols) < threshold ] }  (see "multicollinearity")
        
        corr_dict = {}
        
        if corr_thrs:
            if verbose:
                print('Building correlation table for missing-values...')
            for col in null_cols:
                corr_dict[col] = []
                for tmp in new_X.columns:
                    if corr_thrs < abs(new_X[col].corr(new_X[tmp])) and col != tmp:   
                        corr_dict[col].append(tmp)
        
        ### Begin iterations of MICE
        
        model_score = {}
        
        for i in range(iter):
            if verbose:
                print('Beginning iteration ' + str(i) + ':')
                
            model_score[i] = []
            
            for column in null_cols:
                
                null_rows = null_X[column]                
                not_null_y = new_X.loc[~null_rows, column]
                not_null_X = new_X[~null_rows].drop(column, axis = 1)
                
                train_x, val_x, train_y, val_y = train_test_split(not_null_X, not_null_y, test_size=0.33, random_state=42)
                
                test_x = new_X.drop(column, axis = 1)
                
                if corr_thrs and corr_dict.get(column, []):  
                    corr_cols = corr_dict[column]
                    train_x = train_x[corr_cols]
                    test_x = test_x[corr_cols]
                    val_x = val_x[corr_cols]
                  
                if new_X[column].nunique() > 2:
                    if method == 'Linear':
                        m = LinearRegression(n_jobs = -1)
                    elif method == 'Ridge':
                        m = Ridge()
                        
                    m.fit(train_x, train_y)
                    model_score[i].append(m.score(val_x, val_y))
                    new_X.loc[null_rows,column] = pd.Series(m.predict(test_x))
                    if verbose:
                        print('Model score for ' + str(column) + ': ' + str(m.score(val_x, val_y)))
                    
                elif new_X[column].nunique() == 2:
                    if method == 'Linear':
                        m = LogisticRegression(n_jobs = -1, solver = 'lbfgs')
                    elif method == 'Ridge':
                        m = RidgeClassifier()
                        
                    m.fit(train_x, train_y)
                    model_score[i].append(m.score(val_x, val_y))
                    new_X.loc[null_rows,column] = pd.Series(m.predict(test_x))
                    if verbose:
                        print('Model score for ' + str(column) + ': ' + str(m.score(val_x, val_y)))
                
            if model_score[i] == []:
                model_score[i] = 0
            else:
                model_score[i] = sum(model_score[i])/len(model_score[i])

        return new_X
