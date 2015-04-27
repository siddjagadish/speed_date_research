from joblib import Parallel, delayed
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
import numpy as np
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt

N_PROCESSORS = multiprocessing.cpu_count()

def get_train_test_single_fold(df, split_var, fold):
    train_indices, test_indices = fold
    train_splitters = df[split_var].drop_duplicates().iloc[train_indices]
    train_map = df[split_var].isin(train_splitters)
    return df[train_map], df[~train_map]

def process_single_fold(train, test, classifier_model):
    fold = ModelFold(train, test)
    fold.set_model(classifier_model)
    fold.fit_model()
    fold.mod_predict_proba()
    return fold        

class ClassifierModel(object):
    def __init__(self, mod, predictors, categorical_predictors, outcome_var, **model_params):
        self.mod = mod
        self.model_params = model_params
        self.predictors = predictors
        self.categorical_predictors = categorical_predictors
        self.outcome_var = outcome_var

# TODO: Reset method 
class Analyzer(object):
    def __init__(self, data):
        self.orig_df = data
        self.df = data.copy()

    def set_model(self, classifier_model):
        assert(isinstance(classifier_model, ClassifierModel))
        self.classifier_mod = classifier_model 
        self._is_model_set = True
        self.fit_mod = None 
     
    def get_train_test_dfs(self, split_var, k=5):
        print 'Getting train and test sets...'
        self.n_folds = k
        self.split_var = split_var
        kf = KFold(len(self.df[split_var].drop_duplicates()), n_folds=k, shuffle=True)
        self.train_tests = Parallel(n_jobs=min(N_PROCESSORS,k), backend='threading')(delayed(get_train_test_single_fold)(self.orig_df.copy(), split_var, fold) for fold in kf)

    # TODO: Make sure quantile call works!
    def get_single_col_AUC(self, col, n_replacements=50):
        print 'Getting AUC based on %s only...' % (col)
        labels = self.df[self.classifier_mod.outcome_var].copy()
        predictions = self.df[col].copy()
        get_single_replacement_auc = lambda x: roc_auc_score(labels, predictions.fillna(x))
        n_predictions = len(predictions[~predictions.isnull()]) 
        fillers = predictions[~predictions.isnull()].quantile(np.linspace(0, 1, n_replacements))
        return np.max(fillers.map(get_single_replacement_auc)) 

    def get_var_importance_matrix(self, num_vars=None):
        if self.fit_mod is None:
            raise Exception('Trying to get feature importance matrix before fitting model...call fit_model first')
        if num_vars is None:
            num_vars = len(self.classifier_mod.predictors)
        temp = pd.DataFrame({'feature': self.classifier_mod.predictors, 'importance': self.fit_mod.feature_importances_})  
        return temp.sort('importance', ascending=False).iloc[:num_vars]
         
    def fit_model(self):
        if not self._is_model_set:
            raise Exception('Trying to fit model before setting model...call set_model first') 
        self.df = self.orig_df.copy()  # Reset df for enumeration and such
        mod_fit = self.classifier_mod.mod(**self.classifier_mod.model_params)
        mod_fit.fit(self.df[self.classifier_mod.predictors], self.df[self.classifier_mod.outcome_var]) 
        self.fit_mod = mod_fit 

    def k_fold_cv(self, outcome_metric):
        if self.train_tests is None:
            raise Exception('Trying to run k-fold CV before creating folds...run get_train-test_dfs first')
        print 'Getting K-Fold predictions...'
        self.folds = Parallel(n_jobs=min(N_PROCESSORS, len(self.train_tests)), backend='threading')(delayed(process_single_fold)(train, test, self.classifier_mod) for train, test in self.train_tests)
        return [outcome_metric(fold.test[fold.classifier_mod.outcome_var], fold.test['estimated_prob']) for fold in self.folds]

    def get_df_with_test_probs(self):
        if self.folds is None:
            raise Exception('Trying to get df with probs before fitting k-fold models. Run k_fold_cv first')
        self.df_with_probs = pd.concat([fold.test for fold in self.folds], axis=0)
    
    # Gets repayment rates binned by estimated probability
    def get_binned_repayment_rates(self, group, prob_thresholds, dollar_weighted=True):
        get_num_above_threshold = lambda x: group[group['estimated_prob'] > x].shape[0]
        get_sum_above_threshold = lambda x: sum(group[group['estimated_prob'] > x]['loan_amount'])
        if dollar_weighted:
            return prob_thresholds.map(lambda x: (sum(group[(group['estimated_prob'] > x) & group['made_first_payment']]['loan_amount']) / float(get_sum_above_threshold(x))) if get_sum_above_threshold(x) > 0 else np.nan)
        else:
            return prob_thresholds.map(lambda x: (sum(group[group['estimated_prob'] > x]['made_first_payment']) / float(get_num_above_threshold(x))) if get_num_above_threshold(x) > 0 else np.nan)

    def get_output_matrix(self, prob_thresholds=pd.Series([0.9, 0.925, 0.95, 0.975])):
        grouped = self.df_with_probs.groupby('FICO_tranche')
        prediction_proportions = grouped.apply(lambda x: prob_thresholds.map(lambda y: float(sum(x['estimated_prob'] > y)) / len(x['estimated_prob'])))
        binned_repayment_rates = grouped.apply(lambda x: self.get_binned_repayment_rates(x, prob_thresholds))
        ret_df = pd.concat([prediction_proportions, binned_repayment_rates], axis=1, ignore_index=True)
        ret_df.columns = ['prop_' + str(thresh) for thresh in list(prob_thresholds)] + ['repay_' + str(thresh) for thresh in list(prob_thresholds)]
        return ret_df

    def custom_partial_plot(self, col):
        if self.fit_mod is None:
            raise Exception('Trying to make plots before fitting model...fit model first')
        print 'Plotting %s...' % col
        lb = self.orig_df[col].min()
        ub = self.orig_df[col].max()
        tryme = np.linspace(lb, ub, num=50)
        outcomes = []
        
        df_copy = self.df.copy()
        for x in tryme:
            df_copy[col] = x
            outcomes.append(np.mean(self.fit_mod.predict_proba(df_copy[self.classifier_mod.predictors])[:,1]))

        plt.plot(tryme, outcomes)
        plt.xlabel(col)
        plt.ylabel("Mean Prob")
        plt.savefig("data/%s.png" % col)
        plt.clf()    


# Class that handles a single fold for cross-validation
# TODO: Create reset method
class ModelFold(object):
    def __init__(self, train_df, test_df):
        self._orig_train = train_df.copy()
        self._orig_test = test_df.copy()
        self.train = self._orig_train.copy()
        self.test = self._orig_test.copy()
        self._is_model_set = False
        self.fit_mod = None 
    
    def set_model(self, classifier_model):
        assert(isinstance(classifier_model, ClassifierModel))
        self.classifier_mod = classifier_model 
        self._is_model_set = True
        self.fit_mod = None 

    def fit_model(self):
        if not self._is_model_set:
            raise Exception('Trying to fit model before setting model...call set_model first') 
        print 'Fitting model...'
        mod_fit = self.classifier_mod.mod(**self.classifier_mod.model_params)
        mod_fit.fit(self.train[self.classifier_mod.predictors], self.train[self.classifier_mod.outcome_var]) 
        self.fit_mod = mod_fit
 
    def mod_predict_proba(self, data=None):
        if self.fit_mod is None:
            raise Exception('Trying to predict before fitting model...call fit_mod first')
        if data is None:
            data = self.test
        predictions = self.fit_mod.predict_proba(data[self.classifier_mod.predictors])
        pos_probs = zip(*predictions)[1]
        data['estimated_prob'] = pos_probs
