import ast
import enum
import os
import time
import pickle
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold

class ClassifierType(enum.Enum):
    LR = 1
    KNN = 2
    NN = 3
    CNN = 4

BEST_MODEL_NAME= 'model'
FIT_TIME_NAME = 'fit_time.csv'
BEST_RESULT_NAME= 'best_result.csv'
CV_RESULT_NAME= 'cv_results.csv'
PREDICTIONS_NAME = 'predictions.csv' 
PREDICTIONS_PROB_NAME =  'predictions_prob.csv'
RESULT_DIR_PATH = "results"

class ResultFileManagement:

    def __init__(self, classifier_type, dataset_label):
        self.classifier_type = classifier_type
        self.dataset_label = dataset_label

    def write_best_result(self, best_params, best_score, elapsed_time_ns=None, cpu_process_time_ns=None):
        data = {'best_params': [str(best_params)], 'best_score': [str(best_score)]}
        if(elapsed_time_ns is not None):
            data['elapsed_time_ns'] = [elapsed_time_ns]
        if(cpu_process_time_ns is not None):
            data['cpu_process_time_ns'] = [cpu_process_time_ns]
        df = pd.DataFrame(data)
        self.create_dir()  
        df.to_csv(self.full_file_name(BEST_RESULT_NAME), index=False)

    def read_best_params(self):
        df_ = pd.read_csv(self.full_file_name(BEST_RESULT_NAME))
        return ast.literal_eval(df_['best_params'].iloc[0])

    def read_best_score(self):
        df_ = pd.read_csv(self.full_file_name(BEST_RESULT_NAME))
        return ast.literal_eval(df_['best_score'].iloc[0])

    def write_cv_results(self, cv_results):
        df = pd.DataFrame(cv_results)
        self.create_dir() 
        df.to_csv(self.full_file_name(CV_RESULT_NAME), index=False)

    def read_cv_results(self):
        return pd.read_csv(self.full_file_name(CV_RESULT_NAME))

    def write_model(self, model):
        self.create_dir()
        model_pickle = open(self.full_file_name(BEST_MODEL_NAME), 'wb')
        pickle.dump(model, model_pickle)  
    
    def read_model(self):
        return pickle.load(open(self.full_file_name(BEST_MODEL_NAME), 'rb'))

    def result_exist(self):
        return os.path.exists(self.full_file_name(BEST_RESULT_NAME))

    def model_exist(self):
        return os.path.exists(self.full_file_name(BEST_MODEL_NAME))

    def full_file_name(self, sufix):
        return os.path.join(RESULT_DIR_PATH, f'{self.classifier_type.name}_{self.dataset_label}_{sufix}')
    
    def create_dir(self):
        if not os.path.exists(RESULT_DIR_PATH):
            os.mkdir(RESULT_DIR_PATH)


def grid_search_fit(estimator, param_grid, result_mgt, X, y):

    search = GridSearchCV(estimator = estimator, 
        param_grid = param_grid, scoring = 'accuracy', cv = KFold(n_splits=5), return_train_score=True, n_jobs=4)

    start_time = time.time_ns()    
    start_process_time = time.process_time_ns()

    classifier = search.fit(X, y)
    
    end_process_time = time.process_time_ns()
    elapsed_time = time.time_ns() - start_time
    cpu_process_time =  end_process_time - start_process_time

    result_mgt.write_model(classifier)
    result_mgt.write_best_result(search.best_params_, search.best_score_, elapsed_time, cpu_process_time)
    result_mgt.write_cv_results(search.cv_results_)
    
    return classifier

def random_search_fit(estimator, param_grid, result_mgt, random_state,  X, y):

    search = RandomizedSearchCV(estimator = estimator, 
        param_distributions = param_grid, scoring = 'accuracy', cv = KFold(n_splits=5), return_train_score=True, 
        n_jobs=4, random_state=random_state, n_iter=16)

    start_time = time.time_ns()    
    start_process_time = time.process_time_ns()

    classifier = search.fit(X, y)
    
    end_process_time = time.process_time_ns()
    elapsed_time = time.time_ns() - start_time
    cpu_process_time =  end_process_time - start_process_time

    result_mgt.write_model(classifier)
    result_mgt.write_best_result(search.best_params_, search.best_score_, elapsed_time, cpu_process_time)
    result_mgt.write_cv_results(search.cv_results_)
    
    return classifier


