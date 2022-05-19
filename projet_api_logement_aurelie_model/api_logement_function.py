import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import *
from sklearn.metrics import roc_curve, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
import warnings
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import time
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# ----------------------------------------------------------------------------------
#                        MODELS : METRICS
# ----------------------------------------------------------------------------------
@ignore_warnings(category=UserWarning)
def get_models_regression_linear_grid(X_train, y_train, verbose=False, grid_params=None):
    if verbose: print("LinearRegression")
    if grid_params is None:
        grid_params = { 'linearregression__positive' :   [True, False],
                        'linearregression__normalize' :     [True, False],
                        'linearregression__fit_intercept' : [True, False]}
    grid_pipeline = make_pipeline( LinearRegression())
    grid = GridSearchCV(grid_pipeline,param_grid=grid_params, cv=4)
    grid.fit(X_train, y_train)
    if verbose: print("             DONE")
    return grid

@ignore_warnings(category=UserWarning)
def get_models_regression_random_forest(X_train, y_train, verbose=False, random_state=0, grid_params=None):
    if verbose: print("RandomForestRegressor")
    if grid_params is None:
        grid_params = { 'randomforestregressor__n_estimators' : np.arange(1, 100, 10),
                        'randomforestregressor__max_depth' : [None, 1, 2, 3],
                        'randomforestregressor__max_features' : ['auto', 'sqrt'],
                        'randomforestregressor__criterion' : ['squared_error', 'absolute_error', 'poisson'],
                        'randomforestregressor__min_samples_leaf' : [1],
                        'randomforestregressor__bootstrap' : [True, False],
                        'randomforestregressor__min_samples_split' : [1, 2, 3]}
    grid_pipeline = make_pipeline( RandomForestRegressor(random_state=random_state))
    grid = GridSearchCV(grid_pipeline,param_grid=grid_params, cv=4)
    grid.fit(X_train, y_train)
    if verbose: print("             DONE")
    return grid
# ----------------------------------------------------------------------------------
#                        MODELS : METRICS
# ----------------------------------------------------------------------------------
def get_metrics_for_the_model(model, X_test, y_test, y_pred,scores=None, model_name="", r2=None, full_metrics=False, verbose=0, transformer=None):
    if scores is None:
        scores = defaultdict(list)
    scores["Model"].append(model_name)
        
    if r2 is None:
        r2 = round(model.score(X_test, y_test),3)
        
    if y_pred is None:
        t0 = time.time()
        y_pred = model.predict(X_test)
        t_model = (time.time() - t0)   
        # Sauvegarde des scores
        scores["predict time"].append(time.strftime("%H:%M:%S", time.gmtime(t_model)))
        scores["predict seconde"].append(t_model)
        
    scores["R2"].append(r2)
    scores["MAE"].append(mean_absolute_error(y_test, y_pred))
    mse = mean_squared_error(y_test, y_pred)
    scores["MSE"].append(mse)
    scores["RMSE"].append(np.sqrt(mse))
    scores["Mediane AE"].append(median_absolute_error(y_test, y_pred))

    if full_metrics:
        try:
            y_prob = model.predict_proba(X_test)
        
            for metric in [brier_score_loss, log_loss]:
                score_name = metric.__name__.replace("_", " ").replace("score", "").capitalize()
                try:
                    scores[score_name].append(metric(y_test, y_prob[:, 1]))
                except Exception as ex:
                    scores[score_name].append(np.nan)
                    if verbose > 0:
                        print("005", model_name, score_name, ex)
        except Exception as ex:
            if verbose > 0:
                print("003", model_name, "Proba", ex)
            scores['Brier  loss'].append(np.nan)
            scores['Log loss'].append(np.nan)
                
        for metric in [f1_score, recall_score]:
            score_fc_name = metric.__name__.replace("_", " ").replace("score", "").capitalize()
            av_list = ['micro', 'macro', 'weighted']
            if metric == 3:
                av_list.append(None)
            for average in av_list:
                try:
                    score_name = score_fc_name+str(average)
                    scores[score_name].append(metric(y_test, y_pred, average=average))
                except Exception as ex:
                    if verbose > 0:
                        print("005", model_name, score_name, ex)
                    scores[score_name].append(np.nan)

        # Roc auc  multi_class must be in ('ovo', 'ovr')   
        for metric in [roc_auc_score]:
            score_fc_name = metric.__name__.replace("_", " ").replace("score", "").capitalize()
            for average in ['ovo', 'ovr']:
                try:
                    score_name = score_fc_name+str(average)
                    scores[score_name].append(metric(y_test, y_pred,multi_class= average))
                except Exception as ex:
                    if verbose > 0:
                        print("006", model_name, score_name, ex)
                    scores[score_name].append(np.nan)
    return scores

def get_metrics_for_model(model_dic, X_test, y_test, full_metrics=0, verbose=0):
    score_df = None
    scores = defaultdict(list)
    for model_name, (model, y_pred, r2) in model_dic.items():
        scores = get_metrics_for_the_model(model, X_test, y_test, y_pred, scores,model_name=model_name, r2=r2, full_metrics=full_metrics, verbose=verbose)

    score_df = pd.DataFrame(scores).set_index("Model")
    score_df.round(decimals=3)
    return score_df
# ----------------------------------------------------------------------------------
#                        MODELS : FIT AND TEST
# ----------------------------------------------------------------------------------
from joblib import dump, load
from datetime import datetime

def save_model(model_to_save, file_path, model_save_file_name):
    # Sauvegarde du meilleur modele
    now = datetime.now() # current date and time
    date_time = now.strftime("%Y-%m-%d-%H_%M_%S")
    model_save_file_name = model_save_file_name + date_time + '.joblib'
    # Attention, il faudra mettre à jour les colonnes correspondantes dans le premier if en cas de modification du model
    dump(model_to_save, file_path+model_save_file_name)
    return file_path+model_save_file_name

from os import path
def load_model(model_save_path):
    if path.exists(model_save_path) and path.isfile(model_save_path):
        # Chargement du modèle pré-entrainer
        return load(model_save_path)

def fit_and_test_models(model_list, X_train, Y_train, X_test, Y_test, y_column_name=None, verbose=0, scores=None, metrics=0, transformer=None):
    
    # Sauvegarde des modèles entrainés
    modeldic = {}
    yt = Y_test
    ya = Y_train
    # Sauvegarde des données
    if scores is None:
        scores = defaultdict(list)

    if y_column_name is None:
        y_column_name = ""
    else:
        yt = Y_test[y_column_name]
        ya = Y_train[y_column_name]

    scorelist = []
    for mod_name, model in model_list.items():
        try:
            model_name = mod_name
            if len(y_column_name) > 0:
                model_name = y_column_name+"-"+model_name

            scores["Class"].append(y_column_name)
            scores["Model"].append(mod_name)
            md, score_l = fit_and_test_a_model(model,model_name, X_train, ya, X_test, yt, verbose=verbose, metrics=metrics, transformer=transformer) 
            modeldic[model_name] = md
            scorelist.append(score_l)
        except Exception as ex:
            print(mod_name, "FAILED : ", ex)
    
    for score_l in scorelist:
        for key, val in score_l.items():
            scores[key].append(val)    
    
    return modeldic, scores

@ignore_warnings(category=ConvergenceWarning)
def fit_and_test_a_model(model, model_name, X_train, y_train, X_test, y_test, verbose=0, metrics=0, transformer=None):
    t0 = time.time()
    if verbose:
        print(model_name, "X_train:", X_train.shape,"y_train:", y_train.shape, "X_test:", X_test.shape,"y_test:", y_test.shape)

    if transformer is not None:
        try:
            X_train = transformer.fit_transform(X_train)
            X_test = transformer.fit_transform(X_test)
            if verbose:
                print(model_name, "After transform : X_train:", X_train.shape,"y_train:", y_train.shape, "X_test:", X_test.shape,"y_test:", y_test.shape)
        except:
            pass
    model.fit(X_train, y_train)
    
    r2 = model.score(X_test, y_test)
    if verbose:
        print(model_name+" "*(20-len(model_name))+":", round(r2, 3))
    t_model = (time.time() - t0)
        
    # Sauvegarde des scores
    modeldic_score = {"Modeli":model_name,
                      "R2":r2,
                      "fit time":time.strftime("%H:%M:%S", time.gmtime(t_model)),
                      "fit seconde":t_model}
    
    # Calcul et Sauvegarde des métriques
    if metrics > 0:
        full=metrics > 1
        t0 = time.time()
        model_metrics = get_metrics_for_the_model(model, X_test, y_test, y_pred=None,scores=None, model_name=model_name, r2=r2, full_metrics=full, verbose=verbose, transformer=transformer)
        t_model = (time.time() - t0)   
        modeldic_score["metrics time"] = time.strftime("%H:%M:%S", time.gmtime(t_model))
        modeldic_score["metrics seconde"] = t_model

        for key, val in model_metrics.items():
            if "R2" not in key and "Model" not in key:
                modeldic_score[key] = val[0]

    return model, modeldic_score

# ----------------------------------------------------------------------------------
#                        GRAPHIQUES
# ----------------------------------------------------------------------------------
PLOT_FIGURE_BAGROUNG_COLOR = 'white'
PLOT_BAGROUNG_COLOR = PLOT_FIGURE_BAGROUNG_COLOR


def color_graph_background(ligne=1, colonne=1):
    figure, axes = plt.subplots(ligne,colonne)
    figure.patch.set_facecolor(PLOT_FIGURE_BAGROUNG_COLOR)
    if isinstance(axes, np.ndarray):
        for axe in axes:
            # Traitement des figures avec plusieurs lignes
            if isinstance(axe, np.ndarray):
                for ae in axe:
                    ae.set_facecolor(PLOT_BAGROUNG_COLOR)
            else:
                axe.set_facecolor(PLOT_BAGROUNG_COLOR)
    else:
        axes.set_facecolor(PLOT_BAGROUNG_COLOR)
    return figure, axes