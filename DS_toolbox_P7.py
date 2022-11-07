import pandas as pd 
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt
from IPython.display import display_html
from termcolor import colored
from tabulate import tabulate
import timeit
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn import set_config
set_config(display='diagram')
from sklearn.inspection import permutation_importance

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import learning_curve
from sklearn.feature_selection import VarianceThreshold


# fontion pour afficher plusieurs df à l'horizontal pour meileur visibilité 
def display_dfs(dfs, gap=50, justify='center'):
    html = ""
    for title, df in dfs.items():  
        df_html = df._repr_html_()
        cur_html = f'<div> <h3>{title}</h3> {df_html}</div>'
        html +=  cur_html
    html= f"""
    <div style="display:flex; gap:{gap}px; justify-content:{justify};">
        {html}
    </div>
    """
    display_html(html, raw=True)

#describe pour plusieurs dataframes dans dictionnaire
def describe_all_multi_dfs(dfs):
  
  for keys, df in dfs.items():
      print(f"Le dataset : {colored(keys.upper(),'red')}")
      print(tabulate(df.describe(include='all'),headers='keys',tablefmt='pretty'))
      print('')


# fonction detecte les doublons pour plusieurs dataframe dans un dictionnaire 
def doublons_check_multi_dfs(dfs):
  for keys, df in dfs.items():
    print(f"Les doublons du df {colored(keys.upper(),'red')}: " ,len(df[df.duplicated()]))
    

#Fontion pour detecter les NaN dans les colonnes 
def NaN_columns_check_multi_dfs(dfs):
    
    for keys, df in dfs.items():
        
        print(f"Le dataset : {colored(keys.upper(),'red')}")
        print(f'Les dimensions du {keys} : {df.shape}')
        
        dtypes = df.dtypes
        missing_count = df.isnull().sum()
        value_counts = df.isnull().count()
        missing_pct = (missing_count/value_counts)*100
        missing_total = df.isna().sum().sum()
        df_missing = pd.DataFrame({'Count_NaN':missing_count, 'Pct_NaN':missing_pct, 'Types':dtypes, 'Total_NaN_in_dataset': missing_total})
        df_missing = df_missing.sort_values(by='Pct_NaN', ascending=False)
        
        print('Les valeurs manquantes pour chaques colonnes :')
        print('')
        print(tabulate(df_missing,headers='keys',tablefmt='pretty'))
        
        plt.style.use('ggplot')
        plt.figure(figsize=(14,10))
        plt.title(f'Le pourcentage de valeurs manquantes pour la colonne {keys}', size=20)
        plt.plot( df.isna().sum()/df.shape[0])
        plt.xticks(rotation = 90) 
        plt.show()
        print('')
        print('----------------------------------------------------------------------------------')
        print('')

# fonction pour comparer 2 listes 
def comparaison (list1, list2):
  
  set_list_a = set(list1)
  print(f'Il y {len(set_list_a)} colonnes pour 2015: {set_list_a}')
  set_list_b = set(list2)
  print(f'Il y {len(set_list_b)} colonnes pour 2016 : {set_list_b}')
  # voir la valeur qui est dans les 2 listes
  inter_2_listes = set_list_a.intersection(set_list_b)
  print(f"Il y {len(inter_2_listes)} colonnes qui ont une correspondance entre 2015 et 2016: {inter_2_listes}")
  diff_2_listes1 = set_list_a.difference(set_list_b)
  print(f"Il y {len(diff_2_listes1)} colonnes différentes entre 2015 et 2016 : {diff_2_listes1}")
  diff_2_listes2 = set_list_b.difference(set_list_a)
  print(f"Il y {len(diff_2_listes2)} colonnes différentes entre 2016 et 2015 : {diff_2_listes2}")



#fonction pour comparer 2 listes NaN 
def comparaison_X_train_X_test(list1, list2):
  
  set_list_a = set(list1)
  print(colored(f'Il y {len(set_list_a)} colonnes pour X_train: {set_list_a}', 'blue'))
  set_list_b = set(list2)
  print(colored(f'Il y {len(set_list_b)} colonnes pour X_test : {set_list_b}', 'magenta'))
  # voir la valeur qui est dans les 2 listes
  inter_2_listes = set_list_a.intersection(set_list_b)
  print(f"Il y {len(inter_2_listes)} colonnes qui ont une correspondance entre X_train et X_test: {inter_2_listes}")
  diff_2_listes1 = set_list_a.difference(set_list_b)
  print(colored(f"Il y {len(diff_2_listes1)} colonnes avec NaN différentes entre X_train et X_test: {diff_2_listes1}",'red'))
  diff_2_listes2 = set_list_b.difference(set_list_a)
  print(colored(f"Il y {len(diff_2_listes2)} colonnes avec NaN différentes entre X_test et X_train: {diff_2_listes2}",'yellow'))

 


def memory_optimization(df):
    """
    Method used to optimize the memory usage.
    Parameters:
    -----------------
        df (pandas.DataFrame): Dataset to analyze
        
    Returns:
    -----------------
        df (pandas.DataFrame): Dataset optimized
    """ 
    
    for col in df.columns:
        if df[col].dtype == "int64" and df[col].nunique() == 2:
            df[col] = df[col].astype("int8")
            
    for col in df.columns:
        if df[col].dtype == "float64" and df[col].min() >= -2147483648 and df[col].max() <= 2147483648:
            df[col] = df[col].astype("float32")
            
    return df





def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df





def data_optimize(df, object_option=False):
    """Reduce the size of the input dataframe
    Parameters
    ----------
    df: pd.DataFrame
        input DataFrame
    object_option : bool, default=False
        if true, try to convert object to category
    Returns
    -------
    df: pd.DataFrame
        data type optimized output dataframe
    """

    # loop columns in the dataframe to downcast the dtype
    for col in df.columns:
        # process the int columns
        if df[col].dtype == 'int':
            col_min = df[col].min()
            col_max = df[col].max()
            # if all are non-negative, change to uint
            if col_min >= 0:
                if col_max < np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif col_max < np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif col_max < np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)
                else:
                    df[col] = df[col]
            else:
                # if it has negative values, downcast based on the min and max
                if col_max < np.iinfo(np.int8).max and col_min > np.iinfo(np.int8).min:
                    df[col] = df[col].astype(np.int8)
                elif col_max < np.iinfo(np.int16).max and col_min > np.iinfo(np.int16).min:
                    df[col] = df[col].astype(np.int16)
                elif col_max < np.iinfo(np.int32).max and col_min > np.iinfo(np.int32).min:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col]
                    
        # process the float columns
        elif df[col].dtype == 'float':
            col_min = df[col].min()
            col_max = df[col].max()
            # downcast based on the min and max
            if col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col]

        if object_option:
            if df[col].dtype == 'object':
                if len(df[col].value_counts()) < 0.5 * df.shape[0]:
                    df[col] = df[col].astype('category')

    return df


#Fonction pour réduire la mémoire du df     
def reduce_memory_usage(df, verbose=True):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df




#fonction afficher les valeurs uniques pour chaque colonne 

def unique_multi_cols(df):
      
    """
    Afficher les informations concernant les valeurs uniques pour chaque colonnes

    Parameters:
    -----------------
    df (DataFrame) : Dataframe analysis
        
    Returns:
    -----------------
    Les valeurs uniques pour chaque colonnes
    """ 


    for col in list(df.columns):
      
      pct_nan = (df[col].isna().sum()/df[col].shape[0])
      unique = df[col].unique()
      nunique = df[col].nunique()
  
      print('')
      print(colored(col, 'red'))
      print('') 
      print((f'Le pourcentage NaN : {pct_nan*100}%'))
      print(f'Nombre de valeurs unique : {nunique}')
      print('')
      print(unique)
      print('')
      print('---------------------------------------------------------------------------------------')


# Supprimer les variables avec pourcentage NaN 
def filter_col_pct_NaN(df,threshold_pct_NaN):
  
  """
  Supprimer les variables qui ont pourcentage de valeurs manquantes selon un seuil définis par threshold_pct_NaN

  Parameters:
  -----------------
  df (DataFrame) : Dataframe analysis
  threshold_pct_NaN (float) : Un float entre 0 et 1 (exemple: 0.9, 0.5)
  
        
  Returns:
  -----------------
  Retourne un dataframe avec les variables souhaitées par rapport au seuil définis pourcentage NaN 
  """ 


  df_filter_cols = df[df.columns[df.isna().sum()/df.shape[0] < threshold_pct_NaN]]
  df_col_drops = df[df.columns[df.isna().sum()/df.shape[0] > threshold_pct_NaN]]
  li_col_drops = list(df_col_drops.columns)
  li_col_keep = list(df_filter_cols.columns)
  print(f"Les dimensions après avoir supprimé les colonnes supérieur à {threshold_pct_NaN*100}% de valeurs manquantes :", df_filter_cols.shape)
  print(f"Les colonnes supprimées supérieur à {threshold_pct_NaN*100}% Nan qui ont été suprimés : ",len(li_col_drops), li_col_drops) 
  return df_filter_cols , li_col_drops , li_col_keep



def plot_corrélation_target_negative(df,target,number_var):
  """
  Afficher des heatmaps avec les corrélations negatives la target 

  Parameters:
  -----------------
  df (DataFrame) : Dataframe analysis
  target(string) : le nom de la target (exemple: 'TARGET')
  number_var(int) : nombre entier pour paramétrer corrélation avec target
        
  Returns:
  -----------------
  heatmaps avec le plus corrélation négative
  """ 
  fig, ax = plt.subplots(figsize=(22,12))
  sns.heatmap(df.corr()[[target]].sort_values(target).head(number_var), vmax=1, vmin=-1, cmap='viridis', annot=True, ax=ax)
  ax.invert_yaxis()


def plot_corrélation_target_positive(df,target,number_var):
  """
  Afficher des heatmaps avec les corrélations positives la target 

  Parameters:
  -----------------
  df (DataFrame) : Dataframe analysis
  target(string) : le nom de la target (exemple: 'TARGET')
  number_var(int) : nombre entier pour paramétrer corrélation avec target
        
  Returns:
  -----------------
  heatmaps avec le plus corrélation positive
  """ 
  fig, ax = plt.subplots(figsize=(22,12))
  sns.heatmap(df.corr()[[target]].sort_values(target).tail(number_var), vmax=1, vmin=-1, cmap='viridis', annot=True, ax=ax)
  ax.invert_yaxis()


# Vérifier inf dans dataframe
def check_inf_values(df):

  """
  Afficher les informations concernant si le dafaframe contient des valeurs infinites

  Parameters:
    -----------------
  df (DataFrame) : Dataframe analysis
        
  Returns:
  -----------------
  Le nombre de valeurs infinites dans le dataframe
  """
  
  count = np.isinf(df).values.sum()
  print("Le dataframe contient " + str(count) + " infinite values")





def select_features_variance_constants(threshold, X_train, X_test, X_prod):
  
  sel = VarianceThreshold(threshold=threshold)

  sel.fit(X_train)
  print('Le nombre de features conservées aprés avoir supprimées les variables quasi-constant: ',sum(sel.get_support()))
  # let's print the number of quasi-constant features
  quasi_constant = X_train.columns[~sel.get_support()]
  print('Le nombre de features supprimées quasi-constant: ',len(quasi_constant))
  print(quasi_constant)
  feat_names = X_train.columns[sel.get_support()]
    
  #remove the quasi-constant features
  
  X_train = sel.transform(X_train)
  X_test = sel.transform(X_test)
  #X_prod = sel.transform(X_prod)
  X_prod.drop(columns=quasi_constant, axis=1, inplace=True)
  X_train.shape, X_test.shape, X_prod.shape
  
  
  X_train = pd.DataFrame(X_train, columns=feat_names)
  X_test = pd.DataFrame(X_test, columns=feat_names)
  #print(X_train)
  #print(X_test)
  return X_train, X_test, X_prod





def correlation_features_redondances(X_train, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    
    corr_matrix = X_train.corr()
    
    for i in range(len(corr_matrix.columns)):
    
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr






# Plot NaN pourcentage
def plot_pourcentage_NaN_features(df):

     """
    Afficher les informations concernant les valeurs manquantes des features

    Parameters:
    -----------------
    df (DataFrame) : Dataframe analysis
        
    Returns:
    -----------------
    tabulates les valeurs manquantes des features
    """
    
     plt.figure(figsize=(20,18))
     plt.title('Le pourcentage de valeurs manquantes pour les features', size=20)
     plt.plot((df.isna().sum()/df.shape[0]*100).sort_values(ascending=True))
     plt.xlabel('Features dataset', fontsize=18)
     plt.ylabel('Pourcentage NaN dans features', fontsize=18)
     plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 45-degrees
     plt.show()

     pct_dataset = pd.DataFrame((df.isna().sum()/df.shape[0]*100).sort_values(ascending=False))
     pct_dataset = pct_dataset.rename(columns={0:'Pct_NaN_colonne'})
     pct_dataset =pct_dataset.style.background_gradient(cmap='YlOrRd')
     return pct_dataset



def select_features_model(list1, list2):
    
    """
    Indication sur les features conservées pour le modèle et les features supprimées 

    Parameters:
    -----------------
    list1 (list) : la liste des columns choisies pour le modèle 
    list2 (list) : la liste des columns dans le dataframe
  
        
    Returns:
    -----------------
    Retourne retourne set list des features pour le modèle et features  supprimées 
    """ 
    set_list_a = set(list1)
    print(f'Il y {len(set_list_a)} features à mettre dans le modèle : {set_list_a}')
    set_list_b = set(list2)
    print(f'Il y {len(set_list_b)} colonnes dans le train_set : {set_list_b}')
    diff_2_listes2 = set_list_b.difference(set_list_a)
    print(f"Il y {len(diff_2_listes2)} colonnes à supprimer dans le train_set : {diff_2_listes2}")


# fct cost
from sklearn.metrics import fbeta_score, make_scorer
def costs(X_train, y_train):

    tn, fp, fn, tp = confusion_matrix(X_train, y_train).ravel()

    tp_value = 0
    tn_value = 1
    fp_value = -1
    fn_value = -10

    gain = tp*tp_value + tn*tn_value + fp*fp_value + fn*fn_value

    gain_max = (fp + tn)*tn_value + (fn + tp)*tp_value

    gain_min = (fp + tn)*fp_value + (fn + tp)*fn_value

    gain_norm = (gain - gain_min)/(gain_max - gain_min)

    return gain_norm 

myscorer = make_scorer(costs)




def Gridseach_CV_procedure_eval(pipeline, X_train, y_train, param_grid, cross_validation_design, scoring, metric, name):
  
  """
  Procédure évaluation avec une gridsearchCV afin de de trouver les meileurs hyperparamètres de la pipeline  
  et de maximiser la fonction (métric) evaluation.
  Parameters:
  -----------------
  pipeline (sklearn.pipeline.Pipeline) : la pipeline contient les variables qui vont subir un preprocessing et le type d'algorithme
  (Note: La pipeline n'est pas encore fit)
  
  X_train (dataframe) : dataframe contenant les variables prédictives 
  
  y_train (dataframe) :  la target 
  
  param_grid (dict) : dictionnaire contenant les hyperparamètres pour la recherche par grille 
  
  cross_validation_design (KFold) : définir la procédure de cross validation 
  
  scoring (dict) : dictionnaire contenant les metrics pour la fonction métier 
  
  metric (str) : Réajustez un estimateur en utilisant les meilleurs paramètres trouvés sur l'ensemble de données.
  Pour l'évaluation de métriques multiples, cela doit être un strindiquant le scoreur qui serait utilisé pour trouver les meilleurs paramètres pour réajuster l'estimateur à la fin.

  Returns:
  -----------------
  Retourne les informations sur la pipeline avec les best params et l'évaluation de cette pipeline à l'aide de métrics
  """ 
  
  start_time_model = timeit.default_timer()

  # définir la gridSearch CV 
  grid = GridSearchCV(pipeline, param_grid=param_grid, 
                      cv=cross_validation_design, n_jobs= -1,scoring=scoring, return_train_score=True, refit=metric)
  
  #Entrainement avec les paramètres sélectionnés 
  pipeline_optimizer = grid.fit(X_train, y_train)
  end_time_model =  round(timeit.default_timer() - start_time_model, 3)
  print(name)
  display(pipeline_optimizer)
  print(f"Les meilleurs paramètres pour {name}: ", pipeline_optimizer.best_params_)
  print("Le best_score pour {name} :", pipeline_optimizer.best_score_)
  print("Le temps pour la gridsearhCV: ", end_time_model)
  results = pd.DataFrame(pipeline_optimizer.cv_results_)
  return results 


def evaluation(model, name_model):
    
    ypred_train = model.fit(X_train, y_train)
    ypred_test = model.predict(X_test)

    print(confusion_matrix(y_test, ypred_test))
    
    # Visualisation de la matrice confusion  
    cf_matrix=(confusion_matrix(y_test, ypred_test))
   
    plt.figure(figsize=(14,12))
    ax = plt.axes()
    plt.title(f'La matrice de confusion pour {name_model}', size=20, y=1.1)
    group_names = ['TRUE NEG (client rembourse et crédit accepté par le modèle)', 'FALSE POS (ERROR 1 perte opportunité pour la banque)', 'FALSE NEG (ERROR 2 perte argent pour la banque)', 'TRUE POS (Client pas rembourser et crédit refusé par modèle)']
    group_counts = ['{0:0.0f}'.format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in
                         cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    #cmd = ConfusionMatrixDisplay(cf_matrix, display_labels=['Remboursé(0)','Non remboursé(1)'])
    #cmd
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Reds', xticklabels=["Remboursé(0 = négatif)", "Non remboursé(1 = positif)"], yticklabels=['remboursé(0 = négatif)', 'Non remboursé(1 = positif)'])
    #ax.xaxis.set_ticklabels(["Remboursé(0)", "Non remboursé(1)"]); ax.yaxis.set_ticklabels(['Non remboursé(1)', 'Remboursé(0)']);
    # setting ticks for x-axis
    #ax.set_xticks(['Non remboursé(1)', 'Remboursé(0)'])
    
    # setting ticks for y-axis
    #ax.set_yticks(["Remboursé(0)", "Non remboursé(1)"])
    plt.ylabel('Actual label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.show()
    print( )
    print(classification_report(y_test, ypred_test))

from sklearn.model_selection import RandomizedSearchCV

def RandomizedSearchCV_procedure_eval(pipeline, param_grid, scoring, metric, name):
  
  """
  Procédure évaluation avec une gridsearchCV afin de de trouver les meileurs hyperparamètres de la pipeline  
  et de maximiser la fonction (métric) evaluation.
  Parameters:
  -----------------
  pipeline (sklearn.pipeline.Pipeline) : la pipeline contient les variables qui vont subir un preprocessing et le type d'algorithme
  (Note: La pipeline n'est pas encore fit)
  
  X_train (dataframe) : dataframe contenant les variables prédictives 
  
  y_train (dataframe) :  la target 
  
  param_grid (dict) : dictionnaire contenant les hyperparamètres pour la recherche par grille 
  
  cross_validation_design (KFold) : définir la procédure de cross validation 
  
  scoring (dict) : dictionnaire contenant les metrics pour la fonction métier 
  
  metric (str) : Réajustez un estimateur en utilisant les meilleurs paramètres trouvés sur l'ensemble de données.
  Pour l'évaluation de métriques multiples, cela doit être un string indiquant le scoreur qui serait utilisé pour trouver les meilleurs paramètres pour réajuster l'estimateur à la fin.

  Returns:
  -----------------
  Retourne les informations sur la pipeline avec les best params et l'évaluation de cette pipeline à l'aide de métrics
  """ 
  
  start_time_model = timeit.default_timer()

  # définir la gridSearch CV 
  grid = RandomizedSearchCV(pipeline, param_distributions = param_grid, 
                    cv=stratified_Kfold, n_jobs= -1, scoring=scoring, return_train_score=True, refit=metric, random_state=777)
  
  #Entrainement avec les paramètres sélectionnés 
  pipeline_optimizer = grid.fit(X_train, y_train)
  end_time_model =  round(timeit.default_timer() - start_time_model, 3)
  print(name)
  display(pipeline_optimizer)
  print(f"Les meilleurs paramètres pour {name}: ", pipeline_optimizer.best_params_)
  print(f"Le best_score validation_score_mean pour {name} :", pipeline_optimizer.best_score_)
  print("Le temps pour la RandomizedSearchCV: ", end_time_model)
  results_splits = pd.DataFrame(pipeline_optimizer.cv_results_)[['split0_test_fonction_cout_métier', 'split1_test_fonction_cout_métier', 'split2_test_fonction_cout_métier', 'split3_test_fonction_cout_métier',
       'split4_test_fonction_cout_métier','mean_test_fonction_cout_métier', 'std_test_fonction_cout_métier']].sort_values(by='mean_test_fonction_cout_métier', ascending=False).head(1)
  results_metrics = pd.DataFrame(pipeline_optimizer.cv_results_)[['mean_test_fonction_cout_métier','mean_test_f1', 'mean_test_roc_auc', 'mean_test_recall', 'mean_test_precision']].sort_values(by='mean_test_fonction_cout_métier', ascending=False).head(1)
  

  #pipeline_reg_log_results.sort_values(by='mean_test_fonction_cout_métier', ascending=False).head(1)
  return pipeline_optimizer.best_params_, pipeline_optimizer.best_score_ , end_time_model , results_splits, results_metrics, grid.best_estimator_ 



  

def diagnostique_apprentissage(model,name_model,metric):   
    
    #model.fit(X_train, y_train)
    
    N, train_score, val_score = learning_curve(model, X_train, y_train,
                                              cv=stratified_Kfold, scoring=metric,
                                               train_sizes=np.linspace(0.1, 1, 10))
    
    
    plt.figure(figsize=(12, 8))
    plt.title(f"Diagnostic d'apprentissage pour {name_model} et {metric} ")
    plt.plot(N, train_score.mean(axis=1), label=f'train score {metric}')
    plt.plot(N, val_score.mean(axis=1), label=f'validation score {metric}')
    plt.xlabel('Training examples')
    plt.ylabel(f'Score {metric}')
    plt.legend()

    print("-----------------------------------------------------------------------------------------------------")

from sklearn.model_selection import learning_curve
def diagnostic_apprentissage_learning_curve(model, X_train, y_train, name, scoring):
  
  
  N, train_score, val_score = learning_curve(model, X_train, y_train,
                                              cv=5, scoring=scoring,
                                               train_sizes=np.linspace(0.25, 1.0, 10))
    
    
  plt.figure(figsize=(12, 8))
  plt.title(f'Learning curve {name}', size=16)
  plt.plot(N, train_score.mean(axis=1), label='train score')
  plt.plot(N,val_score.mean(axis=1), label='validation score')
  plt.xlabel('Training examples')
  plt.ylabel('Fonction cout métier')
  plt.legend()




def print_metrics(y_train,y_train_preds,y_test, y_test_preds):
  
  print(colored('train fonction cout métier : {}','blue').format(costs(y_train, y_train_preds)))
  print(colored('train f1: {}','blue').format(f1_score(y_train, y_train_preds)))
  print(colored('train roc_auc_score: {}','blue').format(roc_auc_score(y_train, y_train_preds)))
  print(colored('train accuracy_score: {}','blue').format(accuracy_score(y_train, y_train_preds)))
  print(colored('train recall_score: {}','blue').format(recall_score(y_train, y_train_preds)))
  print(colored('train precision_score : {}','blue').format(precision_score(y_train, y_train_preds)))
  print()

  print(colored('test fonction cout métier : {}','magenta').format(costs(y_test, y_test_preds)))
  print(colored('test f1: {}','magenta').format(f1_score(y_test, y_test_preds)))
  print(colored('test roc_auc_score: {}', 'magenta').format(roc_auc_score(y_test, y_test_preds)))
  print(colored('test accuracy_score: {}','magenta').format(accuracy_score(y_test, y_test_preds)))
  print(colored('test recall_score: {}','magenta').format(recall_score(y_test, y_test_preds)))
  print(colored('test precision_score : {}','magenta').format(precision_score(y_test, y_test_preds)))


def matrix_confusion(y_test, y_test_preds):
  
  # Visualisation de la matrice confusion  
  cf_matrix=(confusion_matrix(y_test, y_test_preds))
  
  plt.figure(figsize=(8,6))
  plt.title(f'La matrice de confusion pour {name}', size=20, y=1.1)
  group_names = ['True neg', 'False pos', 'False neg', 'True pos']
  group_counts = ['{0:0.0f}'.format(value) for value in
                  cf_matrix.flatten()]
  group_percentages = ['{0:.2%}'.format(value) for value in
                       cf_matrix.flatten()/np.sum(cf_matrix)]
  labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]
  labels = np.asarray(labels).reshape(2,2)
  sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Reds')
  
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
  plt.show()

def threshold_selection(y_test, y_prob):
    
    """
    Méthode utilisée pour calculer le seuil par rapport à la fonction cout métier .

    Paramètres:
    -----------------
        df(pandas.DataFrame) : jeu de données à analyser
        y_test (comme un tableau 1d) : valeurs vraies
        y_prob (1d array-like) : valeurs de probabilité prédites
        
    Retour:
    -----------------
        seuil (flottant) : meilleure valeur de seuil
        Seuil (image) : Tracez le seuil et le meilleur score
    """
    
    thresholds = np.arange(0, 1, 0.001)
    scores = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype("int")
        score = costs(y_test, y_pred)
        scores.append(score)
        
    scores = np.asarray(scores)
    ix = np.argmax(scores)
    
    best_threshold, best_score = (thresholds[ix], round(scores[ix], 3))
    
    print("Best results:")
    print("Best Threshold:\t", best_threshold)
    print("Best Score:\t\t", best_score)
    print("\n")
    
    plt.subplots(1, figsize=(6, 5))
    plt.plot(thresholds, scores, lw=1)
    plt.axvline(best_threshold, ls="--", lw=1, c="r")
    plt.title("Threshold")
    plt.xlabel("proba threshold")
    plt.ylabel("Score performed")
    plt.show()
    
    return best_threshold


def evaluation_threshold_solvabilite(model, name_model):
    
    ypred_train = model.fit(X_train, y_train)
    ypred_test = model.predict(X_test)

    # Définir le best threshold
    y_prob = model.predict_proba(X_test)
    threshold = 0.356# definir threshold ici
    # Nouveau ypred_test avec threshold définis 
    ypred_test = [1 if y_prob[i][1]> threshold else 0 for i in range(len(y_prob))]
    ypred_test
    
    #print confusion_matrix
    print(confusion_matrix(y_test, ypred_test))
    
    # Visualisation de la matrice confusion  
    cf_matrix=(confusion_matrix(y_test, ypred_test))
   
    plt.figure(figsize=(14,12))
    ax = plt.axes()
    plt.title(f'La matrice de confusion pour {name_model}', size=20, y=1.1)
    group_names = ['TRUE NEG (client rembourse et crédit accepté par le modèle)', 'FALSE POS (ERROR 1 perte opportunité pour la banque)', 'FALSE NEG (ERROR 2 perte argent pour la banque)', 'TRUE POS (Client pas rembourser et crédit refusé par modèle)']
    group_counts = ['{0:0.0f}'.format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in
                         cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
  
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Reds', xticklabels=["Remboursé(0 = négatif)", "Non remboursé(1 = positif)"], yticklabels=['remboursé(0 = négatif)', 'Non remboursé(1 = positif)'])
    
    plt.ylabel('Actual label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.show()
    print( )
    print(classification_report(y_test, ypred_test))

    

    
      
      
    y_prob = (y_prob >= threshold).astype(int)
    y_prob = y_prob[:, 1] 
    # ROC_CURVE 
    fpr = dict()
    tpr = dict()
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test, y_prob) #roc_curve(y_test, y_prob)#prob_preds.ravel
    auc = round(roc_auc_score(y_test,y_prob), 4)
    
    plt.subplots(1, figsize=(6, 6))
    plt.title("Receiver Operating Characteristic")
    plt.plot(fpr["micro"], tpr["micro"], label="LGBM, AUC="+str(auc))
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend()
    plt.show()
    
    #precision_recall_curve  
    precision, recall, threshold = precision_recall_curve(y_test, y_prob)
    
    plt.plot(threshold, precision[:-1], label='precision')
    plt.plot(threshold, recall[:-1], label='recall')
    plt.legend()
    
    print('------------------------------------------------------------------------------------------------------------------')


from sklearn.inspection import permutation_importance


def plot_features_importance(estimator, name_model, X_train, y_train, scoring=None):
    """
    Generate 1 plots: 
        1. The importance by feature
    
    Parameters
    -----------------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.
        
    name_model : str
        Name of the model as title for the chart.     
        
    X_train : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y_train : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning. 
        
    scoring : string, callable or None, default=None
        Scorer to use. It can be a single string or a callable. 
        If None, the estimator’s default scorer is used. 
        
    Returns:
    -----------------
        None. 
        Plot the graph. 
        
    """     
    # Get the importance by feature
    results = permutation_importance(estimator, X_train, y_train, scoring=scoring)
    
    # Making a dataframe to work easily
    df_importance = pd.DataFrame({
                        "Feature" : X_train.columns,
                        "Importance" : results.importances_mean
                    })
    
    # Sorting by importance before plotting
    df_importance = df_importance.sort_values("Importance")
    
    # Initializing figure    
    fig = plt.subplots(figsize=(12, 8))
    
    plot = sns.barplot(data=df_importance, y=df_importance["Feature"], x=df_importance["Importance"])
    
    plt.title(name_model + " Features Importance", fontdict={ "fontsize": 16, "fontweight": "normal" })
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()



def plot_features_permutation_importance(estimator, name_model, X_train, y_train, scoring=None):
    """
    Generate 1 plots: 
        1. The importance by feature
    
    Parameters
    -----------------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.
        
    name_model : str
        Name of the model as title for the chart.     
        
    X_train : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y_train : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning. 
        
    scoring : string, callable or None, default=None
        Scorer to use. It can be a single string or a callable. 
        If None, the estimator’s default scorer is used. 
        
    Returns:
    -----------------
        None. 
        Plot the graph. 
        
    """     
    # Get the importance by feature
    results = permutation_importance(estimator, X_train, y_train, scoring=scoring)
    
    # Making a dataframe to work easily
    df_importance = pd.DataFrame({
                        "Feature" : X_train.columns,
                        "Importance" : results.importances_mean
                    })
    
    # Sorting by importance before plotting
    df_importance = df_importance.sort_values("Importance")
        
    # Initializing figure    
    fig = plt.subplots(figsize=(12, 8))
        
    plot = sns.barplot(data=df_importance, y=df_importance["Feature"], x=df_importance["Importance"])
        
    plt.title(name_model +"Features Importance", fontdict={ "fontsize": 16, "fontweight": "normal" })
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    








    # Sorting by importance before plotting
    df_importance = df_importance.sort_values("Importance")
    li_feature_not_0 = list(df_importance[df_importance['Importance']!=0]['Feature'])
    
    li_feature_importance = list(df_importance[df_importance['Importance']!=0]['Importance'])
    # Making a dataframe to work easily
    df_importance_1 = pd.DataFrame({ "Feature" : li_feature_not_0,
                            "Importance" : li_feature_importance})
    
    
    # Initializing figure    
    fig = plt.subplots(figsize=(12, 8))
    
    plot = sns.barplot(data=df_importance_1, y=df_importance_1 ["Feature"], x=df_importance_1 ["Importance"])
    
    plt.title(name_model + "Features Importance différentes de 0", fontdict={ "fontsize": 16, "fontweight": "normal" })
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    print(f"Il y a {len(li_feature_not_0)} features qui sont égale à O")
    print('Les variables qui sont différentes de 0 :',li_feature_not_0)