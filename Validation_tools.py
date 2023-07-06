import numpy as np
import pandas as pd
import math
import tqdm
import itertools

from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

def standard(X, features_to_standard, train_index, test_index):
    '''
    Стандартизирует отдельно подтаблицу с номерами строк из train_index и столбцами из features_to_standard, 
    отдельно с номерами строк из test_index и столбцами из features_to_standard
    
    Параметры:
        X (pandas.DataFrame): Таблица, подлежащая стандартизации

        features_to_standard (list, tuple, pandas.RangeIndex, numpy.ndarray): Список столбцов, содержащих данные, 
                                                                              которые будут стандартизированы

        train_index (list, numpy.ndarray, tuple): Список индексов тренировочного набора 
        
        test_index (list, numpy.ndarray, tuple): Список индексов тестового набора
    
    '''
    EPS = 1e-6
        
    train_mean = X.iloc[train_index][features_to_standard].mean(axis = 0)
    train_std = X.iloc[train_index][features_to_standard].std(axis = 0) + EPS
    X.loc[train_index, features_to_standard] = (X.loc[train_index, features_to_standard] - train_mean) / train_std

    if test_index is not None:
        test_mean = X.iloc[test_index][features_to_standard].mean(axis = 0)
        test_std = X.iloc[test_index][features_to_standard].std(axis = 0) + EPS
        X.loc[test_index, features_to_standard] = (X.loc[test_index, features_to_standard] - test_mean) / test_std
        
        return train_mean, train_std, None, None
    
    return train_mean, train_std, test_mean, test_std

def inverse_standard(X, features_to_standard, train_index, test_index,
                     train_mean, train_std, test_mean, test_std):
    '''
    Возвращает данные к исходному масштабу
    
    Параметры: 
        X (pandas.DataFrame): Таблица, подлежащая упомянутой трансформации

        features_to_standard (list, tuple, pandas.RangeIndex, numpy.ndarray): Список столбцов, содержащих данные, 
                                                                              которые были стандартизированы

        train_index (list, numpy.ndarray, tuple): Список индексов тренировочного набора 

        test_index (list, numpy.ndarray): Список индексов тестового набора

        train_mean (list, numpy.ndarray): Средние значения признаков из features_to_standard до стандартизации

        train_std (list, numpy.ndarray): Стандартные отклонения признаков из features_to_standard до стандартизации

        test_mean (list, numpy.ndarray): Средние значения признаков из features_to_standard до стандартизации

        test_std (list, numpy.ndarray): Стандартные отклонения признаков из features_to_standard до стандартизации
    
    '''
    X.loc[train_index, features_to_standard] = X.loc[train_index, features_to_standard] * train_std + train_mean
    
    if test_mean is not None and test_std is not None:
        X.loc[test_index, features_to_standard] = X.loc[test_index, features_to_standard] * test_std + test_mean
        

def teach_and_predict(X, y, train_index, test_index, features_to_standard, model = None, 
                                        to_standard = True, with_pca = True, pca_standard = False, 
                                        n_comps = 10, score_func = accuracy_score, **params):
    '''
    Стандартизует отдельно обучающие блоки и отдельно тестовый блок. После обучает указанную модель 
    с указанными параметрами, считает скор и возвращает данные к исходному масштабу. По сути
    то же, что и teach_and_predict, но принимает данные в удобном во время поиска по сетке формате и 
    при необходимости применяет метод данных компонент
    
    Параметры: 
        X (pandas.DataFrame): Таблица с данными (без целевой переменной)

        y (pandas.Series): Список значений целевой переменной
        
        train_index (list, numpy.ndarray, tuple): Список индексов тренировочного набора 

        test_index (list, numpy.ndarray): Список индексов тестового набора

        features_to_standard (list, tuple, pandas.Index): Список признаков, которые подлежат стандартизации

        model: Что угодно, что имеет методы fit и predict 

        to_standard (bool): Если True, X_test и X_train будут стандартизованы, а в конце приведены к изначальному состоянию

        with_pca (bool): Если True, применяется метод главных компонент 

        pca_standard (bool): Если True, стандартизирует данные после применения метода главных компонент. 
                             Игнорируется, если with_pca == False

        n_comps (int): Количество главных компонент. Игнорируется, если with_pca == False
        
        score_func (func): Метрика качества модели

        **params (dict): Словарь, в котором ключи - это названия гиперпараметров модели, а значения - 
                         список гиперпараметров, которые нужно перебрать
                             
    Возвращаемое значение:
        Точность выбранной модели на тестовом наборе
    '''
    
    if to_standard:
        train_mean, train_std, test_mean, test_std = standard(X, features_to_standard, 
                                                              train_index, test_index)
    
    if with_pca:
        X_before_pca = X.copy()
        train_pca = PCA(n_components = n_comps)
        test_pca = PCA(n_components = n_comps)
        X = pd.DataFrame(0.0, index = np.arange(len(X_before_pca)), 
                         columns = [f'PC{i}' for i in range(1, n_comps + 1)])
        
        X.iloc[train_index] = train_pca.fit_transform(X_before_pca.iloc[train_index])
        X.iloc[test_index] = test_pca.fit_transform(X_before_pca.iloc[test_index])
        pca_columns = [f'PC{i}' for i in range(1, n_comps + 1)]
        X = pd.DataFrame(X, columns = pca_columns)
        
        if pca_standard:
            standard(X, pca_columns, train_index, test_index)

    model = model(**params)
    model.fit(X.iloc[train_index], y.iloc[train_index])
    y_pred = model.predict(X.iloc[test_index])
    
    if with_pca:
        X = X_before_pca
        
    if to_standard:
        inverse_standard(X, features_to_standard, train_index, test_index,
                         train_mean, train_std, test_mean, test_std)

    return score_func(y.iloc[test_index], y_pred)

def GridSearch(X, y, grid_params, features_to_standard, 
              comparator, rand_state, n_splits = 10, folding_func = StratifiedKFold, 
              to_standard = True, with_pca = True, pca_standard = True, worst_metric = -100,
              n_comps = 10, model = None):
    '''
    Проводит поиск лучших гиперпараметров по указанной сетке. 
    
    Параметры: 
        X (pandas.DataFrame): Таблица с данными (без целевой переменной)

        y (pandas.Series): Список значений целевой переменной
        
        grid_params (dict): Словарь, описывающий все возможные гиперпараметры

        features_to_standard (list, tuple, pandas.Index): Список признаков, которые подлежат стандартизации
        
        comparator (function): Функция, сравнивающая два массива. Должна возвращать отрицательное число,
                               если первый список лучше второго в смысле выбранной метрики 
                               
        rand_state (int): Ключ генератора случайных чисел для sklearn.StratifiedKFold
        
        n_splits (int): Количество блоков, по которым будет проходить кросс-валидация
        
        folding_func (fund): Функция, отвечающая за разбивку выборки на K блоков. 
                             KFold для задачи регрессии и StratifiedKFold для классификации
        
        worst_metric (float, int): Значение выбранной метрики, хуже которой быть не может (в смысле выбранной
                                   метрики и выбранного способа сравнения моделей).
                                   Иначе говоря, аналог минус единицы при поиске максимального значения 
                                   в массиве положительных чисел

        to_standard (bool): Если True, X_test и X_train будут стандартизованы, а в конце приведены к изначальному состоянию

        with_pca (bool): Если True, применяется метод главных компонент 

        pca_standard (bool): Если True, стандартизирует данные после применения метода главных компонент. 
                             Игнорируется, если with_pca == False

        n_comps (int): Количество главных компонент. Игнорируется, если with_pca == False

        model: Что угодно, что имеет методы fit и predict 
    
    Возвращаемое значение:
        Кортеж из двух элементов. Первый из них - это словарь лучших гиперпараметров, 
        второй - значения точности модели на валидации
    '''
    cross_val = folding_func(n_splits = n_splits, shuffle = True, random_state = rand_state)

    best_params = {}
    best_scores = np.array([worst_metric] * n_splits) # Инициализируем массив, хуже которого
                                                      # в смысле выбранной метрики быть не может

    # число возможных комбинаций гиперпараметров
    n_combs = math.prod([len(i) for i in grid_params.values()])
    with tqdm(total = n_combs) as psbar: 
        for comb in itertools.product(*grid_params.values()):
            cur_scores = np.array([])
            cur_params = { list(grid_params.keys())[i]: comb[i] for i in range(len(comb)) }
            psbar.set_description(f'Processing {cur_params}')
    
            for train_index, test_index in cross_val.split(X, y):
                score = teach_and_predict(X, y, train_index, test_index, features_to_standard,
                                         model = model, to_standard = to_standard,
                                         with_pca = with_pca, pca_standard = pca_standard,
                                         n_comps = n_comps, **cur_params)
                cur_scores = np.append(cur_scores, score)

            if comparator(best_scores, cur_scores) < 0:
                best_scores = cur_scores
                best_params = cur_params

            psbar.update(1)

    return best_params, best_scores