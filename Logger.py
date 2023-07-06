import sys
import math
import numpy as np

class Logger:
    def __init__(self, estimator = 'XGBClassifier', eda_info = 'Нет', 
                 val_info = '10 блоков, без стандартизации, без PCA',
                 grid = None, best_params = None, comp_info = 'Сравнивались средние значения',
                 scores = None, outfile = r'logs\log_1.txt'):
        
        '''
        
        Параметры:
            estimator (str): Название модели
            
            eda_info (str): Информация о предобработке данных. Например, 'предварительно с помощью некоторого
                            алгоритма были удалены выбросы'
            
            val_info (str): Информация о валидации: кол-во блоков, использование стандартизации и т.д
            
            grid (dict): Сетка, по которой перебирались гиперпараметры 
            
            best_params (dict): Набор параметров, показавший себя лучше остальных 
            
            comp_info (str): Информация о том, как в процессе валидации сравнивались модели с разными параметрами
            
            scores (numpy.ndarray, tuple): Набор значений метрики лучшей модели по время валидации или кортеж вида
                                         (среднее значение, стандартное отклонение)
            
            outfile (str): Путь к файлу, в который вся информация будет записана 
        
        Возвращаемое значение:
            Объект класса 
        
        '''
        
        self.estimator = estimator
        self.eda_info = eda_info
        self.val_info = val_info
        self.grid = grid
        self.best_params = best_params
        self.comp_info = comp_info
        self.scores = scores
        self.outfile = outfile
        
        # Частенько сетка состоит из одной точки, т.е просто проводится валидация с помощью функции
        # для поиска по сетке, ибо я ленивая жопа и не написал функцию просто для валидации
        # (что даже справедливо, ибо зачем)
        # В общем, в таком случае будет лишним выводить и сетку, и лучшие параметры
        # Так что заводим индикатор того, что сетка состоит из одной точки
        
                               # Значения - это списки
        self.only_validation = ( [type(i) for i in self.grid.values()] == [list] * len(self.grid) 
                                 and (math.prod([len(i) for i in self.grid.values()]) == 1) )
        
    def scores_info(self, num_of_tabs = 0):
        '''
        Формирует информацию о метрике лучшей модели. Если информация о метрике представляет из себя
        набор её значений для валидационных блоков, выводится более подробная информация, чем в случае
        когда дано только среднее значение и стандартное отклонение
        
        Параметры:
            num_of_tabs (int): Количество табуляций перед каждой строкой
        
        Возвращаемое значение:
            Строкa с информацией о метрике 
        
        '''
        tabs = '\t' * num_of_tabs
        info = ''
        if type(self.scores) == np.ndarray:
            info += f'{tabs}Точность: {self.scores.mean()} +- {self.scores.std()} (mean +- std)\n'
            info += f'{tabs}Худшая точность: {self.scores.min()}\n'
            info += f'{tabs}Медианная точность: {np.median(self.scores)}\n'
            info += f'{tabs}Лучшая точность: {self.scores.max()}\n'
        else:
            info += f'{tabs}Точность: {self.scores[0]} +- {self.scores[1]} (mean +- std)\n'

        return info
    
    def print_grid(self, stream = sys.stdout):
        '''Выводит информацию о сетке гиперпараметров в указанный поток'''
        stream.write('Сетка:\n')
        for key, value in self.grid.items():
            stream.write(f'\t{key}: {value}\n')
                
    def print_best_params(self, stream = sys.stdout):
        '''Выводит информацию о лучшем наборе гиперпараметров в указанный поток'''
        stream.write('Лучший набор параметров:\n')
        if self.only_validation:
            stream.write('\tСетка состоит из одной точки\n')
            return
        
        for key, value in self.best_params.items():
            stream.write(f'\t{key}: {value}\n')
    
    def make_note(self, print_scores_info = True):
        '''Записывает информацию в файл
        
        Параметры:
            print_scores_info (bool): Если True, вывод информацию о лучшей модели ещё и в sys.stdout
                                      Что (не) происходит в противном случае догадаться не сложно
        
        Возвращаемое значение:
            None
        
        '''
        with open(self.outfile, 'a') as out:
            out.write(f'Модель:\n\t{self.estimator}\n')
            out.write(f'Особенности предобработки данных:\n\t{self.eda_info}\n')
            out.write(f'Особенности валидации:\n\t{self.val_info}\n')
            self.print_grid(stream = out)
            self.print_best_params(stream = out)
            out.write(f'Информация о методе сравнения моделей:\n\t{self.comp_info}\n')
            out.write(f'Информация о метрике:\n{self.scores_info(num_of_tabs = 1)}\n')
            out.write('\n\n' + '*' * 75 + '\n\n')
        
        if print_scores_info:
            print(self.scores_info())