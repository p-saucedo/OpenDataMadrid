import os
import pandas


basedir = os.path.dirname(os.path.abspath(__file__))
datasets = []
columns = ['HORA','CALLE', 'NUMERO', 'LESIVIDAD*']
res_dataset = basedir + '/datasets/complete_dataset.csv'

# 2019 #
d1_file = basedir + '/datasets/AccidentesBicicletas_2019.csv'
d1 = pandas.read_csv(d1_file, index_col=None, header=0, sep=';', usecols = columns)
datasets.append(d1)
# 2018 #
d2_file = basedir + '/datasets/AccidentesBicicletas_2018.csv'
d2 = pandas.read_csv(d2_file, index_col=None, header=0, sep=',', usecols = columns)
datasets.append(d2)
# 2017 #
d3_file = basedir + '/datasets/AccidentesBicicletas_2017.csv'
d3 = pandas.read_csv(d3_file, index_col=None, header=0, sep=',', usecols = columns)
datasets.append(d3)
# 2016 #
d4_file = basedir + '/datasets/AccidentesBicicletas_2016.csv'
d4 = pandas.read_csv(d4_file, index_col=None, header=0, sep=',', usecols = columns)
datasets.append(d4)
# 2015 #
d5_file = basedir + '/datasets/AccidentesBicicletas_2015.csv'
d5 = pandas.read_csv(d5_file, index_col=None, header=0, sep=',', usecols = columns)
datasets.append(d5)


result = pandas.concat(objs = datasets, ignore_index = True)


result.to_csv(path_or_buf=res_dataset, sep=';')
