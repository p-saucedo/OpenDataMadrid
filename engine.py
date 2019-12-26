import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import math
import matplotlib.cm as cm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.mixture import GaussianMixture
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import model_selection
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
import sklearn.metrics
from sklearn.neighbors import KernelDensity
from imblearn.over_sampling import *
import warnings
from watcher import get_logger
from sklearn.model_selection import GridSearchCV

logger = get_logger(__name__)

basedir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(basedir, 'out_csv')
filepath = os.path.join(data_dir, 'geo_out.csv')

class Particion:
    def __init__(self):
        self.X_train = None
        self.Y_train = None
        self.X_validation = None
        self.Y_validation = None

class Engine:

    class RegresionLogistica:
        model = None
        classes = None

        def __init__(self):
            pass

        def setData(self):
            logger.info("Reading {} file for the sake of training, testing and rendering Logistic Regression".format(filepath))
            df = pd.read_csv(filepath, delimiter=';')
            X = np.array(df[["latitude","longitude"]])
            y = np.array(df[["LESIVIDAD*"]].fillna(14)) # 14 significa lo mismo que NaN: sin asistencia sanitaria
           
            self.classes = np.unique(y)

            ct = ColumnTransformer(
                [
                    ('one_hot_encoding',OrdinalEncoder(categories='auto'), [0])
                ],
                remainder = "passthrough"

            )

            y = ct.fit_transform(y) # Lesividad codificada en OneHotEncoder
            
            X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=0.3, random_state=None)
            logger.info("Data partition for Logistic Regression conducted.")

            #sc = StandardScaler()
            #X_train = sc.fit_transform(X_train)
            #X_validation = sc.transform(X_validation)
    
            return X_train, X_validation, Y_train, Y_validation
           

        def train(self, X_train, Y_train):
           
            self.model = linear_model.LogisticRegression(solver='liblinear',
                                                            multi_class='auto',
                                                            max_iter=2000)
            logger.info("Fitting Logistic Regression model to training dataset.")
            self.model.fit(X_train, Y_train.ravel())

        def predict(self, X_validation, Y_validation):
            predictions = self.model.predict(X_validation)
    
            return accuracy_score(Y_validation, predictions)

        def validate(self):
            X_train, X_validation, Y_train, Y_validation, = self.setData()

            self.train(X_train, Y_train)
            acc = self.predict(X_validation, Y_validation)
            logger.info("LogisticRegression accuracy over validation dataset: {}".format(acc*100))
            print("LogisticRegression acc: {}".format(acc*100))

    class RandomForest:
        model = None
        best_model = None
        acc = 0
        def __init__(self):
            pass

        def setData(self, folds, file):
            particiones = []
            logger.info("Reading {} file for the sake of training, testing and rendering Random Forest".format(filepath))
            df = pd.read_csv(file, delimiter=';')
            X = np.array(df[["latitude","longitude"]])
            y = np.array(df[["LESIVIDAD*"]].fillna(14)) # 14 significa lo mismo que Nan: sin asistencia sanitaria

            kf = KFold(n_splits=folds)
            X, y = self.resize(X,y)
            for train_index, test_index in kf.split(X):
                p = Particion()
                X_train, X_validation = X[train_index], X[test_index]
                Y_train, Y_validation = y[train_index], y[test_index]
                p.X_train = X_train
                p.Y_train = Y_train
                p.X_validation = X_validation
                p.Y_validation = Y_validation

                particiones.append(p)
            logger.info("Data partition for Random Forest with Cross-Validation conducted.")

            return particiones

        def resize(self, X, y):
            # Class to perform random over-sampling. Object to over-sample the minority 
            # class(es) by picking samples at random with replacement.
            logger.info("Oversampling imbalanced dataset.")
            sm = RandomOverSampler(random_state=None,
                                    sampling_strategy='auto')

            X_res, y_res = sm.fit_resample(X,y.ravel())


            return X_res, y_res

        def train(self, X_train, Y_train):
            logger.info("Training and fitting RandomForestClassifier.")
            """self.model = RandomForestClassifier(n_estimators = 5, 
                                                random_state = None, 
                                                class_weight='balanced')"""

            parameter_candidates = [
                {
                'n_estimators': [3, 6, 9],
                'min_samples_split': [2,4,6],
                'class_weight': ['balanced', 'balanced_subsample']
                }
            ] 
            self.model = GridSearchCV(estimator=RandomForestClassifier(), 
                                    param_grid = parameter_candidates,
                                    n_jobs= -1)
            
            self.model.fit(X_train, Y_train)

        def predict(self, X_validation, Y_validation):
            logger.info("Testing model overvalidation dataset.")
            predictions = self.model.predict(X_validation)
            return accuracy_score(Y_validation, predictions)

        def validate(self, f, folds = 5):
            # TODO: GridSearch of Hyperparameters to discover the most accurate one.
            # Example: https://github.com/eblancoh/Cube11Paths/blob/master/Auth_Engine/ml_engine/gridsearch_logregr.py
            logger.info("Validating dataset over {} folds.".format(folds))
            particiones = self.setData(folds, f)

            for i in range(folds):
                p = particiones[i]
                self.train(p.X_train, p.Y_train.ravel())
                acc = self.predict(p.X_validation, p.Y_validation.ravel())
                if self.best_model == None:
                    self.best_model = self.model
                    self.acc = acc
                else:
                    if (acc > self.acc):
                        self.best_model = self.model
                        self.acc = acc

            logger.info("Best model has an accuracy of {:.2f}%".format(self.acc*100))
            

        def predict_value(self, X_test):
            return int(self.best_model.predict(X_test)[0])

    class KernelDensity():
        model = None

        def __init__(self):
            pass
            

        def setData(self):
            df = pd.read_csv(filepath, delimiter=';')
            
            X = np.array(df[["latitude","longitude"]])

            self.model = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
            print(self.model.score_samples(X))
            #self.plotData(self.model.score_samples(X))

        def plotData(self):

            df = pd.read_csv(filepath, delimiter=';')
            
            X = np.array(df[["latitude","longitude"]])
            
            data = fetch_species_distributions()

            # Get matrices/arrays of species IDs and locations
            latlon = np.vstack([data.train['dd lat'],data.train['dd long']]).T
            latlon2 = np.vstack([X[:,0],X[:,1]]).T
            print(latlon)
            print(latlon2)
            species = np.array([d.decode('ascii').startswith('micro')
                                for d in data.train['species']], dtype='int')
          
            xgrid, ygrid = construct_grids(data)
            print(xgrid)
            print(ygrid)
            # plot coastlines with basemap
            m = Basemap(projection='cyl', resolution='c',
                        llcrnrlat=ygrid.min(), urcrnrlat=ygrid.max(),
                        llcrnrlon=xgrid.min(), urcrnrlon=xgrid.max())
            m.drawmapboundary(fill_color='#DDEEFF')
            m.fillcontinents(color='#FFEEDD')
            m.drawcoastlines(color='gray', zorder=2)
            m.drawcountries(color='gray', zorder=2)

            # plot locations
            m.scatter(latlon[:, 1], latlon[:, 0], zorder=3,
                    c=species, cmap='rainbow', latlon=True)

            X, Y = np.meshgrid(xgrid[::5], ygrid[::5][::-1])
            land_reference = data.coverages[6][::5, ::5]
            land_mask = (land_reference > -9999).ravel()
            xy = np.vstack([Y.ravel(), X.ravel()]).T
            xy = np.radians(xy[land_mask])

            # Create two side-by-side plots
            fig, ax = plt.subplots(1, 2)
            fig.subplots_adjust(left=0.05, right=0.95, wspace=0.05)
            species_names = ['Bradypus Variegatus', 'Microryzomys Minutus']
            cmaps = ['Purples', 'Reds']

            for i, axi in enumerate(ax):
                axi.set_title(species_names[i])
                
                # plot coastlines with basemap
                m = Basemap(projection='cyl', llcrnrlat=Y.min(),
                            urcrnrlat=Y.max(), llcrnrlon=X.min(),
                            urcrnrlon=X.max(), resolution='c', ax=axi)
                m.drawmapboundary(fill_color='#DDEEFF')
                m.drawcoastlines()
                m.drawcountries()
                
                # construct a spherical kernel density estimate of the distribution
                kde = KernelDensity(bandwidth=0.03, metric='haversine')
                kde.fit(np.radians(latlon[species == i]))

                # evaluate only on the land: -9999 indicates ocean
                Z = np.full(land_mask.shape[0], -9999.0)
                Z[land_mask] = np.exp(kde.score_samples(xy))
                Z = Z.reshape(X.shape)

                # plot contours of the density
                levels = np.linspace(0, Z.max(), 25)
                axi.contourf(X, Y, Z, levels=levels, cmap=cmaps[i])


        def kde(self,x,y,ax):
            xy = np.vstack([x,y])
            print(xy)
            d = xy.shape[0]
            n = xy.shape[1]

            bw = (n * (d + 2) / 4.)**(-1. / (d + 4))

            kde = KernelDensity(bandwidth=bw, metric='euclidean', kernel='gaussian', algorithm='ball_tree')  

            kde.fit(xy.T)

            xmin = x.min()
            xmax = x.max()
            ymin = y.min()
            ymax = y.max()

            X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            positions = np.vstack([X.ravel(), Y.ravel()])

            Z = np.reshape(np.exp(kde.score_samples(positions.T)), X.shape)

            ax.imshow(np.rot90(Z), cmap=plt.cm.inferno,extent=[xmin, xmax, ymin, ymax])
            
            ax.scatter(x, y, c='k', s=5, edgecolor='')



        def run(self):
            fig, axarr = plt.subplots()
            #fig.subplots_adjust(left=0.11, right=0.95, wspace=0.0, bottom=0.18)

            ax = axarr
            df = pd.read_csv(filepath, delimiter=';')
            X = np.array(df['latitude'])
            Y = np.array(df['longitude'])

            self.kde(X,Y,ax)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('scikit-learn')

            ax.set_xlim((40.3,40.6))
            ax.set_ylim((-3.9,-3.5))

            plt.show()


    class KDE():
        model = None
        def __init__(self):
            X = self.setData(file = filepath)
            self.fit(X)


        def setData(self, file):
            logger.info("Reading {} file for the sake of training, testing and rendering KDE".format(file))
            df = pd.read_csv(file, delimiter=';')
            X = np.array(df[["latitude","longitude"]])

            return X

        def fit(self, X):

            bandwidths = 10 ** np.linspace(0, 2, 100)
            parameter_candidates = [
                {
                'bandwidth': bandwidths,
                'kernel': ['gaussian', 'tophat', 'linear']
                }
            ] 
  
            grid = GridSearchCV(estimator=KernelDensity(), 
                                param_grid = parameter_candidates,
                                n_jobs= -1).fit(X)
            
            self.model = KernelDensity(bandwidth = grid.best_params_.get('bandwidth'),
                                        kernel = grid.best_params_.get('kernel'))
            
            self.model.fit(X)
            logger.info("KDE model is fitted")

        def predict_proba(self, X):
            logprobs = np.array(self.model.score_samples(X)).T
            result = np.exp(logprobs)
            return result

        def predict_value(self, X):
            prob = self.predict_proba(X)
            print('Probabilidad de accidente es de {}%'.format(np.around(prob*100,4)[0]))
            return np.around(prob*100,4)

        def validate(self):
            X = self.setData(filepath)

            self.fit(X)
            probas = self.predict_proba(X)

            new_value = np.array([40.42551, -3.69191])
            y = self.predict(new_value.reshape(1,-1))

            print('Punto a predecir: {}'.format(new_value))
            print('Probabilidad de accidente es de {}%'.format(np.around(y*100,4)[0]))