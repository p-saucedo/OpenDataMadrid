
from engine import Engine as eng
import os
import pickle
from watcher import get_logger
from datetime import datetime, timedelta, date
import configparser
from get_coordinates import Get_Coordinates


config = configparser.ConfigParser()
logger = get_logger(__name__)

basedir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(basedir, 'out_csv')
fpath_val = os.path.join(data_dir, 'geo_out.csv')
fpath_conf = os.path.join(basedir, 'config.ini')
datasets_dir = os.path.join(basedir, 'datasets')



# Esta linea indica que dataset cargará toda la APP
fpath_dataset = os.path.join(datasets_dir, 'AccidentesBicicletas_2019.csv')


def save_model():
    eRF = eng.RandomForest()
    eRF.validate(fpath_val, folds=10)

    filename = basedir + '/model.pkl'
    pickle.dump(eRF.best_model, open(filename, 'wb+'))

    logger.info("RandomForestClassifier model trained and saved in {}".format(filename))

def load_model():
    filename = basedir + '/model.pkl'
    model = pickle.load(open(filename, 'rb'))

    logger.info("RandomForestClassifier model loaded successfully.")
    return model

def save_KDEmodel():
    eKDE = eng.KDE()
    eKDE.validate(fpath_val)

    filename = basedir + '/KDEmodel.pkl'
    pickle.dump(eKDE, open(filename, 'wb+'))

    logger.info("KernelDensity model trained and saved in {}".format(filename))

def load_KDEmodel():
    filename = basedir + '/KDEmodel.pkl'
    model = pickle.load(open(filename, 'rb'))

    logger.info("KernelDensity model loaded successfully.")
    return model

def transform_directions():
    Get_Coordinates(fpath_dataset)


def check_updates():
    # Si existe el fichero de configuracion
    update = 0
    today = datetime.now()
    try:
        with open(fpath_conf, 'r') as f:
            config.read(fpath_conf)
            last_upload_file = datetime.strptime(config.get('DATES','last_update'), '%d/%m/%Y')
            if today > (last_upload_file + timedelta(days=31)):
                logger.info("Data needs to be updated")
                transform_directions()
                save_model()
                save_KDEmodel()

                update = 1
            else:
                logger.info("Data is updated since {}".format(last_upload_file))

        if update == 1:
            f.close()
            with open(fpath_conf, 'w') as f:
                config.set('DATES', 'last_update',today.strftime('%d/%m/%Y'))
                config.write(f)
                logger.info("Data updated")

    # Si no existe
    except FileNotFoundError:
        with open(fpath_conf, 'w') as f:
            logger.info("Setting all data")
            config['DATES'] = {'last_update' : today.strftime('%d/%m/%Y')}
        
            config.write(f)
            transform_directions()
            save_model()
            save_KDEmodel()
    finally:
        f.close()
