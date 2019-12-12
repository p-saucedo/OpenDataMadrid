from engine import Engine as eng
import numpy as np
eRL = eng.RegresionLogistica()
eRF = eng.RandomForest()
eKD = eng.KernelDensity()

#eRL.validate()
eRF.validate(1, folds=10)
#eKD.setData()

#p = np.array([40.3731, -3.73081]).reshape(1,-1)
#print(eRF.predict_value(p))
#eKD.plotData()
#eKD.run()