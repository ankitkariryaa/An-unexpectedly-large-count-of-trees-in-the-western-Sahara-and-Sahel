#    Author: Ankit Kariryaa, University of Bremen

from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Nadam

# Optimezers; https://keras.io/optimizers/
adaDelta = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
adam = Adam(lr= 5.0e-05, decay= 0.0, beta_1= 0.9, beta_2= 0.999, epsilon= 1.0e-8)
nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
adagrad = Adagrad(lr=0.01, epsilon=None, decay=0.0)
