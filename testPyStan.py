import pystan
import numpy as np
import pickle

file= 'stan-models/base.stan'

sm = pystan.StanModel(file = file)


with open('stan-models/base.pkl', 'wb') as f:
    pickle.dump(sm, f)

print('Model Compiled and saved...')