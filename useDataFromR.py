import json
import pystan
import pickle# Load model
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import arviz as az

countries = [
	"Denmark",
	"Italy",
	"Germany",
	"Spain",
	"United_Kingdom",
	"France",
	"Norway",
	"Belgium",
	"Austria", 
	"Sweden",
	"Switzerland"
]


# Read json with data
iFile = 'data/stan_data2.json'
with open(iFile) as json_file:
    dataStan = json.load(json_file)

sm = pickle.load(open('stan-models/base.pkl', 'rb'))
fit = sm.sampling(data = dataStan, iter=200, warmup=100,chains=4, thin=4, control= {'adapt_delta': 0.90})
out  = fit.extract()

prediction = out['prediction']
estimatedDeaths = out['E_deaths']
estimatedDeathsCF = out['E_deaths0']


plot_labels  = ["School Closure",
                 "Self Isolation",
                 "Public Events",
                 "First Intervention",
                 "Lockdown", 'Social distancing']

alpha = pd.DataFrame(out['alpha'], columns=plot_labels)


az.plot_forest(fit, var_names=['alpha'], credible_interval=0.9, combined=True )
plt.savefig('results/treeAlphaLog.pdf')

# ------------------------------------------------------------------------
# PLOT!!!
nCountries = len(countries)
for i in range(nCountries):
    N = dataStan['N'][i]

    #Casos!
    predictedCases= (prediction[:,0:N,i]).mean(axis=0)
    predicted_cases_li =np.quantile(prediction[:,0:N,i], q=0.025, axis= 0)
    predicted_cases_ui =np.quantile(prediction[:,0:N,i], q=0.975, axis= 0)


    fig, axs = plt.subplots(1)
    axs.plot(predictedCases, label = 'Prediction')
    axs.plot(predicted_cases_li, label = '0.250')
    axs.plot(predicted_cases_ui, label = '0.725')
    axs.legend()
    axs.set_title('Predicted case for ' + countries[i])
    fig.savefig('figures/python/cases_'+countries[i]+'.pdf')
    plt.close(fig)

    # Deaths!
    predicteDeaths = (estimatedDeaths[:,0:N,i]).mean(axis=0)
    predicteDeathsLi = np.quantile(estimatedDeaths[:,0:N,i], q=0.025, axis= 0)
    predicteDeathsUi = np.quantile(estimatedDeaths[:,0:N,i], q=0.975, axis= 0)

    fig, axs = plt.subplots(1)
    axs.plot(predicteDeaths, label = 'Prediction')
    axs.plot(predicteDeathsLi, label = '0.250')
    axs.plot(predicteDeathsUi, label = '0.725')
    axs.legend()
    axs.set_title('Predicted deaths for ' + countries[i])
    fig.savefig('figures/python/deaths_'+countries[i]+'.pdf')
    plt.close(fig)



print('hello')