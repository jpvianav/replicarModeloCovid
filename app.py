# Librerias standard
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import gamma 
import numpy as np
import pystan
import pickle

from matplotlib import pyplot as plt
import arviz as az


from datetime import datetime  
from datetime import timedelta  
	
# funciones propias
from services import getDataCol
from services import getDataWorld

from methods import poly
from methods import ecdf


def findIndex(d1):
	#Dia en el que se encontro el primer caso de contagio.
	index = d1.query('cases>0').index.tolist()[0] 
	#Encontrar el primer dia en el que las muertes acumuladas suman 10;
	d1['cumdeaths'] = d1['deaths'].cumsum()
	index1 = d1.query('cumdeaths>10').index.tolist()[0] 
	index2 = index1 - 30
	return index, index1, index2




#----------------------------------------------------------------------
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
	"Switzerland", 
	"Colombia"
]


measurements = ['schools_universities', 'travel_restrictions',
	   'public_events', 'sport', 'lockdown',
	   'social_distancing_encouraged', 'self_isolating_if_ill']

#Cantidad de elementos
nMeasurements = len(measurements)
nCountries = len(countries)


# ## Reading all data
d = getDataWorld() #Leer desde internet.
## get CFR
cfrByCountry = pd.read_csv("data/weighted_fatality.csv")
cfrByCountry['country'] = cfrByCountry.iloc[:,1]
cfrByCountry.loc[cfrByCountry.country == 'United Kingdom', 'country'] ='United_Kingdom' #Reemplazar el espacio por un guion bajo.
# Get Serial interval distribution  g with a mean of 6.5 days
serialInterval = pd.read_csv("data/serial_interval.csv")
#Get 
covariates = pd.read_csv('data/interventionsMod.csv') #Depure el contenido para eliminar la informacion no relevante.




# Converts strings to dates
for measi in measurements:
	covariates[measi] = pd.to_datetime(covariates[measi])

## making all covariates that happen after lockdown to have same date as lockdown
for measi in measurements:	
	idx = covariates[measi] > covariates['lockdown']
	covariates.loc[idx, measi] = covariates.loc[idx, 'lockdown']


p = covariates.shape[1] - 1
forecast = 0

# ------------------------------------------------------------------
#Forecaste length: This number include the days with data and the days to forecast. 
N2 = 80 # Increase this for a further forecast
# ------------------------------------------------------------------


dates = {}
reported_cases = {}
deaths_by_country = {}
#Calcular polinomios ortogonales.
x1x2 = poly(np.arange(1,N2+1), p=2)
# dict like data for stan model
stan_data = {
	'M': len(countries), 
	'N': [],
	'p': p,
	'x1': x1x2[:,0], 
	'x2': x1x2[:,1],
	'y': [],
	'covariate1': [],
	'covariate2': [],
	'covariate3': [],
	'covariate4': [],
	'covariate5': [],
	'covariate6': [],
	'covariate7': [],
	'deaths': [],
	'f': [],
	'N0': 6, # N0 = 6 to make it consistent with Rayleigh
	'cases': [],
	'LENGTHSCALE': 7, 
	'SI': serialInterval['fit'][:N2].values, 
	'EpidemicStart': []
}


for country in countries:
	
	# Get CFR by country, in case there's no value use the average
	CFR = cfrByCountry.weighted_fatality[cfrByCountry.country == country].values
	if CFR.shape[0] == 0:
		print('%s has not CFR, average value will be used'%country)
		CFR = cfrByCountry.weighted_fatality.mean()
	else:
		CFR = CFR[0]


	covariates1 = covariates[covariates.Country == country].loc[:, measurements]
	#Encontrar el primer dia con almenos un caso.
	d1 = pd.DataFrame(d[d['countriesAndTerritories'] == country])
	d1['date'] = pd.to_datetime(d1['dateRep'], format = '%d/%m/%Y')
	d1 = d1.sort_values(by='date').reset_index()	
	# First day with cases, first day with more than 10 deaths, start of the Epidemic
	index, index1, index2 = findIndex(d1)
	
	if index2 < 0:
		oneMonth = pd.DataFrame({'date': [d1.loc[index, 'date'] - timedelta(days = i) for i in range(1,16)], 
		'cases': 15 * [0], 
		'deaths': 15 * [0], 
		'cumdeaths': 15 * [0], 
		'countriesAndTerritories': 15 * [country] })
		d1 = (d1.loc[index:, ].append(oneMonth, ignore_index=True))
		d1 = d1.sort_values(by='date').reset_index()
		
		index, index1, index2 = findIndex(d1)


	print("First non-zero cases is on day %d, and 30 days before 5 days is day %d" %(index+1,  index2+1))
	
	d1 = d1.iloc[index2:, ]
	stan_data['EpidemicStart'].append(index1+1-index2)


	for covariatei in measurements:
		d1[covariatei] = 1 * (pd.to_datetime(d1.dateRep, format= '%d/%m/%Y') >= pd.to_datetime(covariates1[covariatei].values[0], format= '%d/%m/%Y'))

	#Almacenar fechas.
	dates[country] = d1.date.values
	# hazard estimation
	N = d1.shape[0]
	print("%s has %d days of data" %(country, N))
	forecast = N2 - N
	if forecast < 0:
		print("ERROR!!!! increasing N2")
		N2 = N
		forecast = N2 - N
	h = np.zeros(N2) # discrete hazard rate from time t = 1, ..., 100
	
	# The infection-to-onset distribution is Gamma distributed with mean 5.1 days and coefficient of variation 0.86.
	mean1 = 5.1
	cv1 = 0.86
	shape1 = cv1**(-2)
	scale1 = mean1/shape1
	x1 = np.random.gamma(shape1, scale = scale1, size = int(5e6))
	# The  onset-to-death  distribution  is  also  Gamma  distributed  with  a  mean  of  18.8  days  and  a coefficient of variation 0.45
	mean2 = 18.8
	cv2 = 0.45
	shape2 = cv2**(-2)
	scale2 = mean2/shape2
	x2 = np.random.gamma(shape2, scale = scale2, size = int(5e6))
	# The infection-to-death distribution is therefore given by:
	# π𝑚∼𝑖f𝑟𝑚⋅(Gamma(5.1,0.86)+Gamma(18.8,0.45))
	f = ECDF(x1+x2)
	convolution = lambda u: (CFR * f(u))
	h[0] = (convolution(1.5) - convolution(0)) 

	for i in range(1, h.size):
		h[i] = (convolution((i + 1)+.5) - convolution((i+1)-.5)) / (1-convolution((i+1)-.5)) #Se suma 1 por el cambio de indices en python.


	s = np.zeros(N2)
	s[0] = 1
	for i in range(1, N2):
		s[i] = s[i-1]*(1-h[i-1])

	f = s * h


	y = np.hstack([d1['cases'].to_numpy(), -1 * np.ones(forecast)])
	reported_cases[country]  = d1.cases.to_numpy()
	deaths = np.hstack([d1['deaths'].to_numpy(), -1 * np.ones(forecast)])
	cases = np.hstack([d1['cases'].to_numpy(), -1 * np.ones(forecast)])
	deaths_by_country[country] = d1['deaths'].to_numpy()

	covariates2 = pd.DataFrame(d1.loc[:, measurements])
	covariates2 = pd.concat([covariates2, covariates2.tail(1).iloc[np.full(forecast,0)]], ignore_index = True) # Completar hasta N2 con la ultima fila.
	# append data
	stan_data['N'].append(N)
	stan_data['y'].append(y[0]) # just the index case!
	# Store data
	stan_data['covariate1'].append(covariates2.iloc[:,0].values.tolist())
	stan_data['covariate2'].append(covariates2.iloc[:,1].values.tolist())
	stan_data['covariate3'].append(covariates2.iloc[:,2].values.tolist())
	stan_data['covariate4'].append(covariates2.iloc[:,3].values.tolist())
	stan_data['covariate5'].append(covariates2.iloc[:,4].values.tolist())
	stan_data['covariate6'].append(covariates2.iloc[:,5].values.tolist())
	stan_data['covariate7'].append(covariates2.iloc[:,6].values.tolist())
	stan_data['f'].append(f.tolist())
	stan_data['deaths'].append(deaths.tolist())
	stan_data['cases'].append(cases.tolist())
	
	stan_data['N2'] = N2
	stan_data['x']=list(range(1,N2+1))



#  La informacion debe ir en tamano N2 x M
for i in range(1,8):
	stan_data['covariate'+str(i)] = (np.array(stan_data['covariate'+str(i)]).T)

stan_data['cases'] = (np.array(stan_data['cases'], dtype= 'int').T)
stan_data['deaths'] = (np.array(stan_data['deaths'], dtype= 'int').T)
stan_data['f'] = np.array(stan_data['f']).T


stan_data['N'] = np.array(stan_data['N']).T 
stan_data['covariate2'] = 0*stan_data['covariate2'] # remove travel bans
stan_data['covariate4'] = 0*stan_data['covariate5'] # remove sport

stan_data['covariate2'] = 1* stan_data['covariate7'] # self-isolating if ill
# create the `any intervention` covariate
stan_data['covariate4'] = 1*((stan_data['covariate1'] + 
							stan_data['covariate3'] +	
							stan_data['covariate5'] + 
							stan_data['covariate6'] +
							stan_data['covariate7'])>0)

stan_data['covariate5'] =  stan_data['covariate5'] 
stan_data['covariate6'] =  stan_data['covariate6'] 
stan_data['covariate7'] =  0 # models should only take 6 covariates

# Load model
sm = pickle.load(open('stan-models/base.pkl', 'rb'))
# Fit models
fit = sm.sampling(data = stan_data, iter=200, warmup=100,chains=4, thin=4, control= {'adapt_delta': 0.90})
# fit = sm.sampling(data = stan_data, iter=10, warmup=2,chains=2, thin=2, control= {'adapt_delta': 0.90})

# Extract information from fited model
out  = fit.extract()

prediction = out['prediction']
estimateddeaths = out['E_deaths']
estimateddeathsCF = out['E_deaths0']


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
dataOut={}
for i in range(nCountries):
	N = stan_data['N'][i]
	country = countries[i]
	datesN = dates[country]
	datesN2 = np.hstack([datesN, np.array([datesN[-1] + np.timedelta64(N2 - N, 'D') for  j in range(1, N2 - N +1)])])

	#Casos!
	predictedcases= (prediction[:,:,i]).mean(axis=0)
	predicted_cases_li =np.quantile(prediction[:,:,i], q=0.025, axis= 0)
	predicted_cases_ui =np.quantile(prediction[:,:,i], q=0.975, axis= 0)

	fig, axs = plt.subplots(1)
	axs.plot(datesN, predictedcases[0:N], label = 'Prediction')
	axs.plot(datesN, predicted_cases_li[0:N], label = '0.025')
	axs.plot(datesN, predicted_cases_ui[0:N], label = '0.975')
	axs.legend()
	axs.set_title('Predicted case for ' + countries[i])
	plt.xticks(rotation = 45)
	fig.savefig('figures/python/cases_'+countries[i]+'.pdf', bbox_inches='tight')
	plt.close(fig)

	# deaths!
	predictedeaths = (estimateddeaths[:,:,i]).mean(axis=0)
	predictedeathsLi = np.quantile(estimateddeaths[:,:,i], q=0.025, axis= 0)
	predictedeathsUi = np.quantile(estimateddeaths[:,:,i], q=0.975, axis= 0)
	actualDeaths = stan_data['deaths'][:N, i]

	fig, axs = plt.subplots(1)
	axs.plot(datesN, predictedeaths[0:N], 'r',  label = 'Prediction')
	axs.plot(datesN, predictedeathsLi[0:N], label = '0.025')
	axs.plot(datesN, predictedeathsUi[0:N], label = '0.975')
	axs.bar(datesN, actualDeaths, label= 'actual')
	axs.legend()
	axs.set_title('Predicted deaths for ' + countries[i])
	plt.xticks(rotation = 45)
	fig.savefig('figures/python/deaths_'+countries[i]+'.pdf', bbox_inches='tight')
	plt.close(fig)

	# Resultados por pais.
	# Desde 1:N es regresión, desde N+1:N2 es extrapolacion.
	dataOut[country] = {
		'cases': predictedcases,
		'deaths': predictedeaths, 
		'dates': datesN2 #Fechas.
		}
