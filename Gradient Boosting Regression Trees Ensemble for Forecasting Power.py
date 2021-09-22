# Balance Authority prediction System

import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np, pandas as pd
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import csv, random, warnings, json
from statistics import mean
from datetime import datetime
from math import sqrt
from colormap import hex2rgb
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates


ROOT_DIR = '/Users/abhishekpadalkar/Documents/IRELAND Admissions/NCI/Course Modules/Modules/Sem 2/DAPA/CA2/Data/'

pd.set_option("display.max_rows", None, "display.max_columns", None)

warnings.filterwarnings("ignore")

def pretty(d, indent=0):
	for key, value in d.items():
		print('\t' * indent + str(key))
		if isinstance(value, dict):
			pretty(value, indent+1)
		else:
			print('\t' * (indent+1) + str(value))


def train_test_split_data(data, test_length_frac):
	random.seed(10)
	
	train_idx = int(len(data)*(1-test_length_frac))
	
	train = data[0:train_idx]
	test = data[train_idx:]
	
	return train, test
	

def x_and_y_split(train, test, task, sub_ba=""): # task means which prediction do we need to make: Net-gen, Forecast error, Demand, Net Interchange, sub-Interchange
	random.seed(10)
	
	# for Net-Gen
	if task == 0:
		X_train = train.iloc[:,[1,2,3,4,5,6,7,8]] # Only columns with Time
		X_test = test.iloc[:,[1,2,3,4,5,6,7,8]]
		y_train = train.iloc[:,11] # Net Gen column
		y_test = test.iloc[:,11]
		return X_train, X_test, y_train, y_test
	
	# for Forecast Error
	elif task == 1:
		X_train = train.iloc[:,[1,2,3,4,5,6,7,8,11]] # Only columns with Time, and Net Gen
		X_test = test.iloc[:,[1,2,3,4,5,6,7,8,11]]
		y_train = train.iloc[:,10] # Forecast column
		y_test = test.iloc[:,10]
		return X_train, X_test, y_train, y_test
		
	# for Demand
	elif task == 2:
		X_train = train.iloc[:,[1,2,3,4,5,6,7,8,10,11]] # Only columns with Time, Net Gen, and Forecast Error
		X_test = test.iloc[:,[1,2,3,4,5,6,7,8,10,11]]
		y_train = train.iloc[:,9] # Demand column
		y_test = test.iloc[:,9]
		return X_train, X_test, y_train, y_test
		
	# for Net Interchange
	elif task == 3:
		X_train = train.iloc[:,[1,2,3,4,5,6,7,8,9,10,11]] # Only columns with Time, Net Gen, Forecast Error, and Demand 
		X_test = test.iloc[:,[1,2,3,4,5,6,7,8,9,10,11]]
		y_train = train.iloc[:,12] # Net Interchange column
		y_test = test.iloc[:,12]
		return X_train, X_test, y_train, y_test

	# for sub Net Interchange
	elif task == 4:
		X_train = train.iloc[:,[1,2,3,4,5,6,7,8,11,12]] # Only columns with Time, Net Gen, and Net Interchange 
		X_test = test.iloc[:,[1,2,3,4,5,6,7,8,11,12]]
		y_train = train.loc[:,[sub_ba]] # Sub Interchange column
		y_test = test.loc[:,[sub_ba]]
		return X_train, X_test, y_train, y_test
	
	
def get_model(nest, mdepth, msamplesplt, lr):
	
	random.seed(10)
	# Build a Gradient Boost Regression Tree model
	params = {'n_estimators': nest,
						'max_depth': mdepth,
						'min_samples_split': msamplesplt,
						'learning_rate': lr,
						'loss': 'ls'}
	reg = ensemble.GradientBoostingRegressor(**params)
	
	return reg

def build_models(train, test, nest, mdepth, msamplesplt, lr):
	random.seed(10)
	
	# Build a model to predict the Net-Gen and Forecast Error: Based upon Time
	X_train, X_test, y_train, y_test = x_and_y_split(train, test, 0)
	net_gen_model = get_model(nest, mdepth, msamplesplt, lr)
	net_gen_model.fit(X_train, y_train)
	
	# Test it
	y_pred = net_gen_model.predict(X_test)
	mse = mean_squared_error(y_test, y_pred)
	print("\nThe mean squared error (MSE) for Net-Generation Prediction on test set: {:.4f}".format(mse))
	print("Root MSE (RMSE): {:.2f}".format(sqrt(mse))+"\n")
	# Plot feature importance
	plot_feature_importance(net_gen_model, X_test, y_test)
	plot_feature_importance(net_gen_model, X_test, y_test)
	
	# Build a model to predict the Forecast Error: Based upon Time
	X_train, X_test, y_train, y_test = x_and_y_split(train, test, 1)
	forecast_err_model = get_model(nest, mdepth, msamplesplt, lr)
	forecast_err_model.fit(X_train, y_train)	
	
	# Test it
	y_pred = forecast_err_model.predict(X_test)
	mse = mean_squared_error(y_test, y_pred)
	print("The mean squared error (MSE) for Forecast Error Prediction on test set: {:.4f}".format(mse))
	print("Root MSE (RMSE): {:.2f}".format(sqrt(mse))+"\n")
	# Plot feature importance
	plot_feature_importance(forecast_err_model, X_test, y_test)


	# Build a model to predict the Demand: Based upon Time, Net-Gen, Forecast Error
	X_train, X_test, y_train, y_test = x_and_y_split(train, test, 2)
	demand_model = get_model(nest, mdepth, msamplesplt, lr)
	demand_model.fit(X_train, y_train)
	
	# Test it
	y_pred = demand_model.predict(X_test)
	mse = mean_squared_error(y_test, y_pred)
	print("The mean squared error (MSE) for Demand Prediction on test set: {:.4f}".format(mse))
	print("Root MSE (RMSE): {:.2f}".format(sqrt(mse))+"\n")
	# Plot feature importance
	plot_feature_importance(demand_model, X_test, y_test)


	# Build a model to predict Net Interchange: Based upon Time, Net-Gen, Forecast Error, Demand
	X_train, X_test, y_train, y_test = x_and_y_split(train, test, 3)
	net_interchange_model = get_model(nest, mdepth, msamplesplt, lr)
	net_interchange_model.fit(X_train, y_train)
	
	# Test it
	y_pred = net_interchange_model.predict(X_test)
	mse = mean_squared_error(y_test, y_pred)
	print("The mean squared error (MSE) for Net-Interchange Prediction on test set: {:.4f}".format(mse))
	print("Root MSE (RMSE): {:.2f}".format(sqrt(mse))+"\n")
	# Plot feature importance
	plot_feature_importance(net_interchange_model, X_test, y_test)

	# For each sub-BA interchange
	if len(train.iloc[:,:].columns.tolist()) > 13:
		dict_sub_int_models = {}
		sub_ba_interchange = train.iloc[:,13:].columns.tolist()
		# Build a model to predict Interchanges in each sub-BA interchanges: Based upon Time, Net-Gen, Interchange
		for i, sub_ba in enumerate(sub_ba_interchange):
#			sub_ba_interchange[i] = sub_ba.split(" : ")[-1]
			X_train, X_test, y_train, y_test = x_and_y_split(train, test, 4, sub_ba=sub_ba)
			sub_int_model = get_model(nest, mdepth, msamplesplt, lr)
			sub_int_model.fit(X_train, y_train)
		
		# Test it	
			y_pred = sub_int_model.predict(X_test)
			mse = mean_squared_error(y_test, y_pred)
			print("The mean squared error (MSE) for "+sub_ba+" Prediction on test set: {:.4f}".format(mse))
			print("Root MSE (RMSE): {:.2f}".format(sqrt(mse))+"\n")
			# Plot feature importance
			plot_feature_importance(sub_int_model, X_test, y_test)
			
			# Save model to the dict
			dict_sub_int_models[sub_ba] = sub_int_model

	return net_gen_model, forecast_err_model, demand_model, net_interchange_model, dict_sub_int_models


def demand_train_model(train, test, nest, mdepth, msamplesplt, lr):
	
	# Build a model to predict the Demand: Based upon Time, Net-Gen, Forecast Error
	X_train, X_test, y_train, y_test = x_and_y_split(train, test, 2)
	demand_model = get_model(nest, mdepth, msamplesplt, lr)
	demand_model.fit(X_train, y_train)
	
	# Test it
	y_pred = demand_model.predict(X_test)
	mse = mean_squared_error(y_test, y_pred)
	
	return float('{:.2f}'.format(sqrt(mse))), list(y_test), list(y_pred)


def get_color(r, g, b):
	r_norm = r/255
	
	g_norm = g/255
	
	b_norm = b/255
	
	return (r_norm, g_norm, b_norm, 1)

	
def plot_graph(dates, test, pred, task, worm, sub_ba=""):
	
	if task == 0:
		y_label = 'Net Generation'
		color = get_color(255, 190, 11)
	elif task == 1:
		y_label = 'Forecast Error'
		color = get_color(251, 86, 7)
	elif task == 2:
		y_label = 'Demand'
		color = get_color(255, 0, 110)
	elif task == 3:
		y_label = 'Net Interchange'
		color = get_color(131, 56, 236)
	elif task == 4:
		y_label = sub_ba+' Interchange (BA-BA)'
		color = get_color(58, 134, 255)
			
	# Plot Actual VS Predicted Values
	font = {'family' : 'Chaparral Pro',   ### Have same font through out for consistency ###
		'size'   : 12}
	plt.rc('font', **font)
	fig, ax = plt.subplots(figsize=(15, 6))
	
	# Set dark background
	plt.style.use("dark_background")
	
	# change grid style
	plt.grid(color='0.3', linestyle='-', linewidth=0.7)
		
	# Specify how our lines should look
	ax.plot(dates, test, color=color, label='Actual '+y_label, marker='.', markersize=4)
	ax.set_xlabel('Time', fontsize=15)
	ax.set_ylabel(y_label+' (MegaWatt / Hour)', fontsize=15)
	ax.set_title(y_label+' Forecast Prediction', fontsize=20, weight="bold")
	ax.grid(True)
	ax.plot(dates, pred, color='0.7', linewidth=2, linestyle='-.', label='Predicted '+y_label+' GBRT')
		
	# Change dates tick style
	ax.xaxis.set_major_formatter(DateFormatter("%d/%m"))
	
	# Weekly or Monthly
	if worm == 0: # Weekly
		#Minor ticks hourly(when we plot a weekly graph)
		ax.xaxis.set_minor_locator(MultipleLocator(1/24))
		#Major ticks Daily (when we plot a weekly graph)
		ax.xaxis.set_major_locator(MultipleLocator(1))
		
		trend_daily = test.rolling(12, center=True).mean() # when we plot week graph
		ax.plot(dates, trend_daily, color=get_color(144, 224, 239), linewidth=2, linestyle='-', label='Trend '+y_label+' Daily')
		
	elif worm == 1: # Monthly
		#Minor ticks daily (when we plot a monthly graph)
		ax.xaxis.set_minor_locator(MultipleLocator(1))
		#Major ticks Weekly(when we plot a monthly graph)
		# Set x-axis major ticks to weekly interval, on Mondays
		ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))
		
		trend_weekly = test.rolling(168, center=True).mean() # when we plot monthly data
		ax.plot(dates, trend_weekly, color=get_color(144, 224, 239), linewidth=2, linestyle='-', label='Trend '+y_label+' Weekly')
		
	
	ax.tick_params(which='major', width=1.5, length=7)
	ax.tick_params(which='minor', width=1, length=4, color='r')
	
	# Redisplay the legend to show our new wind gust line
	ax.legend(loc='upper left', title_fontsize='15', fontsize=15)
	
	plt.show()


def test_for_model_comparison(BA_dict, ba_list, power_consumption_data, test_idx, fnest, fmdepth, fmsamplesplit, flr):
	
	random.seed(10)
	
	BAs_to_ignore = ['AVRN', 'DEAA', 'GLHB', 'GRID', 'GRIF', 'GWA', 'HGMA', 'SEPA', 'WWA', 'YAD'] # Since these BA data doesn't have demand and day-ahead forecast values, but just net gen and interchange values
	
	# List of RMSEs for each BA
	actual_ba_rmse_list = []
	predicted_ba_rmse_list = []	
	
	for ba in ba_list:
		if ba in BAs_to_ignore: # from BA data, select BAs which has demand and forecast values
			continue 
		else:
#			print('For BA:', ba)
			ba_cols = BA_dict[ba][0:2] # Get demand and day-ahead forecast columns
			ba_data = power_consumption_data[ba_cols]
			ba_data_test = ba_data.iloc[test_idx:,:] # Forecast evaluation on test data only 
			ba_data_test = ba_data_test.fillna(ba_data_test.mean())
			rmse_actual = sqrt(mean_squared_error(ba_data_test.iloc[:,0], ba_data_test.iloc[:,1]))
			actual_ba_rmse_list.append(rmse_actual) # for all BAs calculate Forecast error, square it and take mean for that BA, and then sqrt it
			
			# for all BAs build a model with the given hyperparameter and predict the values. calculate rmse.
			# Create a train test data for getting demand rmse for each BA model
			ba_cols_full = power_consumption_data.iloc[:,2:10].columns.tolist() # get time columns
			ba_cols_full = ba_cols_full + BA_dict[ba] # concat ba cols
			ba_data_full = power_consumption_data[ba_cols_full]
			
			for_error = ba_data_full.iloc[:, 8] - ba_data_full.iloc[:, 9] # Demand - Forecast
			ba_data_full[ba+' : Day-ahead demand forecast'] = for_error
			ba_data_full = ba_data_full.rename(columns={ba+' : Day-ahead demand forecast':ba+' : Forecast Error'})
			
			dates = [datetime(int(i["Year"]), int(i["Month"]), int(i["Day"]), int(i["Hour"]), 0, 0, 0) for idx, i in ba_data_full.iterrows()]
			ba_data_full.insert(0, 'DateTime', dates)
			ba_data_full = ba_data_full.fillna(ba_data_full.mean())
			
			ba_train, ba_test = train_test_split_data(ba_data_full, 0.1)
			rmse_pred, y_actual, y_pred = demand_train_model(ba_train, ba_test, fnest, fmdepth, fmsamplesplit, flr)
			predicted_ba_rmse_list.append(rmse_pred)
#			print('Actual RMSE:', rmse_actual, '   Predicted RMSE:', rmse_pred)
#			print()
	
	# take mean of rmses for actual and our predicted model
	# Print Actual mean RMSE and Predicted mean RMSE
	print('\nActual Mean RMSE:', mean(actual_ba_rmse_list))
	print()
	print('Predicted Mean RMSE:', mean(predicted_ba_rmse_list))
			


def plot_tuning_curve(ls_values_with_parameters):
	
	# For AEC
	
	# For actual value at Monday 01.02.2021 # 6458 {got from actual excel datasheet} = 105 {got from test values} index 
	# Actual value on 01.02.2021 00:00 Monday
	actual = 433
	y_axis_rmse_list = []
	x_axis_pred_list = []
	for key in ls_values_with_parameters.keys():
		y_axis_rmse_list.append(key)
		x_axis_pred_list.append(ls_values_with_parameters[key]['y_predicted'][105])
		y_act, y_pred = ls_values_with_parameters[key]['y_actual'], ls_values_with_parameters[key]['y_predicted']
		print(y_act[105], y_pred[105])
	
	data = {'RMSE': y_axis_rmse_list, 'Prediction Score': x_axis_pred_list}
	df = pd.DataFrame(data)
	df = df.sort_values(by = ['RMSE', 'Prediction Score'])
	
	
	
	# Plot Actual VS Predicted Values
	font = {'family' : 'Chaparral Pro',   ### Have same font through out for consistency ###
		'size'   : 12}
	plt.rc('font', **font)
	fig, ax = plt.subplots(figsize=(15, 6))
	
	# Set dark background
	plt.style.use("dark_background")
	
	# change grid style
	plt.grid(color='0.3', linestyle='-', linewidth=0.7)
		
	# Specify how our lines should look
	ax.plot(df['Prediction Score'], df['RMSE'], color='tab:blue', label='', marker='.', markersize=4)
	ax.set_xlabel('Time', fontsize=15)
	ax.set_ylabel(' (MegaWatt / Hour)', fontsize=15)
	ax.set_title(' Forecast Prediction', fontsize=20, weight="bold")
	ax.grid(True)
	ax.tick_params(which='major', width=1.5, length=7)
	ax.tick_params(which='minor', width=1, length=4, color='r')
	
	
	plt.show()


def plot_feature_importance(model, X_test, y_test):
	
	# Plot Actual VS Predicted Values
	font = {'family' : 'Chaparral Pro',   ### Have same font through out for consistency ###
		'size'   : 12}
	plt.rc('font', **font)
	fig, ax = plt.subplots(figsize=(15, 3.5))
	
	# Set dark background
	plt.style.use("dark_background")
	
	# change grid style
	plt.grid(color='0.3', linestyle='-', linewidth=0.7)
	
	feature_importance = model.feature_importances_
	sorted_idx = np.argsort(feature_importance)
	pos = np.arange(sorted_idx.shape[0]) + .5
	plt.subplot(1, 2, 1)
	plt.barh(pos, feature_importance[sorted_idx], align='center')
	plt.yticks(pos, np.array(X_test.columns)[sorted_idx])
	plt.title('Feature Importance (MDI)')

	result = permutation_importance(model, X_test, y_test, n_repeats=10,
									random_state=42, n_jobs=2)
	sorted_idx = result.importances_mean.argsort()
	plt.subplot(1, 2, 2)
	plt.boxplot(result.importances[sorted_idx].T,
				vert=False, labels=np.array(X_test.columns)[sorted_idx])
	plt.title("Permutation Importance (test set)")
	fig.tight_layout()
	plt.show()

if __name__ == "__main__":	

	random.seed(10)
	
	file_name = "new_full_data.csv"
	
	# Get the data
	power_consumption_data = pd.read_csv(ROOT_DIR+file_name)
	
	# Get unique BAs
	BA_dict = {}	# save list of cols for each ba in dict
	ba_list = power_consumption_data.iloc[:,10:].columns.tolist()
	ba = ''
	for bai in ba_list:
		ba = bai.split(" : ")[0]
		if ba in BA_dict:
			BA_dict[ba].append(bai)
		else:
			BA_dict[ba] = [bai]
	
		
	# Ask for which BA
	ba_list = list(BA_dict.keys())
	print("Which BA you need to make predictions for? \n\n")
	
	for i, ba in enumerate(ba_list):
		print(i," : ",ba)
	
	idx = int(input("\nSelect index value corresponding to above BAs: "))
	ba = ba_list[idx]
	print()
	print(ba,"\n")


	# Get sub-data for that particular BA
	ba_cols = power_consumption_data.iloc[:,2:10].columns.tolist() # get time columns
	ba_cols = ba_cols + BA_dict[ba] # concat ba cols
	ba_data = power_consumption_data[ba_cols]

	
	# Calculate Forecast error and replace it with Day-ahead forecast col
	for_error = ba_data.iloc[:, 8] - ba_data.iloc[:, 9] # Demand - Forecast
	ba_data[ba+' : Day-ahead demand forecast'] = for_error
	ba_data = ba_data.rename(columns={ba+' : Day-ahead demand forecast':ba+' : Forecast Error'})
	

	# Create a list of sub-interchange BAs
	if len(ba_data.iloc[:,:].columns.tolist()) > 12:
		sub_ba_interchange = ba_data.iloc[:,12:].columns.tolist()
		for i, sub_ba in enumerate(sub_ba_interchange):
			sub_ba_interchange[i] = sub_ba.split(" : ")[-1]
	else:
		sub_ba_interchange = []
	

	# Create a new datetime column for the dataset and join at first to the dataset
	dates = [datetime(int(i["Year"]), int(i["Month"]), int(i["Day"]), int(i["Hour"]), 0, 0, 0) for idx, i in ba_data.iterrows()]
	ba_data.insert(0, 'DateTime', dates)
	ba_data = ba_data.fillna(ba_data.mean())

	# Train-test split the dataset
	ba_train, ba_test = train_test_split_data(ba_data, 0.1)
	
	# Hyper-parameter tuning
		# n-estimators: 400, 450, 500, 550, 600, 650, 700, 750, 800
#	nest = [400, 450, 500, 550, 600, 650, 700, 750, 800]		
#		# max depth: 1, 2, 3, 4, 5
#	mdepth = [1, 2, 3, 4, 5]	
#		# min samples split: 2, 3, 4, 5, 6, 7
#	msampsplit = [2, 3, 4, 5, 6, 7]
#		# learning rate: 0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005
#	lr = [0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005]
#	
#	ls_values_with_parameters = {}
#	counter = 1
#	# tune demand model
#	for i in lr:
#		for j in msampsplit:
#			for k in mdepth:
#				for l in nest:
#					ls_err, y_actual, y_pred = demand_train_model(ba_train, ba_test, l, k, j, i)
#					print(counter, ": ", ls_err)
#					ls_values_with_parameters[ls_err] = {
#						'nestimator': l,
#						'mdepth': k,
#						'minsamplesplit': j,
#						'learningrate': i,
#						'y_actual': y_actual,
#						'y_predicted': y_pred
#					}
#					counter += 1
#	
#	# Save the tuning values along with prediction to then use them directly in future rather than running this loop again.
#	with open(ROOT_DIR+'hyperptuning.json', 'w') as filep:
#		json.dump(ls_values_with_parameters, filep, indent=4)
		
	# Load the learned tuning parameters
	with open(ROOT_DIR+'hyperptuning.json', 'r') as filep:
		ls_values_with_parameters = json.load(filep)
	
	# min error value from tuned models
	min_err = min(list(ls_values_with_parameters.keys()))
	fnest, fmdepth, fmsamplesplit, flr = ls_values_with_parameters[min_err]['nestimator'], ls_values_with_parameters[min_err]['mdepth'], ls_values_with_parameters[min_err]['minsamplesplit'], ls_values_with_parameters[min_err]['learningrate']
	
	
	# Build models
	net_gen_model, forecast_err_model, demand_model, net_interchange_model, dict_sub_int_models = build_models(ba_train, ba_test, fnest, fmdepth, fmsamplesplit, flr)
	
	# Test models on test data period, ask user for particular period.
	
	# Currently set it for one week {24*7 = 168hrs/rows}
	worm = int(input("Generate weekly or monthly graph? \n0: Weekly \n1: Monthly \nInput: "))
	if worm == 0: # Weekly
		data_test = ba_test.iloc[:168,:]
	if worm == 1: # Monthly
		data_test = ba_test
	data_pred = data_test.copy(deep=True)
	
	# Predict Net_generation and Replace old_values with predicted values
	data_pred.iloc[:,11] = net_gen_model.predict(data_test.iloc[:,[1,2,3,4,5,6,7,8]])
	
	# Predict Forecast Error and Replace old_values with predicted values
	data_pred.iloc[:,10] = forecast_err_model.predict(data_test.iloc[:,[1,2,3,4,5,6,7,8,11]])
	
	# Predict Demand and Replace old_values with predicted values
	data_pred.iloc[:,9] = demand_model.predict(data_test.iloc[:,[1,2,3,4,5,6,7,8,10,11]])
	
	# Predict Net_Interchange and Replace old_values with predicted values
	data_pred.iloc[:,12] = net_interchange_model.predict(data_test.iloc[:,[1,2,3,4,5,6,7,8,9,10,11]])
	
	 # Predict Sub_BAs and Replace old_values with predicted values
	for i, sub_ba in enumerate(sub_ba_interchange):
		data_pred.loc[:,[ba+' : BA-to-BA interchange with ... : '+sub_ba]] = dict_sub_int_models[list(dict_sub_int_models.keys())[i]].predict(data_test.iloc[:,[1,2,3,4,5,6,7,8,11,12]])
	
	# Plot predicted VS actual values
	
		# Net Generation
	plot_graph(data_test["DateTime"], data_test.iloc[:,11], data_pred.iloc[:,11], 0, worm)
	plot_graph(data_test["DateTime"], data_test.iloc[:,11], data_pred.iloc[:,11], 0, worm)
	
		# Forecast
	plot_graph(data_test["DateTime"], data_test.iloc[:,10], data_pred.iloc[:,10], 1, worm)
		
		# Demand
	plot_graph(data_test["DateTime"], data_test.iloc[:,9], data_pred.iloc[:,9], 2, worm)
		
		# Net Interchange
	plot_graph(data_test["DateTime"], data_test.iloc[:,12], data_pred.iloc[:,12], 3, worm)	
		
		# Sub_BA_Interchange
	for sub_ba in sub_ba_interchange:
		plot_graph(data_test["DateTime"], data_test.loc[:,[ba+' : BA-to-BA interchange with ... : '+sub_ba]], data_pred.loc[:,[ba+' : BA-to-BA interchange with ... : '+sub_ba]], 4, worm, sub_ba)
	
	
	# test our prediction RMSE with the actual RMSE
	test_for_model_comparison(BA_dict, ba_list, power_consumption_data, ba_test.index[0], fnest, fmdepth, fmsamplesplit, flr)

	# get the hyper-parameter curve
#	plot_tuning_curve(ls_values_with_parameters)
	
	
		

#Select index value corresponding to above BAs: 0
#
#AEC 
#
#
#The mean squared error (MSE) for Net-Generation Prediction on test set: 38612.7182
#Squareroot Error: 196.50
#
#The mean squared error (MSE) for Forecast Error Prediction on test set: 13241.0768
#Squareroot Error: 115.07
#
#The mean squared error (MSE) for Demand Prediction on test set: 3800.1895
#Squareroot Error: 61.65
#
#The mean squared error (MSE) for Net-Interchange Prediction on test set: 5771.7648
#Squareroot Error: 75.97
#
#The mean squared error (MSE) for AEC : BA-to-BA interchange with ... : MISO Prediction on test set: 4908.1728
#Squareroot Error: 70.06
#
#The mean squared error (MSE) for AEC : BA-to-BA interchange with ... : SOCO Prediction on test set: 6142.6545
#Squareroot Error: 78.38


# AEC After tuning the hyperparameters

#The mean squared error (MSE) for Net-Generation Prediction on test set: 47389.5734
#Root MSE (RMSE): 217.69
#
#The mean squared error (MSE) for Forecast Error Prediction on test set: 12656.1031
#Root MSE (RMSE): 112.50
#
#The mean squared error (MSE) for Demand Prediction on test set: 3533.8754
#Root MSE (RMSE): 59.45
#
#The mean squared error (MSE) for Net-Interchange Prediction on test set: 1984.0056
#Root MSE (RMSE): 44.54
#
#The mean squared error (MSE) for AEC : BA-to-BA interchange with ... : MISO Prediction on test set: 4152.7782
#Root MSE (RMSE): 64.44
#
#The mean squared error (MSE) for AEC : BA-to-BA interchange with ... : SOCO Prediction on test set: 5258.3967
#Root MSE (RMSE): 72.51

	# Second time
#The mean squared error (MSE) for Net-Generation Prediction on test set: 47680.2037
#Root MSE (RMSE): 218.36
#
#The mean squared error (MSE) for Forecast Error Prediction on test set: 12654.3768
#Root MSE (RMSE): 112.49
#
#The mean squared error (MSE) for Demand Prediction on test set: 3425.3028
#Root MSE (RMSE): 58.53
#
#The mean squared error (MSE) for Net-Interchange Prediction on test set: 1976.5474
#Root MSE (RMSE): 44.46
#
#The mean squared error (MSE) for AEC : BA-to-BA interchange with ... : MISO Prediction on test set: 4131.6654
#Root MSE (RMSE): 64.28
#
#The mean squared error (MSE) for AEC : BA-to-BA interchange with ... : SOCO Prediction on test set: 5260.0202
#Root MSE (RMSE): 72.53

	
#				Actual Mean RMSE: 10318.482591454302
#				Predicted Mean RMSE: 10081.204558823529
#			For BA: AEC
#			Actual RMSE: 545.8881808876993    Predicted RMSE: 58.97
#
#			For BA: AECI
#			Actual RMSE: 239.99498637350587    Predicted RMSE: 855.98
#
#			For BA: AVA
#			Actual RMSE: 117.00887959331428    Predicted RMSE: 271.63
#
#			For BA: AZPS
#			Actual RMSE: 155.54434708258748    Predicted RMSE: 440.93
#
#			For BA: BANC
#			Actual RMSE: 84.7250259860014    Predicted RMSE: 238.3
#
#			For BA: BPAT
#			Actual RMSE: 212.64959740095333    Predicted RMSE: 957.73
#
#			For BA: CHPD
#			Actual RMSE: 259.4881975078874    Predicted RMSE: 52.41
#
#			For BA: CISO
#			Actual RMSE: 514.7292269279299    Predicted RMSE: 2612.58
#
#			For BA: CPLE
#			Actual RMSE: 1600.6305711226637    Predicted RMSE: 838.27
#
#			For BA: CPLW
#			Actual RMSE: 168.3566925330855    Predicted RMSE: 101.5
#
#			For BA: DOPD
#			Actual RMSE: 26.321928954591296    Predicted RMSE: 50.87
#
#			For BA: DUK
#			Actual RMSE: 607.3569254105798    Predicted RMSE: 652.5
#
#			For BA: EPE
#			Actual RMSE: 48.818113598309495    Predicted RMSE: 99.19
#
#			For BA: ERCO
#			Actual RMSE: 7329.552134248624    Predicted RMSE: 872.37
#
#			For BA: FMPP
#			Actual RMSE: 126.9095058019277    Predicted RMSE: 201.71
#
#			For BA: FPC
#			Actual RMSE: 1381.3887503535707    Predicted RMSE: 272.82
#
#			For BA: FPL
#			Actual RMSE: 846.7872662132817    Predicted RMSE: 523.26
#
#			For BA: GCPD
#			Actual RMSE: 14.757530367564048    Predicted RMSE: 55.33
#
#			For BA: GVL
#			Actual RMSE: 22.087514179557438    Predicted RMSE: 62.19
#
#			For BA: HST
#			Actual RMSE: 7.114403058457327    Predicted RMSE: 8.72
#
#			For BA: IID
#			Actual RMSE: 8.76171817524601    Predicted RMSE: 62.86
#
#			For BA: IPCO
#			Actual RMSE: 77.68227217818023    Predicted RMSE: 218.68
#
#			For BA: ISNE
#			Actual RMSE: 345.40266595528334    Predicted RMSE: 501.23
#
#			For BA: JEA
#			Actual RMSE: 86.93913731878014    Predicted RMSE: 90.15
#
#			For BA: LDWP
#			Actual RMSE: 279.7615865800361    Predicted RMSE: 245.79
#
#			For BA: LGEE
#			Actual RMSE: 204.71992904361528    Predicted RMSE: 276.51
#
#			For BA: MISO
#			Actual RMSE: 2977.1010496885488    Predicted RMSE: 2659.97
#
#			For BA: NEVP
#			Actual RMSE: 94.68032542541714    Predicted RMSE: 194.61
#
#			For BA: NWMT
#			Actual RMSE: 50.88316339694745    Predicted RMSE: 193.76
#
#			For BA: NYIS
#			Actual RMSE: 624.750004392692    Predicted RMSE: 1381.92
#
#			For BA: PACE
#			Actual RMSE: 712.7744618234874    Predicted RMSE: 742.0
#
#			For BA: PACW
#			Actual RMSE: 105.81481364882428    Predicted RMSE: 282.42
#
#			For BA: PGE
#			Actual RMSE: 241.14874689765986    Predicted RMSE: 318.7
#
#			For BA: PJM
#			Actual RMSE: 3533.191699439168    Predicted RMSE: 2882.37
#
#			For BA: PNM
#			Actual RMSE: 69.87878010775833    Predicted RMSE: 173.48
#
#			For BA: PSCO
#			Actual RMSE: 226.1843772705057    Predicted RMSE: 254.41
#
#			For BA: PSEI
#			Actual RMSE: 2191.2357581861984    Predicted RMSE: 281.91
#
#			For BA: SC
#			Actual RMSE: 128.7027396139491    Predicted RMSE: 288.51
#
#			For BA: SCEG
#			Actual RMSE: 193.44659466073756    Predicted RMSE: 183.86
#
#			For BA: SCL
#			Actual RMSE: 71.37871307250282    Predicted RMSE: 160.54
#
#			For BA: SEC
#			Actual RMSE: 628935.2132667814    Predicted RMSE: 637477.19
#
#			For BA: SOCO
#			Actual RMSE: 1149.746164866864    Predicted RMSE: 1157.96
#
#			For BA: SPA
#			Actual RMSE: 12.28255632194291    Predicted RMSE: 19.02
#
#			For BA: SRP
#			Actual RMSE: 80.15792662696026    Predicted RMSE: 437.21
#
#			For BA: SWPP
#			Actual RMSE: 1876.5416155375272    Predicted RMSE: 2229.66
#
#			For BA: TAL
#			Actual RMSE: 15.692783567316907    Predicted RMSE: 35.22
#
#			For BA: TEC
#			Actual RMSE: 91.16965786697205    Predicted RMSE: 169.44
#
#			For BA: TEPC
#			Actual RMSE: 212.92572089515002    Predicted RMSE: 74.99
#
#			For BA: TIDC
#			Actual RMSE: 14.069032555716984    Predicted RMSE: 66.45
#
#			For BA: TPWR
#			Actual RMSE: 25.03315082498863    Predicted RMSE: 88.2
#
#			For BA: TVA
#			Actual RMSE: 919.2114946815419    Predicted RMSE: 565.73
#
#			For BA: WACM
#			Actual RMSE: 1640.6832316062387    Predicted RMSE: 149.59
#
#			For BA: WALC
#			Actual RMSE: 84.27455879192732    Predicted RMSE: 137.91
#
#			For BA: WAUW
#			Actual RMSE: 8.682276417228378    Predicted RMSE: 19.78
#
#			For BA: California (region)
#			Actual RMSE: 664.3924688700333    Predicted RMSE: 3062.66
#
#			For BA: Carolinas (region)
#			Actual RMSE: 2224.5261452321774    Predicted RMSE: 782.34
#
#			For BA: Central (region)
#			Actual RMSE: 1882.0212790775904    Predicted RMSE: 2018.44
#
#			For BA: Florida (region)
#			Actual RMSE: 2244.3067063278113    Predicted RMSE: 690.44
#
#			For BA: Mid-Atlantic (region)
#			Actual RMSE: 3533.191699439168    Predicted RMSE: 2882.28
#
#			For BA: Midwest (region)
#			Actual RMSE: 3278.8718217571036    Predicted RMSE: 2762.5
#
#			For BA: New England (region)
#			Actual RMSE: 337.015122263914    Predicted RMSE: 502.19
#
#			For BA: Northwest (region)
#			Actual RMSE: 3057.260312365262    Predicted RMSE: 1944.3
#
#			For BA: New York (region)
#			Actual RMSE: 624.750004392692    Predicted RMSE: 1363.52
#
#			For BA: Southeast (region)
#			Actual RMSE: 966.0330142922392    Predicted RMSE: 870.09
#
#			For BA: Southwest (region)
#			Actual RMSE: 719.2317755949717    Predicted RMSE: 994.94
#
#			For BA: Tennessee (region)
#			Actual RMSE: 919.2114946815419    Predicted RMSE: 565.55
#
#			For BA: Texas (region)
#			Actual RMSE: 7329.552134248624    Predicted RMSE: 872.37
#
#			For BA: United States Lower 48 (region)
#			Actual RMSE: 12276.220488529963    Predicted RMSE: 2961.0


# STARTED @1:45pm

# ENDED @3:05PM

# Need to reach 1890 models to complete hypertuning.