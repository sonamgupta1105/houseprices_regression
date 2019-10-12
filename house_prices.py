#!/usr/bin/env python
# coding: utf-8

# In[192]:


# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


# In[53]:


# Read the input csv file into pandas dataframe
wa = pd.read_csv("ds_candidate_exercise_predictions.csv")
wa.head()
wa.fillna(0)


# In[54]:


# Recreate the predictive model, hence disregard the Prediction column
# There are some columns like PropertyID, and Usecode that will not affect
# the analysis, hence dropping them. Usecode is 9 since all the houses
# in this dataset are single family homes.
wa.drop(wa.columns[[0, 5, 24]], axis = 1, inplace = True)
wa.info()


# In[55]:


# To summarize the data and get the basic statistics
wa.describe().T


# In[56]:


get_ipython().run_line_magic('matplotlib', 'inline')

wa.hist(bins=50, figsize=(20,15))
plt.savefig("attribute_histogram_plots")
plt.show()
# Some of the graphs look left skewed. 


# In[57]:


# Create a correlation matrix to see how correlated the predictor variables
# are to the dependent variable which is SaleDollarCnt
corr_mat_prices = wa.corr()
corr_mat_prices["SaleDollarCnt"].sort_values(ascending= False)


# #### From the correlations, we see that house prices increase with increase in median home value, finished square footage of the house, total number of bathrooms and median income of households also affect it. The other variables that have correlation close to 0 indicates no linear relation. The negative correlation between house prices and longitude, year the houses were built, houses with children under 18 seem to lower the house prices.

# In[58]:


# Correlation scatter plot with the most correlated variable, in this case
# BGMedHomeValue
wa.plot(kind="scatter", x="BGMedHomeValue", y="SaleDollarCnt", alpha=0.5)
plt.savefig('MedianValue vs Home Price Value.png')


# ##### From the above graph, we can see a very linear relation, indicating strong correlation

# In[59]:


# From the correlation matrix, we see censusblockgroup has no correlation, that
# means we do not need that column for our predictive model.
# For the purpose of this analysis, Transdate and ZoneCodeCounty can be dropped too

wa.drop(wa.columns[[1,2,3]], axis = 1, inplace = True)
wa.info()


# In[60]:


# Checking for missing values, NaN
# Ref: https://datascience.stackexchange.com/questions/11928/valueerror-input-contains-nan-infinity-or-a-value-too-large-for-dtypefloat32
np.isnan(wa)
wa[:] = np.nan_to_num(wa)


# In[61]:


# Creating X and Y data for building the model
#wa.columns
X = wa[['BedroomCnt', 'BathroomCnt', 'FinishedSquareFeet', 'GarageSquareFeet',
       'LotSizeSquareFeet', 'StoryCnt', 'BuiltYear', 'ViewType', 'Latitude',
       'Longitude', 'BGMedHomeValue', 'BGMedRent', 'BGMedYearBuilt',
       'BGPctOwn', 'BGPctVacant', 'BGMedIncome', 'BGPctKids', 'BGMedAge']]
Y = wa['SaleDollarCnt']


# In[62]:


# Splitting the dataset into training and testing
X_train, X_test, y_train,y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)


# In[63]:


# Fitting the linear regression model

regr = LinearRegression()
regr.fit(X_train, y_train)


# In[64]:


# Predictions on testing set
y_predictions = regr.predict(X_test)
# Calculate the R^2 score for linear regression model
print('R squared: %.4f' % regr.score(X_test, y_test))


# In[65]:


# Calculate mean squared error:
mse = mean_squared_error(y_predictions, y_test)
rmse = np.sqrt(mse)
print(rmse)


# In[66]:


# Calculate Mean Absolute error
mae = mean_absolute_error(y_predictions, y_test)
print(mae)


# #### From the above metrics and score, we get a pretty good R^2 value of 68.13. The MAE of 143728.49 indicates that we may have some predictions that might have been on target, but there are some outliers. From RMSE, we can say that the model predicted sale prices of houses in test set within $251833 of actual prices

# In[67]:


# Plot outputs
plt.scatter(regr.predict(X_train), regr.predict(X_train)-y_train,  color='black', s=40, alpha=0.5)
plt.scatter(regr.predict(X_test), regr.predict(X_test)-y_test,  color='green', s=40)
plt.hlines(y=0, xmin=0, xmax=50)
plt.title('Residual plot using training and testing set')
plt.ylabel('Residuals')
plt.show()


# #### From the looks of residual plot, the data points are not randomly scattered towards zero for both x and Y axes. Thus, a way to remove that can be handling or removing outliers since that may be causing the model to overestimate. 

# In[68]:


# Calculate Z-score as a way to remove outliers from the original data

z_score = np.abs(stats.zscore(wa))
print(z_score)


# In[15]:


# Setting threshold to be 5
# Ref: https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba
thold = 5
print(np.where(z_score > 5))


# In[69]:


#print(z_score[37][5])


# In[70]:


# removing outliers using z-score calculated above
wa_outliers = wa[(z_score < 5).all(axis = 1)]


# In[71]:


print(wa.shape)
print(wa_outliers.shape)


# #### 454 outliers were removed
# Now Fitting the linear regression model again with the new data

# In[136]:


X_out = wa_outliers[['BedroomCnt', 'BathroomCnt', 'FinishedSquareFeet', 'GarageSquareFeet',
       'LotSizeSquareFeet', 'StoryCnt', 'BuiltYear', 'ViewType', 'Latitude',
       'Longitude', 'BGMedHomeValue', 'BGMedRent', 'BGMedYearBuilt',
       'BGPctOwn', 'BGPctVacant', 'BGMedIncome', 'BGPctKids', 'BGMedAge']]
Y_out = wa_outliers['SaleDollarCnt']


# In[137]:


X_train_out, X_test_out, y_train_out, y_test_out = train_test_split(X_out, Y_out, test_size = 0.2, random_state = 42)


# In[74]:


# Fitting the linear regression model

regr_out = LinearRegression()
regr_out.fit(X_train_out, y_train_out)


# In[75]:


y_pred_out= regr_out.predict(X_test_out)
# Calculate the R^2 score for linear regression model
print('R squared: %.4f' % regr_out.score(X_test_out, y_test_out))


# In[76]:


mse_out = mean_squared_error(y_pred_out, y_test_out)
rmse_out = np.sqrt(mse_out)

print('Root Mean Squared Error:')
print(rmse_out)
mae_out = mean_absolute_error(y_pred_out, y_test_out)
print('Mean Absolute Error:')
print(mae_out)


# In[77]:


# Plot outputs
plt.scatter(regr_out.predict(X_train_out), regr_out.predict(X_train_out)-y_train_out,  color='black', s=40, alpha=0.5)
plt.scatter(regr_out.predict(X_test_out), regr_out.predict(X_test_out)-y_test_out,  color='green', s=40)
plt.hlines(y=0, xmin=0, xmax=50)
plt.title('Residual plot using training(Blue) and testing(Green) set with outliers removed')
plt.ylabel('Residuals')
plt.xlabel('X')
plt.show()


# After removing outliers, all the scores definitely improved and from the residuals plot, the data points are more towards zero
# and mostly randomly scattered

# #### Building Random Forest Regressor Model Without Outliers in the Data

# In[78]:


# Fit the Random forest model
regr_rf = RandomForestRegressor(random_state = 42)
regr_rf.fit(X_train_out, y_train_out)


# In[83]:


y_pred_out_rf= regr_rf.predict(X_test_out)
# Calculate R squared for random forest regressor
print('R squared": %.4f' % regr_rf.score(X_test_out, y_test_out))


# Calculate the absolute errors
errors = abs(y_pred_out_rf - y_test_out)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# In[84]:


# Plot outputs
plt.scatter(regr_rf.predict(X_train_out), regr_rf.predict(X_train_out)-y_train_out,  color='black', s=40, alpha=0.5)
plt.scatter(regr_rf.predict(X_test_out), regr_rf.predict(X_test_out)-y_test_out,  color='green', s=40)
plt.hlines(y=0, xmin=0, xmax=50)
plt.title('Residual plot using training(Blue) and testing(Green) set with outliers removed')
plt.ylabel('Residuals')
plt.xlabel('X')
plt.show()


# In[85]:


# REf: https://hackernoon.com/predicting-the-price-of-houses-in-brooklyn-using-python-1abd7997083b
regressionTree_imp = regr_rf.feature_importances_
plt.figure(figsize=(16,6))
plt.yscale('log',nonposy='clip')
plt.bar(range(len(regressionTree_imp)),regressionTree_imp,align='center')
plt.xticks(range(len(regressionTree_imp)),X_out,rotation='vertical')
plt.title('Feature Importance')
plt.ylabel('Importance')
plt.show()


# ### Finding patterns in the features that may increase or decrease the accuracy of the model by looking at the prediction errors -- Random Forest Predicted Values

# In[102]:


# Append the predictions to testing set
X_test_out['rf_predictions'] = y_pred_out_rf[:]


# ### Creating a dataframe with rows for highest prediction values produced by random forest regressor. Threshold 95% percentile

# In[130]:


X_test_out.rf_predictions.quantile(.95)
error_df_highestpredictions = X_test_out.loc[X_test_out['rf_predictions'] >= 1243969.999999998]
error_df_highestpredictions.columns


# In[131]:


error_df_highestpredictions[['BedroomCnt', 'BathroomCnt', 'BuiltYear']].hist(bins = 100, figsize=(16,12))


# #### Using different set of features for highest predictions vs lowest predictions on house prices

# In[132]:


error_df_highestpredictions[['Longitude','BGMedAge', 'BGMedHomeValue', 'BGMedIncome', 'BGMedYearBuilt']].hist(bins = 100, figsize=(16,12))


# In[134]:


X_test_out[['Longitude','BGMedAge', 'BGMedHomeValue', 'BGMedIncome', 'BGMedYearBuilt']].hist(bins = 100, figsize=(16,12))


# # Performing Ridge Regression

# In[175]:


# Split the data for Ridge regression
X_train_ridge,X_test_ridge,y_train_ridge,y_test_ridge=train_test_split(X_out, Y_out,test_size = 0.2, random_state=3)


# In[176]:


# Fit the Ridge Regression model
rr = Ridge(alpha=0.01) # higher the alpha value, more restriction on the coefficients; low alpha > more generalization
rr.fit(X_train_ridge, y_train_ridge)


# In[178]:


# Calculating scores
#Ref:https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b
ridge_train_score = rr.score(X_train_ridge,y_train_ridge)
ridge_test_score = rr.score(X_test_ridge, y_test_ridge)

print("ridge regression train score:", ridge_train_score)
print("ridge regression test score:", ridge_test_score)


# In[181]:


predictions_ridge = rr.predict(X_test_ridge)


# In[183]:


X_test_ridge.head()
# Append the predictions to testing set
X_test_ridge['rf_predictions'] = predictions_ridge[:]


# #### Finding patterns in the features using Ridge regression model

# In[186]:


X_test_ridge.rf_predictions.quantile(.95) #1186760.4061410949
error_df_highestpredictions = X_test_ridge.loc[X_test_ridge['rf_predictions'] >= 1186760.4061410949]


# In[190]:


error_df_highestpredictions[['BedroomCnt', 'BathroomCnt', 'FinishedSquareFeet', 'GarageSquareFeet',
       'LotSizeSquareFeet', 'StoryCnt', 'BuiltYear', 'ViewType', 'Latitude',
       'Longitude', 'BGMedHomeValue', 'BGMedRent', 'BGMedYearBuilt',
       'BGPctOwn', 'BGPctVacant', 'BGMedIncome', 'BGPctKids', 'BGMedAge']].hist(bins = 100, figsize=(16,12))


# In[191]:


X_test_ridge[['BedroomCnt', 'BathroomCnt', 'FinishedSquareFeet', 'GarageSquareFeet',
       'LotSizeSquareFeet', 'StoryCnt', 'BuiltYear', 'ViewType', 'Latitude',
       'Longitude', 'BGMedHomeValue', 'BGMedRent', 'BGMedYearBuilt',
       'BGPctOwn', 'BGPctVacant', 'BGMedIncome', 'BGPctKids', 'BGMedAge']].hist(bins = 100, figsize=(16,12))


# In[ ]:




