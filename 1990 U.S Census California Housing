#!/usr/bin/env python
# coding: utf-8

# ## Predicting Housing Prices in California using Neural Network and 𝑘-Nearest Neighbor 
# 
# Angeline Diongson

# ## Introduction
# 
# Rising housing costs have become a pressing issue now more than ever, especially in urban areas with denser populations. This has led to increased interest in understanding the factors influencing housing prices and developing predictive models that forecast prices based on observed trends of the housing market.
# The dataset "fetch_california_housing" is derived from sklearn: https://scikit-learn.org/dev/modules/generated/sklearn.datasets.fetch_california_housing.html based on the 1990 U.S census and is comprised of 20,640 samples and 9 features such as MedInc', 'HouseAge', and 'AveRooms.
# 
# In this project, two machine learning techniques will be used to predict housing prices: Neural Networks (NN) and 𝑘-Nearest Neighbors (kNN). Neural networks entails configuring multiple layers in a model, compiling it, and then training it to make accurate predictions. The 𝑘NN algorithm, tunes the hyperparameter 
# 𝑘, and is used to determine the number of nearest neighbors to consider when making predictions. Both methods will be tested to compare their performance in predicting housing prices, allowing to identify the most effective approach for this relevant problem. I infer that the neural network will accurately predict housing prices on the test dataset due to the algorithm's capability to predict based on learned patterns. 
# 
# With this analysis, this model could be used as a tool to  predict housing prices based on various features that may affect costs in other urban areas of interest.

# The data will be split to train-test, with the test-size being 30% of the data for the Neural Network build:

# In[1]:


# Imports libraries and dataset from sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Loads California dataset from sklearn
homes = fetch_california_housing()
X, y = homes.data, homes.target


# In[2]:


print(homes.feature_names[0:9])


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=60)


# In[37]:


import pandas as pd

# Converts training data to a DataFrame
df_train = pd.DataFrame(X_train, columns=homes.feature_names)
df_train['Target'] = y_train

print(df_train.head())

# Converts test data to a DataFrame
df_test = pd.DataFrame(X_test, columns=homes.feature_names)
df_test['Target'] = y_test


print(df_test.head())


# ## Data Overview
# The first row in the test data shows the MedInc of 2.67 refers to the median income of households in the block group where the medianincome is $26,765.  
# 
# The median age of houses in that area is around 20 years, the average rooms per household in that area has around 5.3 rooms, with the average number of bedrooms per household being approximately 1.1. The total population in that area is 919 where the average number of occupants per house is 2.56. The latitude and longitude places this area in North California and the median house value in that area equates to $96,400.
# 

# In[38]:


print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# ## Preprocessing for Neural Network Modeling
# 
# The data will be standardized as part of the NN preprocessing step for the model.

# In[39]:


# Data scaled for Neural Network
scaler = StandardScaler() # tool from  sklearn.preprocessing that removes the mean and scales to unit variance.
X_train_scaled = scaler.fit_transform(X_train) # calculates mean and std. deviation of features
X_test_scaled = scaler.transform(X_test) # incorporates the standardization formula


# Prior to building a neural network, the layers of the model must be configured then compiled. Numpy and tesor flow seeds will be set so re-runing the cells result in starting with the same initial values of all the parameters to beestimated, which gives reproducibility of the same results.

# In[40]:


# Tensorflow and Keras imported to create NN for regression
import tensorflow as tf
from tensorflow.keras import layers

np.random.seed(123) # Set the seed in NumPy
tf.random.set_seed(1234)  # Set the seed in TensorFlow

# Building the neural network
mynet = tf.keras.Sequential([
    layers.Dense(units=128, activation='relu', input_shape=(X_train_scaled.shape[1],)), # layer 1, testing with wider first layer to get subsequent narrower layers 
    layers.Dense(units=64, activation='relu'), # layer 2
    layers.Dense(units=1)  # Single output for price prediction
])

# Compiling the model
mynet.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Viewing the model summary
mynet.summary()


# ## Training and Fitting the Model
# For fitting the model,the fit method of object mynet will be used.

# In[41]:


# the model will train for 10 epochs, with a 30% split, verbose displays progress for each epoch
mynet.model = mynet.fit(X_train_scaled, y_train, epochs=10, validation_split=0.3, verbose=1)


# As we can see above, the gaps between training and validation loss/MAE throughout 10 epochs such that validation loss and MAE decreased to 0.2955 and 0.3765, respectively, indicating a trend towards good generalization such that there is no overfitting, though some fluctuations in highlight potential noise or complexity in the data.
# 
# ## Data Analysis
# - Increasing epochs may further reduce loss and monitor for signs of overfitting.
# - Tuning the Hyperparameter, which will be done in kNN may optimize performance, in addition to adjusting number of units in the layers. 
# The model as shown below is evaluated on performance on the test dataset: 

# In[42]:


# Evaluates the model mean absolute error on the test set
test_loss, test_mae = mynet.evaluate(X_test_scaled, y_test, verbose=2)

# Print the results
print("Test Mean Absolute Error:", test_mae)


# The low Mean Squared Error (MSE) on the test dataset is 0.3117, indicating better predictions where the Mean Absolute Error is 0.3824 meaning that on average, the model's predictions are off by approximately 0.3824 units. Below, the model mynet is used to evaluate the performance of the regression model through root mean squared error (RMSE).

# In[43]:


# Predicting and evaluation 
from sklearn.metrics import mean_squared_error
import numpy as np

y_pred_mynet = mynet.predict(X_test_scaled)
rmse_mynet = np.sqrt(mean_squared_error(y_test, y_pred_mynet))
print(f"{rmse_mynet}")


# The NN RMSE of 0.558 indicates the model's predictions are further from actual values.

# ## Implementing 𝑘-NN
# k-Nearest Neighbors will be used for regression. The hyperparameter 𝑘, will be finetuned to find its optimal value, and then fit onto the model with that value of 𝑘.

# In[44]:


# Imports the KNeighbors package 
from sklearn.neighbors import KNeighborsRegressor
 
# Sets the seed     
np.random.seed(123)

tr = np.random.choice(20640, size=14448, replace=False)

# Display the first few elements of the sample
print(tr[:6])


# Vector kvec of possible values for the tuning parameter 𝑘
# will be created for the kNN model.Included could be values up to the size of the training sample train (14448). nk is the length of kvec (14448).

# In[48]:


kvec = range(1,14448)
nk = len(kvec)


# Vectors for outRMSE and inRMSE will be created for each 𝑘∈ kvec and fit on the train data.The out-of-sample root mean squared error 𝑜𝑢𝑡𝑅𝑀𝑆𝐸𝑘 and as in-sample root mean squared error 𝑖𝑛𝑅𝑀𝑆𝐸𝑘 will also be computed, the memory for both are first pre-allocated. The for loop below fits the kNN model on nthe  train data for each 𝑘∈kvec. This is important because the optimal 𝑘 is calculated and to evaluate the performance of the model across a  range of k values.

# In[57]:


# Arrays pre-allocated for RMSE
outRMSE = np.zeros(nk)  # Out-of-sample (test) RMSE
inRMSE = np.zeros(nk)   # In-sample (train) RMSE


# In[59]:


# Iterate over all k values
for i, k in enumerate(kvec):
    # Train kNN model
    kmod = KNeighborsRegressor(n_neighbors=k, weights='uniform')
    kmod.fit(X_train, y_train)
    


# In[63]:


# Predictions for test and train data
y_test_pred = kmod.predict(X_test)
y_train_pred = kmod.predict(X_train)
  
  # Calculate RMSE for test and train data
outRMSE[i] = np.sqrt(mean_squared_error(y_test, y_test_pred))
inRMSE[i] = np.sqrt(mean_squared_error(y_train, y_train_pred))

# prints RMSE of first 5
print("Out-of-sample RMSE:", outRMSE[:5])
print("In-sample RMSE:", inRMSE[:5])


# ## Data Analysis
# - The out-of-sample (test) RMSE values decrease as 𝑘
# increases, starting from 1.29 and then approaching 1.09. This suggests that the model's performance improves as the number of neighbors increases, leading to more stable and generalized predictions.
# - The in-sample (train) RMSE values increases as 𝑘 increases, sarting from 0 and rises to 0.85 for 𝑘= 5. This suggests that with smaller values of 𝑘, the model is fiting the train data too closely--overfitting.
# - Optimal 𝑘 would be the value that minimizes out-of-sample RMSE while staying within range of an acceptable in-sample error.

# In[72]:


# Visualization of RMSE

plt.figure(figsize=(12, 6))

# Plot out-of-sample RMSE
plt.plot(kvec, outRMSE, label='Test RMSE (out-of-sample)', color='red')

# Plot in-sample RMSE
plt.plot(kvec, inRMSE, label='Train RMSE (in-sample)', color='blue')

# Add labels and legend
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('RMSE')
plt.title('RMSE vs. Number of Neighbors (k)')
plt.legend()
plt.show()


# For small 𝑘 values on the left, both train and test RMSE values are higher suggesting that the model may be overfitting the training data. As 𝑘 increases, te RMSE decreases initially, but at very high values of 𝑘,  both train and test RMSE increase significantly indicating underfitting as the model generalizes. From this, the best 𝑘 value can be solved that optimally fits the model without over-generalization. 

# In[64]:


# Finds the index of the minimum value in outRMSE
k_index = np.argmin(outRMSE)  

# Gets k value
kbest = kvec[k_index]

# Prints the result
print(f"Best value of 𝑘  is: {kbest}")


# The best value of 𝑘 is 93, meaning that using 93 neighbors to predict housing prices in the California housing dataset is the best generalization value to reduce model variance and prevent overfitting. With a smaller 𝑘, the model becomes more generalized and reduces variance with the cost of increasing bias.

# ## Visualization of Comparison
# The model is compared based on the performance of kNN and NN models in terms of RMSE and MSE. Below also shows the predictions vs. true values.

# In[73]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_knn, label='kNN Predictions', alpha=0.7, color='purple')
plt.scatter(y_test, y_pred_mynet, label='NN Predictions', alpha=0.7, color='teal')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='pink', label='Ideal ')
plt.xlabel('True Prices')
plt.ylabel('Predicted Prices')
plt.legend()
plt.show()


# The neural network shows closer clustering around the ideal line with lower spread of predictions compared to the kNN prediction.
# The kNN predictions are more dispersed and less aligned with the ideal line, especially around extreme values for the target variable indicating relatively higher errors compared to the neural network. The straight line clustering of both methods at the 5 on the "True Prices" axis in the plot indicate the model predictions are strugglingto generalize for higher true price values. This could be a result of both kNN and NN models having difficulty predicting the prices for the higher end of the housing price in the dataset.

# ## Conclusion
# From the calculations above, the Neural Network model showed decreases over 10 epochs, indicating that it was learning from the training data. During the 10th epoch, the validation loss and validation MAE stabilize at 0.2955 and 0.3765, respectively. These metrics suggest that the model generalizes well to unseen data. In analyzing the scatterplot comparison of the two models, the neural network  demonstrates closer clustering around the ideal line compared to the kNN predictions, showing that the neural network achieves higher accuracy in predictions.
# The neural network outperforms the kNN model in terms of accuracy, as seen from the scatter plot and the training/validation loss metrics. Though not to the approximate predictions of the train data, the neural network demonstrates more of an advantage over the kNN model in predicting housing prices, owing to its ability to "mimic" the trends observed in this dataset--which was the inference that was not proven. Perhaps to improve model performance, the complexity could be increased by adding more layers, and increasing Epochs, further engineering the target variable and cross-validation for both methods.

# In[ ]:




