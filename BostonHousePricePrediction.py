##### Import libraries necessary for this project #######

#STandard data manipulation libraries
import numpy as np
import pandas as pd

#Import import standard visuals library and novel visuals code visuals.py
import seaborn as sns
import matplotlib.pyplot as plt
import Visuals as vs

#Import scikit-learn libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import ShuffleSplit
#------------------------------------------------------------------------------------------------------

#Function to fit Decision Tree Regression based on certain parameters
def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 42)

    # Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth':[1,2,3,4,5,6,7,8,9,10]}

    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # Create the grid search cv object --> GridSearchCV()
    grid = GridSearchCV(estimator=regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_
#--------------------------------------------------------------------------------------------------------------
# Create new similar function
def fit_model_2(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """

    # Create cross-validation sets from the training data
    # ShuffleSplit works iteratively compared to KFOLD
    # It saves computation time when your dataset grows
    # X.shape[0] is the total number of elements
    # n_iter is the number of re-shuffling & splitting iterations.
    cv_sets = ShuffleSplit(X.shape[0], test_size = 0.20, random_state = 42)

    # TODO: Create a decision tree regressor object
    # Instantiate
    regressor = DecisionTreeRegressor(random_state=0)

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    dt_range = range(1, 11)
    params = dict(max_depth=dt_range)

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
    # We initially created performance_metric using R2_score
    scoring_fnc = make_scorer(performance_metric)

    # TODO: Create the grid search object
    # You would realize we manually created each, including scoring_func using R^2
    rand = RandomizedSearchCV(regressor, params, cv=cv_sets, scoring=scoring_fnc)

    # Fit the grid search object to the data to compute the optimal model
    rand = rand.fit(X, y)

    # Return the optimal model after fitting the data
    return rand.best_estimator_

#---------------------------------------------------------------------------------------------
#Create metric to evaluarte the performance of each regression model
def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true (y_true) and predicted (y_predict) values based on the metric chosen. """
    from sklearn.metrics import r2_score

    score = r2_score(y_true, y_predict)
    
    # Return the score
    return score
#-----------------------------------------------------------------------------------------------------
#Function to compare the predicted price to its nearest neighbours in the dataset
def nearest_neighbor_price(x):
    num_neighbors=5
    # x is your vector and X is the data set.
    def find_nearest_neighbor_indexes(x, X):
        # Instantiate
        neigh = NearestNeighbors(num_neighbors)
        # Fit
        neigh.fit(X)
        distance, indexes = neigh.kneighbors(x)
        return indexes
        # This returns, the position, say for example [4, 55, 22]
        # array([[357, 397, 356, 141, 395]])
    indexes = find_nearest_neighbor_indexes(x, features)
    # Create list
    sum_prices = []
    # Loop through the array
    for i in indexes:
        # Append the prices to the list using the index position i
        sum_prices.append(prices[i])
    # Average prices
    neighbor_avg = np.mean(sum_prices)
    # Return average
    return neighbor_avg
#----------------------------------------------------------------------------------------------------
################### Complete Algorithm ###################################

#Load data and seperate into feature and target variable
dataset = 'housing.csv'
data = pd.read_csv(dataset)
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)



#Rudimentary EDA to explore corrolations in the data
cols = data.columns
cm = np.corrcoef(data.values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                cbar=True,
                annot=True,
                square=True,
                yticklabels=cols,
                xticklabels=cols)

sns.pairplot(data, height=2.5)
plt.tight_layout()

vs.ModelLearning(features, prices) #Display how the model quality changes with differing parameters
vs.ModelComplexity(features, prices)

# Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state = 42)

# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)
reg_2 = fit_model_2(X_train, y_train) 

#Input new data to be predicted 
RM = int(input('Rooms: '))
LSTAT = int(input('Poverty %: '))
PTRATIO = int(input('Pupil teacher ratio: '))
client_data = [[RM,LSTAT,PTRATIO]] #Input client data

#Predict selling price of client house and compare to neighbouring houses in the dataset (houses with similar feature values)
value = nearest_neighbor_price(client_data)
price = reg.predict(client_data)[0]

#Print result
print (f"\nPredicted selling price for your home: ${round(price,0)}, compared to its neighbours at ${value}")
print(" ")
print(" ") 

#Sensitivity test
vs.PredictTrials(features, prices, fit_model, client_data)

