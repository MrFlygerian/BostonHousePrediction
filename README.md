# BostonHousePrediction
This is regression training using boston house price data from 1978. It's main purpose is to predic the price of a house based on the number of rooms in the house, the relative poverty percentage in the house's neighborhood and the average class size, represented as the pupil-teacher ratio. The regressor used here is a Decision Tree regressor.

There is also some data analaysis performed near the beginning of the script, to give a brief overview of the data its distribution and correlations, as well as some information about the nature of the models being used. The Model Learning and Model Complexity graphs are used for model analysis and are generated from functions in the visuals file. They give a description of how the model predictive power changes as a function of kew Decision Tree parameters (in this case the max depth is the parameter being principally examined).
When the Model Learning script is run, some warning will appear. Also, for the Model Learning graphs, there is no traning score for the oth training size. This is purely due to the inability of the regressor to provide score for null traning points, and is nothing to be concerned aboout.

For the error analysis, a function from the aforementioned visuals file is used to run the prediction 10 times using variable random states and the variance calculated. This nicely describes the models' sensitivity in both monetary and percentage terms.

To run the prediction algorithm simply use import and make sure to have all the dependencies (the data and the visuals file) saved in the same directory. All libraries used here are fairly standard. 

This algorithm was adapated from a towardsdatascience article by Victor Roman (the URL is available below). The article is rather brief and bereft of detail, and significant research had to be done to figure out why parts of the instructions didn't work. All the relevant material is available in the article links, and apart from some minor design details little revision was required.

https://towardsdatascience.com/machine-learning-project-predicting-boston-house-prices-with-regression-b4e47493633d
