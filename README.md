# Wine Score Classification Project
by: Morgan Cross

This project is designed to predict the scoring of wine using the top 100 wines from 1988 to 2019 from winespectator.com. 

-----
## Project Overview:

#### Objectives:
- Document code, process (data acquistion, preparation, exploratory data analysis and statistical testing, modeling, and model evaluation), findings, and key takeaways in a Jupyter Notebook Final Report.
- Create modules (wrangle.py) that make the process repeateable and the report (notebook) easier to read and follow.
- Ask exploratory questions of the data that will help you understand more about the attributes and drivers of customers churning. Answer questions through charts and statistical tests.
- Construct a model to predict customer churn using classification techniques, and make predictions for a group of customers.
- Refine work into a Report, in the form of a jupyter notebook, that you will walk through in a 5 minute presentation to a group of collegues and managers about the work you did, why, goals, what you found, your methdologies, and your conclusions.
- Be prepared to answer panel questions about your code, process, findings and key takeaways, and model.

#### Project Deliverables:
- this README.md walking through the project details
- final_report.ipynb displaying the process, findings, models, key takeaways, recommendation and conclusion
- wrangle.py with all data acquisition and preparation functions used
- working_report.ipynb showing all work throughout the pipeline

-----
## Executive Summary:
Goals
- Identify causes of wine score
- Build a model to predict bottle score

Key Findings
 - Highly scored wines tend to rank higher year after year
 - Highly scored wines also show trends of costing more than others

Takeaways
-  My best model was able to predict this scoring with 44% accuracy, but requires more data to hone in on what makes a wine truly great. 

Recommendation
- Locate and utilize data on a wine's flavor profile in order to better encompass the scoring criteria

-----
## Data Dictionary:
| Target | Type | Description |
| ---- | ---- | ---- |
| score | int | scoring of the wine |


| Feature Name | Type | Description |
| ---- | ---- | ---- |
| aged | int | difference between vintage and issue year |
| Australia | int | 1 if the bottle is from Austrailia, 0 if not |
| California | int | 1 if the bottle is from California, 0 if not |
| France | int | 1 if the bottle is from France, 0 if not |
| issue_year | int | year the wine was available for purchase |
| Italy | int | 1 if the bottle is from Italy, 0 if not |
| note | string | taster's comments |
| price | float | price in USD of one bottle of wine |
| red | int | 1 if the bottle is a red, 0 if not |
| Spain | int | 1 if the bottle is from Spain, 0 if not |
| top100_rank | int | rank of the wine on the top 100 list |
| top100_year | int | year the wine made the top 100 list |
| vintage | int | year the grapes were collected for the bottle |
| Washington | int | 1 if the bottle is from Washington, 0 if not |
| white | int | 1 if the bottle is a white, 0 if not |
| wine | string | name of the bottle of wine |
| winery | string | name of the winery |

-----
## Planning
 - Create deliverables:
     - README
     - final_report.ipynb
     - working_report.ipynb
     - py files
 - Bring over functional wrangle.py, explore.py, and model.py files
 - Acquire the data from the Code Up database via the acquire function
 - Prepare and split the data via the prepare function
 - Explore the data and define hypothesis. Run the appropriate statistical tests in order to accept or reject each null hypothesis. Document findings and takeaways.
 - Model a baseline in predicting score and document the error.
 - Fit and train three (3) classification models to predict score on the train dataset.
 - Evaluate the models by comparing the train and validation data.
 - Select the best model and evaluate it on the train data.
 - Develop and document all findings, takeaways, recommendations and next steps.

-----
## Data Aquisition and Preparation
Files used:
 - wrangle.py

For acquisition, I called my acquire function from wrangle.py. This function:
- web scrapes the top 100 wines from winespectator.com from 1988 to 2019
- creates a local CSV of the table, if not already saved locally
- provided 3200 bottles of wine worth of data over 16 different features

For preparation, I called my prepare function from wrangle.py. This function:
- handles nulls and other missing values
- corrects input values to match column format
- renames columns to be human readable
- creates dummies for locations and colors
- removes outliars in price
- feature engineers 'aged'
- uses NLP to capture top words in tasters' notes
- splits data into train, validate, and test datasets

-----
## Data Exploration
Files used:
- explore.py

Questions Addressed:
1. Do red wines score higher than other wines?
2. Is there a linear relationship between a wine's price and its score?
3. Do highly scored wines rank higher?
4. Does location play a role in score?
5. How do words in the tasters' notes change for wines above and below the average score?


### Test 1: T-Test - Red Wine Score
- A T-Test evaluates if there is a difference in the means of two continuous variables. This test is looking at a two samples and one tail.
- This test returns a p-value and a t-statistic.
- This test will compare the average red wine score against the average non-red wine score.
- Confidence level is 95%
- Alpha is 0.05

Hypothesis:
 - The null hypothesis is the mean score of red wine is equal to or less than all other wines mean score.
 - The alternate hypothesis is the mean score of red wine is greater than all other wines mean score.

Results: 
- p-value is less than alpha
- t-statistic is positive
- I rejected the Null Hypothesis, suggesting the mean score of red wine is greater than all other wines mean score.

### Test 2: Spearman's Correlation - Score vs. Price
- This test evaluates if there is a linear relationship between two variables.
- This test returns a correlation and a p-value.
- This test will compare the score and price features.
- Confidence level is 95%
- Alpha is 0.05

Hypothesis:
- The null hypothesis is there is not a linear relationship between a wine's score and its price.
- The alternative hypothesis is there is a linear relationship between a wine's score and its price.

Results: 
- p-value is less than alpha
- I rejected the Null Hypothesis, suggesting there is a linear relationship between a wine's score and its price.

### Test 3: T-Test - Score vs. Rank
- A T-Test evaluates if there is a difference in the means of two continuous variables. This test is looking at a two samples and one tail.
- This test returns a p-value and a t-statistic.
- This test will compare top and bottom rank average scores.
- Confidence level is 95%
- Alpha is 0.05

Hypothesis:
 - The null hypothesis is the average score of wines ranking in the top 50 is less than or equal to the average score of wines ranking in the bottom 50.
 - The alternate hypothesis is the average score of wines ranking in the top 50 is greater than the average score of wines ranking in the bottom 50.

Results: 
- p-value is less than alpha
- t-statistic is positive
- I rejected the Null Hypothesis, suggesting the average score of wines ranking in the top 50 is greater than the average score of wines ranking in the bottom 50.

### Test 4: T-Test - Score vs. Location
- A T-Test evaluates if there is a difference in the means of two continuous variables. This test is looking at a two samples and one tail.
- This test returns a p-value and a t-statistic.
- This test will compare the average score from French wines against the population average score.
- Confidence level is 95%
- Alpha is 0.05

Hypothesis:
 - The null hypothesis is the average score of wines from France is lower than or equal to the populations average score.
 - The alternate hypothesis is the average score of wines from France is greater than the populations average score.

Results: 
- p-value is less than alpha
- t-statistic is positive
- I rejected the Null Hypothesis, suggesting the average score of wines from France is greater than the populations average score.

### Takeaways from exploration:
- Red wines score higher than other wines on average.
- There is a linear relationship between price and score.
- Higher ranked wines have a higher average score.
- French wines tend to score higher than the population average.
- There are some words that show up more often in above average scoring wines than below. However, these lines are blurry.

-----
## Modeling:
### Model Preparation:

### Baseline:
Baseline Results
- Train score feature's mode is 91.
- The baseline accuracy is 15.4%.

Selected features to input into models:
- vintage
- price
- top100_rank
- red
- white
- location dummies
- note dummies

#### Model 1: Logistic Regression
- Hyperparameters: C = 1.0, random_state = 123

#### Model 2: Decision Tree
- Hyperparameters: max_depth = 4, random_state = 123 

#### Model 3: Random Forest
- Hyperparameters: max_depth = 7, min_samples_leaf = 5, random_state = 123

### Selecting the Best Model:
|          |   train_accuracy |   train_rmse |   validate_accuracy |   validate_rmse |
|:---------|-----------------:|-------------:|--------------------:|----------------:|
| baseline |         0.153605 |     4.82497  |          nan        |       nan       |
| logit    |         0.415361 |     1.33183  |            0.339593 |         1.40422 |
| tree     |         0.443574 |     1.12862  |            0.43662  |         1.13004 |
| forest   |         0.718913 |     0.999739 |            0.446009 |         1.19007 |

The Logistic Regression model performed the best for recall.

### Testing the Model:
|          |   accuracy |    rmse |
|:---------|-----------:|--------:|
| train    |   0.443574 | 1.12862 |
| validate |   0.43662  | 1.13004 |
| test     |   0.383412 | 1.23207 |

-----
## Conclusion:
Wine scoring is a point of pride for wineries everywhere. My best model was able to predict this scoring with 44% accuracy, but requires more data to hone in on what makes a wine truly great. 

#### Recommendations
- Add taster evaluation data on the profile (dry, sweet, full-bodied, etc.) for a professional assessment and metric to compare the wines
- Build out metrics that numerically evaluate what the criteria are for scoring

#### Next Steps
- Look into websites such as vivino.com to gather customer taste profile data
- Configure the location and winery data into a numerical form and see how that effects the model

-----
## How to Recreate:
1. Utilize the following files found in this repository:
- final_report.ipynb
- wrangle.py
- explore.py
- model.py

2. Run the final_report.ipynb notebook.