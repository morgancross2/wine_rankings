import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as met
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def scale_data(train, val, test, cols_to_scale):
    '''
    This function takes in train, validate, and test dataframes as well as a
    list of features to be scaled via the MinMaxScalar. It then returns the 
    scaled versions of train, validate, and test in new dataframes. 
    '''
    # create copies to not mess with the original dataframes
    train_scaled = train.copy()
    val_scaled = val.copy()
    test_scaled = test.copy()
    
    # create the scaler and fit it
    scaler = MinMaxScaler()
    scaler.fit(train[cols_to_scale])
    
    # use the scaler to scale the data and resave
    train_scaled[cols_to_scale] = pd.DataFrame(scaler.transform(train[cols_to_scale]),
                                               columns = train[cols_to_scale].columns.values).set_index([train.index.values])
    val_scaled[cols_to_scale] = pd.DataFrame(scaler.transform(val[cols_to_scale]),
                                               columns = val[cols_to_scale].columns.values).set_index([val.index.values])
    test_scaled[cols_to_scale] = pd.DataFrame(scaler.transform(test[cols_to_scale]),
                                               columns = test[cols_to_scale].columns.values).set_index([test.index.values])
    
    return train_scaled, val_scaled, test_scaled


def run_models(train, val, test):
    '''
    This function takes in train, validate, and test datasets, scales them, and 
    runs three models (DecisionTree, RandomForest, Logistic Regression) on the 
    winespectator dataset. It returns two dataframes. The first is the results
    of the models on train and validate, and the second is the results of the best
    model on test.
    '''
    # set the features to be scaled
    cols = train.columns.tolist()[4:]
    cols.append('vintage')
    # scale them
    train_scaled, val_scaled, test_scaled = scale_data(train[cols], val[cols], test[cols], cols)
    
    # create list of features to not be put through the model
    drop_list = [
             'score',
             'issue_year',
             'top100_year',
    ]
    
    # set up X and y values using the drop_list just created
    X_train, y_train = train_scaled.drop(columns=drop_list), train.score
    X_val, y_val = val_scaled.drop(columns=drop_list), val.score
    X_test, y_test = test_scaled.drop(columns=drop_list), test.score

    # create a dict to hold all model results
    results = {}

    # Create baseline metric from mode
    thing = pd.DataFrame(train.score)
    thing.columns = ['actual']
    thing['pred'] = round(train.score.mean())
    # Add baseline to results
    results['baseline'] = {'train_accuracy': met.accuracy_score(thing.actual,thing.pred),
                            'train_rmse': met.mean_squared_error(thing.actual, thing.pred)
                        }

    # Create Logistic Regression Model
    logit = LogisticRegression(random_state=123)
    # Fit it to the train data
    logit.fit(X_train, y_train)
    # Run train
    logit_accuracy = met.accuracy_score(y_train, logit.predict(X_train))
    logit_rmse = met.mean_squared_error(y_train, logit.predict(X_train), squared=False)
    # Run validate
    val_acc = met.accuracy_score(y_val, logit.predict(X_val))
    val_rmse = met.mean_squared_error(y_val, logit.predict(X_val), squared=False)
    # Add to results
    results['logit'] = {'train_accuracy': logit_accuracy,
                                        'train_rmse': logit_rmse,
                                        'validate_accuracy' : val_acc,
                                        'validate_rmse': val_rmse}
                    

    # Create Decision Tree model
    tree = DecisionTreeClassifier(max_depth=4, random_state=123)
    # Fit it
    tree.fit(X_train, y_train)
    # Add to results
    results['tree'] = {'train_accuracy' : met.accuracy_score(y_train, tree.predict(X_train)), 
                                        'train_rmse': met.mean_squared_error(y_train, tree.predict(X_train), squared=False),
                                        'validate_accuracy' : met.accuracy_score(y_val, tree.predict(X_val)), 
                                        'validate_rmse': met.mean_squared_error(y_val, tree.predict(X_val), squared=False)
                        }

    # Create Random Forest model
    forest = RandomForestClassifier(max_depth=7, min_samples_leaf=5, random_state=123)
    # Fit it
    forest.fit(X_train, y_train)
    # Add to results
    results['forest'] = {'train_accuracy' : met.accuracy_score(y_train, forest.predict(X_train)), 
                                'train_rmse': met.mean_squared_error(y_train, forest.predict(X_train), squared=False),
                                'validate_accuracy' : met.accuracy_score(y_val, forest.predict(X_val)), 
                                'validate_rmse': met.mean_squared_error(y_val, forest.predict(X_val), squared=False)
                            }


    # Create best model dict to become dataframe of results
    final = {}
    # Create Decision Tree model
    tree = DecisionTreeClassifier(max_depth=4, random_state=123)
    # Fit it
    tree.fit(X_train, y_train)
    # Add to final
    final['train'] = {'accuracy' : met.accuracy_score(y_train, tree.predict(X_train)), 
                      'rmse': met.mean_squared_error(y_train, tree.predict(X_train), squared=False)}
    final['validate'] = {'accuracy' : met.accuracy_score(y_val, tree.predict(X_val)), 
                         'rmse': met.mean_squared_error(y_val, tree.predict(X_val), squared=False)}
    final['test'] = {'accuracy' : met.accuracy_score(y_test, tree.predict(X_test)), 
                     'rmse': met.mean_squared_error(y_test, tree.predict(X_test), squared=False)}
    
    # return model resutls for train and validate, and best model results with test
    return pd.DataFrame(results).T, pd.DataFrame(final)