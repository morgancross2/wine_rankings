import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import nltk

def get_q1_viz(train):
    '''
    This function takes in train and returns the visualization for question 1
    in the winespectator.com dataset.
    '''
    # make it big
    plt.figure(figsize=(12,8))
    # plot hist for red wines and non-red wines
    sns.histplot(data=train[train.red == 1], x='score', color='darkred')
    sns.histplot(data=train[train.red == 0], x='score', color='pink')
    # find score averages for red and non-red wines
    red_avg = train[train.red == 1].score.mean()
    other_avg = train[train.red == 0].score.mean()
    # plot lines at those averages and label them
    plt.axvline(x= red_avg, color='red', lw=3)
    plt.text(93.2,213, '''Average Red Wine Score''')
    plt.axvline(x=other_avg, color='black', lw=3)
    plt.text(89.3,213, '''Average Non-Red Wine Score''')
    # label the graph
    plt.title('Red wines have a higher average score than other wines')
    # show the graph
    plt.show()
    
def get_q1_stats(train):
    '''
    This function takes in train and returns the statistical results for question 1
    in the winespectator.com dataset.
    '''
    # create the samples
    reds = train[train.red == 1].score
    other = train[train.red == 0].score
    # set alpha
    α = 0.05
    # run the levene test to check for equal variances
    s, pval = stats.levene(reds, other)
    # run the ttest based on the levene results
    t, p = stats.ttest_ind(reds, other, equal_var=pval > α)
    # evaluate results based on the t-statistic and the p-value
    if ((t > 0) & (p/2 < α)):
        print('''Reject the Null Hypothesis.
    Findings suggest the mean score of red wine is greater than the mean score of all other wines. ''')
    else:
        print('''Fail to reject the Null Hypothesis.
    Findings suggest the mean score of red wine is less than or equal to the mean score of all other wines.''')
    print()
    
    
def get_q2_viz(train):
    '''
    This function takes in train and returns the visualization for question 2
    in the winespectator.com dataset.
    '''
    # make it big
    plt.figure(figsize=(12,8))
    # plot the box plot
    sns.boxplot(data=train, x='score', y='price', color='darkred')
    # give it a title
    plt.title('Price and score have a positive linear relationship')
    # show it
    plt.show()
    
def get_q2_stats(train):
    '''
    This function takes in train and returns the statistical results for question 2
    in the winespectator.com dataset.
    '''
    # set alpha
    α = 0.05
    # run the spearman's correlation test on price and score
    r, p = stats.spearmanr(train.price, train.score)
    # evaluate the results against the pvalue
    if p < α:
        print('''Reject the Null Hypothesis.
    Findings suggest there is a linear relationship between a wine's score and its price.
    Spearman's Correlation was: '''+ str(round(r,3)))
    else:
        print('''Fail to reject the Null Hypothesis.
    Findings suggest there is not a linear relationship between a wine's score and its price.''')
    print()
    
def get_q3_viz(train):
    '''
    This function takes in train and returns the visualization for question 3
    in the winespectator.com dataset.
    '''
    # make it big
    plt.figure(figsize=(12,8))
    # plot it
    sns.scatterplot(data=train, x='top100_year', y='top100_rank', hue='score').invert_yaxis()
    # add a horizonal line at 50
    plt.axhline(50, color='black')
    # add space for the labels
    plt.xlim(1984.5,2020)
    # add the labels
    plt.text(1984.8,48, 'Top 50')
    plt.text(1984.8,53, 'Bottom 50')
    # give it a title
    plt.title('Higher scoring wines also rank higher')
    # move the legend
    plt.legend(loc='upper left', title='score', framealpha=1)
    # show it
    plt.show()
    
def get_q3_stats(train):
    '''
    This function takes in train and returns the statistical results for question 3
    in the winespectator.com dataset.
    '''
    # create the samples
    top = train[train.top100_rank <= 50].score
    bottom = train[train.top100_rank > 50].score
    # set alpha
    α = 0.05
    # run the levene test to check for equal variances
    s, pval = stats.levene(top, bottom)
    # run the ttest based on the levene results
    t, p = stats.ttest_ind(top, bottom, equal_var=pval > α)
    # evaluate results based on the t-statistic and the p-value
    if ((t > 0) & (p/2 < α)):
        print('''Reject the Null Hypothesis.
    Findings suggest the mean score of wines ranking in the top 50 is higher.''')
    else:
        print('''Fail to reject the Null Hypothesis.
    Findings suggest the mean score of wines ranking in the top 50 is lower or equal to the bottom 50.''')
    print()
    
    
def get_q4_viz(train):
    '''
    This function takes in train and returns the visualization for question 4
    in the winespectator.com dataset.
    '''
    # make it big
    plt.figure(figsize=(12,8))
    # plot a histograph for each country
    sns.histplot(data=train[train.France == 1], x='score', label = 'France', color='darkred')
    sns.histplot(data=train[train.California == 1], x='score', label = 'California', color='lightgreen')
    sns.histplot(data=train[train.Italy == 1], x='score', label = 'Italy', color='blue')
    sns.histplot(data=train[train.Australia == 1], x='score', label = 'Australia', color = 'green')
    sns.histplot(data=train[train.Washington == 1], x='score', label = 'Washington', color='yellow')
    sns.histplot(data=train[train.Spain == 1], x='score', label = 'Spain', color='pink')
    # add a line to show the population average and label it
    plt.axvline(train.score.mean(), color='red', lw=3)
    plt.text(92.9,100, 'Average Score')
    # show the legend
    plt.legend()
    # show the whole graph
    plt.show()
    
def get_q4_stats(train):
    '''
    This function takes in train and returns the statistical results for question 4
    in the winespectator.com dataset.
    '''
    # create the samples
    french = train[train.France == 1].score
    pop = train.score.mean()
    # set alpha
    α = 0.05
    # run the ttest based on the levene results
    s, p = stats.ttest_1samp(french, pop)
    # evaluate results based on the t-statistic and the p-value
    if (p < α):
        print('''Reject the Null Hypothesis.
    Findings suggest the mean score of wines from France is higher than the population average.''')
    else:
        print('''Fail to reject the Null Hypothesis.
    Findings suggest the mean score of wines from France is lower than or equal to the population average.''')
    print()
    
def get_q5_viz(train):
    '''
    This function takes in train and returns the visualization for question 5
    in the winespectator.com dataset.
    '''
    # create an empty string to hold all the notes
    big_note = ''
    # loop through the notes adding them together
    for note in train[train.score > train.score.mean()].note:
        big_note += note
    # make them into a dataframe
    top = (pd.Series(nltk.ngrams(big_note.split(), 2)).value_counts().head(20))
    # plot them by frequency
    top.sort_values(ascending=False).plot.barh(color='darkred', width=.9, figsize=(12, 8))
    # give it title and label the axis
    plt.title('20 Most frequently occuring wine bigrams for above average wine scores')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurances')

    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)
    plt.show()
    
    # do it all again for below average score wines
    big_note = ''
    for note in train[train.score < train.score.mean()].note:
        big_note += note
    top = (pd.Series(nltk.ngrams(big_note.split(), 2)).value_counts().head(20))

    top.sort_values(ascending=False).plot.barh(color='darkred', width=.9, figsize=(12, 8))

    plt.title('20 Most frequently occuring wine bigrams for below average wine scores')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurances')

    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)
    plt.show()
    
