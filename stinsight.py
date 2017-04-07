#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
To highlight occurences of the word "Assumptions" run this in your VIM :-
:match Todo /Assumptions/
'''

# Dataset credits :- http://www3.dsi.uminho.pt/pcortez/Downloads.html
# Dataset credits :- https://archive.ics.uci.edu/ml/datasets/Student+Performance


####################################################################################################
'''IMPORTING THE PANDAS AND LOADING THE CSV DATASET FILE'''


import pandas as pd

#df = pd.read_csv("/home/pranav/Desktop/student_data.csv")
df = pd.read_csv("student_data.csv")


######################################################################################################
'''UNDERSTANDING DATASET'''


#print(df.shape)
        # (395, 31)
#print(df.describe())
#print(df["passed"].value_counts())
        # yes 265
        # no 130

'''
Looking at the results obtained in this section, following observations were made :-

- There are no null values to be taken care of.
(By seeing total number of rows from 'df.shape' and count in each column of 'df.describe()')

- Numerical data is present in only 13 columns out of total 31 columns, so appropriate
assumptions have to be taken to analyse the data and convert them into corresponding numerical values.

- The data is rarely skewed as in nearly all cases, the mean is almost equal to median.

- The number of people passing the course are signficantly higher than the ones failing,
so data needs to be appropriately changed so that in both test and training data, distribution
is somewhat proportional.

- There are 31 features, some of these wouldn't be too much correlated with the "passed" feature,
these features would be dropped later.
'''


######################################################################################################
'''PREPROCESSING THE DATA AND CONVERTING STRING LITERALS TO CORRESPONDING INT VALUES'''


#print(df.head())

def convert_yes_no(col_name) :
    df[col_name] = df[col_name].apply(lambda x : yes_or_no[x])

global yes_or_no
yes_or_no = {"yes" : 1, "no" : 0}

col_name = ["passed", "internet", "romantic", "schoolsup", "famsup", "paid", "activities", "nursery", "higher"]

for col in col_name :
    convert_yes_no(col)


#print(df["Mjob"].value_counts())
        #other 141
        #services 103
        #at_home 59
        #teacher 58
        #health 34
#print(df["Fjob"].value_counts())
        #other 217
        #services 111
        #teacher 29
        #at_home 20
        #health 18


job_assist_studies_level = {"services" : 1, "other" : 0, "teacher" : 2, "at_home" : 2, "health" : 0}
# Assumptions :-
# If the Parent is at house for most of the time - Expected to assist his ward more.
# Else if the Parent is him/her self a teacher - Expected to be a better guide of ward.
# Else - 9-5 job i.e. no time for family or assisting the ward in studies.


#print(df["famsize"].value_counts())
        #GT3 281
        #LE3 114

support_family_type = {"GT3" : 1, "LE3" : 0}
# Assumptions :-
# To simplify the analysis, 
# GT3 i.e. family size greater than 3 is assumed to comprise of Mother, Father, Two siblings or more, thus providing more guidance to ward for studies.
# LE3 i.e. family size less than equal to 3 is assumed to comprise of Father, Mother and a single child, thus providing less guidance to ward for studies.


parent_both_or_not = {"A" : 0, "T" : 1}
# Assumptions :-
# If parents are living together, the child can be taken care of in a better way and responsibilities can be shared.


urban_rural = {"U" : 1, "R" : 0}
# Assumptions :-
# It is general observation that people in urban areas have more exposure thanks to good peer-study groups.


reason_school = {"course" : 2, "reputation" : 1, "home" : 0, "other" : 0}
# Assumptions :-
# The ward would have been more interested in studies if his preferable course or school would have been choosen.
# In all other cases, he relunctantly accepted the course making it a choice by either neutral or somewhat negative mindset.

guardian_assist = {"mother" : 1, "other" : 1, "father" : 2}
# Assumptions :-
# As according to expected strictness from guardian towards studies.

school = {"GP" : 0, "MS" : 1}
# Assumptions :-
# Baseed on the reviews on their facebook pages, quality of education at these schools was assumed.


gender = {"F" : 0, "M" : 1}


global column_to_int
column_to_int = ["Mjob", "Fjob", "famsize", "Pstatus", "address", "reason", "guardian", "school", "sex"]


global dict_to_use
dict_to_use = [job_assist_studies_level] * 2 + [support_family_type, parent_both_or_not, urban_rural, reason_school, guardian_assist, school, gender]


def convert_to_int(i) :
    df[column_to_int[i]] = df[column_to_int[i]].apply(lambda x : (dict_to_use[i])[x])


l = len(column_to_int)

for i in range(l) :
    convert_to_int(i)

#print(df.head())

corr = df.corr("pearson")
#print(corr.head())


#########################################################################################################
'''REMOVING THOSE FEATURES WHOSE CORRELATION WITH "passed" COLUMN, HAS ABSOLUTE VALUE LESS THAN 0.05 i.e. WOULD HAVE NEGLIGIBLE EFECT COMPARED TO OTHER FEATURES'''


all_columns = list(df.columns[:-1])
columns_to_drop = []

for i in all_columns :
    if abs(corr[i]["passed"]) < 0.05 :
        columns_to_drop.append(i)

for i in columns_to_drop :
    df.drop(i, axis = 1, inplace = True)

#print(df.shape)
#print(df.head())
#print(df.corr("pearson"))
        # Found out those features, on which "passed" feature hardly depends upon.


##########################################################################################################
'''SPLITTING THE DATA INTO TRAINING AND TESTING DATA'''


from sklearn.model_selection import train_test_split
# sklearn.cross_validation is set to be deprecated in next update i.e. 0.20

global X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = train_test_split(df[list(df.columns[:-1])], df[df.columns[-1]], test_size = 0.4, random_state = 0)

#print(X_train.count(axis = 0))
        #237
#print(X_test.count(axis = 0))
        #158


##########################################################################################################
'''IMPLEMENTATION OF DIFFERENT MODELS AND COMPARIOSN BETWEEN THEM'''

import time
from sklearn.metrics import accuracy_score


def time_for_each_model_implementation(model_func, best_accuracy_score, best_model, best_model_time) :
    start_time = time.time()
    model_func.fit(X_train, Y_train)
    Y_pred = model_func.predict(X_test)
    end_time = time.time()

    time_taken = end_time - start_time
    score = accuracy_score(Y_test, Y_pred)


    #print("Total execution time : {:.1f}" .format((time_taken) * 1000), "milli seconds")
    #print("Accuracy is {:.2f}%" .format(score * 100))
    #print('\n' * 2)

    # Ouput obtained after un-commneting the above lines can be seen in "Which is the best model to use ?"
    # section of the README.md of 'Stinsight' repository at 
    # https://github.com/prnvdixit/Stinsight/blob/master/README.md


    # Only if the accuracy of the present model is better or same than all previous ones
    # AND the time taken in fitting and predicting is BETTER than previous ones, we will
    # include the model
    if score >= best_accuracy_score and time_taken < best_model_time :
        best_accuracy_score = score
        best_model = model_func
        best_model_time = time_taken

    return (best_accuracy_score, best_model, best_model_time)


from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# The list storing the model-functions and their names
classifier = [()] * 7

# Storing the known models for given classification problem
classifier[0] = (SVC(kernel = "linear"), "Support Vector Machine - Linear")
classifier[1] = (SVC(kernel = "poly"), "Support Vector Machine - Polynomial")
classifier[2] = (SVC(kernel = "rbf"), "Support Vector Machine - Radial Basis Function")

classifier[3] = (LogisticRegression(), "Logistic Regression")
classifier[4] = (DecisionTreeClassifier(), "Decision Tree")
classifier[5] = (RandomForestClassifier(n_estimators = 100), "Random Forest")
classifier[6] = (RandomForestClassifier(n_estimators = 25, min_samples_split = 25, max_depth = 7, max_features = 1), "Random Forest (Optimised)")


# The variables deciding which out of all the models is best among all
best_model = classifier[0][0]
best_model_name = classifier[0][1]
best_model_time = 1
best_accuracy_score = 0.0


for (model_func, model_name) in classifier :
    #print(model_name, '\n')
    (best_accuracy_score, best_model, best_model_time) = time_for_each_model_implementation(model_func, best_accuracy_score, best_model, best_model_time)
    if best_model == model_func :
        best_model_name = model_name

print("Based on the accuracy and time taken, the best model is", best_model_name, "\n\nConfusion Matrix :-")
best_model.fit(X_train, Y_train)
Y_pred = best_model.predict(X_test)


###########################################################################################################
'''CONFUSION MATRIX OF MODEL'''


from collections import Counter, OrderedDict
# Counter to store the Confusion-matrix and OrderedDict to have the column-names show
# up at start of the data (By inserting column header at first before inserting the Confusion matrix data into the dict)

confusion_matrix = Counter()

true = [1, 3]

train_true = [x in true for x in Y_train]
pred_true = [x in true for x in Y_pred]

for t, p in zip(train_true, pred_true) :
    confusion_matrix[t, p] += 1

confusion_matrix = dict(confusion_matrix)

confusion_dict = OrderedDict()
confusion_dict[("Expected-Predicted")] = "Frequency"
# Adding the column header to the OrderedDict initially.

for (i, j) in confusion_matrix.items() :
    confusion_dict[i] = j

#print(confusion_dict)

conf_mat_df = pd.DataFrame.from_dict(confusion_dict, orient = 'index').rename(columns = {0 : ""})

print(conf_mat_df.head(), '\n')


#################################################################################################################
'''TESTING THE ACCURACY OF THE MODEL'''


from sklearn.metrics import accuracy_score

print("Accuracy of the model is {:.2f}%" .format(accuracy_score(Y_test, Y_pred) * 100))

'''
FINAL OUTPUT :-

Based on the accuracy and time taken, the best model is Logistic Regression

Confusion Matrix :-

Expected-Predicted  Frequency
(False, True)              42
(True, False)              12
(False, False)              7
(True, True)               97

Accuracy of the model is 69.62%

'''
