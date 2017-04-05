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
#print(df.describe())
#print(df["passed"].value_counts())


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
#print(df["Fjob"].value_counts())


job_assist_studies_level = {"services" : 1, "other" : 0, "teacher" : 2, "at_home" : 2, "health" : 0}
# Assumptions :-
# If the Parent is at house for most of the time - Expected to assist his ward more
# Else if the Parent is him/her self a teacher - Expected to be a better guide of ward
# Else - 9-5 job i.e. no time for family or assisting the ward in studies


#print(df["famsize"].value_counts())


support_family_type = {"GT3" : 1, "LE3" : 0}
# Assumptions :-
# To simplify the analysis, 
# GT3 i.e. family size greater than 3 is assumed to comprise of Mother, Father, Two siblings or more, thus providing more support to ward for studies.
# LE3 i.e. family size less than equal to 3 is assumed to comprise of Father, Mother and a single child, thus providing less support to ward for studies..


parent_both_or_not = {"A" : 0, "T" : 1}
# Assumptions :-
# If parents are living together, the child can be taken care of in a better way and responsibilities can be shared.


urban_rural = {"U" : 1, "R" : 0}
# Assumptions :-
# It is general observation that people in urban areas have more exposure thanks to good peer-study groups


reason_school = {"course" : 2, "reputation" : 1, "home" : 0, "other" : 0}
# Assumptions :-
# The ward would have been more interested in studies if his preferable course or school would have been choosen
# In all other cases, he relunctantly accepted the course

guardian_assist = {"mother" : 1, "other" : 1, "father" : 2}
# Assumptions :-
# As according to expected strictness from guardian towards studies

school = {"GP" : 0, "MS" : 1}
# Assumptions :-
# Baseed on the reviews on their facebook pages, quality of education at these schools was assumed


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


##########################################################################################################
'''SPLITTING THE DATA INTO TRAINING AND TESTING DATA'''


from sklearn.model_selection import train_test_split
# sklearn.cross_validation is set to be deprecated in next update i.e. 0.20

X_train, X_test, Y_train, Y_test = train_test_split(df[list(df.columns[:-1])], df[df.columns[-1]], test_size = 0.4, random_state = 0)

#print(X_train.count(axis = 0)) = 237
#print(X_test.count(axis = 0))  = 158


##########################################################################################################
'''SVM MODEL IMPLEMENTATION'''


from sklearn.svm import SVC

classifier = SVC(kernel = "linear")
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)


###########################################################################################################
'''CONFUSION MATRIX OF MODEL'''


from collections import Counter, OrderedDict

confusion_matrix = Counter()

true = [1, 3]

train_true = [x in true for x in Y_train]
pred_true = [x in true for x in Y_pred]

for t, p in zip(train_true, pred_true) :
    confusion_matrix[t, p] += 1

confusion_matrix = dict(confusion_matrix)

confusion_dict = OrderedDict()
confusion_dict[("Expected-Predicted")] = "Frequency"

for (i, j) in confusion_matrix.items() :
    confusion_dict[i] = j

#print(confusion_dict)

conf_mat_df = pd.DataFrame.from_dict(confusion_dict, orient = 'index').rename(columns = {0 : ""})

print(conf_mat_df.head(), '\n')


#################################################################################################################
'''TESTING THE ACCURACY OF THE MODEL'''


from sklearn.metrics import accuracy_score

print("Accuracy of the model is {:.2f}%" .format(accuracy_score(Y_test, Y_pred) * 100))
