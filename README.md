# Stinsight
A Machine Learning project implemented for the Artificial Intelligence Society DataSci '17 competition. The UCI Student Performance Data Set was used to effectively predict if a student is expected to fail the course or not, using school reports and questionnaires. Different classification models were compared. The best model, Logistic Regression achieved about 70% accuracy.


### Prerequisites

This project requires Python 3.4 and the following Python libraries installed:

- [Pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)


## Install

On Unix systems, the above libraries can be installed using these (or corresponding ones for your distribution) commands,

```
sudo apt-get install build-essential python3-dev python3-setuptools python3-numpy python3-scipy libatlas-dev libatlas3gf-base

sudo apt-get install python3-pip

sudo -H pip3 install -U scikit-learn

sudo -H pip3 install pandas
```

## Observations and Evaluating Criterions

### 1. In which class of problems, does this qualify ?

This is a classification problem. The reason for terming it as a classification problem being, we are asked to identify students who might end up failing the final exam. 

Thus, we need to identify such students and intervene before its too late. Or in other words, we have to classify whole group of students into two sections - Ones who are expected to fail the course and others who are expected to pass the course.

### 2. Observations from Given Data

- Total number of students: 395
- Total number of features: 30
- Target: 1 ("passed")
- Number of students who passed: 265
- Number of students who failed: 130
- Number of columns with numeral values: 13

### 3. What are the features and Relevant Target ?

```
Feature column(s):-
['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason',
 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']

Target column: passed
```


### 4. Which is the best model to use ?

```

Support Vector Machine - Linear 

Total execution time : 23.7 milli seconds
Accuracy is 69.62%


Support Vector Machine - Polynomial 

Total execution time : 478.0 milli seconds
Accuracy is 63.92%


Support Vector Machine - Radial Basis Function 

Total execution time : 8.7 milli seconds
Accuracy is 66.46%


Logistic Regression 

Total execution time : 2.7 milli seconds
Accuracy is 69.62%


Decision Tree 

Total execution time : 1.7 milli seconds
Accuracy is 56.33%


Random Forest 

Total execution time : 201.7 milli seconds
Accuracy is 69.62%


Random Forest (Optimised) 

Total execution time : 47.6 milli seconds
Accuracy is 63.29%

```

Based on the statistics obtained, Logistic Regression provides the best performance and also in least time. 

The other models didn't performed well in comparison with Logistic regression. Though, Logistic Regression appears to be a clear winner in both measures used to decide the best model i.e. performance and training and testing time, I would've still choosen to trade off the training time for the higher performance. The reason being that failing a course in school can have some adverse effect on mental and pyschological health of student, making accuracy a dominant factor in comparing different models.

Thus, as the predictions needs to be very accurate, we would have to value the accuracy more as compared to training and testing time.


## Appendix

Attributes for Data Set along with the Data Set itself can be found at [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Student+Performance)


## Author

* **Pranav Dixit** - [Github](https://github.com/prnvdixit) - [Linkedin](https://www.linkedin.com/in/prnvdixit/)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Paulo Cortez - For making the Data Set public by donating it to the UCI ML Repository.
