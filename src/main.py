#Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import joblib
from imblearn.over_sampling import SMOTE
warnings.filterwarnings('ignore')
plt.style.use('ggplot')


#Reading Data
train = pd.read_csv('C:\\Users\\Asus\\Desktop\\Quitjob\\data\\train.csv',sep=',',na_values=['NaN'])
test = pd.read_csv('C:\\Users\\Asus\\Desktop\\Quitjob\\data\\test.csv',sep=',',na_values=['NaN'])


#Checking Train Dataframe
train.head()

#GLOBAL FUNCTIONS
#Data check
def data_check(df):
    print('dataframe shape', df.shape)
    print()
    print(df.describe)
    print()
    print('Dataframe NaN Values')
    print(df.isna().sum())
    print()
    print(df.info())


#Gender
def gender(gender):
    if gender == 'Male':
        return 'Male'
    elif gender == 'Female':
        return 'Female'
    else:
        return 'Unspecified'


#Enrolled at University
def course(course):
    if course == 'no_enrollment':
        return 'no_enrollment'
    elif course == 'Full time course ':
        return 'FullTime'
    elif course== 'Part time course':
        return 'PartTime'
    else:
        return 'Unknow'


#Education Level
def education_level(level):
    if level == 'NaN':
        return 'Unknow'
    else:
        return level


#MajorDiscipline
def major_discipline(discipline):
    if discipline == 'NaN':
        return 'Unknow'
    else:
        return discipline



#Experience
def experience(exp):
    if exp == 'NaN':
        return 'Unknow'
    else:
        return exp


#Company Size
def company_size(size):
    if size == 'NaN':
        return 'Unknow'
    else:
        return size


#Company Type
def company_type(company_type):
    if company_type == 'NaN':
        return 'Unknow'
    else:
        return company_type


#Last Job
def last_job(last_job):
    if last_job == 'NaN':
        return 'Unknow'
    else:
        return last_job


#Discretize experience
def discrete_exp(exp):
    if exp=='<1'      :   return '<1'
    if exp=='1'       :   return '1-10'
    if exp=='2'       :   return '1-10'
    if exp=='3'       :   return '1-10'
    if exp=='4'       :   return '1-10'
    if exp=='5'       :   return '1-10'
    if exp=='6'       :   return '1-10'
    if exp=='7'       :   return '1-10'
    if exp=='8'       :   return '1-10'
    if exp=='9'       :   return '1-10'
    if exp=='10'      :   return '1-10'
    if exp=='11'      :   return '11-20'
    if exp=='12'      :   return '11-20'
    if exp=='13'      :   return '11-20'
    if exp=='14'      :   return '11-20'
    if exp=='15'      :   return '11-20'
    if exp=='16'      :   return '11-20'
    if exp=='17'      :   return '11-20'
    if exp=='18'      :   return '11-20'
    if exp=='19'      :   return '11-20'
    if exp=='20'      :   return '11-20'
    if exp=='>20'     :   return '>20'
    else              :   return 'Unknow'


#Job Quitting
def job_quit(scientist):
    if scientist == 1.0:
        return 'Job Change'
    else:
        return 'Not Changing'


# Apply the fillna method to incluce NaN string
train = train.fillna('NaN')

#Applying Functions to delete NaN values
train['gender'] = train['gender'].apply(gender)

train['enrolled_university'] = train['enrolled_university'].apply(course)

train['education_level'] = train['education_level'].apply(education_level)

train['major_discipline'] = train['major_discipline'].apply(major_discipline)

train['experience'] = train['experience'].apply(experience)

train['company_size'] = train['company_size'].apply(company_size)

train['company_type'] = train['company_type'].apply(company_type)

train['last_new_job'] = train['last_new_job'].apply(last_job)

train['experience_summary'] = train['experience'].apply(discrete_exp)


#Dropping enrolle_id and City code
train = train.drop(columns=['enrollee_id','city'])


#Target Variable
color = ['green','red']
sns.countplot(data=train, x='target',hue='target',palette=color)
plt.title('TARGET VARIABLE')
plt.ylabel('Looking for a job or not count')
plt.show()


#Target Variable
sns.countplot(data=train, x='target',hue='gender',palette='rocket')
plt.title('TARGET VARIABLE')
plt.ylabel('Looking for a job or not count')
plt.show()


#Target Variable
sns.countplot(data=train, x='target',hue='relevent_experience',palette='Set1')
plt.title('TARGET VARIABLE')
plt.ylabel('Looking for a job or not count')
plt.show()


#Target Variable
sns.countplot(data=train, x='target',hue='enrolled_university',palette='Set3')
plt.title('TARGET VARIABLE')
plt.ylabel('Looking for a job or not count')
plt.show()


#Target Variable
sns.countplot(data=train, x='target',hue='education_level',palette='magma')
plt.title('TARGET VARIABLE')
plt.ylabel('Looking for a job or not count')
plt.show()


#City development index
sns.distplot(train['city_development_index'], color = 'blue',kde=True,)
plt.xlabel('City Development Index')
plt.title('CITY DEVELOPMENT INDEX')
plt.show()


#Training Hours distribution
sns.distplot(train['training_hours'], color = 'green')
plt.title('TRAINING HOURS DISTRIBUTION FOR ALL EMPLOYEES', fontsize = 15)
plt.show()


#Training Hours distribution for people who left job or not
#filtering employees by target
left = train[train['target']==1]
no_left = train[train['target']!=1]

#People who left
sns.distplot(left['training_hours'], color = 'darkred')
plt.title('TRAINING HOURS DISTRIBUTION FOR EMPLOYEES WHO LEFT JOB', fontsize = 15)
plt.show()

#People who don´t left
sns.distplot(no_left['training_hours'], color = 'blue')
plt.title('TRAINING HOURS DISTRIBUTION FOR EMPLOYEES WHO NOT LEFT JOB', fontsize = 15)
plt.show()


#Gender Variable
color_gender = ['skyblue','gray','pink']
sns.countplot(data=train, x='gender',palette=color_gender)
plt.title('GENDER')
plt.ylabel('Gender Count')
plt.show()


#Experience Variable
sns.countplot(data=train, x='relevent_experience',palette='viridis')
plt.title('Work Experience on the Field')
plt.ylabel('Gender Count')
plt.xlabel('Experience Relevance')
plt.show()

#enrolled Variable
sns.countplot(data=train, x='education_level',palette='magma')
plt.title('EDUCATION LEVEL')
plt.ylabel('Count')
plt.xlabel('Education Description')
plt.show()


#enrolled Variable
sns.countplot(data=train, x='enrolled_university',palette='rainbow')
plt.title('UNIVERSITY ENROLLED')
plt.ylabel('Count')
plt.xlabel('Enrollement Situation')
plt.show()


#Major Discipline
sns.countplot(data=train, x='major_discipline',palette='rainbow')
plt.title('MAJOR DISCIPLINE')
plt.ylabel('Count')
plt.xlabel('Education Description')
plt.xticks(rotation=90)
plt.show()

#experience Variable
sns.countplot(data=train, y='experience',palette='rainbow_r')
plt.title('EXPERIENCE')
plt.ylabel('Time in Years')
plt.xlabel('Count')
plt.show()


#Experience Variable
sns.countplot(data=train, x='experience_summary',palette='rainbow_r')
plt.title('EXPERIENCE')
plt.ylabel('Count')
plt.xlabel('Periods of Time in Years')
plt.xticks(rotation=90)
plt.show()


#Company Size
sns.countplot(data=train, x='company_size',palette='rainbow')
plt.title('COMPANY SIZE')
plt.ylabel('Count')
plt.xlabel('Number of Employees')
plt.xticks(rotation=90)
plt.show()


#Company type
sns.countplot(data=train, x='company_type',palette='rainbow_r')
plt.title('COMPANY TYPE')
plt.ylabel('Count')
plt.xlabel('Company')
plt.xticks(rotation=90)
plt.show()


#enrolled Variable
sns.countplot(data=train, x='last_new_job',palette='magma')
plt.title('LAST JOB ENTER')
plt.ylabel('Count')
plt.xlabel('Periods of Time in Years')
plt.xticks(rotation=90)
plt.show()


#last job
sns.countplot(data=train, x='last_new_job',palette='magma',hue='target')
plt.title('LAST JOB ENTER')
plt.ylabel('Count')
plt.xlabel('Periods of Time in Years')
plt.xticks(rotation=90)
plt.show()


#Adding additional df to handle the model
model_df = train

#Creating Dummy variables
model_df = pd.get_dummies(model_df)


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier


#Data Split

#Input Values
X = model_df.drop(columns=['target'])

#Target Variable
y = model_df['target']
y.columns = ['target']


#Train and Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)



#RANDOM FOREST MODEL
random_classifier = RandomForestClassifier(n_estimators=5, random_state=0)
random_classifier.fit(X_train, y_train)
random_prediction = random_classifier.predict(X_test)


#Model Accuracy
print(confusion_matrix(y_test, random_prediction))
print(classification_report(y_test, random_prediction))
print('Model Accuracy: ',accuracy_score(y_test, random_prediction))


#Lets oversampe
oversample = SMOTE()
smote = SMOTE(random_state = 0)
X_smote, y_smote = smote.fit_resample(X,y)

#Checking
print('X_smote shape', X_smote.shape)
print('y_smote shape', y_smote.shape)


#Train and Test Split
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.3,random_state=0)

#RANDOM FOREST MODEL
random_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
random_classifier.fit(X_train, y_train)
random_prediction = random_classifier.predict(X_test)

#Model Accuracy
print(confusion_matrix(y_test, random_prediction))
print(classification_report(y_test, random_prediction))
print('Model Accuracy: ',accuracy_score(y_test, random_prediction))


#Let´s Create an entire Function for the model, if we want to use it on another notebook
def hrAnalytics_randomForest():
    #Input Values
    X = model_df.drop(columns=['target'])
    #Target Variable
    y = model_df['target']
    y.columns = ['target']

    #Lets oversampe
    oversample = SMOTE()
    smote = SMOTE(random_state = 0)
    X_smote, y_smote = smote.fit_resample(X,y)

    #Train and Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.30,random_state=0)

    #RANDOM FOREST MODEL
    random_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
    random_classifier.fit(X_train, y_train)
    random_prediction = random_classifier.predict(X_test)

    #Model Accuracy
    print('RANDOM FOREST CLASSIFIER FOR HR ANALYTICS - JOB CHANGE TRAIN DATAFRAME')
    print(confusion_matrix(y_test, random_prediction))
    print(classification_report(y_test, random_prediction))
    print('Model Accuracy: ',accuracy_score(y_test, random_prediction))

    #Save Model
    joblib.dump(random_classifier, 'hr_analytics_random_forest.joblib')

    return random_classifier


#MODEL
hrAnalytics_randomForest()


def hrAnalytics_Knn():
    from sklearn.neighbors import KNeighborsClassifier
    #Input Values
    X = model_df.drop(columns=['target'])
    #Target Variable
    y = model_df['target']
    y.columns = ['target']

    #Lets oversampe
    oversample = SMOTE()
    smote = SMOTE(random_state = 0)
    X_smote, y_smote = smote.fit_resample(X,y)

    #Train and Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.30,random_state=0)

    #KNN MODEL
    knn_classifier = KNeighborsClassifier(n_neighbors=138)
    knn_classifier.fit(X_train, y_train)
    knn_prediction = random_classifier.predict(X_test)

    #Model Accuracy
    print('K-NEAREST NEIGHBOR CLASSIFIER FOR HR ANALYTICS - JOB CHANGE TRAIN DATAFRAME')
    print(confusion_matrix(y_test, knn_prediction))
    print(classification_report(y_test, knn_prediction))
    print('Model Accuracy: ',accuracy_score(y_test, knn_prediction))

    #Save Model
    joblib.dump(knn_classifier, 'hr_analytics_knn.joblib')

    return knn_classifier


#Knearest Neighbor Model
hrAnalytics_Knn()

def hrAnalytics_lr():
    from sklearn.linear_model import LogisticRegressionCV
    #Input Values
    X = model_df.drop(columns=['target'])
    #Target Variable
    y = model_df['target']
    y.columns = ['target']

    #Lets oversampe
    oversample = SMOTE()
    smote = SMOTE(random_state = 0)
    X_smote, y_smote = smote.fit_resample(X,y)

    #Train and Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.30,random_state=0)

    #LOGISTIC REGRESSION MODEL
    lr_classifier =  LogisticRegressionCV(cv=5, random_state=0)
    lr_classifier.fit(X_train, y_train)
    lr_prediction = random_classifier.predict(X_test)

    #Model Accuracy
    print('LOGISTIC REGRESSION NEIGHBOR CLASSIFIER FOR HR ANALYTICS - JOB CHANGE TRAIN DATAFRAME')
    print(confusion_matrix(y_test, lr_prediction))
    print(classification_report(y_test, lr_prediction))
    print('Model Accuracy: ',accuracy_score(y_test, lr_prediction))

    #Save Model
    joblib.dump(lr_classifier, 'hr_analytics_logisticregression.joblib')

    return lr_prediction

hrAnalytics_lr()


#Inspecting Test Dataset, with no target variable
test.head()


# Apply the fillna method to incluce NaN string
test = test.fillna('NaN')

#Applying Functions to delete NaN values
test['gender'] = test['gender'].apply(gender)

test['enrolled_university'] = test['enrolled_university'].apply(course)

test['education_level'] = test['education_level'].apply(education_level)

test['major_discipline'] = test['major_discipline'].apply(major_discipline)

test['experience'] = test['experience'].apply(experience)

test['company_size'] = test['company_size'].apply(company_size)

test['company_type'] = test['company_type'].apply(company_type)

test['last_new_job'] = test['last_new_job'].apply(last_job)

test['experience_summary'] = test['experience'].apply(discrete_exp)

#shape
print('Test dataset shape: ',test.shape)


#Adding additional df to handle the model
test_df = test

#Dropping some uselless columns to fit the model
test_df = test_df.drop(columns=['enrollee_id', 'city'])

#Creating Dummy variables
test_df = pd.get_dummies(test_df)


#Fitting Model to the Test Dataset
test_set_predictions = random_classifier.predict(test_df)

#Creating Dataframe with resuts
quit_job = pd.DataFrame(test_set_predictions)
quit_job.columns = ['Quit']

#Appending survive dataframe to test
test = test.join(quit_job)

#Creating Plot to show predictions
test['JobQuit'] = test.Quit.apply(job_quit)
color = ['#81B622','#900C3F']
sns.countplot(data=test, x='JobQuit',palette=color)
plt.title('DATA SCIENTISTS JOB CHANGE PREDICTION')


#Final Result
final_result = test[['enrollee_id','Quit']]
print(final_result.head(10))

#Dowloading CSV file
final_result.to_csv(r'C:\\Users\\Asus\\Desktop\\Quitjob\\data\\predictions.csv')
