# Loan-Status-Prediction
## Task:
Build a Machine Learning Model to predict the Loan status i.e. whether the loan is approved or not given the set of features.</br>
This comes under Supervised learning and also a Classification problem as the output will have only 2 types of results 1(approved) and 0(Not Approved).</br>
We will be using Support Vector Machine(SVM) for this classification problem. The main work of SVM is to create a line called hyperplane such that on its one side would be one set of results and on another side, it would be another set of results.
## Datasets:
The dataset is taken from Kaggle, it contains 614 records and 13 features. The features are:</br>
Loan_ID: AlphaNumeric </br>
Gender: Male or Female</br>
Married: Yes or No</br>
Dependents: 0, 1, 2 and 3+</br>
Education: Graduate or Not Graduate</br>
Self_Employed: Yes or No</br>
ApplicantIncome: Integer Value Depicting Salary</br>
CoapplicantIncome: Integer Value Depicting Salary</br>
LoanAmount: Float Value</br>
Loan_Amount_term: Float Value</br>
Credit_History: 1 or 0</br>
Propert_Area: Urban, Semiurban and Rural</br>
Loan_Status: Y(Approved) or N(Not Approved)</br></br>

Link: https://www.kaggle.com/datasets/ninzaami/loan-predication
## Steps Involved:
### Import Dependencies:
Libraries need in this problem are:</br>
NumPy</br>
Pandas</br>
seaboran</br>
train_test_split from sklearn.model_selection</br>
svm from sklearn</br>
accuracy_score from sklearn.metrics</br>
### Data Collection and Processing:
Import the data using pandas, read_csv function. Since this data contains lots of textual data lots of preprocessing steps are involved before building the model.</br>
. Check about the data using shape and describe function.</br>
. Check for <b>Missing Values</b>. We are going to drop the rows which contain NaN.</br>
. Change the label of Y and N to 1 and 0 respectively</br>
. We need to convert our dependant column containing '3+' to 4 as the machine can't understand 3+.</br>
. Perform Some <b>Data Visualization</b>, plotted countplot using Seaborn, to know how each feature is linked to the loan_Status column.</br>
. We need to convert all the remaining textual data to numerical data using the 'replace' function.</br>
. Finally, separate the label(Y) and the data(X).</br>
### Splitting the dataset into training and test set:
This would be performed using train_test_split function. We will split according to the Y, such that both test and train sets will have an equal ratio of 0 and 1.  We are taking 10% of the data for tests and 90% for training.
### Model Training and Evaluation:
We build the model using Support Vector Model, keeping the kernel value 'Linear'.</br>
Model Evaluation is done on Training and test data. </br>
Accuracy Score of Trained Data:  0.7986111111111112</br>
Accuracy Score of test data:  0.8333333333333334
### Building the predictive System:
The system is built using the random data from the dataset, feeding it to the model. And subsequently, predicting the outcome. </br></br></br>




Reference: Project 5. Loan Status Prediction using Machine Learning with Python | Machine Learning Project, Siddhardhan, https://www.youtube.com/watch?v=XckM1pFgZmg&list=PLfFghEzKVmjvuSA67LszN1dZ-Dd_pkus6&index=7




