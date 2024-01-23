## Case Study: Predictive Analysis in Banking Sector

In this practical application, my goal is to compare the performance of the classifiers: K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines. 

I utilized a dataset related to marketing bank products over the telephone. The dataset comes from the UCI Machine Learning repository [link](https://archive.ics.uci.edu/ml/datasets/bank+marketing).  The data is from a Portugese banking institution and is a collection of the results of multiple marketing campaigns.  I used the article accompanying the dataset [here](CRISP-DM-BANK.pdf) for more information on the data and features.

[Link](notebooks/ModelComparison.ipynb) to the Jupyter Notebook with all the analysis and models 

## Overall acceptance rate - Has the client subscribed a term deposit? (binary: "yes","no")
The overall acceptance rate for this dataset is unbalanced. 88.3% of the customers did not subscribe the term deposit, 11.7% subscribed.

1. % no: 39922 (88.302%)
2. % yes: 5289 (11.698%)

## Exploraty Data Analysis
To better understand the independent features, I conducted an exploratory data analysis. Below are the key findings.

#### Acceptance By Age
Younger Age Groups: younger individuals are more likely to have subscribed to the term deposit.
Middle Age Groups: people in these age groups have a low to moderate likelihood of subscribing to the term deposit.-
Older Age Groups: older individuals are more likely to subscribe.

#### Acceptance by Job
Students and retired individuals have the highest acceptance rates, significantly above the baseline, indicating a strong inclination towards accepting term deposit offers within these groups. Management also has a higher than average acceptance rate, while blue-collar workers are just above average. Job categories such as services, housemaid, and unemployed fall below the baseline, indicating a lower likelihood of accepting term deposit offers.

#### Acceptance by Marital
Single customers show the highest propensity to subscribe to a term deposit, followed by divorced customers, both above the baseline acceptance rate. Married customers are approached most often but have a lower acceptance rate compared to the other groups.

#### Acceptance by level of Education
Individuals with tertiary education are the most likely to subscribe to a term deposit, significantly surpassing the acceptance rates of those with secondary or primary education. Those with unknown education levels still have a relatively high likelihood of subscription, potentially indicating that factors other than formal education influence their decision to subscribe. Individuals with primary education are the least likely to subscribe to term deposits among the groups represented.

#### Acceptance by customer having or not a housing Loan
Customers without a housing loan have a higher rate of accepting term deposit offers compared to those with a housing loan. This could imply that customers without the financial commitment of a housing loan might be more inclined to invest in term deposits.

#### Summary of Exploratory Data Analysis
In conclusion, the profile of a customer more likely to subscribe to a term deposit includes being single or divorced, having a tertiary level of education, not having a housing loan, and belonging to certain job categories such as students or retirees. These factors may contribute to a customer's financial stability and freedom, as well as their knowledge and attitudes towards saving and investment, which in turn influence their decision to subscribe to term deposits. It's important to note that these are trends and generalizations, and individual decisions to subscribe to term deposits will also be influenced by a range of personal factors and circumstances.

## Classification models
After exploratory data analysis, I prepared the data to create the four classification models. I followed the following steps to create the models:

### Data Preparation and Splitting
I define the DataFrame `X` with attributes like age, job, marital status, and education, and the Series `y` as a binary outcome. I use `train_test_split` to divide the dataset into a training set (70%) and a test set (30%), ensuring the class proportions are consistent using `stratify=y` and setting a `random_state` for reproducibility.

### Feature Encoding and Scaling
For preprocessing, I initialize a `ColumnTransformer` to apply `OneHotEncoder` to categorical variables and `StandardScaler` to numerical features. This ensures categorical data is transformed into a numerical format and all features are normalized for the machine learning algorithms.

### Model Training and Evaluation
I employ a `Pipeline` to streamline preprocessing and model fitting. Using `GridSearchCV`, I optimize hyperparameters for various classifiers: Decision Tree, Logistic Regression, Support Vector Machine, and K-Nearest Neighbors, selecting the best model based on accuracy. Each model's performance is cross-validated to ensure reliability.

## Models Performance
In my final evaluation, I compared four classification models: Decision Tree, Logistic Regression, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN), using three different scoring metrics: accuracy, F1 score, and recall. My goal was to identify the model that most accurately predicts customers likely to accept a bank's term deposit offer.

The Support Vector Machine achieved the highest accuracy, suggesting it is the most reliable for general classification tasks in this context. However, due to class imbalance, accuracy alone might not be a comprehensive metric.

For the F1 score, which balances precision and recall, the Decision Tree model performed the best, closely followed by the Support Vector Machine. This indicates that the Decision Tree model has a good balance of precision and recall, but still has room for improvement in identifying true positives.

When focusing on the recall score, which is critical for minimizing false negatives and capturing as many positive instances (i.e., potential customers) as possible, the Decision Tree again outperformed the others, with the K-Nearest Neighbors model as the runner-up.

Given the business objective of maximizing the identification of customers who will accept the offer, I recommend prioritizing the Decision Tree model for its superior recall score. Its ability to capture a higher number of positive instances makes it the most suitable choice for this specific outcome, despite some trade-offs in precision and accuracy.

## Ranking for Recall Score:
- Decision Tree
- KNeighborsClassifier
- Logistic Regression
- Support Vector Machine