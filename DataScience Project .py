#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[2]:


import numpy as np
import pandas as pd
import sklearn.linear_model

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


# In[3]:


from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


# In[4]:


trainingData = pd.read_csv("train.csv")
trainingData.describe()
trainingData[:5]


# In[4]:


trainingData.shape


# In[5]:


trainingData.info()


# Looking at the data we can see that the number of non-null values vary from column to column meaning we need to introduce some pre-processing steps (such as imputation and encoding for objects)  in order to fill in null values. Our target variable is a binary value therefore we want to use classificaiton methods on this dataset. There are also lots of features, we need to identify and shave down features that don't contribute to our models performance.

# In[5]:


X_train = trainingData.drop("h1n1_vaccine", axis=1) # drop labels for the training set
y_train = trainingData["h1n1_vaccine"].copy() # save the labels


# In[7]:


X_train.dtypes
X_train["age_group"].value_counts()


# In[8]:


X_train["education"].value_counts()


# In[9]:


X_train["h1n1_concern"].value_counts()


# In[10]:


X_train["h1n1_knowledge"].value_counts()


# In[11]:


X_train["income_poverty"].value_counts()


# In[12]:


X_train["employment_status"].value_counts()

age_categories = ["18 - 34 Years", "35 - 44 Years", "45 - 54 Years", "55 - 64 Years", "65+ Years"]
education_categories = ["< 12 Years", "12 Years", "Some College", "College Graduate"]
income_categories = ["Below Poverty", "<= $75,000, Above Poverty", "> $75,000"]
ordinalOrders = [
    age_categories,  
    education_categories,  
    income_categories  
]


# 3 objects have Categorical ordinal data types, need to establish order. Not going to count employment_status as not in labour force could refer to retirees and children two distinct groups. Preprocessing multiple ordinal columns creating tons of errors. Using OneHotEncoder for education and income columns.
# 

# In[13]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer


numericalColumns = ['h1n1_concern', 'h1n1_knowledge', 'behavioral_antiviral_meds', 'behavioral_avoidance', 'behavioral_face_mask'
                    , 'behavioral_wash_hands', 'behavioral_large_gatherings', 'behavioral_outside_home', 'behavioral_touch_face', 'doctor_recc_h1n1', 'doctor_recc_seasonal',
                    'chronic_med_condition', 'child_under_6_months', 'health_worker', 'health_insurance', 'opinion_h1n1_vacc_effective', 'opinion_h1n1_risk'
                   , 'opinion_h1n1_sick_from_vacc', 'opinion_seas_vacc_effective', 'opinion_seas_risk','opinion_seas_sick_from_vacc',
                   'household_adults','household_children']
nominalColumns = ['race', 'sex','marital_status','rent_or_own','employment_status']
ordinalColumns = ['age_group', 'education', 'income_poverty']








numericalPipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')),
                               ('scaler', StandardScaler())])
nominalPipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                             ('encoder', OneHotEncoder(handle_unknown='ignore'))])
ordinalPipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=ordinalOrders, handle_unknown='use_encoded_value', unknown_value=-1)),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('numerical_transformer', numericalPipeline, numericalColumns),
    ('nominal_transformer', nominalPipeline, nominalColumns),
    ('ordinal_transformer', ordinalPipeline, ordinalColumns),
])


# In[14]:




lr_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=800))
])

lr_model.fit(X_train, y_train);


# In[15]:




scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='accuracy')


print("Mean Accuracy:", scores.mean())


# Mean Accuracy: 0.8358708545877528 on the training set for logistic regression.

# In[16]:


from sklearn.ensemble import RandomForestClassifier

rf_model_v1 = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=0))
])

rf_model_v1.fit(X_train, y_train)



scores = cross_val_score(rf_model_v1, X_train, y_train, cv=5, scoring='accuracy')


print("Mean Accuracy:", scores.mean())


# Mean Accuracy: 0.8496418752187169 for the random forest model, decent performance.

# In[17]:


from sklearn.svm import SVC

svm_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC())
])

svm_model.fit(X_train, y_train)


# In[18]:



scores = cross_val_score(svm_model, X_train, y_train, cv=5, scoring='accuracy')

print("Mean Accuracy:", scores.mean())


# Mean Accuracy: 0.843526157949773 using the SVC model.

# In[9]:


from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import make_scorer

 
roc_auc_scorer = make_scorer(roc_auc_score)

# Logistic Regression Model
lr_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=800))
])

# Compute AUC-ROC scores using cross-validation
lr_roc_auc_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring=roc_auc_scorer)
print("Logistic Regression AUC-ROC Scores:", lr_roc_auc_scores)
print("Mean AUC-ROC:", lr_roc_auc_scores.mean())

# Random Forest Model
rf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=0))
])

# Compute AUC-ROC scores using cross-validation
rf_roc_auc_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring=roc_auc_scorer)
print("Random Forest AUC-ROC Scores:", rf_roc_auc_scores)
print("Mean AUC-ROC:", rf_roc_auc_scores.mean())

# Support Vector Machine (SVM) Model
svm_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC(probability=True))  # Set probability=True for ROC-AUC
])

# Compute AUC-ROC scores using cross-validation
svm_roc_auc_scores = cross_val_score(svm_model, X_train, y_train, cv=5, scoring=roc_auc_scorer)
print("SVM AUC-ROC Scores:", svm_roc_auc_scores)
print("Mean AUC-ROC:", svm_roc_auc_scores.mean())


# Logistic Regression AUC-ROC Scores: [0.70187859 0.66870258 0.68269717 0.676667   0.68360207]
# Mean AUC-ROC: 0.6827094827304416
# Random Forest AUC-ROC Scores: [0.72125735 0.69762957 0.69200936 0.68991312 0.70180489]
# Mean AUC-ROC: 0.7005228577117072
# SVM AUC-ROC Scores: [0.70455102 0.67952742 0.68339127 0.69046626 0.69501157]
# Mean AUC-ROC: 0.690589509470874
# 
# From the results of the three I'm going to go ahead and continue using random forest classifier as Model A as on average Random Forest classifier had the best performance.

# In[8]:



rf_model_v1 = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=0))
])

rf_model_v1.fit(X_train, y_train)

# Get feature importances
feature_importances = rf_model_v1.named_steps['classifier'].feature_importances_


all_features = (
    numericalColumns +
    list(rf_model_v1.named_steps['preprocessor']
             .named_transformers_['nominal_transformer']
             .named_steps['encoder']
             .get_feature_names(input_features=nominalColumns)) +
    list(rf_model_v1.named_steps['preprocessor']
             .named_transformers_['ordinal_transformer']
             .named_steps['encoder']
             .categories_)
)

# Create a DataFrame to visualize feature importances
importances_df = pd.DataFrame({'Feature': all_features, 'Importance': feature_importances})

# Sort the DataFrame by importance in descending order
importances_df = importances_df.sort_values(by='Importance', ascending=False)

# Visualize the feature importances
print(importances_df)


# Judging from these metrics of feature importance I will utilize features for model A that hold an importance value of over 0.05:
# 
# | Feature                      | Feature Name                    | Importance |
# | ----------------------------- | ------------------------------- | ---------- |
# | 9                            | doctor_recc_h1n1                | 0.102280   |
# | 16                           | opinion_h1n1_risk               | 0.080846   |
# | 14                           | health_insurance                | 0.063734   |
# | 15                           | opinion_h1n1_vacc_effective      | 0.061304   |
# 

# In[7]:


from sklearn.feature_selection import SelectFromModel

selected_features = ["doctor_recc_h1n1", "opinion_h1n1_risk", "health_insurance", "opinion_h1n1_vacc_effective"]
X_train_selected = X_train[selected_features]

selected_features_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor2 = ColumnTransformer([
    ('selected_features_transformer', selected_features_transformer, selected_features)
])


rf_model_A = Pipeline([
    ('preprocessor', preprocessor2),
    ('classifier', RandomForestClassifier(random_state=0))
])


rf_model_A.fit(X_train_selected, y_train)


param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=rf_model_A, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train_selected, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_
best_rf_model_A = grid_search.best_estimator_


# In[10]:



test_data = pd.read_csv("test.csv")
X_test = test_data.drop("h1n1_vaccine", axis=1) 
y_test = test_data["h1n1_vaccine"].copy()


y_test_predicted = rf_model_A.predict(X_test)
accuracy_score(y_test, y_test_predicted)
score(y_test,y_test_predicted)



# Accuracy not a good measure for imbalanced datasets important evaluation metrics below

# In[23]:


from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, average_precision_score

# Predict the probabilities and class labels on the test set
y_pred_proba_A = best_rf_model_A.predict_proba(X_test)[:, 1]
y_pred_labels_A = best_rf_model_A.predict(X_test)


auc_roc = roc_auc_score(y_test, y_pred_proba_A)


precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_A)
auc_pr = average_precision_score(y_test, y_pred_proba_A)  # Use average_precision_score


accuracy = accuracy_score(y_test, y_pred_labels_A)

# Calculate sensitivity and specificity
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_labels_A).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

# Calculate F1 score
f1 = f1_score(y_test, y_pred_labels_A)

#Evalation Metrics
print("AUC-ROC:", auc_roc)
print("AUC-PR:", auc_pr)
print("Accuracy:", accuracy)
print("Sensitivity (True Positive Rate):", sensitivity)
print("Specificity (True Negative Rate):", specificity)
print("F1 Score:", f1)


# model A's performance on the test set was pretty bad in terms of AUC-PR, Sensitivity and f1 score metrics. This could mean we are dealing with need to better change our model to fit an imbalanced dataset.

# In[24]:


vaccine_counts = trainingData["h1n1_vaccine"].value_counts()


print(vaccine_counts)


# Sure enough we are dealing with an extremely imbalanced dataset, we'll use methods for model B to counteract this and add an additional feature.

# Running the same features to see the difference in performance when using the same preprocessing / transformer with gradient boosting.

# In[25]:


selected_features = ["doctor_recc_h1n1", "opinion_h1n1_risk", "health_insurance", "opinion_h1n1_vacc_effective"]
X_train_selected = trainingData[selected_features]
y_train = trainingData["h1n1_vaccine"].copy()

numericalPipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])


preprocessorModelB = ColumnTransformer([
    ('numerical_transformer', numericalPipeline, selected_features)
])


gb_model_B = Pipeline([
    ('preprocessor', preprocessorModelB),
    ('classifier', GradientBoostingClassifier(random_state=0))
])

gb_model_B.fit(X_train_selected, y_train)

param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 4, 5]
}

grid_search = GridSearchCV(estimator=gb_model_B, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train_selected, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_
best_gb_model_B = grid_search.best_estimator_


test_data = pd.read_csv("test.csv")
X_test = test_data[selected_features]
y_test = test_data["h1n1_vaccine"].copy()


y_pred_proba_B = best_gb_model_B.predict_proba(X_test)[:, 1]
y_pred_labels_B = best_gb_model_B.predict(X_test)


auc_roc = roc_auc_score(y_test, y_pred_proba_B)


precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_B)
auc_pr = average_precision_score(y_test, y_pred_proba_B)


accuracy = accuracy_score(y_test, y_pred_labels_B)


tn, fp, fn, tp = confusion_matrix(y_test, y_pred_labels_B).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)


f1 = f1_score(y_test, y_pred_labels_B)


print("AUC-ROC:", auc_roc)
print("AUC-PR:", auc_pr)
print("Accuracy:", accuracy)
print("Sensitivity (True Positive Rate):", sensitivity)
print("Specificity (True Negative Rate):", specificity)
print("F1 Score:", f1)


# Using Gradient Boosting didn't have a significant effect on the performance of this model, so I will implement a custom transformer to see if it will improve the performance. I will also use the 5 features with the highest level of importance according to my RF model I ran earlier.

# In[26]:


X_trainmodelB = ["doctor_recc_h1n1", "opinion_h1n1_risk", "health_insurance", "opinion_h1n1_vacc_effective","opinion_seas_risk"]


# In[27]:


from sklearn.base import BaseEstimator, TransformerMixin

class NewPolyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, degree=None):
        self.degree = degree

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.degree is None:
            return X
        else:
            return np.hstack((X, X**self.degree))


# In[28]:



modelBColumns = ["doctor_recc_h1n1", "opinion_h1n1_risk", "health_insurance", "opinion_h1n1_vacc_effective","opinion_seas_risk"]
X_train_selected_df = X_train[modelBColumns]


preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', SimpleImputer(strategy='mean'), modelBColumns)
        
    ],
    remainder='passthrough'
)


gb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('poly', NewPolyTransformer(degree=2)),
    ('classifier', GradientBoostingClassifier())
])


gb_pipeline.fit(X_train_selected_df, y_train)


y_pred = gb_pipeline.predict(X_train_selected_df)


y_pred_proba = gb_pipeline.predict_proba(X_train_selected_df)


# In[29]:


param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 4, 5]
}


grid_search = GridSearchCV(estimator=gb_model_B, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

grid_search.fit(X_train_selected_df, y_train)


best_params = grid_search.best_params_
best_score = grid_search.best_score_
best_gb_model_B = grid_search.best_estimator_


# In[30]:


from sklearn.metrics import auc


y_pred_proba_B = gb_pipeline.predict_proba(X_train_selected_df)[:, 1]
y_pred_labels_B = gb_pipeline.predict(X_train_selected_df)


auc_roc_B = roc_auc_score(y_train, y_pred_proba_B)


precision_B, recall_B, _ = precision_recall_curve(y_train, y_pred_proba_B)
auc_pr_B = auc(recall_B, precision_B)


accuracy_B = accuracy_score(y_train, y_pred_labels_B)


tn_B, fp_B, fn_B, tp_B = confusion_matrix(y_train, y_pred_labels_B).ravel()
sensitivity_B = tp_B / (tp_B + fn_B)
specificity_B = tn_B / (tn_B + fp_B)


f1_B = f1_score(y_train, y_pred_labels_B)

#Evaluation Metrics
print("AUC-ROC (Model B):", auc_roc_B)
print("AUC-PR (Model B):", auc_pr_B)
print("Accuracy (Model B):", accuracy_B)
print("Sensitivity (True Positive Rate) (Model B):", sensitivity_B)
print("Specificity (True Negative Rate) (Model B):", specificity_B)
print("F1 Score (Model B):", f1_B)


# Model B performed slightly worse than model A on the unseen data, the features were selected using the RF_model's importance values reported above.

# In[34]:


reTestData = pd.read_csv("test.csv")
X_test = reTestData.drop("h1n1_vaccine", axis=1) 
y_test = reTestData["h1n1_vaccine"].copy() 

y_pred_proba_B = gb_pipeline.predict_proba(X_test)[:, 1]
y_pred_labels_B = gb_pipeline.predict(X_test)


auc_roc_B = roc_auc_score(y_test, y_pred_proba_B)


precision_B, recall_B, _ = precision_recall_curve(y_test, y_pred_proba_B)
auc_pr_B = auc(recall_B, precision_B)




tn_B, fp_B, fn_B, tp_B = confusion_matrix(y_test, y_pred_labels_B).ravel()
sensitivity_B = tp_B / (tp_B + fn_B)
specificity_B = tn_B / (tn_B + fp_B)


f1_B = f1_score(y_test, y_pred_labels_B)

balanced_accuracy = (sensitivity_B+specificity_B/2)


print("AUC-ROC (Model B):", auc_roc_B)
print("AUC-PR (Model B):", auc_pr_B)
print("Sensitivity (True Positive Rate) (Model B):", sensitivity_B)
print("Specificity (True Negative Rate) (Model B):", specificity_B)
print("F1 Score (Model B):", f1_B)
# Balanced accuracy = (Sensitivity + Specificity) / 2
print("Balanced Accuracy", balanced_accuracy)


# In[36]:


from sklearn.metrics import roc_curve

fpr_A, tpr_A, _ = roc_curve(y_test, y_pred_proba_A)
roc_auc_A = auc(fpr_A, tpr_A)

fpr_B, tpr_B, _ = roc_curve(y_test, y_pred_proba_B)
roc_auc_B = auc(fpr_B, tpr_B)


precision_A, recall_A, _ = precision_recall_curve(y_test, y_pred_proba_A)
pr_auc_A = average_precision_score(y_test, y_pred_proba_A)

precision_B, recall_B, _ = precision_recall_curve(y_test, y_pred_proba_B)
pr_auc_B = average_precision_score(y_test, y_pred_proba_B)


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr_A, tpr_A, color='darkorange', lw=2, label=f'Model A (AUC = {roc_auc_A:.2f})')
plt.plot(fpr_B, tpr_B, color='navy', lw=2, label=f'Model B (AUC = {roc_auc_B:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')


plt.subplot(1, 2, 2)
plt.plot(recall_A, precision_A, color='darkorange', lw=2, label=f'Model A (AUC = {pr_auc_A:.2f})')
plt.plot(recall_B, precision_B, color='navy', lw=2, label=f'Model B (AUC = {pr_auc_B:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')

plt.tight_layout()


plt.show()


# As we can see the performance on both the ROC Curve and the AUC is slightly better under Model B which utilized an extra feature + custom transformers. I learnt from this project how much an imbalanced dataset can affect the performance of a model and certain metrics such as Accuracy. The most effective methods of boosting performance in the case of this project was removing features, hypertuning parameters and the Gradient Boosting model. The biggest problem was trying to combat undersampling as the target was heavily skewed towards one class. In the future I'd like to work to try to do more feature engineering when there are so many different variables within the dataset and see it's effect on my model's performance .
# 
# 
# 
# 
