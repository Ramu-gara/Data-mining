#  E-commerce Fraudulent Transactions Analysis


 
# Project Summary
	
The purpose of this project is to develop a real-time fraudulent e-commerce transaction detection system using machine learning. This system will analyze customer data, transaction details, and historical patterns to identify suspicious activity with high accuracy. It will predict outcomes of fraudulent e-commerce transactions using data retrieved with the API for https://www.kaggle.com/datasets/shriyashjagtap/fraudulent-e-commerce-transactions/data?select=Fraudulent_E-Commerce_Transaction_Data.csv
Using either a classification tree or a logistic regression model is suitable for our analysis since the outcome variable categorizes transactions as either fraudulent or non-fraudulent. Given the extensive pool of predictor variables, employing elimination and search algorithms to streamline our selection process could prove beneficial. Surprisingly, our findings did not demonstrate any advantages for the backward elimination algorithm. However, implementing cross-validation effectively diminished the number of predictors integrated into the classification tree, consequently mitigating the risk of overfitting.
This project serves as a vital shield against financial losses for online businesses. By employing advanced algorithms and machine learning models, it scrutinizes transaction patterns to identify suspicious activities in real time. This proactive approach helps mitigate risks associated with fraudulent transactions, safeguarding both the business and its customers. Additionally, it enhances customer trust and loyalty by ensuring a secure shopping environment. Ultimately, the project's utility lies in its ability to minimize financial losses, protect sensitive data, and uphold the integrity of e-commerce platforms.

 
# Introduction

The "Fraudulent E-Commerce Transactions" synthetic dataset serves as a simulated representation of transactional data typically encountered within an e-commerce platform, with a primary emphasis on fraud detection. This dataset encompasses a diverse array of features commonly observed in transactional records, supplemented by additional attributes deliberately engineered to facilitate the construction and evaluation of fraud detection algorithms. 

In our investigation, we leveraged two prominent machine learning algorithms, namely decision tree and logistic regression, to discern fraudulent transactions from legitimate ones within the dataset. Our objective was to develop robust models capable of accurately identifying instances of fraudulent activity, thereby enhancing the security and integrity of e-commerce platforms. Utilizing the decision tree algorithm, we employed grid search cross-validation to optimize model parameters and enhance predictive performance.

The application of this technique yielded a notable accuracy rate of 90%, underscoring the efficacy of decision tree-based approaches in discerning fraudulent transactions. Additionally, we explored the utility of logistic regression, a widely utilized statistical method for binary classification tasks. Despite employing backward elimination to refine predictor variables, the logistic regression model failed to exhibit substantial improvements in accuracy compared to the decision tree approach. 
Through our investigation, we aim to provide insights into the efficacy of different machine learning algorithms for fraud detection in e-commerce transactions. Our findings not only underscore the importance of algorithm selection but also highlight the impact of feature engineering and model optimization techniques in enhancing the accuracy and reliability of fraud detection systems.




# Main Chapter

# Develop understanding of purpose of  project

Our primary goal is to support online businesses in safeguarding against financial losses by effectively identifying fraudulent transactions. 
Our Outcome Objective is to Predict whether a transaction is fraudulent or legitimate to enable proactive fraud detection and prevention measures. 
Our Long-term Objective of this project is to enhance prediction accuracy over time through continual data collection, model refinement, and adaptation to evolving fraud tactics

# Obtain Data for Analysis:

We acquired the dataset for our analysis from Kaggle, a renowned platform for data science and machine learning datasets. The dataset, titled "Fraudulent E-Commerce Transactions," is accessible via the following link: [Kaggle Dataset](https://www.kaggle.com/datasets/shriyashjagtap/fraudulent-e-commerce-transactions/data?select=Fraudulent_E-Commerce_Transaction_Data_2.csv). 

# Feature Details:
**1. Transaction ID:** A unique identifier assigned to each transaction. 
**2. Customer ID:** A unique identifier associated with each customer. 
**3. Transaction Amount:** The total monetary value of the transaction. 
**4. Transaction Date:** The date and time when the transaction occurred. 
**5. Payment Method:** The mode of payment utilized for the transaction, such as credit card, PayPal, etc. 
**6. Product Category:** The category to which the product involved in the transaction belongs.
**7. Quantity:** The number of products involved in the transaction.
**8. Customer Age:** The age of the customer conducting the transaction. 
**9. Customer Location:** The geographical location of the customer. 
**10. Device Used:** The type of device (e.g., mobile, desktop) employed for the transaction. 
**11. Address:** The IP address of the device utilized for the transaction. 
**12. Shipping Address:** The address to which the product was shipped. 
**13. Billing Address:** The address associated with the payment method.
**14. Is Fraudulent:** A binary indicator denoting whether the transaction is fraudulent (1 for fraudulent, 0 for legitimate). 
**15. Account Age Days:** The duration of the customer's account existence in days at the time of the transaction. 
**16. Transaction Hour:**  The hour of the day when the transaction was initiated. These features encompass a comprehensive array of transactional attributes essential for fraud detection and analysis.

# Class Imbalance: 
A notable observation emerged regarding the class distribution, revealing a substantial disparity between fraudulent and legitimate transactions. Specifically, the dataset comprised 1,222 fraudulent transactions and 22,412 legitimate transactions.
<img width="525" alt="image" src="https://github.com/user-attachments/assets/c3852cb5-fa35-47fc-b172-6ea4750a4620" />

Recognizing the need to rectify the class imbalance issue, we employed oversampling techniques to augment the representation of fraudulent transactions within the dataset. This strategic approach aimed to create a more equitable distribution between fraudulent and legitimate transactions, fostering improved model training and evaluation.
  
<img width="526" alt="image" src="https://github.com/user-attachments/assets/644354a8-4c2d-4f9d-b0e5-1c95ab4ca4fc" />

# Explore, Clean and Preprocess Data:
In our column name review, we streamlined multi-word column names into single-word formats using underscores for improved clarity and consistency. During our data exploration phase, subsets of the dataset were accessed and examined to gain insights into its structure and contents. Utilizing the describe() function, we generated comprehensive statistics for both the dataset as a whole and individual columns, including measures such as mean, standard deviation, and quartile values. This statistical analysis provided a detailed overview of the dataset's numerical attributes, aiding in subsequent modeling and analysis tasks. By standardizing column names and conducting thorough data exploration and statistical analysis, we ensured the dataset was well-prepared for further investigation and modeling efforts.

<img width="526" alt="image" src="https://github.com/user-attachments/assets/4b9a3101-7460-4146-b53a-b58aaaabb632" />
<img width="526" alt="image" src="https://github.com/user-attachments/assets/f6e7968d-25e4-421f-8d73-b553ba72086e" />

We assessed the presence of null values within the dataset using the code fraud_df.isnull().any(), revealing the absence of any null values across all columns. This indicates that the dataset is complete, with no missing data points, ensuring the integrity of subsequent analyses.

<img width="526" alt="image" src="https://github.com/user-attachments/assets/759fc73c-b51f-4865-b2f5-870a1d38f911" />

After conducting outlier detection using Min-Max scaling with a threshold of 3 standard deviations from the mean, it was determined that no outliers were present in the dataset. This suggests that the data points within the dataset fall within a reasonable range of values without exhibiting extreme deviations. Further analysis may explore alternative outlier detection techniques or refine the threshold to ensure comprehensive data examination.

<img width="525" alt="image" src="https://github.com/user-attachments/assets/e61834f3-7c27-4eba-ac9b-7066b5754350" />

We employed the `dtype` function to determine the data types of variables within the dataset. This step allowed us to gain insights into the nature of each attribute, aiding in subsequent data preprocessing and analysis tasks.

<img width="525" alt="image" src="https://github.com/user-attachments/assets/86ca04fe-fb3e-406a-bab5-e7dd53726a2a" />

We utilized the `pd.get_dummies()` function to convert categorical variables into dummy/indicator variables within the DataFrame `fraud_df`. Specifically, we converted the columns 'Payment_Method,' 'Product_Category,' and 'Device_Used' into dummy variables while dropping the first category to avoid multicollinearity issues. This transformation facilitated the incorporation of categorical data into our analysis, enabling more robust modeling and prediction capabilities.

# Reduce Data Dimension:

Dimension reduction entails the removal of extraneous variables or columns from the dataset that hold no significance for the data mining (DM) analysis. In the case of the `fraud_df` DataFrame, columns such as "Transaction ID," "Customer ID," "Customer Location," "IP Address," "Shipping Address," and "Billing Address" were identified as dispensable and subsequently dropped. This process optimizes the dataset by retaining only pertinent features, enhancing the efficiency and efficacy of subsequent analytical procedures and modeling endeavors.

<img width="525" alt="image" src="https://github.com/user-attachments/assets/23754e80-0d68-4f79-bc11-53ee7ec4193f" />

<img width="526" alt="image" src="https://github.com/user-attachments/assets/c89b6aea-46eb-4006-a216-2110de2b0a51" />

# Determine the Data Mining Task

The data mining task for fraud detection in ecommerce transactions primarily involved classification, where the objective was to predict whether a transaction is fraudulent or not based on various features. This task is commonly addressed using supervised learning algorithms. In this particular analysis, decision tree and logistic regression algorithms were employed to classify transactions into fraudulent or non-fraudulent categories based on their respective characteristics. The decision tree algorithm partitions the feature space into regions, while logistic regression models the probability of a transaction being fraudulent. Both approaches contribute to enhancing fraud detection capabilities by learning patterns and relationships within the dataset to make accurate predictions.


# 1.	Partition Data

To mitigate the risk of overfitting, we adopted a partitioning strategy to develop our data, utilizing the train_test_split function with a test-size parameter set at 40%. This partitioning approach ensures that the dataset is divided into separate 'Training' and 'Validation' sets, with the Training partition comprising 60% of the data for model development and the Validation partition containing 40% for assessing performance on unseen data. By employing this methodology, we aim to strike a balance between model complexity and generalization, enhancing the robustness and reliability of our fraud detection model.


# 2.	Techniques

In our analysis, we employed a combination of techniques to enhance the effectiveness of fraud detection in ecommerce transactions. Specifically, we utilized decision tree algorithms to capture complex relationships within the data and Logistic Regression with backward elimination to refine model features. Additionally, we leveraged GridSearchCV to optimize model hyperparameters, ensuring optimal performance. These techniques collectively enabled us to develop robust and accurate fraud detection models capable of effectively identifying fraudulent transactions while minimizing false positives.

# Algorithm and Measures:
For our fraud detection task in ecommerce transactions, one of the techniques we utilized involved decision trees, complemented by the application of GridSearchCV for hyperparameter optimization. Decision trees are adept at capturing intricate patterns within the dataset, making them a suitable choice for classification tasks such as fraud detection. GridSearchCV allowed us to systematically explore various hyperparameter combinations, ensuring the optimal configuration for our decision tree model. This combination of techniques enabled the development of a robust fraud detection system tailored to the nuances of the dataset  .

<img width="518" alt="image" src="https://github.com/user-attachments/assets/1bc24013-41b1-48ad-872b-cf87692a7ebf" />

<img width="285" alt="image" src="https://github.com/user-attachments/assets/fae296e8-0889-4a00-a2a8-12321397090b" />

For the training partition, the confusion matrix reveals an accuracy of 0.7662 for the smaller decision tree model. The matrix illustrates the model's predictive performance, with 4633 true negatives (correctly predicted non-fraudulent transactions), 6232 true positives (correctly predicted fraudulent transactions), 2203 false positives (incorrectly predicted as fraudulent), and 1112 false negatives (incorrectly predicted as non-fraudulent). The misclassification rate for the training partition is approximately 23.39%. Similarly, for the validation partition, the confusion matrix indicates an accuracy of 0.7709, showcasing the model's ability to generalize to unseen data. The matrix displays 3060 true negatives, 4228 true positives, 1422 false positives, and 744 false negatives. The misclassification rate for the validation partition is approximately 22.91%. These metrics provide insights into the performance of the decision tree model in detecting fraudulent transactions, highlighting areas for further refinement and optimization.

Upon recognizing the limitations of the accuracy achieved with the smaller decision tree, we proceeded to implement GridSearchCV. This strategic approach resulted in a notable enhancement in model performance, achieving an accuracy score of 0.8336. The hyperparameters optimized through GridSearchCV, including a maximum depth of 12, no minimum impurity decrease, and minimum samples split of 3, played a pivotal role in achieving this improved accuracy. This outcome underscores the efficacy of hyperparameter tuning in refining the decision tree model, ultimately bolstering its predictive accuracy in identifying fraudulent transactions.

# Note: The diagram for the classification tree generated using Grid Search is currently unavailable and is too large to display directly. Please run the corresponding code segment for the classification tree employing the Grid Search algorithm to view the diagram.

<img width="525" alt="image" src="https://github.com/user-attachments/assets/8c3b29f4-eb20-459f-b8af-2ef8a9f3b525" />

For the training partition, the confusion matrix reveals an accuracy of 0.7662 for the smaller decision tree model. The matrix illustrates the model's predictive performance, with 4633 true negatives (correctly predicted non-fraudulent transactions), 6232 true positives (correctly predicted fraudulent transactions), 2203 false positives (incorrectly predicted as fraudulent), and 1112 false negatives (incorrectly predicted as non-fraudulent). The misclassification rate for the training partition is approximately 23.39%. Similarly, for the validation partition, the confusion matrix indicates an accuracy of 0.7709, showcasing the model's ability to generalize to unseen data. The matrix displays 3060 true negatives, 4228 true positives, 1422 false positives, and 744 false negatives. The misclassification rate for the validation partition is approximately 22.91%. These metrics provide insights into the performance of the decision tree model in detecting fraudulent transactions, highlighting areas for further refinement and optimization. 

<img width="471" alt="image" src="https://github.com/user-attachments/assets/04bf97ae-d5de-4108-86a0-078f0cff3e6e" />

# Logistic Regression

For our fraud detection task in ecommerce transactions, Other  technique we utilized involved logistic regression, complemented by the application of backward elimination for parameters optimization. Logistic regression can capture interactions between predictor variables through techniques like polynomial features or interaction terms, allowing it to model more complex relationships between the predictors and the outcome in cases of fraud detection. Backward elimination allowed us to systematically explore various combinations of parameters, ensuring the optimal parameters for our logistic regression model. This combination of techniques enabled the development of a robust fraud detection system tailored to the nuances of the dataset.

<img width="525" alt="image" src="https://github.com/user-attachments/assets/f697d8a2-14d7-4f94-b9eb-486ded5de0ff" />

# The mathematical equation of the trained logistic regression model is 

Logit = 0.483 + 0.002 Transaction_Amount  + 0.009 Customer_Age – 0.006 Account_Age_Days  - 0.076 Transaction_Hour + 0.18 payment_Method_bank_transfer – 0.161 payment_Method_credit_card – 0.081 payment_Method_debit_card – 0.05 product_category_electronics + 0.06 product_category_health_beauty + 0.085 product_category_home_garden – 0.069 product_category_toys_games +0.201 Device_Used_mobile +0.003 Device_used_tablet

Then in Python, we made predictions and identified probabilities p(0) and p(1)  for the validation data set.
Here are the first 20 classifications for the validation partition.

<img width="525" alt="image" src="https://github.com/user-attachments/assets/6b9a9131-4bfa-49ff-89e4-821af0ecfbb6" />

In this table, "Actual" represents the true fraudulent transactions, while "Classification" represents the predicted fraudulent transactions. The columns "p(0)" and "p(1)" represent the probabilities assigned to the classes "Legitimate " (0) and "Fraud " (1), respectively.

From the table, it can be observed that in the majority of cases, the predicted fraudulent transaction matches the actual fraudulent transaction. The model tends to assign higher probabilities to the correct class, indicating that it is generally confident in its predictions. However, there are almost 50% instances where the model misclassifies the fraudulent transactions, such as in rows 398 ,7443  where the actual is "legitimate" but the model predicts "Fraud". Overall, the model demonstrates good performance in predicting fraudulent transactions, as evidenced by the high percentage of correct classifications. 

<img width="259" alt="image" src="https://github.com/user-attachments/assets/44b851bb-594c-485b-b931-f6c950f111e8" />

 For the training partition, the confusion matrix reveals an accuracy of 0.7079 for the logistic regression model with all predictors. The matrix illustrates the model's predictive performance, with 4860 true negatives (correctly predicted non-fraudulent transactions), 5178 true positives (correctly predicted fraudulent transactions), 1976 false positives (incorrectly predicted as fraudulent), and 2166 false negatives (incorrectly predicted as non-fraudulent). The misclassification rate for the training partition is approximately 29.21%. Similarly, for the validation partition, the confusion matrix indicates an accuracy of 0.7070, showcasing the model's ability to generalize to unseen data. The matrix displays 3220 true negatives, 3464 true positives, 1262 false positives, and 1508 false negatives. The misclassification rate for the validation partition is approximately 29.30%. These metrics provide insights into the performance of the logistic regression model in detecting fraudulent transactions, highlighting areas for further refinement and optimization

<img width="525" alt="image" src="https://github.com/user-attachments/assets/822848a9-f7d3-4ba7-a5d6-e5d4839f279b" />

Here, we plotted a gains and lift chart which effectively illustrates the predictive model's capability in identifying fraudulent transactions. From the lift chart we can see that in the top 10 percentile of the records, this model helps to predict a fraudulent transaction accurately 1.8 times higher than when the records are predicted at random. similarly, at every decile of records, we can observe the accuracy of this model to help predict fraudulent transactions. Gains charts can reveal whether a model is well-calibrated, meaning the predicted probabilities align well with the actual cases of fraudulent transactions. This is a well-calibrated model as it is exhibiting a smooth gains curve as a poorly calibrated model might show erratic behavior or deviation from the diagonal line.

Upon recognizing the limitations of the accuracy achieved with the logistic regression model with all predictors, we proceeded to implement a backward elimination algorithm. This strategic approach resulted in a notable enhancement in model performance, achieving a score of 22474.42. The parameters eliminated through Backward elimination including payment_method_credit_card and Device_used_tablet played a pivotal role in achieving this improved score. This outcome underscores the efficacy of parameter elimination tuning in refining the logistic regression model, ultimately bolstering its predictive accuracy in identifying fraudulent transactions.

 <img width="525" alt="image" src="https://github.com/user-attachments/assets/325dc7f4-a458-486a-94c3-d9dc2bc2eff1" />

<img width="525" alt="image" src="https://github.com/user-attachments/assets/5fc825e8-bf22-4991-966e-9b1cb541a64f" />

# The  mathematical  equation  of  logistic regression using  backward elimination algorithm is :

Logit = 0.949 + 0.002 Transaction_Amount  – 0.006 Account_Age_Days  - 0.076 Transaction_Hour – 0.001 payment_Method_credit_card – 0.23 payment_Method_debit_card – 0.054 product_category_electronics  + 0.088 product_category_health_beauty + 0.081 product_category_home_garden – 0.066 product_category_toys_games +0.156 Device_Used_mobile – 0.071 Device_used_tablet

Based on the comparison between the logistic regression models before and after applying the backward elimination algorithm, several variables were removed in the process. The backward elimination algorithm is a method used in statistical models to simplify the model by iteratively removing the least significant variables, based on certain criteria like the p-value.

# Variables Removed Through Backward Elimination:

**1.	Customer Age** - This variable was included in the initial model but is absent in the model after backward elimination, suggesting that customer age may not have a significant impact on the prediction of the outcome in the presence of other variables.
**2.	Payment Method: Bank Transfer** - The variable indicating whether the payment was made through bank transfer was also removed, indicating that this method of payment does not significantly affect the logistic regression model’s predictions when other variables are considered.
**3.	Device Used: Mobile** - Initially considered in the model, the specific usage of mobile devices was later removed, implying that the use of mobile devices, compared to other devices, does not uniquely influence the model's predictive power

These eliminations indicate a refinement of the model to focus on variables that most significantly predict the outcome, enhancing model efficiency and potentially improving its performance by reducing overfitting.
 
<img width="504" alt="image" src="https://github.com/user-attachments/assets/3b708aba-8f98-48d5-a001-06790145ecaa" />

For the training partition utilizing one of the best logistic regression models obtained through backward elimination, the confusion matrix reveals an accuracy of 0.7110. This matrix depicts the model's predictive performance, with 4874 true negatives (non-fraudulent transactions correctly predicted), 5208 true positives (fraudulent transactions correctly predicted), 1962 false positives (non-fraudulent transactions incorrectly predicted as fraudulent), and 2136 false negatives (fraudulent transactions incorrectly predicted as non-fraudulent). The misclassification rate for the training partition is approximately 28.89%. Similarly, for the validation partition employing the best predictors obtained through backward elimination, the confusion matrix indicates an accuracy of 0.7117. This matrix illustrates the model's ability to generalize to unseen data, with 3237 true negatives, 3491 true positives, 1245 false positives, and 1481 false negatives. The misclassification rate for the validation partition is approximately 28.84%. These metrics provide valuable insights into the performance of the logistic regression model optimized through backward elimination, facilitating further analysis and refinement as needed.

# Interpret Results:

The confusion matrix small classification tree indicates a good fit with an accuracy close to 76% for both training and validation partitions and here there is no possibility of overfitting. However, we further  improved the prediction power of the algorithm using GridsearchCV()

 <img width="280" alt="image" src="https://github.com/user-attachments/assets/67a06959-ba8b-4d51-a58a-ff893d8b628d" />

The confusion matrix for best classification tree with Grid search  has improved accuracy scores close to 85% for both training and partition validation and here as well we clearly see there is no possibility of overfitting.

<img width="443" alt="image" src="https://github.com/user-attachments/assets/aedf17a9-f5f8-4c0a-8566-5ce5d61a751f" />

 The confusion matrix for the logistic regression model with all predictors has an accuracy score close to 70% for both training and validation partitions however we do not see any overfitting in this model as well but using backward elimination we plan to eliminate the predictors and achieve higher accuracy scores

 <img width="259" alt="image" src="https://github.com/user-attachments/assets/f0843993-bfd6-4797-9b51-1f35161f3d45" />

The confusion matrix for the improved logistic regression model with  backward elimination by reducing predictors has an accuracy score of 71% for both training and validation partitions. However, we see that there is no chance of overfitting here.

<img width="504" alt="image" src="https://github.com/user-attachments/assets/aba3ba58-c4da-4ac8-8192-80058d6255c2" />

 
# Conclusion
 
**Classification Tree:** 
By implementing the classification tree, with two different outcomes, we can conclude that our data is best used for predicting fraudulent transactions, we can predict the new data with an accuracy of 77%. However on improving the classification tree using GridSearchCV() function and optimizing the parameters we have been able to achieve an accuracy score of 85%. Although the prediction accuracy is good enough for us to make a more precise prediction, there might still be some things that could be improved in predicting whether a transaction is a legitimate or a fraudulent transaction.

**Logistic Regression:**
The logistic regression model, with an accuracy of around 70%, shows good results using the full model for predicting fraudulent transactions. The backward elimination algorithm, however, provided no advantages in eliminating predictors as the accuracy score has just improved by 1 percentage point and the logistic regression model did not provide acceptably confident results for predicting fraudulent transactions, as there was higher misclassification of records

<img width="439" alt="image" src="https://github.com/user-attachments/assets/640b92ba-1cb5-44c9-9e77-2d4d9db0181e" />

<img width="475" alt="image" src="https://github.com/user-attachments/assets/37fdeff4-40b5-41f2-8be6-b433dfc2327b" />

<img width="440" alt="image" src="https://github.com/user-attachments/assets/7fe90a8c-5fad-44da-a103-7e8b4296faf7" />

<img width="498" alt="image" src="https://github.com/user-attachments/assets/30ea906f-9840-412f-b548-ea396119cbf6" />

# Summary: 
The decision tree model, particularly with grid search optimization, proved to be effective in accurately predicting fraudulent transactions. However, the logistic regression model did not perform as well, achieving an accuracy of only 70%, which might be lower than expected.
Although logistic regression offers probability statistics as outputs, which can be valuable for certain applications, the decision tree's visual representation and its ability to capture the importance of individual variables in predicting outcomes make it a preferable choice in this scenario.Ultimately, the recommendation is to use the decision tree model for predicting whether a transaction is legitimate or fraudulent due to its higher accuracy and intuitive visual representation. This choice aligns with the goal of accurately identifying fraudulent transactions, which is of utmost importance in e-commerce fraud detection. 

# Bibliography


https://www.kaggle.com/datasets/shriyashjagtap/fraudulent-e-commerce-transactions
https://ieeexplore.ieee.org/document/9591720


