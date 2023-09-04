# HOUSE_PRICE_PREDICTION

This GitHub repository contains a machine learning project that utilizes the XGBoost algorithm to predict housing prices in California using the California Housing dataset. The repository includes all the necessary code and resources to train and evaluate the model.

![HOUSE_PRICE](house.png)

# OBJECTIVE
The objective of a house price prediction project is to develop a model that can accurately estimate or predict the market value or selling price of residential properties based on various input features. This prediction can be used for a variety of purposes, including real estate investment, property appraisal, and market analysis. The primary goal is to provide valuable insights into property valuation, enabling informed decision-making for buyers, sellers, and real estate professionals.

[Open in Google Colab](https://colab.research.google.com/drive/1vOWM59iAa5UW9FpPasvjIckijM0tjIP7?usp=sharing)
![Google Collab](ML_PROJECT_House_Price_Prediction.ipynb)

# DATASET
The California Housing dataset is a popular dataset often used for regression tasks. It contains data related to housing prices in various districts of California, along with several features such as population, median income, and more.

# DEPENDENCIES
* Numpy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

# DATA VISUALISATION
we have used Python's Data visualization libraries such as:- Seaborn and Matplotlib, to create informative and visually appealing plots. We explore various types of visualizations, including:
* **Box Plot:-** A box plot is a powerful visualization for understanding the spread and central tendency of data. Here, box plot is created between 'Price' and 'HouseAge'.
* **Scatter Plot:-** Scatter plots are used to visualize the relationship between two numeric variables. Here, Scatter plot has been used to show relation between Actual prices and Predicted price.
* **Heat Map:-** Heat maps are a valuable tool for visualizing the correlation between numeric variables. They provide insights into how variables relate to each other.
* **Histogram:-** Histograms are useful for understanding the distribution of a single numeric variable. They display the frequency or probability distribution of that variable.

# DATA PREPROCESSING

Before building our predictive models, it's essential to preprocess the data to ensure it's in a suitable format for training and testing. In this project, we employed two fundamental preprocessing techniques:

## Feature Scaling with StandardScaler

Standard scaling, also known as Z-score normalization, is a preprocessing technique used to standardize the range of independent variables or features in the dataset. This process transforms the features to have a mean of 0 and a standard deviation of 1. Standardization ensures that all features contribute equally to machine learning model training and avoids problems caused by features with different scales.

In our project, we applied StandardScaler from the scikit-learn library to scale our features. This step was essential as it allowed our machine learning models to converge efficiently and deliver improved predictive performance.

## Train-Test Split

To evaluate the performance of our machine learning models and assess their ability to generalize to new data, we employed the train-test split technique. This method involves dividing our dataset into two subsets: a training set and a testing set.

The training set is used to train our models, allowing them to learn patterns and relationships within the data. The testing set, which remains unseen during training, is used to evaluate the model's performance and assess its ability to make accurate predictions on new, unseen data.

In our project, we utilized the `train_test_split` function from the scikit-learn library to perform this split. We allocated 80% of the data to the training set and reserved the remaining 20% for testing. The use of a random seed ensured the reproducibility of our results.

These data preprocessing steps were essential in ensuring the quality and reliability of our machine learning models, enabling us to make informed predictions with confidence.

# BUILDING THE MODEL USING "XGBoost"
XGBoost is known for its exceptional predictive accuracy. It excels in capturing complex relationships between features and the target variable, which is crucial in accurately estimating house prices that can be influenced by a multitude of factors. XGBoost employs an ensemble learning approach, which combines the predictions of multiple decision trees to create a strong predictive model. This ensemble method helps reduce overfitting and enhances model generalization. XGBoost incorporates L1 (Lasso) and L2 (Ridge) regularization techniques, which help prevent overfitting by penalizing overly complex models. This is essential in maintaining the model's ability to generalize well to new, unseen data.

# LESSONS LEARNED
* **Data Quality Matters:** The quality of our data has a significant impact on the performance of your machine learning model. Addressing missing values, outliers, and noisy data is crucial for building an accurate model.
* **Feature Engineering:** Thoughtful feature engineering can greatly improve model performance. Creating meaningful and relevant features or transforming existing ones can make a substantial difference in predictive accuracy.
* **Data Splitting Strategy:** Properly splitting your data into training and testing sets is vital for model evaluation. Consideration of stratified sampling and cross-validation can help mitigate issues related to data partitioning.
* **Interpretable Models:** In some cases, interpretability may be as important as predictive accuracy, especially in domains where model explanations are crucial for decision-making.

# CONTRIBUTION
Feel free to contribute to this repository by opening issues, proposing improvements, or adding new features. Pull requests are welcome!

# LICENSE
This project is licensed under the MIT License - see the LICENSE file for details.

# ACKNOWLEDGEMENTS
* The California Housing dataset is available from the Scikit-Learn Datasets.
* XGBoost: XGBoost Documentation




























