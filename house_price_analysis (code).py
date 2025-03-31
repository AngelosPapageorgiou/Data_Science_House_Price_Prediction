import pandas as pd
import numpy as np
# Set file location

file_path = r'C:\Users\aggel\Desktop\house-price-prediction\Housing.csv'

# Load data

data = pd.read_csv(file_path)

# Check the first rows of the dataset

print(data.head())

# General information about the dataset

print(data.info())

# 545 rows , from 0 to 544
# 13 columns
# Numerical Columns (int64) : price, area, bedrooms, bathrooms, stories, parking
# Categorical Columns (object) : mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea, furnishingstatus

# Checking for missing values

print(data.isnull().sum())

# No missing values

import matplotlib.pyplot as plt
import seaborn as sns

# Visualization of the house price distribution (SalePrice)

plt.figure(figsize=(10, 6))
sns.histplot(data['price'], kde=True, color='blue')
plt.title('House Price Distribution')
plt.xlabel('House Price')
plt.ylabel('Frequency')
plt.show()

# Scatter plot price vs area

sns.scatterplot(data=data, x='area', y='price', color='blue')
plt.title('Price vs Area')
plt.xlabel('Area (in square feet)')
plt.ylabel('Price')
plt.show()

# Scatter plot price vs bedrooms

sns.scatterplot(data=data, x='bedrooms', y='price', color='green')
plt.title('Price vs Bedrooms')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Price')
plt.show()

# Scatter plot price vs bathrooms

sns.scatterplot(data=data, x='bathrooms', y='price', color='red')
plt.title('Price vs Bathrooms')
plt.xlabel('Bathrooms')
plt.ylabel('Price')
plt.show()

# Compute the correlation matrix 
corr_matrix = data.select_dtypes(include=['number']).corr()

# Create Heatmap

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Create Boxplots for categorical features

categorical_features = ["furnishingstatus", "airconditioning", "basement", "guestroom", "hotwaterheating", "mainroad", "prefarea"]

plt.figure(figsize=(15,10))

for i, feature in enumerate(categorical_features, 1):

 plt.subplot(3, 3, i)
 sns.boxplot(x=data[feature], y=data["price"])
 plt.title(f"Price vs {feature}")

plt.tight_layout()
plt.show()

## Furnishingstatus ##

# Furnishing status Distribution

furnishingstatus_distr = data['furnishingstatus'].value_counts()
print("Furnishing status distribution")
print(furnishingstatus_distr)
print("\n")

# Furnished houses

furnished = data[data['furnishingstatus'] == 'furnished']['price']

# Semi-Furnished houses

semi_furnished = data[data['furnishingstatus'] == 'semi-furnished']['price']

# Unfurnished houses

unfurnished = data[data['furnishingstatus'] == 'unfurnished']['price']

# Statistics for Furnished houses

q1_furnished = furnished.quantile(0.25)

median_furnished = np.median(furnished)

q3_furnished = furnished.quantile(0.75)

iqr_furnished = q3_furnished - q1_furnished

upper_bound_furnished = q3_furnished + 1.5 * iqr_furnished

print("Statistics for furnished houses")
print(f"Q1 : {q1_furnished}")
print(f"Median : {median_furnished}")
print(f"Q3 : {q3_furnished}")
print(f"IQR : {iqr_furnished}")
print(f"Upper bound : {upper_bound_furnished}")
print("\n")

# Statistics for semi-furnished houses

q1_semi_furnished = semi_furnished.quantile(0.25)

median_semi_furnished = np.median(semi_furnished)

q3_semi_furnished = semi_furnished.quantile(0.75)

iqr_semi_furnished = q3_semi_furnished - q1_semi_furnished

upper_bound_semi_furnished = q3_semi_furnished + 1.5 * iqr_semi_furnished

print("Statistics for semi-furnished houses")
print(f"Q1 :{q1_semi_furnished}")
print(f"Median : {median_semi_furnished}")
print(f"Q3 : {q3_semi_furnished}")
print(f"IQR : {iqr_semi_furnished}")
print(f"Upper bound : {upper_bound_semi_furnished}")
print("\n")

# Statistics for unfurnished

q1_unfurnished = unfurnished.quantile(0.25)

median_unfurnished = np.median(unfurnished)

q3_unfurnished = unfurnished.quantile(0.75)

iqr_unfurnished = q3_unfurnished - q1_unfurnished

upper_bound_unfurnished = q3_unfurnished + 1.5 * iqr_unfurnished

print("Statistics for unfurnished houses")
print(f"Q1 : {q1_unfurnished}")
print(f"Median : {median_unfurnished}")
print(f"Q3 : {q3_unfurnished}")
print(f"IQR : {iqr_unfurnished}")
print(f"Upper bound : {upper_bound_unfurnished}")
print("\n")

# outliers for furnished

outliers_furnished = furnished[furnished > upper_bound_furnished]

print("Outliers for furnished houses :")
print(outliers_furnished)
print("\n")

# outliers for semi-furnished

outliers_semi_furnished = semi_furnished[semi_furnished > upper_bound_semi_furnished]

print("Outliers for semi furnished houses : ")
print(outliers_semi_furnished)
print("\n")

# outliers for unfurnished

outliers_unfurnished = unfurnished[unfurnished > upper_bound_unfurnished]

print("outliers for unfurnished houses :")
print(outliers_unfurnished)
print("\n")

## Airconditioning ## 

# Distribution of airconditioning

air_condition_dist = data['airconditioning'].value_counts()
print("Air conditioning distribution:")
print(air_condition_dist)
print("\n")

# Houses with air conditioning
with_air_condition = data[data['airconditioning'] == 'yes']['price']
# Houses without air conditioning
without_air_condition = data[data['airconditioning'] == 'no']['price']

# Calculating Statistics for houses with air conditioning

import numpy as np

q1_with_air_condition = with_air_condition.quantile(0.25)

median_with_air_condition = np.median(with_air_condition)

q3_with_air_condition = with_air_condition.quantile(0.75)

iqr_with_air_condition = q3_with_air_condition - q1_with_air_condition

upper_bound_with_air_condition = q3_with_air_condition + 1.5 * iqr_with_air_condition

print("Statistics for houses with air conditioning")
print(f"Q1: {q1_with_air_condition}")
print(f"Median: {median_with_air_condition}")
print(f"Q3: {q3_with_air_condition}")
print(f"IQR: {iqr_with_air_condition}")
print(f"Upper Bound: {upper_bound_with_air_condition}")
print("\n")

# Calculating Statistics for the houses without air conditioning

q1_without_air_condition = without_air_condition.quantile(0.25)

median_without_air_condition = np.median(without_air_condition)

q3_without_air_condition = without_air_condition.quantile(0.75)

iqr_without_air_condition = q3_without_air_condition - q1_without_air_condition

upper_bound_without_air_condition = q3_without_air_condition + 1.5 * iqr_without_air_condition

print("Statistics for houses without air conditioning")
print(f"Q1: {q1_without_air_condition}")
print(f"Median: {median_without_air_condition}")
print(f"Q3: {q3_without_air_condition}")
print(f"IQR :{iqr_without_air_condition}")
print(f"Upper Bound: {upper_bound_without_air_condition}")
print("\n")

# Calculating outliers for houses with air conditioning

outliers_with_air_condition = with_air_condition[with_air_condition > upper_bound_with_air_condition]

# Calculating outliers for houses without air conditioning

outliers_without_air_condition = without_air_condition[without_air_condition > upper_bound_without_air_condition]

print("Outliers for houses with air conditioning:")
print(outliers_with_air_condition)
print("\n")

print("Outliers for houses without air conditioning:")
print(outliers_without_air_condition)
print("\n")


## Basement ##

# Basement Distribution

basement_dist = data['basement'].value_counts()
print("Basement Distribution:")
print(basement_dist)
print("\n")

# Houses with basement

with_basement = data[data['basement'] == 'yes']['price']

# Houses without basement

without_basement = data[data['basement'] == 'no']['price']

# Calculating Statistics for houses with basement

q1_with_basement = with_basement.quantile(0.25)

median_with_basement = np.median(with_basement)

q3_with_basement = with_basement.quantile(0.75)

iqr_with_basement = q3_with_basement - q1_with_basement

upper_bound_with_basement = q3_with_basement + 1.5 * iqr_with_basement

print("Statistics for houses with basement:")
print(f"Q1 : {q1_with_basement}")
print(f"Median : {median_with_basement}")
print(f"Q3 : {q3_with_basement}")
print(f"IQR : {iqr_with_basement}")
print(f"Upper Bound : {upper_bound_with_basement}")
print("\n")

# Calculating Statistics for houses without basement 

q1_without_basement = without_basement.quantile(0.25)

median_without_basement = np.median(without_basement)

q3_without_basement = without_basement.quantile(0.75)

iqr_without_basement = q3_without_basement - q1_without_basement

upper_bound_without_basement = q3_without_basement + 1.5 * iqr_without_basement

print("Statistics for houses without basement :")
print(f"Q1 : {q1_without_basement}")
print(f"Median : {median_without_basement}")
print(f"Q3 : {q3_without_basement}")
print(f"IQR : {iqr_without_basement}")
print(f"Upper Bound : {upper_bound_without_basement}")
print("\n")

# Calculating outliers for houses with basement 

outliers_with_basement = with_basement[with_basement > upper_bound_with_basement]

print("Outliers for houses with basement")
print(outliers_with_basement)
print("\n")

# Calculating outliers for houses without basement

outliers_without_basement = without_basement[without_basement > upper_bound_without_basement]

print("Outliers for houses without basement :")
print(outliers_without_basement)
print("\n")



## Guestroom ##

# Guestroom distribution

guestroom_dist = data['guestroom'].value_counts()

print("Guestroom Distribution")
print(guestroom_dist)
print("\n")

# Houses with guestroom

with_guestroom = data[data['guestroom'] == 'yes']['price']

# Houses without guestroom

without_guestroom = data[data['guestroom'] == 'no']['price']

# Calculating statistics for houses with guestroom

q1_with_guestroom = with_guestroom.quantile(0.25)

median_with_guestroom = np.median(with_guestroom)

q3_with_guestroom = with_guestroom.quantile(0.75)

iqr_with_guestroom = q3_with_guestroom - q1_with_guestroom

upper_bound_with_guestroom = q3_with_guestroom + 1.5 * iqr_with_guestroom

print("Statistics for houses with guestroom")
print(f"Q1 : {q1_with_guestroom}")
print(f"Median : {median_with_guestroom}")
print(f"Q3 : {q3_with_guestroom}")
print(f"IQR : {iqr_with_guestroom}")
print(f"Upper bound : {upper_bound_with_guestroom}")
print("\n")

# Calculating statistics for houses without guestroom

q1_without_guestroom = without_guestroom.quantile(0.25)

median_without_guestroom = np.median(without_guestroom)

q3_without_guestroom = without_guestroom.quantile(0.75)

iqr_without_guestroom = q3_without_guestroom - q1_without_guestroom

upper_bound_without_guestroom = q3_without_guestroom + 1.5 * iqr_without_guestroom

print("Statistics for houses without guestroom")
print(f"Q1 : {q1_without_guestroom}")
print(f"Median : {median_without_guestroom}")
print(f"Q3 : {q3_without_guestroom}")
print(f"IQR : {iqr_without_guestroom}")
print(f"Upper bound : {upper_bound_without_guestroom}")
print("\n")

# Calculating outliers for houses with guestroom

outliers_with_guestroom = with_guestroom[with_guestroom > upper_bound_with_guestroom]

print("Outliers for houses without guestroom")
print(outliers_with_guestroom)
print("\n")

# Calculating outliers for houses without guestroom

outliers_without_guestroom = without_guestroom[without_guestroom > upper_bound_without_guestroom]

print("Outliers for houses without guestroom")
print(outliers_without_guestroom)
print("\n")

## Hot Water Heating ##

# Hot water heating distribution

hotwaterheating_dist = data['hotwaterheating'].value_counts()
print("Hot Water Heating Distribution :")
print(hotwaterheating_dist)
print("\n")

# Houses with hot water heating

with_hotwaterheating = data[data['hotwaterheating'] == 'yes']['price']

print(with_hotwaterheating.value_counts)

# Houses without hot water heating

without_hotwaterheating = data[data['hotwaterheating'] == 'no']['price']

# Statistics for houses with hot water heating

q1_with_hotwaterheating = with_hotwaterheating.quantile(0.25)

median_with_hotwaterheating = np.median(with_hotwaterheating)

q3_with_hotwaterheating = with_hotwaterheating.quantile(0.75)

iqr_with_hotwaterheating = q3_with_hotwaterheating - q1_with_hotwaterheating

upper_bound_with_hotwaterheating = q3_with_hotwaterheating + 1.5 * iqr_with_hotwaterheating

print("Statistics for houses with hot water heating")
print(f"Q1 : {q1_with_hotwaterheating}")
print(f"Median : {median_with_hotwaterheating}")
print(f"Q3 : {q3_with_hotwaterheating}")
print(f"IQR : {iqr_with_hotwaterheating}")
print(f"Upper bound : {upper_bound_with_hotwaterheating}")
print("\n")

# Statistics for houses without hot water heating 

q1_without_hotwaterheating = without_hotwaterheating.quantile(0.25)

median_without_hotwaterheating = np.median(without_hotwaterheating)

q3_without_hotwaterheating = without_hotwaterheating.quantile(0.75)

iqr_without_hotwaterheating = q3_without_hotwaterheating - q1_without_hotwaterheating

upper_bound_without_hotwaterheating = q3_without_hotwaterheating + 1.5 * iqr_without_hotwaterheating

print("Statistics for houses without hot water heating")
print(f"Q1 : {q1_without_hotwaterheating}")
print(f"Median : {median_without_hotwaterheating}")
print(f"Q3 : {q3_without_hotwaterheating}")
print(f"IQR : {iqr_without_hotwaterheating}")
print(f"Upper Bound : {upper_bound_without_hotwaterheating}")
print("\n")

# Outliers for houses with hot water heating

outliers_with_hotwaterheating = with_hotwaterheating[with_hotwaterheating > upper_bound_with_hotwaterheating]

print("Outliers for houses with hot water heating :")
print(outliers_with_hotwaterheating)
print("\n")

# Outliers for houses without hot water heating

outliers_without_hotwaterheating = without_hotwaterheating[without_hotwaterheating > upper_bound_without_hotwaterheating]

print("Outliers for houses without hot water heating")
print(outliers_without_hotwaterheating)
print("\n")


## Main Road ##

# Main road distribution
mainroad_dist = data['mainroad'].value_counts()

print("Main road distribution :")
print(mainroad_dist)
print("\n")

# Houses that are on main road

mainroad = data[data['mainroad'] == 'yes']['price']

# Houses that are not on main road 

no_mainroad = data[data['mainroad'] == 'no']['price']

# Statistics for houses that are on main road

q1_mainroad = mainroad.quantile(0.25)

median_mainroad = np.median(mainroad)

q3_mainroad = mainroad.quantile(0.75)

iqr_mainroad = q3_mainroad - q1_mainroad

upper_bound_mainroad = q3_mainroad + 1.5 * iqr_mainroad

print("Statistics for houses that are on main road :")
print(f"Q1 : {q1_mainroad}")
print(f"Median : {median_mainroad}")
print(f"Q3 : {q3_mainroad}")
print(f"IQR : {iqr_mainroad}")
print(f"Upper bound : {upper_bound_mainroad}")
print("\n")

# Statistics for houses that are not on main road

q1_no_mainroad = no_mainroad.quantile(0.25)

median_no_mainroad = np.median(no_mainroad)

q3_no_mainroad = no_mainroad.quantile(0.75)

iqr_no_mainroad = q3_no_mainroad - q1_no_mainroad

upper_bound_no_mainroad = q3_no_mainroad + 1.5 * iqr_no_mainroad

print("Statistics for houses that are not on main road :")
print(f"Q1 : {q1_no_mainroad}")
print(f"Median : {median_no_mainroad}")
print(f"Q3 : {q3_no_mainroad}")
print(f"IQR : {iqr_no_mainroad}")
print(f"Upper bound : {upper_bound_no_mainroad}")
print("\n")

# Outliers for houses that are on main road 

outliers_mainroad = mainroad[mainroad > upper_bound_mainroad]

print("Outliers for houses that are on main road :")
print(outliers_mainroad)
print("\n")

# Outliers for houses that are not on main road

outliers_no_mainroad = no_mainroad[no_mainroad > upper_bound_no_mainroad]

print("Outliers for houses that are not on main road :")
print(outliers_no_mainroad)
print("\n")

## Preferred Area ##

# Preferred area distribution

prefarea_distr = data['prefarea'].value_counts()

print("Preferred area distribution :")
print(prefarea_distr)
print("\n")

# Houses that are in preferred area

prefarea = data[data['prefarea'] == 'yes']['price']

# Houses that are not in preferred area

no_prefarea = data[data['prefarea'] == 'no']['price']

# Statistics for houses that are in preferred area

q1_prefarea = prefarea.quantile(0.25)

median_prefarea = np.median(prefarea)

q3_prefarea = prefarea.quantile(0.75)

iqr_prefarea = q3_prefarea - q1_prefarea

upper_bound_prefarea = q3_prefarea + 1.5 * iqr_prefarea

print("Statistics for houses that are in preferred area :")
print(f"Q1 : {q1_prefarea}")
print(f"Median : {median_prefarea}")
print(f"Q3 : {q3_prefarea}")
print(f"IQR : {iqr_prefarea}")
print(f"Upper bound : {upper_bound_prefarea}")
print("\n")

# Statistics for houses that are not in preferred area 

q1_no_prefarea = no_prefarea.quantile(0.25)

median_no_prefarea = np.median(no_prefarea)

q3_no_prefarea = no_prefarea.quantile(0.75)

iqr_no_prefarea = q3_no_prefarea - q1_no_prefarea

upper_bound_no_prefarea = q3_no_prefarea + 1.5 * iqr_no_prefarea

print("Statistics for houses that are not in preferred area :")
print(f"Q1 : {q1_no_prefarea}")
print(f"Median : {median_no_prefarea}")
print(f"Q3 : {q3_no_prefarea}")
print(f"IQR : {iqr_no_prefarea}")
print(f"Upper bound : {upper_bound_no_prefarea}")
print("\n")

# Outliers for houses that are in preferred area

outliers_prefarea = prefarea[prefarea > upper_bound_prefarea]

print("Outliers for houses that are in prefarea :")
print(outliers_prefarea)
print("\n")

# Outliers for houses that are not in preferred area 

outliers_no_prefarea = no_prefarea[no_prefarea > upper_bound_no_prefarea]

print("Outliers for houses that are not in preferred area :")
print(outliers_no_prefarea)
print("\n")

#######################################################################################################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

# Handle categorical variables with LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
columns_to_encode = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
for col in columns_to_encode:
    data[col] = le.fit_transform(data[col])

# Split into X (features) and y (target)
X = data.drop('price', axis=1)
y = data['price']

# Split into Train/Validation/Test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% Train, 30% Temp (validation + test)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # Split the 30% into 50% validation, 50% test

# Scale the data for X
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train the Gradient Boosting Model
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train_scaled, y_train)

# Make predictions with the Gradient Boosting model
y_pred_gb = gb_model.predict(X_test_scaled)

# Evaluate the Gradient Boosting Model
mae_gb = mean_absolute_error(y_test, y_pred_gb)
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = mse_gb ** 0.5
r2_gb = r2_score(y_test, y_pred_gb)

# Print evaluation results
print(f"Gradient Boosting Model Evaluation Results:")
print(f"MAE: {mae_gb}")
print(f"MSE: {mse_gb}")
print(f"RMSE: {rmse_gb}")
print(f"R-squared: {r2_gb}")

# Cross-validation for the Gradient Boosting Model
gb_cv_scores = cross_val_score(gb_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
gb_cv_rmse = (-gb_cv_scores.mean()) ** 0.5
print(f"Gradient Boosting CV RMSE: {gb_cv_rmse}")

# Create DataFrame with actual and predicted values
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_gb})

# Sort DataFrame by index to maintain correct order
comparison_df['Index'] = y_test.index
comparison_df = comparison_df.sort_values(by='Index').drop('Index', axis=1)

# Calculate difference between actual and predicted values
comparison_df['Difference'] = abs(comparison_df['Actual'] - comparison_df['Predicted'])
comparison_df['Percentage_Difference'] = (comparison_df['Difference'] / comparison_df['Actual']) * 100

# Set a threshold to consider values as "close"
threshold_percentage = 10  # percentage %

# Filter to show only values that are close
close_predictions = comparison_df[comparison_df['Percentage_Difference'] <= threshold_percentage]

# Display close predictions with proper formatting
print("\nActual and predicted values that are very close (difference <= {}%):".format(threshold_percentage))
print(close_predictions[['Actual', 'Predicted', 'Percentage_Difference']].to_string(
    formatters={
        'Actual': '${:.2f}'.format,
        'Predicted': '${:.2f}'.format,
        'Percentage_Difference': '{:.2f}%'.format
    }
))

# Alternatively, you can export the results to a csv file
# close_predictions.to_csv('close_predictions.csv', index=False)
