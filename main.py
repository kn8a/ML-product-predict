# pip install -r requirements.txt

from data import process_dataset
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import learning_curve
import seaborn as sns #Training curve
import matplotlib.pyplot as plt #Scatter graph
import easygui #GUI
from tqdm import tqdm #Progress bar
import logging
import warnings
warnings.filterwarnings("ignore")

#1. Set CPU cores
num_cores = 8
os.environ["LOKY_MAX_CPU_COUNT"] = str(num_cores)

#! LOAD DATA
# 1.Specify the file path
file_path = 'shopping_trends_updated.csv'

# 2.Apply operations from data.py
df = process_dataset(file_path)

# 3.Make a copy of the DataFrame
df2 = df.copy()


#! ML MODEL
#1. Encode the categorical features as numbers
categorical_mappings = {
    'Gender': {'Male': 0, 'Female': 1},
    'Region': {'North East': 0, 'Midwest': 1, 'South': 2, 'West': 3},
    'Age Group': {'0-18': 0, '19-27': 1, '28-36': 2, '37-45': 3, '46-55': 4, '55+': 5},
    'Season': {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3},
}

for column, mapping in categorical_mappings.items():
    df2[column] = df2[column].map(mapping)

#2. Define X and y 
X = df2[['Gender', 'Region', 'Age Group', 'Season']]  
y = df2['Item Purchased']

#2.5. Define target
y_colors = df2['Color']  

#3. Split data
with tqdm(total=100, desc="Splitting data") as pbar:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pbar.update(50)  
    X_train, X_test, y_train_colors, y_test_colors = train_test_split(X, y_colors, test_size=0.2, random_state=42)
    pbar.update(50)

#3.5 Hyperparameter tuning
param_grid = {                  #QUICK         BEST       LET PROGRAM DECIDE ON BEST PARAM        
    'max_iter': [1, 20, 90],    # 10           90         20,90,200   
    'max_depth': [None, 2, 4],  # None         4          2,4,10
}

#4. Train
hgboost = HistGradientBoostingClassifier()
grid_search = GridSearchCV(hgboost, param_grid, cv=5, verbose=20)
grid_search.fit(X_train, y_train)

#5. Get the best hyperparameters
best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

#6. Train a model with the best hyperparameters
hgboost = HistGradientBoostingClassifier(**best_params)
hgboost.fit(X_train, y_train)

#6.5 Print the accuracy of the tuned model
print("Tuned Product prediction accuracy:", hgboost.score(X_test, y_test))

#7. Cross-validation using KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

#8. Perform cross-validation
with tqdm(total=kf.get_n_splits(), desc="Cross-validation") as pbar:
    cv_scores = []
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        hgboost_cv = HistGradientBoostingClassifier(**best_params)
        hgboost_cv.fit(X_train_fold, y_train_fold)
        score = hgboost_cv.score(X_val_fold, y_val_fold)
        cv_scores.append(score)
        # Increment the progress bar
        pbar.update(1) 

# Print the cross-validation scores
print("Cross-Validation Scores:", cv_scores)


#! TRAIN COLORS MODEL
# Set up logging
logging.basicConfig(level=logging.INFO)
logging.info("Color model training initiated.")
#9. Train color model with logging
color_model = HistGradientBoostingClassifier()
logging.info("Training...")
color_model.fit(X_train, y_train_colors)
# Log information during training
logging.info(f"Color prediction accuracy: {color_model.score(X_test, y_test_colors)}")


#! PRODUCT SCATTER GRAPH - uncomment code below to render graph
# feat_1 = 'Season'
# feat_2 = 'Age Group'
# feat_3 = 'Region'

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Map products to integers
# product_mapping = {p: i for i, p in enumerate(np.unique(y_test))}  

# # Encode test labels 
# y_test_encoded = [product_mapping[y] for y in y_test]

# # Plot 
# ax.scatter(X_test[feat_1], X_test[feat_2], X_test[feat_3],
#            c=y_test_encoded, cmap='tab10') 

# # Get colormap
# cmap = plt.cm.get_cmap('tab10', len(product_mapping))

# # Define handles 
# handles = [plt.Rectangle((0,0),1,1, color=cmap(i))  
#            for i in range(len(product_mapping))]

# labels = list(product_mapping.keys())

# ax.legend(handles, labels)
# ax.set_xlabel(feat_1)
# ax.set_ylabel(feat_2)
# ax.set_zlabel(feat_3)
# plt.title("Products Distribution")
# plt.show()


#! COLOR SCATTER GRAPH - uncomment code below to render graph
# fig_color = plt.figure()
# ax_color = fig_color.add_subplot(111, projection='3d')

# # Predict colors using the trained color_model
# predicted_colors = color_model.predict(X_test)

# # Map colors to integers
# color_mapping = {c: i for i, c in enumerate(np.unique(predicted_colors))}

# # Encode predicted colors
# predicted_colors_encoded = [color_mapping[c] for c in predicted_colors]

# # Plot the color_model scatterplot
# ax_color.scatter(X_test[feat_1], X_test[feat_2], X_test[feat_3],
#                  c=predicted_colors_encoded, cmap='tab10')

# # Get colormap
# cmap_color = plt.cm.get_cmap('tab10', len(color_mapping))

# # Define handles for legend
# handles_color = [plt.Rectangle((0, 0), 1, 1, color=cmap_color(i))
#                  for i in range(len(color_mapping))]

# labels_color = list(color_mapping.keys())

# ax_color.legend(handles_color, labels_color)
# ax_color.set_xlabel(feat_1)
# ax_color.set_ylabel(feat_2)
# ax_color.set_zlabel(feat_3)
# plt.title("Colors Distribution")
# plt.show()


#! LEARNING CURVE GRAPH - uncomment code below to render graph
# # Function to plot learning curve
# def plot_learning_curve(estimator, X, y, title="Learning Curve", cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
#     train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
#         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, return_times=True, scoring='accuracy'
#     )
    
#     # Calculate mean and standard deviation of training scores and test scores
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
    
#     # Plot the learning curve
#     plt.figure(figsize=(10, 6))
#     plt.title(title)
#     plt.xlabel("Training examples")
#     plt.ylabel("Score")
#     plt.grid()

#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1, color="r")
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

#     plt.legend(loc="best")
#     return plt

# # Plot learning curve
# plot_learning_curve(hgboost, X, y, title="Learning Curve", cv=5, n_jobs=-1)
# plt.show()


#! GUI
from gui import *

#5. Loop to ask the user if they want to select new criteria
while True:
    # Predictions for user selection
    user_data = [
        gender_mapping.get(user_selection['Gender'], None),
        region_mapping.get(user_selection['Region'], None),
        age_group_mapping.get(user_selection['Age Group'], None),
        season_mapping.get(user_selection['Season'], None)
    ]

    print("Selected:", user_data)

    product = hgboost.predict([user_data])[0]
    color = color_model.predict([user_data])[0]
    
    print("Recommended Product:", product)
    print("Recommended Color:", color)
    
    # Import the image processing module
    from image import process_image

    # Use the process_image function
    process_image(product, color)

    # Return predicted product,color and corresponding image
    easygui.msgbox(msg=color + " " + product, title='Recommended product', ok_button='OK',
                   image="./assets/output.png", root=None)

    # Ask the user to continue or exit
    choice = easygui.ynbox(msg="Do you want to select new criteria?", title="New Criteria", choices=('Yes, Try another', 'No, Exit'))
    if choice:
        # Get new user selection
        user_selection = get_user_selection()
    else:
        # End the program
        break
