# General purpose packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import randint

# Image processing packages
from skimage import io, color
from skimage.transform import resize
from skimage.segmentation import slic
from skimage.color import label2rgb
from skimage.filters import try_all_threshold, sobel
from skimage import exposure

# Preprocessing modeling packages
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Modeling packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier

# Test metrics packages
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_curve, auc, accuracy_score, mean_squared_error as MSE, classification_report

##########################################

df = pd.read_csv('signatures_data.csv', index_col=0) # Open the data frame with first column as index
print(df.head())
print(df.shape)

# Get the image of the row id name (the image has to be stored in the directory), return it as an array

# FUNCTION 1
def get_image(row_id):
    filename = "{}.jpeg".format(row_id)
    img = io.imread(filename)
    img = resize(img, (200,200), anti_aliasing=True) # resize image
    return np.array(img)

# Check the function in the first cat image and the first dog image

other_1_row = df[df["label"] == 0].index[0]
other_1 = get_image(other_1_row)
other_1.shape
plt.imshow(other_1)
plt.show()

personal_10_row = df[df["label"] == 1].index[9]
personal_10 = get_image(personal_10_row)
personal_10.shape
plt.imshow(personal_10)
plt.show()

####################################################

# Inspect converting to grayscale

other_1_grey = color.rgb2gray(other_1)
plt.imshow(other_1_grey, cmap=plt.cm.gray)
plt.show()

personal_10_grey = color.rgb2gray(personal_10)
plt.imshow(personal_10_grey, cmap=plt.cm.gray)
plt.show()

####################################################

# Apply edge detection

other_1_sobel = sobel(other_1_grey)
plt.imshow(other_1_sobel, cmap=plt.cm.gray)
plt.show()

personal_10_sobel = sobel(personal_10_grey)
plt.imshow(personal_10_sobel, cmap=plt.cm.gray)
plt.show()

#######################################################################3

# Create a function that grab all the features of the RGB resized image and the superpixeled image,
# then it flatten both together into the original row of the data frame. In that way every feature is converted
# to a column of the data frame and it can be used in a machine learning model.

# FUNCTION 2
def create_features(img):
    # 0. flatten all features of the RGB image
    # color_features = img.flatten()
    # 1. convert image to grayscale
    grey_image = color.rgb2gray(img)
    # 2. get the grey features
    grey_features = grey_image.flatten() 
    # 3. get the sobel features from the grayscale image
    sobel_features = sobel(grey_image).flatten()
    # 4. combine the RGB and the HOG features into a single array
    flat_features = np.hstack((grey_features, sobel_features))

    return flat_features

# Check the function in the first image
other_1_features = create_features(other_1)
other_1_features.shape

###############################################################

# Now we use functions 1 and 2 in one single new function to generate a matrix with one row for every image
# and one column for every feature of the images. This can be used for amchine learning

# FUNCTION 3
def create_feature_matrix(label_dataframe):
    feature_list = []

    for img_id in label_dataframe.index:
        # 1. Apply function 1 (convert image to array)
        img = get_image(img_id)
        # 2. Apply function 2 (generate features and stack them)
        img_features = create_features(img)
        # 3. Append img features to the empty list
        feature_list.append(img_features)

    # Convert the list of arrays into an array matrix
    feature_matrix = np.array(feature_list)

    return feature_matrix

# Apply the function to all the images id in the data frame 
# (remember images must be also in the directory)

features_matrix = create_feature_matrix(df)

# Inspect the matrix and their rows shape

type(features_matrix)
features_matrix.shape # 80 thousand columns each one row!! that´s big data for sure!

features_matrix[0].shape
features_matrix[19].shape
features_matrix[28].shape
features_matrix[31].shape

#########################

# Resizing the matrix with Standard Scaler and Principal Component Analisis (PCA): reducing feature numbers

ss = StandardScaler() # Standardize features by removing the mean and scaling to unit variance
stand_matrix = ss.fit_transform(features_matrix)

pca = PCA(n_components = 160) # reduce to 40 features
pca_matrix = pca.fit_transform(stand_matrix)

pca_matrix.shape


################################################################################################################

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(pca_matrix,
                                                    df.label.values,
                                                    test_size = .3,
                                                    random_state = 123)

# Check the split
pd.Series(y_train).value_counts()
pd.Series(y_test).value_counts()

################################################################################################################

# MODEL 0: K NEARIEST NEIGHBOR CLASSIFIER

# CrossValidation for the knn

param_grid = {"n_neighbors": range(1,22),
              "leaf_size": range(1,50),
              "p": [1,2]}

knn = KNeighborsClassifier()

knn_cv = RandomizedSearchCV(knn, param_grid, random_state= 1234, cv = 5) # generate a tree model and test in 5 folders the best params of the grid
knn_cv.fit(X_train, y_train)

print("Tuned KNN Parameters: {}".format(knn_cv.best_params_))
print("Best score is {}".format(knn_cv.best_score_))


################################################################################################################

# MODEL 1: DECISION TREE CLASSIFIER

# CrossValidation for the decision tree

param_grid = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

cart = DecisionTreeClassifier(random_state=1234)

tree_cv = RandomizedSearchCV(cart, param_grid, random_state= 1234, cv = 5) # generate a tree model and test in 5 folders the best params of the grid
tree_cv.fit(X_train, y_train)

print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))


################################################################################################################

# MODEL 2: LOGISTIC REGRESSION CLASSIFIER

# CrossValidation for the logistic regression

param_grid = {"dual": [True, False],
              "max_iter": randint(100, 150),
              "C": randint(1, 3)}

log = LogisticRegression(random_state=1234)

log_cv = RandomizedSearchCV(log, param_grid, random_state= 1234, cv = 5) # generate a log reg and test in 5 folders the best params of the grid
log_cv.fit(X_train, y_train)

print("Tuned Logistic Regression Parameters: {}".format(log_cv.best_params_))
print("Best score is {}".format(log_cv.best_score_))


##################################################################################################################

# MODEL 3: SUPPORT VECTOR MACHINE CLASSIFIER

# CrossValidation for the SVM

param_grid = {"gamma": ['scale', 'auto'],
              "kernel" : ['linear', 'poly', 'rbf', 'sigmoid'],
              "C": randint(1, 100)}

svm = SVC(random_state=1234)

svm_cv = RandomizedSearchCV(svm, param_grid, random_state= 1234, cv = 5) # generate a log reg and test in 5 folders the best params of the grid
svm_cv.fit(X_train, y_train)

print("Tuned SVM Parameters: {}".format(svm_cv.best_params_))
print("Best score is {}".format(svm_cv.best_score_))

##################################################################################################################

# MODEL 4: RANDOM FOREST CLASSIFIER

#Cross Validation for the Random Forest model
param_grid = {"n_estimators": randint(50,200),
              "criterion": ['gini', 'entropy'],
              "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]}

forest = RandomForestClassifier(random_state=1234)
forest_cv = RandomizedSearchCV(forest, param_grid, random_state=1234, cv=5)
forest_cv.fit(X_train, y_train)

print("Tuned Random Forest Parameters: {}".format(forest_cv.best_params_))
print("Best score is {}".format(forest_cv.best_score_))
##################################################################################################################


classifiers = [('K Neariest Neighbouss', knn_cv), ('Logistic Regression', log_cv), ('Support Vector Machine', svm_cv), ('Decision Tree', tree_cv), ('Random Forest', forest_cv)]


for clsf_name, clsf in classifiers:
    # Fit the training data
    clsf.fit(X_train, y_train)
    # Predict in the trin data
    y_pred_train = clsf.predict(X_train)
    # Predict in the test data
    y_pred_test = clsf.predict(X_test)
    # Calculate MSE in train
    mse_train =MSE(y_pred_train, y_train)
    # Calculate MSE in test
    mse_test = MSE(y_pred_test, y_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_pred_test, y_test)
    # Print MSE train
    print('{:s} mean squared error in train data : {:.3f}'.format(clsf_name, mse_train))
    # Print MSE test
    print('{:s} mean squared error in test data: {:.3f}'.format(clsf_name, mse_test))
    # Print accuracy
    print('{:s} accuracy in test data: {:.3f}'.format(clsf_name, accuracy))


############################################################################################################

# As logistic regression performed the best, we keep it and make a confussion matrix

# Probabilities of abel 1 in the test set
probabilities = log_cv.predict_proba(X_test)
y_proba = probabilities[:,1] # probabilities of 1 (bigcoin) in the test tada
print(y_proba)

# ROC curve
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba, pos_label = 1)
roc_auc = auc(false_positive_rate, true_positive_rate) # area under the curve

plt.title("ROC curve: area under the curve")
roc_plot = plt.plot(false_positive_rate,
                    true_positive_rate,
                    label = 'AUC = {:0.2}'.format(roc_auc))
plt.legend(loc=0)
plt.plot([0,1], [0,1], ls = '')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# Confussion matrix
y_pred_log_cv = log_cv.predict(X_test)
print( classification_report(y_pred_log_cv, y_test) )
