# ML7374_OnlineShoppersIntentions_Classification
This repository contains an implementation of various machine learning models, including data preprocessing, oversampling techniques, feature selection, and classifiers. The following models and techniques are included:

Data Preprocessing
Synthetic Minority Over-sampling Technique (SMOTE)
Radial Basis Oversampling
Minimum Redundancy Maximum Relevance (mRMR) feature selection
Naive Bayes Classifier
Logistic Regression
Support Vector Machine (SVM)

## Table of Contents
Classes and Methods
Preprocess
SMOTE
Radial Basis Oversampling
mRMR
Naive Bayes
Logistic Regression
SVM



## Classes and Methods
### Preprocess
This class contains methods for data preprocessing.

`train_test_split(X, Y)`: Splits the dataset into training and testing sets.
`extract_class(X, Y, class_)`: Extracts rows belonging to a particular class.
`pages_per_sec(dataset, x, y)`: Divides two columns and replaces one of the columns.
`rare_categories(X, categories)`: Combines rare categories into one category.
`add_prefix(X, categories)`: Adds the category name as a prefix to categorical values.
`log_normalize(X, num_columns)`: Performs log normalization on numerical columns.

### SMOTE
This class implements the SMOTE algorithm for oversampling.

`__init__(self, x, y, min_class, num_columns, cat_columns, cat_start_index)`: Initializes the SMOTE class.
`extract_class(X, Y, class_)`: Extracts rows belonging to a particular class.
`one_hot_encoding(x, categories)`: One-hot-encodes categorical columns.
`median_std(x)`: Finds the median of the standard deviations of numerical columns.
`neighbours(x_num, x_cat)`: Finds the 2 nearest neighbors of each point and returns their Euclidean distance.
`points(neighbour_points, x, column_names)`: Picks two points for each point between their 2 nearest neighbors.
`smote_nc()`: Generates new points using SMOTE.

### Radial Basis Oversampling
This class implements radial basis oversampling for handling class imbalance.

`__init__(self, X, Y, minority_class_label, majority_class_label, lamda, count, step_size, cat_start_index, categorical_columns)`: Initializes the Radial Basis Oversampling class.
`label_encoding()`: Label encodes the categorical values.
`inverse_label_encoding(x)`: Inverses label encoding.
`extract_class(X, Y)`: Extracts rows belonging to a particular class.
`score(point)`: Calculates the potential of a point.
`points()`: Generates new points.

### mRMR
This class implements the mRMR feature selection algorithm.

`__init__(self, X, Y, num_of_features, numerical_columns)`: Initializes the mRMR class.
`binning(X, numerical_columns)`: Discretizes numerical columns.
`mutual_information(X, Y)`: Calculates mutual information between two columns.
`max_mi(X, Y)`: Finds the column with maximum mutual information with the target column.
`score(old_set, new_column, target)`: Calculates the score of each column using redundancy and relevance.
`select()`: Selects the top features.

### Naive Bayes
This class implements the Naive Bayes classifier.

`__init__(self, xtrain, ytrain, xtest, ytest, numerical_columns, categorical_columns, cat_start_index)`: Initializes the Naive Bayes class.
`one_hot_encoding(x, categories)`: One-hot-encodes categorical columns.
`extract_class(X, Y, class_)`: Extracts rows belonging to a particular class.
`fit_dist(mean, std, value)`: Finds the probability of a value by fitting a normal distribution.
`prior_prob(y)`: Calculates prior probabilities.
`conditional_probability(x, y, class_, cat_columns)`: Calculates conditional probabilities.
`probabilites(x, y, num_columns, cat_columns)`: Stores the probabilities in respective variables.
`num_dist(x, y)`: Finds the mean and std of numerical columns.
`naive_bayes(x, probability_true, probability_false, prior_true, prior_false)`: Predicts the class.
`run_model()`: Runs the Naive Bayes model.

### Logistic Regression
This class implements the Logistic Regression model.

`__init__(self, Xtrain, Ytrain, Xtest, Ytest, maxiteration, tolerance, learningrate, categories)`: Initializes the Logistic Regression class.
`one_hot_encoding(x)`: One-hot-encodes categorical columns.
`add_bias(X)`: Adds a column of ones to the dataframe.
`cost_function(x, y)`: Calculates the cost function.
`sigmoid_function(x)`: Calculates the sigmoid function.
`derivative_cost(x, y)`: Calculates the derivative of the cost function.
`gradient_descent(x, y)`: Performs gradient descent to optimize the weights.
`predict(x)`: Predicts the class.
`evaluvate(ytest, ypred)`: Evaluates the model's performance.
`run_model()`: Runs the Logistic Regression model.

### SVM
This class implements the Support Vector Machine (SVM) model.

`__init__(self, c, sig, x, y, xtest, tol, categorical_columns)`: Initializes the SVM class.
`one_hot_encoding(x)`: One-hot-encodes categorical columns.
`training_data_generator(X, Y, samplesize)`: Generates training data.
`gaussian_kernel(x, y)`: Calculates the Gaussian kernel.
`F(x)`: Calculates the decision function.
`train(x, y, max_passes)`: Trains the SVM model.
