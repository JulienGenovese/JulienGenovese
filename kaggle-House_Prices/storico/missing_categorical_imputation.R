library(tidyverse)
library(caret)
library(skimr)
library(fastDummies)
library(EnsCat)
library(proxy)
library(class)
library(FastKNN)
cat("\014")
rm(list = ls())

trainset<-read.csv2("./input/train.csv", sep =",")
testset<-read.csv2("./input/test.csv", sep =",")

target <- trainset %>% pull(LotFrontage)
attributes <- trainset[,c(1:6)]
k_neighbors = 5

#possible_aggregation_method = list("mean", "median", "mode")
#number_observations = length(target)
#is_target_numeric = sum(apply(as.matrix(target), FUN = is.numeric, MARGIN = 1)) > 0

#nrow(attributes) != number_observations

# Get the distance matrix and check whether no error was triggered when computing it
#distances = distance_matrix(attributes, "euclidean", "hamming")

# Create the function.
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

weighted_hamming <- function(data) {
  " Compute weighted hamming distance on categorical variables. For one variable, it is equal to 1 if
        the values between point A and point B are different, else it is equal the relative frequency of the
        distribution of the value across the variable. For multiple variables, the harmonic mean is computed
        up to a constant factor.
        @params:
            - data = a data frame of categorical variables
        @returns:
            - distance_matrix = a distance matrix with pairwise distance for all attributes"
  
  #TO BE IMPLEMENTED
}


distance_matrix <- function(dataset_temp, numeric_distance = "euclidean", categorical_distance = "jaccard") {
  "Compute the pairwise distance attribute by attribute in order to account for different variables type:
        - Continuous
        - Categorical
        For ordinal values, provide a numerical representation taking the order into account.
        Categorical variables are transformed into a set of binary ones.
        If both continuous and categorical distance are provided, a Gower-like distance is computed and the numeric
        variables are all normalized in the process.
        If there are missing values, the mean is computed for numerical attributes and the mode for categorical ones.
        
        params:
            - data                  = R dataframe to compute distances on.
            - numeric_distances     = the metric to apply to continuous attributes.
                                      euclidean and cityblock available.
                                      Default = euclidean
            - categorical_distances = the metric to apply to binary attributes.
                                      jaccard, hamming, weighted-hamming and euclidean
                                      available. Default = jaccard
        returns:
            - the distance matrix
    "
possible_continuous_distances = list("euclidean", "maximum", "manhattan", "canberra", "minkowski")
possible_binary_distances = list("euclidean", "jaccard", "hamming", "weighted-hamming")
number_of_variables = ncol(dataset_temp)
number_of_observations = nrow(dataset_temp)

# Get the type of each attribute (Numeric or categorical)
cat <- sapply(dataset_temp, is.factor) #Select categorical variables
is_all_categorical = sum(cat) == number_of_variables
is_all_numerical = sum(cat) == 0
is_mixed_type = !is_all_categorical & !is_all_numerical

# Check the content of the distances parameter
if ((numeric_distance %in% possible_continuous_distances)==FALSE)
  {
  print(paste(paste("The continuous distance",numeric_distance,sep = " "),"is not supported.",sep= " ")) 
  return(NULL)
  }
else if ((categorical_distance %in% possible_binary_distances)==FALSE)
  {
  print(paste(paste("The binary distance",categorical_distance,sep = " "),"is not supported.",sep= " ")) 
  return(NULL)
  }

# Separate the data frame into categorical and numeric attributes and normalize numeric data
if (is_mixed_type){
  number_of_categorical_var = sum(cat)
  number_of_numerical_var = number_of_variables - number_of_categorical_var
  data_numeric <- dataset_temp[,!cat]
  data_numeric <- scale(data_numeric)
  data_categorical <- dataset_temp[,cat]
}


# "Dummifies" categorical variables in place
if (!is_all_numerical & !(categorical_distance == 'hamming' | categorical_distance == 'weighted-hamming'))
  {
  if (is_mixed_type)
    {
    data_categorical <- fastDummies::dummy_cols(data_categorical,remove_selected_columns=TRUE)
    } 
  else
    {
      dataset_temp <- fastDummies::dummy_cols(dataset_temp,remove_selected_columns=TRUE)
    }
  }
else if (!is_all_numerical & categorical_distance == 'hamming')
  {
  if (is_mixed_type)
    {
    col_names <- names(data_categorical)
    data_categorical[col_names] <- lapply(data_categorical[col_names] , factor)
    indx <- sapply(data_categorical, is.factor)
    data_categorical[indx] <- lapply(data_categorical[indx], function(x) as.numeric(x))
    }
  else
    {
    col_names <- names(dataset_temp)
    dataset_temp[col_names] <- lapply(dataset_temp[col_names] , factor)
    indx <- sapply(dataset_temp, is.factor)
    dataset_temp[indx] <- lapply(dataset_temp[indx], function(x) as.numeric(x))
    }
  }

if (is_all_numerical)
  {
  result_matrix = as.matrix(dist(dataset_temp, method = numeric_distance, p=3))
  }
else if (is_all_categorical)
  {
  if (categorical_distance == "weighted-hamming")
    {  
    result_matrix = weighted_hamming(data)
    }
  else if (categorical_distance == "hamming")
    {
    n <- nrow(dataset_temp)
    result_matrix <- matrix(nrow=n, ncol=n)
    for(i in seq_len(n - 1))
      for(j in seq(i, n))
        result_matrix[j, i] <- result_matrix[i, j] <- sum(dataset_temp[i,] != dataset_temp[j,])
    }
  else
    {
    result_matrix = as.matrix(dist(dataset_temp, method = categorical_distance))
    }
  }
else
  {
  result_numeric = as.matrix(dist(data_numeric, method = numeric_distance,p=3))

  if (categorical_distance == "weighted-hamming")
    {
    result_categorical = weighted_hamming(data_categorical)
    }
  else if (categorical_distance == "hamming")
    {
    n <- nrow(dataset_temp)
    result_categorical <- matrix(nrow=n, ncol=n)
    for(i in seq_len(n - 1))
      for(j in seq(i, n))
        result_categorical[j, i] <- result_categorical[i, j] <- sum(data_categorical[i,] != data_categorical[j,])
    }
  else
    {
    result_categorical = as.matrix(dist(data_categorical, method = categorical_distance))
    }
  result_matrix = result_numeric * number_of_numerical_var + result_categorical * number_of_categorical_var
  }
# Fill the diagonal with NaN values
diag(result_matrix) <- NaN

return (result_matrix)
}

knn_impute <- function(target, attributes, k_neighbors, aggregation_method="mean", numeric_distance="euclidean",
               categorical_distance="jaccard", missing_neighbors_threshold = 0.5) 
{
  "Replace the missing values within the target variable based on its k nearest neighbors identified with the
        attributes variables. If more than 50% of its neighbors are also missing values, the value is not modified and
        remains missing. If there is a problem in the parameters provided, returns None.
        If to many neighbors also have missing values, leave the missing value of interest unchanged.
        params:
            - target                        = a vector of n values with missing values that you want to impute. The length has
                                              to be at least n = 3.
            - attributes                    = a data frame of attributes with n rows to match the target variable
            - k_neighbors                   = the number of neighbors to look at to impute the missing values. It has to be a
                                              value between 1 and n.
            - aggregation_method            = how to aggregate the values from the nearest neighbors (mean, median, mode)
                                              Default = mean
            - numeric_distances             = the metric to apply to continuous attributes.
                                              euclidean and cityblock available.
                                              Default = euclidean
            - categorical_distances         = the metric to apply to binary attributes.
                                              jaccard, hamming, weighted-hamming and euclidean
                                              available. Default = jaccard
            - missing_neighbors_threshold   = minimum of neighbors among the k ones that are not also missing to infer
                                              the correct value. Default = 0.5
        returns:
            target_completed        = the vector of target values with missing value replaced. If there is a problem
                                      in the parameters, return None"

# Get useful variables
possible_aggregation_method = list("mean", "median", "mode")
number_observations = length(target)
is_target_numeric = sum(apply(as.matrix(target), FUN = is.numeric, MARGIN = 1)) > 0
# Check for possible errors
if (number_observations < 3)
  {
  print("Not enough observations.")
  return(NULL)
  }
if (nrow(attributes) != number_observations)
  {
  print("The number of observations in the attributes variable is not matching the target variable length.")
  return(NULL)
  }
if ((k_neighbors > number_observations) | (k_neighbors < 1))
  {
  print("The range of the number of neighbors is incorrect.")
  return(NULL)
  }
if ((aggregation_method %in% possible_aggregation_method)==FALSE)
  {
  print("The aggregation method is incorrect.")
  return(NULL)
  }
if (!is_target_numeric & aggregation_method != "mode")
  {
  print("The only method allowed for categorical target variable is the mode.")
  return(NULL)
  }

# Get the distance matrix and check whether no error was triggered when computing it
distances = distance_matrix(attributes, numeric_distance, categorical_distance)
if (is.null(distances))
{
  print("null distance")
  return(NULL)
}

n = ncol(distances)
for (i in 1:length(target))
{
  value = target[i]
  if (is.na(value))
  {
    # matrix of neighbours
    closest_to_target = k.nearest.neighbors(i, distances, k = k_neighbors)
    neighbors = target[closest_to_target]
    count=0
    for (p in 1:length(neighbors))
    {
      miss = neighbors[p]
      if (is.na(miss))
      {
        count = count+1
      }
    }
    # Compute the right aggregation method if at least more than 50% of the closest neighbors are not missing
    if (count >= missing_neighbors_threshold * k_neighbors)
    {
      next
    }
    else if (aggregation_method == "mean")
    {
      target[i] = mean(na.omit(neighbors))
    }
    else if (aggregation_method == "median")
    {
      target[i] = median(na.omit(neighbors))
    }
    else
    {
      target[i] = getmode(na.omit(neighbors))
    }
  }
}
return(target)
}

tmp = knn_impute(target, attributes, k_neighbors, aggregation_method="mean", numeric_distance="maximum",
           categorical_distance="hamming", missing_neighbors_threshold = 0.5) 
