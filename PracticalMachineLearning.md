# Practical Machine Learning
Chris Benson  
Wednesday, September 23, 2015  

#Introduction
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, our goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

For this project, we are given data from accelerometers on the belt, forearm, arm, and dumbell of 6 research study participants. Our training data consists of accelerometer data and a label identifying the quality of the activity the participant was doing. Our testing data consists of accelerometer data without the identifying label. Our goal is to predict the labels for the test set observations.The goal of this project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. 

Below is the code for cleaning the data (some basic exploratory data anlysis had taken place before hand), creating the model, perfroming cross validation and estimating the out-of-sample error, and making the predictions on the test data provided by the project.

#Data Preparation
First we load the required packages load and read in the source (raw) training and testing data:


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
src_train <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!",""))
src_test <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!",""))
```

We will reduce the number of features by removing variables with nearly zero variance, variables that are close to NA, and variables that don't make sense as a predictor (not necessarily in that order). 

```r
#Remove column variables that don't make sense as a predictor.The first five variables
src_train <- src_train[, -(1:5)]
src_test <- src_test[, -(1:5)]


#Remove variables that are are close to NA
colNA <- sapply(src_train, function(x) mean(is.na(x))) > 0.90
src_train <- src_train[, colNA==F]
src_test <- src_test[, colNA==F]

#Remove near zero value variables
nzv <- nearZeroVar(src_train)
src_train <- src_train[, -nzv]
src_test <- src_test[, -nzv]
```


Because we want to be able to estimate the out-of-sample error, we randomly split the full training data (src_train) into a smaller training set (train_set 75%) and create a validation set from the remaining data (test_set 25%):


```r
set.seed(1234)
training_data <- createDataPartition(y=src_train$classe, p=0.75, list=F)
train_set <- src_train[training_data, ]
test_set <- src_train[-training_data, ]
```


#Model Building and Cross validation
We initially choose a Random Forest model, to see if it would have acceptable performance. Anything with an accuracy greater than 97% would be good. Then fit the model on train_set, and instruct the "train" function to train the model. We then create a cross validation set and check the out of sample error rate hopefully lower than 3%. We then run a confusionMatrix to check the accuracy   


```r
#Fit model on train_set
model_fit <- randomForest(classe ~. , data=train_set)

#Use the 25% test_set to test the predictions
prediction_validation <- predict(model_fit, test_set, type = "class")

#Create a cross validations based on the predictions and the out of sample error
cross_validation <- table(prediction_validation, test_set$classe)
out_of_sample_error <- 1 - (sum(diag(cross_validation))/ length(prediction_validation))
cross_validation
```

```
##                      
## prediction_validation    A    B    C    D    E
##                     A 1395    1    0    0    0
##                     B    0  948    5    0    0
##                     C    0    0  850    5    0
##                     D    0    0    0  799    0
##                     E    0    0    0    0  901
```

```r
out_of_sample_error
```

```
## [1] 0.002243067
```

```r
confusionMatrix(test_set$classe, prediction_validation)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    0    0    0    0
##          B    1  948    0    0    0
##          C    0    5  850    0    0
##          D    0    0    5  799    0
##          E    0    0    0    0  901
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9978         
##                  95% CI : (0.996, 0.9989)
##     No Information Rate : 0.2847         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9972         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9993   0.9948   0.9942   1.0000   1.0000
## Specificity            1.0000   0.9997   0.9988   0.9988   1.0000
## Pos Pred Value         1.0000   0.9989   0.9942   0.9938   1.0000
## Neg Pred Value         0.9997   0.9987   0.9988   1.0000   1.0000
## Prevalence             0.2847   0.1943   0.1743   0.1629   0.1837
## Detection Rate         0.2845   0.1933   0.1733   0.1629   0.1837
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9996   0.9973   0.9965   0.9994   1.0000
```

We used the fitted model to predict the label ("classe") in train_set using the validatin test data (test_set), and show the confusion matrix to compare the predicted versus the actual labels.

As can be seen from the confusion matrix this model is very accurate. Experimenting with PCA and other models we did not see the accuracy that Random Forests provided. Because the test data (validation data) was around 99.78% accurate, we expected nearly all of the submitted test cases to be correct. It turned out they were all correct. The out of sample error rate 0.2243067% which is exaclty 100% minus the accuracy. This is a very good model. 

#Making Test Set Predictions
Now, use the model fit on the actual Test data provided by the source to predict the label for the observations, and write those predictions to individual files:


```r
predict_test <- predict(model_fit, src_test, type = "class")

# create function to write predictions to files
pml_write_files <- function(x) {
  n <- length(x)
  for(i in 1:n) {
    filename <- paste0("Test\\problem_id_", i, ".txt")
    write.table(x[i], file=filename, quote=F, row.names=F, col.names=F)
  }
}

# create prediction files to submit
pml_write_files(predict_test)
```
