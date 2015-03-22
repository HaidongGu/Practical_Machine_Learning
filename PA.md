# Practical Machine Learning: Peer Assessment
Haidong Gu  
22, March, 2014  

## Requirement
please see [Coursera Website](https://class.coursera.org/predmachlearn-012/human_grading/view/courses/973547/assessments/4/submissions) for details.

## Loading and Processing

load librarires

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
set.seed(8880)
```

set working directory, it will be different for each person

```r
setwd("/Users/hgu/Downloads/")
paste("working dir is", getwd())
```

```
## [1] "working dir is /Users/hgu/Downloads"
```

 1. Load the data


```r
if (!file.exists("./pml-training.csv")) {
    download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
        destfile = "./pml-training.csv")
}
if (!file.exists("./pml-testing.csv")) {
    download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
        destfile = "./pml-testing.csv")
}

data_training <- read.csv("./pml-training.csv")
data_testing <- read.csv("./pml-testing.csv")
```

 2. Process/transform the data

remove columns with all missing values or without keywords from predication.

```r
missing_value_col <- sapply(data_testing, function (x) any(is.na(x) | x == ""))
predictors <- names(missing_value_col)[!missing_value_col]
predictors <- predictors[grepl("belt|forearm|arm|dumbbell", predictors)]

data_training <- data_training[c("classe", predictors)]

dim(data_training)
```

```
## [1] 19622    53
```

```r
names(data_training)
```

```
##  [1] "classe"               "roll_belt"            "pitch_belt"          
##  [4] "yaw_belt"             "total_accel_belt"     "gyros_belt_x"        
##  [7] "gyros_belt_y"         "gyros_belt_z"         "accel_belt_x"        
## [10] "accel_belt_y"         "accel_belt_z"         "magnet_belt_x"       
## [13] "magnet_belt_y"        "magnet_belt_z"        "roll_arm"            
## [16] "pitch_arm"            "yaw_arm"              "total_accel_arm"     
## [19] "gyros_arm_x"          "gyros_arm_y"          "gyros_arm_z"         
## [22] "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
## [25] "magnet_arm_x"         "magnet_arm_y"         "magnet_arm_z"        
## [28] "roll_dumbbell"        "pitch_dumbbell"       "yaw_dumbbell"        
## [31] "total_accel_dumbbell" "gyros_dumbbell_x"     "gyros_dumbbell_y"    
## [34] "gyros_dumbbell_z"     "accel_dumbbell_x"     "accel_dumbbell_y"    
## [37] "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"   
## [40] "magnet_dumbbell_z"    "roll_forearm"         "pitch_forearm"       
## [43] "yaw_forearm"          "total_accel_forearm"  "gyros_forearm_x"     
## [46] "gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"     
## [49] "accel_forearm_y"      "accel_forearm_z"      "magnet_forearm_x"    
## [52] "magnet_forearm_y"     "magnet_forearm_z"
```

 3. partition the training data with 60% training, 40% probing.


```r
partition <- createDataPartition(data_training$classe, p=0.6, list=FALSE)
data_probing <- data_training[-partition, ]
data_training <- data_training[partition, ]
```

## Modeling

building model with Random Forest ("rf") machine leaning techique. __It will be a good model  if the error is less than 2.5%.__


```r
model <- train(classe ~ ., data = data_training, method = "rf", prox = TRUE, 
               trControl = trainControl(method = "cv", number = 4, allowParallel = TRUE))
```

### Evaluate the model on the training data


```r
training_result <- predict(model, data_training)
confusionMatrix(training_result, data_training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3348    0    0    0    0
##          B    0 2279    0    0    0
##          C    0    0 2054    0    0
##          D    0    0    0 1930    0
##          E    0    0    0    0 2165
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

### Evaluate the model on the probing data


```r
probing_result <- predict(model, data_probing)
confusionMatrix(probing_result, data_probing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2222    9    0    0    0
##          B    7 1507   16    0    1
##          C    3    2 1343   12    5
##          D    0    0    9 1273    5
##          E    0    0    0    1 1431
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9911         
##                  95% CI : (0.9887, 0.993)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9887         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9955   0.9928   0.9817   0.9899   0.9924
## Specificity            0.9984   0.9962   0.9966   0.9979   0.9998
## Pos Pred Value         0.9960   0.9843   0.9839   0.9891   0.9993
## Neg Pred Value         0.9982   0.9983   0.9961   0.9980   0.9983
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2832   0.1921   0.1712   0.1622   0.1824
## Detection Prevalence   0.2843   0.1951   0.1740   0.1640   0.1825
## Balanced Accuracy      0.9970   0.9945   0.9892   0.9939   0.9961
```

### Final Mode

```r
varImp(model)
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 52)
## 
##                      Overall
## roll_belt             100.00
## pitch_forearm          59.62
## yaw_belt               53.57
## magnet_dumbbell_y      44.02
## magnet_dumbbell_z      43.26
## pitch_belt             43.22
## roll_forearm           42.89
## accel_dumbbell_y       23.58
## accel_forearm_x        16.92
## roll_dumbbell          16.56
## magnet_dumbbell_x      15.94
## magnet_forearm_z       15.47
## magnet_belt_z          14.05
## total_accel_dumbbell   13.56
## accel_dumbbell_z       13.41
## accel_belt_z           13.05
## gyros_belt_z           11.54
## yaw_arm                10.93
## magnet_belt_y          10.57
## magnet_belt_x          10.03
```

```r
model$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, proximity = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.88%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3341    4    2    0    1 0.002090800
## B   23 2247    8    1    0 0.014041246
## C    0   16 2029    9    0 0.012171373
## D    0    1   23 1901    5 0.015025907
## E    0    1    2    8 2154 0.005080831
```

__OOB estimate of error rate is 0.88%.__

## Apply model to testing data and Submit

```r
testing_result <- predict(model, data_testing)
```

prepare for submission

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=file.path("/Users/hgu/Downloads/answers2", filename),quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(testing_result)
```

## Summary
* I use _Random Forest_ to to build our model.
* I was expecting error rate to below 2.5%, and my model achieves 0.88%.
