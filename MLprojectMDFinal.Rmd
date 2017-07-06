---
title: "Machine Learning Project"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```


## Machine Learning Analysis of Body Movements

This project involves analysing the body movements of people performing exercises in different styles, as quantified by gyroscope readings from smartphones. The goal is to build a predictive model of the classe (movment style, of which there are 5 labelled A-E) based on the numerical data provided in 159 features.

First, load libraries and import data.

```{r}
## Load required libraries
library(lubridate)
library(randomForest)
library(stats)
library(caret)

training <- read.csv("pml-training.csv", stringsAsFactors = FALSE)
testing <- read.csv("pml-testing.csv", stringsAsFactors = FALSE)
```


## Data Processing

The goal is to predict classe (the manner in which an exercise was performed) based on variables in training. First examine the relative occurrence of each classe. 

``` {r}
table(training$classe)
```

We can see there is fairly even distribution of classes represented (A is a little high) but if desired/necessary we ccould possibly correct for this with some sampling.

For the purpose of cross validation prior to prediction, split the data into training and validation sets.

```{r}
set.seed(123)
inTrain <- createDataPartition(y=training$classe, p=0.7, list = FALSE)
trainset <- training[inTrain,]
validset <- training[-inTrain,]


## Check that the classes are similarly represented in both sets
table(trainset$classe)/sum(table(trainset$classe))
table(validset$classe)/sum(table(validset$classe))
```

Thus, the classes are represented similarly in the training and validation sets.

The data frame contains two types of record, given by the value of new_window. If new_window = yes, a large set of averages are recorded, but this is relatively rare amongst the data set. Thus, we copy all the new_window data to a new dataframe (for analysis at another time if required). 

```{r}
##Split data into summary and rename training set as rawdata to clarify
trainsummary <- trainset[trainset$new_window == "yes",]
validsummary <- validset[validset$new_window == "yes",]
```

We now clean up the data by first eliminate NA-rich columns and and empty fields values (these features contain the averages, which are too sparse to be useful for creating a model).

```{r}
## Eliminate columns with any NA values
NAvals <- sapply(trainset, function(x) sum(is.na(x)))
trainset <- trainset[, NAvals == 0]
validset <- validset[, NAvals == 0]
testing <- testing[, NAvals == 0]

# Clean up data - identify features as factor and time if necessary
trainset$user_name <- as.factor(trainset$user_name)
trainset$classe <- as.factor(trainset$classe)
trainset$cvtd_timestamp <- dmy_hm(trainset$cvtd_timestamp)

validset$user_name <- as.factor(validset$user_name)
validset$classe <- as.factor(validset$classe)
validset$cvtd_timestamp <- dmy_hm(validset$cvtd_timestamp)

testing$user_name <- as.factor(testing$user_name)
testing$cvtd_timestamp <- dmy_hm(testing$cvtd_timestamp)

# Remove character variables which are all blank in the raw data (plus "new_window")
elim <- sapply(trainset, function(x) is.character(x))
trainset[,elim] <- list(NULL)
validset[,elim] <- list(NULL)
testing[,elim] <- list(NULL)
```

These steps leave 59 variables, a mix of POSIXct/num/int/factor. We can now use this cleaned up dataset to build a machine learning predictor of classe.


## Building a model

For a binary classification problem, it is often effection to try a random forest. I will do this using all variables except the timestamps (which are complex) and the identidy of the operator (this may not be a useful predictor in future). The choice of random forest was made as it has a relatively strong classification perfomance with a short training time.

```{r cache=TRUE}
RFmod <- randomForest(classe ~., data = trainset[6:59], importance = TRUE)

## Use model to make predictions of training and test sets
## In sample and out of sample errors
RFconf <- predict(RFmod, trainset[,6:58])
RFvalid <- predict(RFmod, validset[,6:58])
RFpred <- predict(RFmod, testing[,6:58])
```

We can now evaluation the models performance against both the training and validation sets.

```{r}
table(RFconf, trainset$classe)


table(RFvalid, validset$classe)
```

We see  that the model is able to fit the training set with 100 % accuracy, whilst in the validation set we have some errors - 18 misidentified points out of 5885. This is equivalent to error rate of 0.3% (i.e. an accuracy of 99.7%). This means we would expect to be able to correctly predict 20 instances with a probability of (1 - (18/5885))^20 = 0.941, i.e. we expect on average to get 19/20 correct.

This isn't a bad model but we may be able to improve things. One option (previously erroneously employed by the author) to improve accuracy would be to include more data when generating the model - however this places a limitation on our ability to validate the model prior to making predictions. 

A common cause of errors is due to overfitting of noise in the training set. Thus we can use cross validation to determine the minimum set of parameters necessary to make good predictions and avoid fitting too much noise. To do this we employ the rfcv function (which unfortunately is super slow). This function performs n-fold cross validation (here we chose 5-fold) whilst removing variables from the model in order of their importance.

```{r cross_validation, cache=TRUE, eval= FALSE}

## Cross validation - for feature selection
set.seed(42)
xdata <-  trainset[, 6:58]
ydata <-  trainset[, 59]

## This object contains the error rates and predictions when varying
## the number a parameters (ranked by variable importance)
rfcvTrain <- rfcv(xdata, ydata, cv.fold=5)

results = data.frame(rfcvTrain$n.var, rfcvTrain$error.cv)

results

with(rfcvTrain, plot(n.var, error.cv))
```

NB Unfortunately the rfcv function was generating errors through knitr but below I inlcude the plot that was generated above when running through the console:

```{r errorplot}
library(png)
img = readPNG("Cross validation with rfcv on trainset.png")

h<-dim(img)[1]
w<-dim(img)[2]

par(mar=c(0,0,0,0), xpd=NA, mgp=c(0,0,0), oma=c(0,0,0,0), ann=F)
plot.new()
plot.window(0:1, 0:1)

usr<-par("usr")    
rasterImage(img, usr[1], usr[3], usr[2], usr[4])
```

Thus, the error rate reduces slightly down to 13 variables before increasing as we approach fewer and fewer variables.


We extract a list of the relative importance of each variable (plot decrease in Gini):

```{r}
imp <- importance(RFmod, scale = FALSE)

## Sort by Mean Decrease in Gini
sortimp <- sort(imp[,4], decreasing = TRUE)

par(cex = 0.5)
dotchart(sort(imp[,4]), xlab = "Mean Decrease in GINI")

```

We can now try building the model again with the 13 top variables:

```{r}
toptraindata <- cbind(trainset[,names(sortimp[1:13])], classe = trainset$classe)
topvaliddata <- cbind(validset[,names(sortimp[1:13])], classe = validset$classe)
# testingdata <- cbind(testing[,names(sortimp[1:13])], classe = testing$classe)

## Rebuild models and predictions using this new model
RFmodTop <- randomForest(classe ~., data = toptraindata, importance = TRUE)
RFconfTop <- predict(RFmodTop, toptraindata[,-ncol(toptraindata)])
RFvalidTop <- predict(RFmodTop, topvaliddata[, -ncol(topvaliddata)])
# RFpredTop <- predict(RFmodTop, testingdata[, -ncol(testingdata)])

table(RFconfTop, toptraindata$classe)
## Still models training data with 100% accuracy

table(RFvalidTop, topvaliddata$classe)
```

This new model predicts with an error rate of 9/5885 (99.8% accurate). We now anticipate the probability of getting 20/20 correct to be (1 - (9/5885))^20 = 0.970, which is slightly better and should be sufficient for our purposes. We might find futher improvements in performance by more accurately tuning which parameters are included and cross validating with different samples of the full training dataset. Other ways to fit the data include multinomial logistic regression which may give a slightly more intuitive interpretation of how variables map to the outcome (random forest models are hard to interpret).

