## Practical Machine Learning
## Course Project
## denis.maurin@gmail.com

library(Hmisc)

## Load the datasets
trainingC0 <- read.csv("pml-training.csv", na.strings=c("", "NA", "NULL","#DIV/0"))
testingC0 <- read.csv("pml-testing.csv", na.strings=c("", "NA", "NULL","#DIV/0"))

## Have a look at the data
describe(trainingC0)

## Cleanup the dataset by removing useless columns or columns with too many NA
## First remove the columns which have no physical meaning and therefore should be excluded from the model
## typically all timestamps, window information and username
trainingC1 <- subset(trainingC0, select=-c(X, user_name, new_window, num_window, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp))

## Then remove columns with NA - we reach 153 columns
trainingC2 <- trainingC1[, colSums(is.na(trainingC1))==0]

## They all look nice 53 columns, all numeric except the classe column obviously
head(trainingC2,5)
dim(trainingC2)

## Identify if there are near zero variance variables in the remaining set
nzv <- nearZeroVar(trainingC2, saveMetrics=TRUE)
nzv
## We can see that none of them fall into that category so we are keeping all predictors 

## Check if some variables are highly correlated
corMat <- cor(na.omit(trainingC2[sapply(trainingC2, is.numeric)]))
cor_var <- findCorrelation(corMat, cutoff = .90, verbose = FALSE)
trainingC3 <- trainingC2[,-cor_var] ## Remove the useless ones
names(trainingC2[c(cor_var)]) ## 7 variables being removed 

# Before we move forward, let's replicate the same filtering/cleaning functions to the test file for later
testingC1 <- subset(testingC0, select=-c(X, user_name, new_window, num_window, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp))
testingC2 <- testingC1[, colSums(is.na(trainingC1))==0] ## using the columns excluded in the training set
testingC3 <- testingC2[,-cor_var] ## removing the highly correlated values

## Now let's split the set into a training set and a test set
## This is addressing cross validation
inTrain <- createDataPartition(y=trainingC3$classe, p=0.75, list=FALSE)
trainS <- trainingC3[inTrain,]
testS <- trainingC3[-inTrain,]

## Now let's try different techniques and check which one is the best

## Regression trees
## ================
modFitTREE <- rpart(classe ~ ., data=trainS, method="class")
print(modFitTREE)

## Plot
plot(modFitTREE, uniform=TRUE, main="Classification Tree")
text(modFitTREE, use.n=TRUE, all=TRUE, cex=.8)

## Predict
predictTREE <- predict(modFitTREE, newdata=testS, type="class")

## Result
confusionMatrix(predictTREE, testS$classe) ## Accuracy around 73%

## Random forest
## ==============
modFitRF <- randomForest(classe ~ ., data=trainS, method="class")
print(modFitRF$finalModel)
predictRF <- predict(modFitRF, newdata=testS, type="class")
confusionMatrix(predictRF, testS$classe) ## Accuracy around 99.65%
varImp(modFitRF)
varImpPlot(modFitRF)


# Now apply to the provided test set for grading using random forests, our best algorithm
predictTEST <- predict(modFitRF, newdata=testingC3, type="class")




