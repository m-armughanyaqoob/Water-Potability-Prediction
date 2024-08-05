library(ggplot2)
library(mice)
library(missForest)
library(randomForest)
library(corrplot)
library(caret)
library(moments)
library(rpart)
library(rpart.plot)
library(pROC)
library(ROCR)

# Loading the dataset and checking the dimensions and general summary
water.quality <- read.csv('C:/Users/ashis/Downloads/water_potability.csv')
head(water.quality)
summary(water.quality)
nrow(water.quality)
dim(water.quality)

water.quality_na_dropped <- na.omit(water.quality)

# Printing out all the NAN counts
sum(apply(water.quality, 1, anyNA))
sum(is.na(water.quality$ph))
sum(is.na(water.quality$Hardness))
sum(is.na(water.quality$Solids))
sum(is.na(water.quality$Chloramines))
sum(is.na(water.quality$Sulfate))
sum(is.na(water.quality$Conductivity))
sum(is.na(water.quality$Organic_carbon))
sum(is.na(water.quality$Trihalomethanes))
sum(is.na(water.quality$Turbidity))
sum(is.na(water.quality$Potability))

# Pair plots for the dataset
pairs(water.quality)

# Creating a plot of missing value rows
md.pattern(water.quality, rotate.names = TRUE)

# Imputing missing data using missForest
water.quality_original <- water.quality

Random.seed <- c("Mersenne-Twister", 530)
set.seed(530)
imputation_result <- missForest(water.quality, xtrue = water.quality_original)
water.quality <- imputation_result$ximp

# Viewing the random forest imputation error
imputation_result$OOBerror

# Verifying that the imputation has left no NAN values
md.pattern(water.quality, rotate.names = TRUE)

# ==========================
# Exploratory Data Analysis
# ==========================

ggplot(water.quality) + geom_bar(aes(x = factor(Potability)), fill='lightblue', color='black', stat='count') +
  geom_text(stat='count',aes(x = factor(Potability), label=..count..), vjust=2) +
  xlab('Potable') + ylab('Count')


print("Non-Potable")
summary(subset(water.quality, Potability == 0))
print("Potable")
summary(subset(water.quality, Potability == 1))


correlations <- cor(water.quality)
corrplot(correlations, method="number", col=colorRampPalette(c("blue","red"))(20))


features <- c("ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity")
for (feature in features) {
  p <- ggplot(water.quality, aes_string(x=feature, fill="factor(Potability)")) + 
    geom_density(alpha = 0.3) +
    labs(title = paste("KDE of", feature, "based on Potability"), x = feature) +
    scale_fill_discrete(name = "Potability", labels = c("Not Potable", "Potable"))
  print(p)
}


skewness_values <- sapply(water.quality[, 1:9], skewness)
print(skewness_values)
kurtosis_values <- sapply(water.quality[, 1:9], kurtosis)
print(kurtosis_values)

# =========================
# Logistic Regression Model (50/50 Split)
# =========================

Random.seed <- c("Mersenne-Twister", 530)
set.seed(530)
indices <- sample(1:nrow(water.quality), 0.5 * nrow(water.quality), replace = FALSE)
train_set <- water.quality[indices, ]
test_set <- water.quality[-indices, ]

logit.train <- glm(Potability ~ ., data=train_set, family = "binomial")
summary(logit.train)
logLik(logit.train)
with(logit.train, pchisq(null.deviance - deviance, df.null - df.residual, lower.tail = FALSE))

# Performing predictions and displaying the confusion table and performance parmeters
predictions <- predict(logit.train, test_set, type="response")
predictions.binary <- ifelse(predictions > 0.5, 1, 0)
cftable <- table(predictions.binary, test_set$Potability)

accuracy <- sum(diag(cftable))/sum(cftable)
sensitivity<-cftable[1]/(cftable[1] + cftable[2])
specificity <- cftable[4]/(cftable[3] + cftable[4])
ppv <- cftable[1]/(cftable[1] + cftable[3])
npv <- cftable[4]/(cftable[2] + cftable[4])
print(cftable)
sprintf("accuracy = %s", accuracy)
sprintf("sensitivity = %s", sensitivity)
sprintf("specificity = %s", specificity)
sprintf("ppv = %s", ppv)
sprintf("npv = %s", npv)

# Same for the training set
predictions <- predict(logit.train, train_set, type="response")
predictions.binary <- ifelse(predictions > 0.5, 1, 0)
cftable <- table(predictions.binary, train_set$Potability)
accuracy <- sum(diag(cftable))/sum(cftable)
sensitivity<-cftable[1]/(cftable[1] + cftable[2])
specificity <- cftable[4]/(cftable[3] + cftable[4])
ppv <- cftable[1]/(cftable[1] + cftable[3])
npv <- cftable[4]/(cftable[2] + cftable[4])
print(cftable)
sprintf("accuracy = %s", accuracy)
sprintf("sensitivity = %s", sensitivity)
sprintf("specificity = %s", specificity)
sprintf("ppv = %s", ppv)
sprintf("npv = %s", npv)

# AUC/ROC Curves for the train and test sets respectively
predictions <- predict(logit.train, train_set, type="response")
train.roc <- roc(train_set$Potability, predictions)
plot.roc(train.roc, main="Train ROC", col=par("fg"), plot=TRUE, print.auc=FALSE, legacy.axes=FALSE, asp =NA)
plot.roc(smooth(train.roc), col="blue", add=TRUE, plot=TRUE, print.auc=TRUE, legacy.axes=TRUE, asp=NA)
legend("bottomright", legend=c("Empirical", "Smoothed"),
       col=c(par("fg"), "blue"), lwd=2)
train.roc

predictions <- predict(logit.train, test_set, type="response")
test.roc <- roc(test_set$Potability, predictions)
plot.roc(test.roc, main="Test ROC", col=par("fg"), plot=TRUE, print.auc=FALSE, legacy.axes=FALSE, asp =NA)
plot.roc(smooth(test.roc), col="blue", add=TRUE, plot=TRUE, print.auc=TRUE, legacy.axes=TRUE, asp=NA)
legend("bottomright", legend=c("Empirical", "Smoothed"),
       col=c(par("fg"), "blue"), lwd=2)
test.roc

# =========================
# Logistic Regression Model (80/20 Split)
# =========================

set.seed(530)
indices <- sample(1:nrow(water.quality), 0.8 * nrow(water.quality), replace = FALSE)
train_set <- water.quality[indices, ]
test_set <- water.quality[-indices, ]

logit.train <- glm(Potability ~ ., data=train_set, family = "binomial")
summary(logit.train)

predictions <- predict(logit.train, test_set, type="response")
predictions.binary <- ifelse(predictions > 0.5, 1, 0)
cftable <- table(predictions.binary, test_set$Potability)

accuracy <- sum(diag(cftable))/sum(cftable)
sensitivity<-cftable[1]/(cftable[1] + cftable[2])
specificity <- cftable[4]/(cftable[3] + cftable[4])
ppv <- cftable[1]/(cftable[1] + cftable[3])
npv <- cftable[4]/(cftable[2] + cftable[4])
print(cftable)
sprintf("accuracy = %s", accuracy)
sprintf("sensitivity = %s", sensitivity)
sprintf("specificity = %s", specificity)
sprintf("ppv = %s", ppv)
sprintf("npv = %s", npv)

predictions <- predict(logit.train, train_set, type="response")
predictions.binary <- ifelse(predictions > 0.5, 1, 0)
cftable <- table(predictions.binary, train_set$Potability)
accuracy <- sum(diag(cftable))/sum(cftable)
sensitivity<-cftable[1]/(cftable[1] + cftable[2])
specificity <- cftable[4]/(cftable[3] + cftable[4])
ppv <- cftable[1]/(cftable[1] + cftable[3])
npv <- cftable[4]/(cftable[2] + cftable[4])
print(cftable)
sprintf("accuracy = %s", accuracy)
sprintf("sensitivity = %s", sensitivity)
sprintf("specificity = %s", specificity)
sprintf("ppv = %s", ppv)
sprintf("npv = %s", npv)

# AUC/ROC Curves for training and testing sets respectively
predictions <- predict(logit.train, train_set, type="response")
train.roc <- roc(train_set$Potability, predictions)
plot.roc(train.roc, main="Train ROC", col=par("fg"), plot=TRUE, print.auc=FALSE, legacy.axes=FALSE, asp =NA)
plot.roc(smooth(train.roc), col="blue", add=TRUE, plot=TRUE, print.auc=TRUE, legacy.axes=TRUE, asp=NA)
legend("bottomright", legend=c("Empirical", "Smoothed"),
       col=c(par("fg"), "blue"), lwd=2)
train.roc

predictions <- predict(logit.train, test_set, type="response")
test.roc <- roc(test_set$Potability, predictions)
plot.roc(test.roc, main="Test ROC", col=par("fg"), plot=TRUE, print.auc=FALSE, legacy.axes=FALSE, asp =NA)
plot.roc(smooth(test.roc), col="blue", add=TRUE, plot=TRUE, print.auc=TRUE, legacy.axes=TRUE, asp=NA)
legend("bottomright", legend=c("Empirical", "Smoothed"),
       col=c(par("fg"), "blue"), lwd=2)
test.roc

# =========================
# Random Forest Model (50/50 Split)
# =========================

# Splitting the dataset
set.seed(530)
indices <- sample(1:nrow(water.quality), 0.5 * nrow(water.quality), replace = FALSE)
train_set <- water.quality[indices, ]
test_set <- water.quality[-indices, ]

# Training the model
set.seed(530)
rf.model <- randomForest(factor(Potability) ~ ., data = train_set, ntree = 500, mtry = 3, importance = TRUE)
print(rf.model)

# Performing predictions and generating the confusion matrix
predictions <- predict(rf.model, train_set)
confusionMatrix(predictions, as.factor(train_set$Potability))

predictions <- predict(rf.model, test_set)
confusionMatrix(predictions, as.factor(test_set$Potability))

# Generating AUC/ROC Curves
predictions <- as.numeric(predict(rf.model, train_set))
train.roc <- roc(train_set$Potability, predictions)
plot.roc(train.roc, main="Train ROC", col=par("fg"), plot=TRUE, print.auc=TRUE, legacy.axes=FALSE, asp =NA)
train.roc

predictions <- as.numeric(predict(rf.model, test_set))
test.roc <- roc(test_set$Potability, predictions)
plot.roc(test.roc, main="Test ROC", col=par("fg"), plot=TRUE, print.auc=TRUE, legacy.axes=FALSE, asp =NA)
test.roc

# Plotting error rates of the model
plot(rf.model)

# Code to tune the model by changing mtry
set.seed(530)
tuneRF(train_set[, -10], as.factor(train_set[, 10]), stepFactor=1.5, improve=1e-20, ntreeTry=500, plot=TRUE, trace=TRUE)

# Viewing number of nodes in the tree
hist(treesize(rf.model), main = "No. of Nodes for the Trees", col = "lightblue")

# Checking the variable importance of our dataset
importance(rf.model)
varImpPlot(rf.model, sort=TRUE, n.var=9, main="Most important variables in our dataset")

# Checking the partial dependence of the top 3 variables
partialPlot(rf.model, train_set, Sulfate, 1)
partialPlot(rf.model, test_set, Sulfate, 1)

partialPlot(rf.model, train_set, ph, 1)
partialPlot(rf.model, test_set, ph, 1)

partialPlot(rf.model, train_set, Hardness, 1)
partialPlot(rf.model, test_set, Hardness, 1)

# =========================
# Random Forest Model (80/20 Split)
# =========================

set.seed(530)
indices <- sample(1:nrow(water.quality), 0.8 * nrow(water.quality), replace = FALSE)
train_set <- water.quality[indices, ]
test_set <- water.quality[-indices, ]

set.seed(530)
rf.model <- randomForest(factor(Potability) ~ ., data = train_set, ntree = 500, mtry = 3, importance = TRUE)
print(rf.model)

predictions <- predict(rf.model, train_set)
confusionMatrix(predictions, as.factor(train_set$Potability))

predictions <- predict(rf.model, test_set)
confusionMatrix(predictions, as.factor(test_set$Potability))

# AUC/ROC Curves
predictions <- as.numeric(predict(rf.model, train_set))
train.roc <- roc(train_set$Potability, predictions)
plot.roc(train.roc, main="Train ROC", col=par("fg"), plot=TRUE, print.auc=TRUE, legacy.axes=FALSE, asp =NA)
train.roc

predictions <- as.numeric(predict(rf.model, test_set))
test.roc <- roc(test_set$Potability, predictions)
plot.roc(test.roc, main="Test ROC", col=par("fg"), plot=TRUE, print.auc=TRUE, legacy.axes=FALSE, asp =NA)
test.roc

plot(rf.model)

# We tune and retrain the model with the updaed hyperparameters
set.seed(530)
tuneRF(train_set[, -10], as.factor(train_set[, 10]), stepFactor=1.5, improve=1e-20, ntreeTry=500, plot=TRUE, trace=TRUE)

set.seed(530)
rf.model <- randomForest(factor(Potability) ~ ., data = train_set, ntree = 500, mtry = 5, importance = TRUE)
print(rf.model)

predictions <- predict(rf.model, train_set)
confusionMatrix(predictions, as.factor(train_set$Potability))

predictions <- predict(rf.model, test_set)
confusionMatrix(predictions, as.factor(test_set$Potability))

# Checking the AUC again
predictions <- as.numeric(predict(rf.model, test_set))
test.roc <- roc(test_set$Potability, predictions)
plot.roc(test.roc, main="Test ROC", col=par("fg"), plot=TRUE, print.auc=TRUE, legacy.axes=FALSE, asp =NA)
test.roc

hist(treesize(rf.model), main = "No. of Nodes for the Trees", col = "lightblue")

# Checking the important variables and partial dependence plots again
importance(rf.model)
varImpPlot(rf.model, sort=TRUE, n.var=9, main="Most important variables in our dataset")

partialPlot(rf.model, train_set, Sulfate, 1)
partialPlot(rf.model, test_set, Sulfate, 1)

partialPlot(rf.model, train_set, ph, 1)
partialPlot(rf.model, test_set, ph, 1)

partialPlot(rf.model, train_set, Hardness, 1)
partialPlot(rf.model, test_set, Hardness, 1)

# =========================
# CART Model (50/50 and 80/20 Split)
# =========================

# Split the data into training (80%) and testing (20%) sets
set.seed(530) # for reproducibility
train_index_80 <- createDataPartition(water.quality$Potability, p = 0.8, list = FALSE)
train_data_80 <- water.quality[train_index_80, ]
test_data_80 <- water.quality[-train_index_80, ]

set.seed(530) # for reproducibility
train_index_50 <- createDataPartition(water.quality$Potability, p = 0.5, list = FALSE)
train_data_50 <- water.quality[train_index_50, ]
test_data_50 <- water.quality[-train_index_50, ]

# Function to train and evaluate CART model
train_and_evaluate_CART <- function(train_data, test_data) {
  # Train the CART model
  cart_model <- rpart(Potability ~ ., data = train_data, method = "class")
  
  # Display the CART model
  print("CART Model:")
  print(cart_model)
  
  # Plot the CART model
  print("Plot of the CART Model:")
  rpart.plot(cart_model)
  
  # Predictions on the training data
  train_predictions <- predict(cart_model, train_data, type = "class")
  
  # Calculate training accuracy
  train_accuracy <- mean(train_predictions == train_data$Potability)
  print(paste("Training Accuracy:", train_accuracy))
  
  # Predictions on the testing data
  test_predictions <- predict(cart_model, test_data, type = "class")
  
  # Calculate testing accuracy
  test_accuracy <- mean(test_predictions == test_data$Potability)
  print(paste("Testing Accuracy:", test_accuracy))
  
  # Other metrics (e.g., confusion matrix)
  confusion_matrix <- table(test_data$Potability, test_predictions)
  print("Confusion Matrix:")
  print(confusion_matrix)
  
  # Calculate sensitivity, specificity, PPV, NPV for testing data
  TP <- confusion_matrix[2, 2]
  TN <- confusion_matrix[1, 1]
  FP <- confusion_matrix[1, 2]
  FN <- confusion_matrix[2, 1]
  
  sensitivity <- TP / (TP + FN) * 100
  specificity <- TN / (TN + FP) * 100
  PPV <- TP / (TP + FP) * 100
  NPV <- TN / (TN + FN) * 100
  
  print(paste("Testing Dataset:"))
  print(paste("Accuracy:", paste(round(test_accuracy * 100, 2), "%", sep = "")))
  print(paste("Sensitivity:", paste(round(sensitivity, 2), "%", sep = "")))
  print(paste("Specificity:", paste(round(specificity, 2), "%", sep = "")))
  print(paste("PPV:", paste(round(PPV, 2), "%", sep = "")))
  print(paste("NPV:", paste(round(NPV, 2), "%", sep = "")))
  
  # Calculate sensitivity, specificity, PPV, NPV for training data
  TP_train <- sum(train_data$Potability[train_predictions == "1"])
  TN_train <- sum(train_data$Potability[train_predictions == "0"] == "0")
  FP_train <- sum(train_data$Potability[train_predictions == "1"] == "0")
  FN_train <- sum(train_data$Potability[train_predictions == "0"])
  
  sensitivity_train <- TP_train / (TP_train + FN_train) * 100
  specificity_train <- TN_train / (TN_train + FP_train) * 100
  PPV_train <- TP_train / (TP_train + FP_train) * 100
  NPV_train <- TN_train / (TN_train + FN_train) * 100
  
  print(paste("Training Dataset:"))
  print(paste("Accuracy:", paste(round(train_accuracy * 100, 2), "%", sep = "")))
  print(paste("Sensitivity:", paste(round(sensitivity_train, 2), "%", sep = "")))
  print(paste("Specificity:", paste(round(specificity_train, 2), "%", sep = "")))
  print(paste("PPV:", paste(round(PPV_train, 2), "%", sep = "")))
  print(paste("NPV:", paste(round(NPV_train, 2), "%", sep = "")))
  
  # AUC and ROC curve
  pred <- predict(cart_model, test_data, type="prob")[,2]
  roc_obj <- prediction(pred, test_data$Potability)
  perf <- performance(roc_obj, "tpr", "fpr")
  auc <- performance(roc_obj, "auc")
  
  plot(perf, main="ROC Curve", col="blue", lwd=2)
  abline(a=0, b=1, lwd=2, lty=2, col="gray")
  print(paste("AUC:", auc@y.values[[1]]))
}

# Train and evaluate CART model with 80-2 0 split
print("Results for 80-20 Split:")
train_and_evaluate_CART(train_data_80, test_data_80)

# Train and evaluate CART model with 50-50 split
print("Results for 50-50 Split:")
train_and_evaluate_CART(train_data_50, test_data_50)