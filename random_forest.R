
library(dplyr)
library(caret)
library(tidyr)
library(rsample)
library(randomForest)
library(pROC)
library(PRROC)

# importul setului de date inițial
data <- read.csv("~/Desktop/licenta/csv_licenta/ds_fraud.csv")

data <- data %>%
  mutate(across(where(is.character), as.factor))

# eșantionarea unui subset de 50% din date pentru a gestiona resursele
set.seed(123)
sample_index <- createDataPartition(data$is_fraud, p = 0.5, list = FALSE)
fraud_data_sampled <- data[sample_index, ]


# functia pentru metrici
compute_metrics <- function(y_true, y_pred_prob, threshold = 0.5) {
  y_pred <- ifelse(y_pred_prob > threshold, 1, 0)
  y_pred <- as.factor(as.vector(y_pred))
  
  confusion <- confusionMatrix(y_pred, y_true)
  f1_score <- confusion$byClass["F1"]
  
  pr <- pr.curve(scores.class0 = y_pred_prob[y_true == 1], 
                 scores.class1 = y_pred_prob[y_true == 0], 
                 curve = TRUE)
  pr_auc <- pr$auc.integral
  
  y_pred_prob <- as.numeric(y_pred_prob)  # Ensure y_pred_prob is a numeric vector
  roc_obj <- roc(y_true, y_pred_prob)
  auc_roc <- auc(roc_obj)
  
  list(confusion = confusion, f1_score = f1_score, pr_auc = pr_auc, auc_roc = auc_roc)
}


# impartirea datelor in train,validation,test
set.seed(123)
fraud_split <- initial_split(fraud_data_sampled, prop = 0.7, strata = is_fraud) 
fraud_train <- training(fraud_split)
remaining_data <- testing(fraud_split)

set.seed(123)
remaining_split <- initial_split(remaining_data, prop = 0.5, strata = is_fraud) 
fraud_valid <- training(remaining_split)
fraud_test <- testing(remaining_split)

# stergere NAs
fraud_train <- na.omit(fraud_train)
fraud_valid <- na.omit(fraud_valid)
fraud_test <- na.omit(fraud_test)


fraud_train$is_fraud <- factor(fraud_train$is_fraud)
fraud_valid$is_fraud <- factor(fraud_valid$is_fraud)
fraud_test$is_fraud <- factor(fraud_test$is_fraud)

# basic rf
set.seed(123)
m1_fraud_rf <- randomForest(
  formula = is_fraud ~ .,
  data = fraud_train
)
print(m1_fraud_rf)
plot(m1_fraud_rf)

# predictii pe validare
pred_rf <- predict(m1_fraud_rf, newdata = fraud_valid, type = "prob")
metrics_rf <- compute_metrics(fraud_valid$is_fraud, pred_rf[, "1"])
print(metrics_rf$confusion)
print(paste("AUC-ROC:", metrics_rf$auc_roc))
print(paste("PR AUC:", metrics_rf$pr_auc)) 
print(paste("F1 Score:", metrics_rf$f1_score))

# tuning pt random forest
set.seed(123)
levels(fraud_train$is_fraud) <- make.names(levels(fraud_train$is_fraud))
levels(fraud_valid$is_fraud) <- make.names(levels(fraud_valid$is_fraud))

best_mtry <- NULL
best_nodesize <- NULL
best_auc <- 0

for (mtry in c(2, 3, 4, 5)) {
  for (nodesize in c(1, 5, 10)) {
    set.seed(123)
    model <- randomForest(
      is_fraud ~ .,
      data = fraud_train,
      mtry = mtry,
      ntree = 500,
      nodesize = nodesize
    )
    pred_probs <- predict(model, newdata = fraud_valid, type = "prob")
    auc <- auc(roc(fraud_valid$is_fraud, pred_probs[, "X1"]))
    
    if (auc > best_auc) {
      best_auc <- auc
      best_mtry <- mtry
      best_nodesize <- nodesize
    }
  }
}

print(paste("Best mtry:", best_mtry))
print(paste("Best nodesize:", best_nodesize))
print(paste("Best AUC on validation set:", best_auc))

# antrenarea modelului cu hiperparametri
class_weights <- c("X0" = 1, "X1" = 10)
set.seed(123)
rf_model_weighted <- randomForest(
  is_fraud ~ .,
  data = bind_rows(fraud_train, fraud_valid),
  mtry = best_mtry,
  ntree = 500,
  nodesize = best_nodesize,
  classwt = class_weights
)

print(rf_model_weighted)


# predictii pe noul set de date
pred_probs_new <- predict(rf_model_weighted, newdata = fraud_valid, type = "prob")

# aplicarea unui threshold
threshold <- 0.5
predicted_fraud <- ifelse(pred_probs_new[, "X1"] > threshold, 1, 0)

print("Summary of predicted frauds:")
print(table(predicted_fraud))

# evaluare model
pred_probs_rf <- predict(rf_model_weighted, newdata = fraud_valid, type = "prob")
metrics_rf_final <- compute_metrics(fraud_valid$is_fraud, pred_probs_rf[, "X1"])
print(metrics_rf_final$confusion)
print(paste("AUC-ROC:", metrics_rf_final$auc_roc))
print(paste("PR AUC:", metrics_rf_final$pr_auc))
print(paste("F1 Score:", metrics_rf_final$f1_score))


#salvarea modelului
#save(rf_model_weighted, file = "rf_model_weighted.RData")

