library(xgboost)
library(vtreat)
library(lime)
library(pdp)
library(gridExtra)
library(vip)
library(caret)
library(pROC)
library(PRROC)
library(rsample)
library(dplyr)

# importarea datasetului
data <- read.csv("~/Desktop/licenta/csv_licenta/ds_fraud.csv")

categorical_vars <- c("category", "weekday", "gender", "age_group", "hour_group", "city_pop_category","is_fraud")
data[categorical_vars] <- lapply(data[categorical_vars], factor)



# Ensure target variable is numeric
data$is_fraud <- as.numeric(data$is_fraud) - 1

# Sample 10% of the original dataset
set.seed(123)
sample_index <- createDataPartition(data$is_fraud, p = 0.5, list = FALSE)
fraud_data_sampled <- data[sample_index, ]

# Perform the initial split
set.seed(123)
fraud_split <- initial_split(fraud_data_sampled, prop = 0.7, strata = is_fraud)
fraud_train <- training(fraud_split)
remaining_data <- testing(fraud_split)

set.seed(123)
remaining_split <- initial_split(remaining_data, prop = 0.5, strata = is_fraud)
fraud_valid <- training(remaining_split)
fraud_test <- testing(remaining_split)

# Remove NA values
fraud_train <- na.omit(fraud_train)
fraud_valid <- na.omit(fraud_valid)
fraud_test <- na.omit(fraud_test)

# Elimină nivelurile goale din coloana weekday pentru fraud_test
fraud_test$weekday <- droplevels(fraud_test$weekday)
fraud_train$weekday <- droplevels(fraud_train$weekday)
fraud_valid$weekday <- droplevels(fraud_valid$weekday)

features_train <- model.matrix(~ . - 1, data = fraud_train %>% select(-is_fraud))
response_train <- fraud_train$is_fraud

features_valid <- model.matrix(~ . - 1, data = fraud_valid %>% select(-is_fraud))
response_valid <- fraud_valid$is_fraud

features_test <- model.matrix(~ . - 1, data = fraud_test %>% select(-is_fraud))
response_test <- fraud_test$is_fraud



# Function to compute metrics
compute_metrics <- function(y_true, y_pred_prob, threshold = 0.5) {
  y_pred <- ifelse(y_pred_prob > threshold, 1, 0)
  y_pred <- as.factor(as.vector(y_pred))
  y_true <- as.factor(as.vector(y_true))
  
  levels(y_true) <- c("0", "1")
  levels(y_pred) <- c("0", "1")
  
  confusion <- confusionMatrix(y_pred, y_true)
  f1_score <- confusion$byClass["F1"]
  
  pr <- pr.curve(scores.class0 = y_pred_prob[y_true == 1], 
                 scores.class1 = y_pred_prob[y_true == 0], 
                 curve = TRUE)
  pr_auc <- pr$auc.integral
  
  roc_obj <- roc(as.numeric(as.character(y_true)), y_pred_prob)
  auc_roc <- auc(roc_obj)
  
  list(confusion = confusion, f1_score = f1_score, pr_auc = pr_auc, auc_roc = auc_roc)
}


scale_pos_weight <- sum(response_train == 0) / sum(response_train == 1)

# modelul xgboost initial
set.seed(123)
xgb.fit1 <- xgboost(
  data = features_train,
  label = response_train,
  nrounds = 1000,
  objective = "binary:logistic",
  eval_metric = "auc",
  scale_pos_weight = scale_pos_weight,
  verbose = 0
)

# evaluare pe validation
pred_valid <- predict(xgb.fit1, features_valid)
metrics_xgb_initial <- compute_metrics(response_valid, pred_valid)
print(metrics_xgb_initial$confusion)
print(paste("AUC-ROC:", metrics_xgb_initial$auc_roc))
print(paste("PR AUC:", metrics_xgb_initial$pr_auc)) 
print(paste("F1 Score:", metrics_xgb_initial$f1_score))

# grid-ul de tuning
hyper_grid_xg <- expand.grid(
  eta = c(.01, .05, .1),
  max_depth = c(3, 5, 7),
  min_child_weight = c(1, 3, 5),
  subsample = c(.7, .8, .9), 
  colsample_bytree = c(.7, .8, .9)
)

best_params <- list()
best_auc <- 0

for(i in 1:nrow(hyper_grid_xg)) {
  params <- list(
    eta = hyper_grid_xg$eta[i],
    max_depth = hyper_grid_xg$max_depth[i],
    min_child_weight = hyper_grid_xg$min_child_weight[i],
    subsample = hyper_grid_xg$subsample[i],
    colsample_bytree = hyper_grid_xg$colsample_bytree[i],
    scale_pos_weight = scale_pos_weight,
    objective = "binary:logistic",
    eval_metric = "auc"
  )
  
  set.seed(123)
  xgb.fit <- xgboost(
    params = params,
    data = features_train,
    label = response_train,
    nrounds = 1000,
    verbose = 0
  )
  
  pred_valid <- predict(xgb.fit, features_valid)
  auc <- auc(roc(response_valid, pred_valid))
  
  if (auc > best_auc) {
    best_auc <- auc
    best_params <- params
  }
}

print(best_params)
print(paste("Best AUC on validation set:", best_auc))

# Definire parametrii optimi
best_params <- list(
  eta = 0.01,
  max_depth = 7,
  min_child_weight = 5,
  subsample = 0.8,
  colsample_bytree = 0.8,
  scale_pos_weight = 168.7353,
  objective = "binary:logistic"  # Obiectivul pentru clasificarea binară
)

# antrenare model final cu parametri optimi si 5000 arbori
set.seed(123)
xgb.fit_final <- xgboost(
  params = best_params,
  data = features_train,
  label = response_train,
  nrounds = 5000,
  verbose = 1
)

# matricea de importanta a variabilelor
importance_matrix <- xgb.importance(model = xgb.fit_final)
xgb.plot.importance(importance_matrix, top_n = 10, measure = "Gain")


# predictia pe validare
pred_valid <- predict(xgb.fit_final, features_valid)
response_test <- as.factor(response_valid)
metrics_xgb_test <- compute_metrics(response_valid, pred_valid)
print(metrics_xgb_test$confusion)
print(paste("AUC-ROC:", metrics_xgb_test$auc_roc))
print(paste("PR AUC:", metrics_xgb_test$pr_auc)) 
print(paste("F1 Score:", metrics_xgb_test$f1_score))


#modelul este cel mai bun, facem precictia si pe testare

# predictia pe testare 
pred_test <- predict(xgb.fit_final, features_test)
response_test <- as.factor(response_test)
metrics_xgb_test <- compute_metrics(response_test, pred_test)
print(metrics_xgb_test$confusion)
print(paste("AUC-ROC:", metrics_xgb_test$auc_roc))
print(paste("PR AUC:", metrics_xgb_test$pr_auc)) 
print(paste("F1 Score:", metrics_xgb_test$f1_score))


save(xgb.fit_final, file = "xgb.fit_final.RData")
