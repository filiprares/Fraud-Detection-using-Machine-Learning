
library(rsample)
library(dplyr)
library(caret)
library(gbm)
library(pROC)
library(ggplot2)
library(vip)
library(pdp)
library(gridExtra)
library(PRROC)

set.seed(123)

table(ds_fraud$is_fraud)

# sample 50% din setul de date initial 
sample_index <- createDataPartition(ds_fraud$is_fraud, p = 0.5, list = FALSE)
fraud_data_sampled <- ds_fraud[sample_index, ]

categorical_vars <- c("category", "weekday", "gender", "age_group", "hour_group", "city_pop_category", "is_fraud")
fraud_data_sampled[categorical_vars] <- lapply(fraud_data_sampled[categorical_vars], factor)

# transformarea variabilei target in variabila logical
fraud_data_sampled$is_fraud <- as.logical(as.integer(fraud_data_sampled$is_fraud) - 1)

table(fraud_data_sampled$is_fraud)

#stergem NAs
fraud_data_sampled <- fraud_data_sampled[complete.cases(fraud_data_sampled), ]




set.seed(123)
fraud_split <- initial_split(fraud_data_sampled, prop = 0.7)
fraud_train <- training(fraud_split)
temp_data <- testing(fraud_split)

set.seed(123)
temp_split <- initial_split(temp_data, prop = 0.5)
fraud_valid <- training(temp_split)
fraud_test <- testing(temp_split)



# functie pentru metrici
compute_metrics <- function(y_true, y_pred_prob, threshold = 0.5) {
  y_pred <- ifelse(y_pred_prob > threshold, TRUE, FALSE)
  y_pred <- factor(y_pred, levels = levels(factor(y_true)))  
  confusion <- confusionMatrix(y_pred, as.factor(y_true))
  f1_score <- confusion$byClass["F1"]
  
  pr <- pr.curve(scores.class0 = y_pred_prob[y_true == 1], 
                 scores.class1 = y_pred_prob[y_true == 0], 
                 curve = TRUE)
  pr_auc <- pr$auc.integral
  
  y_pred_prob <- as.numeric(y_pred_prob)  
  roc_obj <- roc(y_true, y_pred_prob)
  auc_roc <- auc(roc_obj)
  
  list(confusion = confusion, f1_score = f1_score, pr_auc = pr_auc, auc_roc = auc_roc)
}

# gbm model de baza
set.seed(123)
gbm.fit_basic <- gbm(
  formula = is_fraud ~ ., 
  distribution = "bernoulli",
  data = fraud_train,
  n.trees = 100,
  interaction.depth = 10,
  shrinkage = 0.1
)
print(gbm.fit_basic)

#evaluarea modelului
preds_basic <- predict(gbm.fit_basic, newdata = fraud_valid, type = "response")
confusion_basic <- compute_metrics(fraud_valid$is_fraud, preds_basic)
print(confusion_basic$confusion)
print(paste("AUC-ROC:", confusion_basic$auc_roc))
print(paste("PR AUC:", confusion_basic$pr_auc))
print(paste("F1 Score:", confusion_basic$f1_score))

# grid-ul de hyperparametri
hyper_grid <- expand.grid(
  n.trees = c(50, 100, 200),
  interaction.depth = c(1, 3, 5),
  shrinkage = c(0.01, 0.1),
  n.minobsinnode = c(10, 20)
)

results <- data.frame(n.trees = numeric(), interaction.depth = numeric(), shrinkage = numeric(),
                      n.minobsinnode = numeric(), Specificity = numeric(), AUC_ROC = numeric())

# tuning-ul hyperparametrilor
for (i in 1:nrow(hyper_grid)) {
  params <- hyper_grid[i, ]
  cat("Testing hyperparameters: n.trees =", params$n.trees, 
      "interaction.depth =", params$interaction.depth, 
      "shrinkage =", params$shrinkage, 
      "n.minobsinnode =", params$n.minobsinnode, "\n")
  if (params$n.trees == 200 && params$interaction.depth == 5 && params$shrinkage == 0.1 && params$n.minobsinnode == 10) {
    cat("Stopping before hyperparameters: n.trees = 200, interaction.depth = 5, shrinkage = 0.1, n.minobsinnode = 10\n")
    break
  }
  
  
  set.seed(123)
  gbm_model <- gbm(
    formula = is_fraud ~ ., 
    distribution = "bernoulli",
    data = fraud_train,
    n.trees = params$n.trees,
    interaction.depth = params$interaction.depth,
    shrinkage = params$shrinkage,
    n.minobsinnode = params$n.minobsinnode,
    cv.folds = 0,  
    n.cores = NULL,
    verbose = FALSE
  )
  
  best_iter <- gbm.perf(gbm_model)
  cat("Best iteration:", best_iter, "\n")
  
  preds <- predict(gbm_model, newdata = fraud_valid, n.trees = best_iter, type = "response")
  evaluation <- compute_metrics(fraud_valid$is_fraud, preds)
  results <- rbind(results, c(params, Specificity = evaluation$confusion$byClass["Specificity"], AUC_ROC = evaluation$auc_roc))
}

print(results)

# antrenare model final
set.seed(123)
final_gbm_model <- gbm(
  formula = is_fraud ~ ., 
  distribution = "bernoulli",
  data = fraud_train,
  n.trees = 100,
  interaction.depth = 5,
  shrinkage = 0.10,
  n.minobsinnode = 10,
  cv.folds = 0, 
  n.cores = NULL,
  verbose = TRUE
)

best_iter <- gbm.perf(final_gbm_model, method = "OOB", plot.it = FALSE)

# evaluare pe validare
preds_final <- predict(final_gbm_model, newdata = fraud_valid, n.trees = best_iter, type = "response")
final_confusion <- compute_metrics(fraud_valid$is_fraud, preds_final)
print(final_confusion$confusion)
print(paste("Final AUC-ROC:", final_confusion$auc_roc))
print(paste("Final PR AUC:", final_confusion$pr_auc))
print(paste("Final F1 Score:", final_confusion$f1_score))


