  library(dplyr)
  library(caret)
  library(tidyr)
  library(rsample)
  library(rpart)
  library(rpart.plot)
  library(pROC)
  library(ROSE)
  library(randomForest)
  library(PRROC)
  library(tree)
  library(ipred)
  

  compute_metrics <- function(y_true, y_pred_prob, threshold = 0.5) {
    y_pred <- ifelse(y_pred_prob > threshold, 1, 0)
    y_pred <- as.factor(as.vector(y_pred))
    
    confusion <- confusionMatrix(y_pred, y_true)
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
  
  
  table(ds_fraud$is_fraud)
  
  # facem un sample de 50% din setul de date initial din cauza faptului ca este costisitor ca resurse
  sample_index <- createDataPartition(ds_fraud$is_fraud, p = 0.5, list = FALSE)
  fraud_data_sampled <- ds_fraud[sample_index, ]
  
  set.seed(123)
  fraud_split <- initial_split(fraud_data_sampled, prop = 0.7, strata = is_fraud) 
  fraud_train <- training(fraud_split)
  remaining_data <- testing(fraud_split)
  
  set.seed(123)
  remaining_split <- initial_split(remaining_data, prop = 0.5, strata = is_fraud) 
  fraud_valid <- training(remaining_split)
  fraud_test <- testing(remaining_split)

  fraud_train$is_fraud <- as.factor(fraud_train$is_fraud)
  fraud_valid$is_fraud <- as.factor(fraud_valid$is_fraud)
  fraud_test$is_fraud <- as.factor(fraud_test$is_fraud)
  
  #incep cu Rpart, antrenament si tuning pana obtin rezultate bune
  
  rpart_grid <- expand.grid(cp = seq(0.01, 0.1, by = 0.01))
  
  
  evaluate_model <- function(cp, train_data, valid_data) {
    control <- rpart.control(minsplit = 20, minbucket = 7, maxdepth = 30, cp = cp)
    model <- rpart(is_fraud ~ ., data = train_data, method = "class", control = control)
    predictions <- predict(model, newdata = valid_data, type = "class")
    confusion <- confusionMatrix(predictions, as.factor(valid_data$is_fraud), positive = "1")
    sensitivity <- confusion$byClass["Sensitivity"]
    return(list(model = model, sensitivity = sensitivity))
  }
  
  # grid search 
  best_cp <- NULL
  best_sensitivity <- 0
  best_model <- NULL
  for (cp in rpart_grid$cp) {
    result <- evaluate_model(cp, fraud_train, fraud_valid)
    if (result$sensitivity > best_sensitivity) {
      best_sensitivity <- result$sensitivity
      best_cp <- cp
      best_model <- result$model
    }
  }
  #cel mai bun cp dupa specificitate
  print(paste("Best cp:", best_cp, "with sensitivity:", best_sensitivity))
  
  # Plot the best model
  rpart.plot(best_model)
  
  # Combine the training and validation sets
  final_train_data <- bind_rows(fraud_train, fraud_valid)
  
  # Train the final model on the combined dataset
  final_model <- rpart(is_fraud ~ ., data = final_train_data, method = "class", control = rpart.control(cp = best_cp))
  
  # Plot the final model
  rpart.plot(final_model)
  
  # evaluarea modelului pe validare
  test_predictions_rpart <- predict(final_model, newdata = fraud_valid, type = "prob")[,2]
  metrics_rpart <- compute_metrics(fraud_valid$is_fraud, test_predictions_rpart)
  print(metrics_rpart$confusion)
  print(paste("AUC-ROC:", metrics_rpart$auc_roc))
  print(paste("PR AUC:", metrics_rpart$pr_auc))
  print(paste("F1 Score:", metrics_rpart$f1_score))
  
  
  # Function to train and evaluate a tree model with control parameters
  evaluate_tree_model <- function(mincut, minsize, train_data, valid_data) {
    model <- tree(is_fraud ~ ., data = train_data, 
                  control = tree.control(nobs = nrow(train_data), mincut = mincut, minsize = minsize))
    predictions <- predict(model, newdata = valid_data, type = "class")
    confusion <- confusionMatrix(predictions, as.factor(valid_data$is_fraud), positive = "1")
    sensitivity <- confusion$byClass["Sensitivity"]
    return(list(model = model, sensitivity = sensitivity))
  }
  
  # Adjust the grid search parameters for a smaller number of attributes
  tune_grid <- expand.grid(mincut = seq(1, 10, by = 1),
                           minsize = seq(2, 20, by = 2))
  
  # Remove invalid combinations where mincut is greater than half of minsize
  tune_grid <- tune_grid[tune_grid$mincut <= tune_grid$minsize / 2, ]
  
  # Initialize variables to store the best parameters and model
  best_mincut <- NULL
  best_minsize <- NULL
  best_sensitivity <- 0
  best_tree_model <- NULL
  
  # grid search
  for (i in 1:nrow(tune_grid)) {
    mincut <- tune_grid$mincut[i]
    minsize <- tune_grid$minsize[i]
    result <- evaluate_tree_model(mincut, minsize, fraud_train, fraud_valid)
    if (result$sensitivity > best_sensitivity) {
      best_sensitivity <- result$sensitivity
      best_mincut <- mincut
      best_minsize <- minsize
      best_tree_model <- result$model
    }
  }
  # printarea celor mai buni hiperparametri
  print(paste("Best mincut:", best_mincut, "Best minsize:", best_minsize, "with sensitivity:", best_sensitivity))
  

  final_train_data <- bind_rows(fraud_train, fraud_valid)
  
  # antrenare model final
  final_model <- tree(is_fraud ~ ., data = final_train_data, 
                      control = tree.control(nobs = nrow(final_train_data), mincut = best_mincut, minsize = best_minsize))
  
  # evaluarea modelului pe validare
  test_predictions_tree <- predict(final_model, newdata = fraud_valid, type = "vector")[,2]
  metrics_tree <- compute_metrics(fraud_valid$is_fraud, test_predictions_tree)
  print(metrics_tree$confusion)
  print(paste("AUC-ROC:", metrics_tree$auc_roc))
  print(paste("PR AUC:", metrics_tree$pr_auc))
  print(paste("F1 Score:", metrics_tree$f1_score))
  
  # model bagging
  set.seed(123)
  bagged_m1_fraud <- bagging(is_fraud ~ ., data = fraud_train, coob = TRUE)
  pred_bagged_m1_fraud <- predict(bagged_m1_fraud, newdata = fraud_valid, type = "prob")[,2]
  metrics_bagging <- compute_metrics(fraud_valid$is_fraud, pred_bagged_m1_fraud)
  print(metrics_bagging$confusion)
  print(paste("AUC-ROC:", metrics_bagging$auc_roc))
  print(paste("PR AUC:", metrics_bagging$pr_auc))
  print(paste("F1 Score:", metrics_bagging$f1_score))
  
  ntree <- seq(10, 50, by = 1)
  misclassification <- vector(mode = "numeric", length = length(ntree))
  for (i in seq_along(ntree)) {
    set.seed(123)
    model <- bagging(is_fraud ~., data = fraud_train, coob = TRUE, nbag = ntree[i])
    misclassification[i] = model$err
  }
  plot(ntree, misclassification, type="l", lwd="2")
  
  # Approx 34 numarul perfect de bags
  # Antrenam modelul pentru 34 de bags
  
  # Combinare validare si train pentru antrenamentul final
  final_train_data <- bind_rows(fraud_train, fraud_valid)
  
  bagged_m1_fraud_34 <- bagging(is_fraud ~ ., data = final_train_data, coob = TRUE, nbag = 34)
  
  # Am obtinut modelul final, facem predictia pe setul hold-out de test
  pred_bagged_m1_fraud_34 <- predict(bagged_m1_fraud_34, newdata = fraud_valid, type = "prob")[,2]
  
  # metrici pe validare
  metrics_bagging_final <- compute_metrics(fraud_valid$is_fraud, pred_bagged_m1_fraud_34)
  print(metrics_bagging_final$confusion)
  print(paste("AUC-ROC:", metrics_bagging_final$auc_roc))
  print(paste("PR AUC:", metrics_bagging_final$pr_auc))
  print(paste("F1 Score:", metrics_bagging_final$f1_score))
  
  
  
  
  
  # antrenarea modelului final
  final_model <- tree(is_fraud ~ ., data = final_train_data, 
                      control = tree.control(nobs = nrow(final_train_data), mincut = 5, minsize = 10))
  
  # evaluare model pe validare
  test_predictions_tree <- predict(final_model, newdata = fraud_valid, type = "vector")[,2]
  metrics_tree <- compute_metrics(fraud_valid$is_fraud, test_predictions_tree)
  print(metrics_tree$confusion)
  print(paste("AUC-ROC:", metrics_tree$auc_roc))
  print(paste("PR AUC:", metrics_tree$pr_auc))
  print(paste("F1 Score:", metrics_tree$f1_score))