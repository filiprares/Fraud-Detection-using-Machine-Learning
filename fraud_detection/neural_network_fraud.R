
library(keras)
library(tensorflow)
library(caret)
library(pROC)
library(rsample)
library(tidymodels)
library(ROSE)
library(reticulate)
library(PRROC)

# functie pentru mentinerea reproductibilitatii
set_random_seeds <- function(seed) {
  set.seed(seed)
  tensorflow::tf$random$set_seed(seed)
}

# functie pentru calcularea metricilor
compute_metrics <- function(y_true, y_pred_prob, threshold = 0.5) {
  y_pred <- ifelse(y_pred_prob > threshold, 1, 0)
  y_pred <- as.vector(y_pred)
  
  confusion <- confusionMatrix(as.factor(y_pred), as.factor(y_true))
  f1_score <- confusion$byClass["F1"]
  
  pr <- pr.curve(scores.class0 = y_pred_prob[y_true == 1], 
                 scores.class1 = y_pred_prob[y_true == 0], 
                 curve = TRUE)
  pr_auc <- pr$auc.integral
  
  y_pred_prob <- as.vector(y_pred_prob) 
  roc_obj <- roc(y_true, y_pred_prob)
  auc_roc <- auc(roc_obj)
  
  list(confusion = confusion, f1_score = f1_score, pr_auc = pr_auc, auc_roc = auc_roc)
}

# importarea setului de date
data <- read.csv("~/Desktop/licenta/csv_licenta/ds_fraud.csv")


data <- data %>% 
  mutate(across(where(is.character), as.factor))








set_random_seeds(123)

use_condaenv("NN_env", required = TRUE)

#variabile categorice
categorical_vars <- c("category", "weekday", "gender", "age_group", "hour_group", "city_pop_category")

#one hot encoding pt variabile categorice
dummy_data <- dummyVars("~ .", data = ds_fraud[categorical_vars])
fraud_data_one_hot <- data.frame(predict(dummy_data, newdata = ds_fraud))

fraud_data_combined <- cbind(ds_fraud, fraud_data_one_hot)

fraud_data_combined <- fraud_data_combined[, !sapply(fraud_data_combined, is.factor)]



# split initial
set.seed(123)
fraud_split <- initial_split(fraud_data_combined, prop = 0.7, strata = is_fraud) 
fraud_train <- training(fraud_split)
remaining_data <- testing(fraud_split)


# spit in validation si testing
set.seed(123)
remaining_split <- initial_split(remaining_data, prop = 0.5, strata = is_fraud) 
fraud_valid <- training(remaining_split)
fraud_test <- testing(remaining_split)


x_train <- model.matrix(~ . - 1, data = fraud_train %>% select(-is_fraud))
y_train <- fraud_train$is_fraud

x_valid <- model.matrix(~ . - 1, data = fraud_valid %>% select(-is_fraud))
y_valid <- fraud_valid$is_fraud

x_test <- model.matrix(~ . - 1, data = fraud_test %>% select(-is_fraud))
y_test <- fraud_test$is_fraud

fraud_data_combined_app <- model.matrix(~ . - 1, data = fraud_data_combined_app %>% select(-is_fraud))

build_model <- function(units1, units2, dropout_rate, input_shape) {
  model <- keras_model_sequential() %>%
    layer_dense(units = units1, activation = 'relu', input_shape = input_shape) %>%
    layer_dense(units = units2, activation = 'relu') %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_dense(units = 1, activation = 'sigmoid')
  
  model %>% compile(
    optimizer = optimizer_adam(),
    loss = 'binary_crossentropy',
    metrics = c('accuracy')
  )
  
  return(model)
}

# definim si compilam modelul
set_random_seeds(123)
model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = 'relu', input_shape = c(ncol(x_train))) %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 1, activation = 'sigmoid')

model %>% compile(
  optimizer = tensorflow::tf$keras$optimizers$legacy$Adam(learning_rate = 0.001),
  loss = 'binary_crossentropy',
  metrics = c('accuracy') 
)

# antrenam modelul
history <- model %>% fit(
  x_train,
  y_train,
  epochs = 20,
  batch_size = 64,
  validation_data = list(x_valid, y_valid),
  shuffle = FALSE
)


# definim input shape
input_shape <- c(ncol(x_train))

# construirea modelului
final_model <- build_model(128, 64, 0.3, input_shape)


final_history <- final_model %>% fit(
  x_train,
  y_train,
  epochs = 6,
  batch_size = 64,
  validation_data = list(x_valid, y_valid),
  shuffle = FALSE,
  class_weight = list("0" = 1, "1" = 5)
)

# predictia pe setul de test
pred_prob <- final_model %>% predict(fraud_data_combined_app)



# metricile
metrics <- compute_metrics(y_test, pred_prob, threshold = 0.5)
print(metrics$confusion)
print(paste("AUC-ROC:", metrics$auc_roc))
print(paste("PR AUC:", metrics$pr_auc))
print(paste("F1 Score:", metrics$f1_score))



