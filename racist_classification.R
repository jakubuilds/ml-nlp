library(tidyverse)
library(rsample)
library(recipes)
library(textclean)
library(textrecipes)
library(themis)
library(tidymodels)
library(discrim)
library(doMC)
library(tictoc)
library(reticulate)
library(keras)
library(tfdatasets)


source("hatespeech_functions.R")
use_virtualenv("./venv")


# Data Prep --------------------------------------------------------------------
# Load data
# df_raw <- read_csv("data/01_imported/train.csv") %>%
#   mutate(
#     hate = factor(ifelse(label == 1, "Hate", "Other")),
#     tweet_clean = sanitize_tweets(tweet)
#   )
# saveRDS(df_raw, "data/02_classification/data_labeled_cleaned.rds")
df_raw <- read_rds("data/02_classification/data_labeled_cleaned.rds") %>%
  select(tweet_clean, label) %>%
  rename(txt = tweet_clean,
         lbl = label)

# Specify training/test sets
set.seed(8675309)
df_split <- df_raw %>%
  filter(nchar(txt) > 0 & nchar(txt) <= 280) %>%
  initial_split(prop = .8, strata = lbl)
df_train <- training(df_split)
df_test <- testing(df_split)

# Create validation sets
folds <- vfold_cv(df_train, strata = lbl, v = 5)


# Exploration ------------------------------------------------------------------
# df_eda <- df_train %>%
#   mutate(
#     n_words = tokenizers::count_words(tweet),
#     n_words2 = tokenizers::count_words(tweet_clean))
# 
# df_eda %>%
#   ggplot(aes(n_words2)) +
#   geom_bar() +
#   labs(x = "Number of words per post",
#        y = "Number of posts")


#  ML Models -------------------------------------------------------------------
# Recipe
hate_rec <- recipe(hate ~ tweet_clean, data = df_train) %>%
  step_upsample(hate, over_ratio = .5) %>%
  step_tokenize(tweet_clean) %>%
  step_tokenfilter(tweet_clean, max_tokens = 1e3) %>%
  step_tfidf(tweet_clean)

lasso_tuned_wf <- lasso_tuner(recipe = hate_rec,
                          folds = folds_strat,
                          n = 5,
                          optimizer = "std_err")
tic()
registerDoMC(cores = 6)
lasso_tuned_rs <- fit_resamples(
  lasso_tuned_wf,
  folds_strat,
  control = control_resamples(save_pred = TRUE)
)
toc()
registerDoSEQ()

fitted_lasso <- fit(lasso_spec, df_train)

fitted_terms <- fitted_lasso %>%
  extract_fit_parsnip() %>%
  tidy() %>%
  arrange(estimate)



# Dense Neural Network ---------------------------------------------------------
max_words <- 20000
max_length <- 30

# Recipe
dnn_rec <- recipe(~ txt, data = df_train) %>%
  step_tokenize(txt) %>%
  step_tokenfilter(txt, max_tokens = max_words) %>%
  step_sequence_onehot(txt, sequence_length = max_length)

dnn_prep <- prep(dnn_rec)

fit_

dnn_train <- bake(dnn_prep, new_data = NULL, composition = "matrix")


# Model
dense_model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words + 1,
                  output_dim = 12,
                  input_length = max_length) %>%
  layer_flatten() %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

dense_model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

dense_history <- dense_model %>%
  fit(
    x = dnn_train,
    y = df_train$label,
    batch_size = 512,
    epochs = 20,
    validation_split = 0.25,
    verbose = FALSE
  )

plot(dense_history). #probably settle on 7 or 8 epochs

# Re-specify validation split using tidymodels framework
set.seed(123)
txt_val <- validation_split(df_train, strata = label)

txt_analysis <- bake(dnn_prep, 
                     new_data = analysis(txt_val$splits[[1]]),
                     composition = "matrix")
txt_assess <- bake(dnn_prep,
                   new_data = assessment(txt_val$splits[[1]]),
                   composition = "matrix")

# Pull labels
label_analysis <- analysis(txt_val$splits[[1]]) %>% pull(label)
label_assess <- assessment(txt_val$splits[[1]]) %>% pull(label)

# Refit
val_history <- dense_model %>%
  fit(
    x = txt_analysis,
    y = label_analysis,
    batch_size = 512,
    epochs = 10,
    validation_data = list(txt_assess, label_assess),
    verbose = FALSE
  )
plot(val_history)



val_res <- keras_predict(dense_model, txt_assess, label_assess)

# Assess
metrics(val_res, state, .pred_class)

val_res %>%
  conf_mat(state, .pred_class) %>%
  autoplot(type = "heatmap")

val_res %>%
  roc_curve(truth = state, .pred_1) %>%
  autoplot() +
  labs(
    title = "Receiver operator curve for Kickstarter blurbs"
  )


# Application using cross-validation
set.seed(345)
df_folds <- vfold_cv(df_train, v = 5)




# Map
cv_fitted <- df_folds %>%
  mutate(validation = map(splits, fit_split, dnn_prep))
cv_fitted %>%
  unnest(validation)

cv_fitted %>%
  unnest(validation) %>%
  group_by(.metric) %>%
  summarize(
    mean = mean(.estimate),
    n = n(),
    std_err = sd(.estimate) / sqrt(n)
  )

# Final test
d_test <- bake(dnn_prep, new_data = df_test, composition = "matrix")
final_res <- keras_predict(dense_model, d_test, df_test$label)
final_res %>%
  metrics(state, .pred_class, .pred_1)
val_res %>%
  metrics(state, .pred_class, .pred_1)
train_data <- df_train %>%
  select(hate, tweet_clean)
x_train <- data.matrix(train_data %>% select(-hate))
y_train <- data.matrix(train_data %>% select(hate))


# Manual inspection of poor fits
df_bind <- final_res %>%
  bind_cols(df_test %>% select(-label))

# false negatives
df_bind %>%
  filter(state == 1, .pred_1 < .2) %>%
  select(text) %>%
  slice_sample(n = 10)
#false positives
df_bind %>%
  filter(state == 0, .pred_1 > .8) %>%
  select(text) %>%
  slice_sample(n = 10)


# Long Short-Term Memory (LSTM) ------------------------------------------------
max_words = 2e4
max_length = 30
set.seed(12345)

# Recipe
deep_rec <- recipe(~ txt, data = df_train) %>%
  step_tokenize(txt) %>%
  step_tokenfilter(txt, max_tokens = max_words) %>%
  step_sequence_onehot(txt, sequence_length = max_length)

deep_prep <- prep(deep_rec)
deep_train <- bake(deep_prep, new_data = NULL, composition = "matrix")

# Specify validation split
vsplit <- validation_split(df_train, strata = lbl)
#create datasets
df_analysis <- bake(deep_prep, new_data = analysis(vsplit$splits[[1]]),
                    composition = "matrix")
df_assess <- bake(deep_prep, new_data = assessment(vsplit$splits[[1]]),
                  composition = "matrix")
#extra labels from each dataset
lbl_analysis <- analysis(vsplit$splits[[1]]) %>% pull(lbl)
lbl_assess <- assessment(vsplit$splits[[1]]) %>% pull(lbl)


# Initial model
lstm_mod <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words + 1, output_dim = 32) %>%
  layer_lstm(units = 32) %>%
  layer_dense(units = 1, activation = "sigmoid")
lstm_mod %>%
  compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = c("accuracy")
  )

# Initial fit
lstm_history <- lstm_mod %>%
  fit(
    df_analysis,
    lbl_analysis,
    epochs = 10,
    validation_data = list(df_assess, lbl_assess),
    batch_size = 512,
    verbose = FALSE
  )
lstm_history
plot(lstm_history)


# Add dropout model
lstm2_mod <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words + 1, output_dim = 32) %>%
  layer_lstm(units = 32, dropout = 0.4, recurrent_dropout = 0.4) %>%
  layer_dense(units = 1, activation = "sigmoid")
lstm2_mod %>%
  compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = c("accuracy")
  )

# Initial fit
lstm2_history <- lstm2_mod %>%
  fit(
    df_analysis,
    lbl_analysis,
    epochs = 10,
    validation_data = list(df_assess, lbl_assess),
    batch_size = 512,
    verbose = FALSE
  )
lstm2_history
plot(lstm2_history)


# Bidirectional model
bilstm_mod <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words + 1, output_dim = 32) %>%
  bidirectional(layer_lstm(units = 32, dropout = 0.4,
                           recurrent_dropout = 0.4)) %>%
  layer_dense(units = 1, activation = "sigmoid")

bilstm_mod %>%

bilstm_history <- bilstm_mod %>%
  fit(
    df_analysis,
    lbl_analysis,
    epochs = 10,
    validation_data = list(df_assess, lbl_assess),
    batch_size = 512,
    verbose = FALSE
  )
bilstm_history


# Final model
final_mod <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words + 1, output_dim = 32) %>%
  bidirectional(layer_lstm(units = 32, dropout = 0.4,
                           recurrent_dropout = 0.4,
                           return_sequences = TRUE)) %>%
  bidirectional(layer_lstm(units = 32, dropout = 0.4,
                           recurrent_dropout = 0.4)) %>%
  layer_dense(units = 1, activation = "sigmoid")

final_mod %>%
  compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = c("accuracy")
  )

final_history <- final_mod %>%
  fit(
    df_analysis,
    lbl_analysis,
    epochs = 10,
    validation_data = list(df_assess, lbl_assess),
    batch_size = 512,
    verbose = FALSE
  )
final_history

# Compare models
results <- map(
  list(lstm_mod, 
       lstm2_mod,
       bilstm_mod,
       final_mod),
  ~keras_predict(.x, df_assess, lbl_assess))

all_mods_results <-  bind_rows(
    results[[1]] %>% mutate(model = "lstm"),
    results[[2]] %>% mutate(model = "lstm-dropout"),
    results[[3]] %>% mutate(model = "bi-lstm"),
    results[[4]] %>% mutate(model = "final")
  )

all_mods_results %>% 
  group_by(model) %>%
  metrics(state, .pred_class)

all_mods_results %>%
  group_by(model) %>%
  roc_curve(truth = state, .pred_1) %>%
  autoplot() +
  labs(
    title = "Receiver operator curve for hateful speech"
  )

val_ltsm <- keras_predict(ltsm_mod, df_assess, lbl_assess)



# LSTM
#source: https://www.jla-data.net/eng/vocabulary-based-text-classification/
# ltsm_rec <- recipe(~ tweet, data = df_train) %>%
#   step_tokenize(tweet) %>%
#   step_sequence_onehot(tweet, sequence_length = 30)
# 
lstm_mod <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words + 1, output_dim = 32) %>%
  layer_lstm(units = 32) %>%
  layer_dense(units = 1, activation = "sigmoid")

lstm_mod %>%
  compile(optimizer = "rmsprop",
          loss = "binary_crossentropy",
          metrics = c("accuracy"))

history <- lstm_mod %>%  # fit the model (this will take a while...)
  fit(x_train, 
      y_train, 
      epochs = 25, 
      batch_size = nrow(train_data)/5, 
      validation_split = 1/5)

summary(model)

# Models
nb_spec <- naive_Bayes() %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_fit <- wf %>%
  add_model(nb_spec) %>%
  fit(data = df_train)


# Evaluation
set.seed(8675309)
folds <- vfold_cv(df_train)
folds_strat <- vfold_cv(df_train, strata = hate)

nb_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(nb_spec)

nb_rs <- fit_resamples(
  nb_wf,
  folds_strat,
  control = control_resamples(save_pred = TRUE)
)

nb_rs_metrics <- collect_metrics(nb_rs)
nb_rs_predictions <- collect_predictions(nb_rs)

nb_rs_metrics


# Plot curve
nb_rs_predictions %>%
  group_by(id) %>%
  roc_curve(truth = hate, .pred_Hate) %>%
  autoplot() +
  labs(
    color = NULL,
    title = "ROC curve for Hateful Tweets",
    subtitle = "Each resample fold show in a different color"
  )

# confusion matrix
conf_mat_resampled(nb_rs, tidy = FALSE) %>%
  autoplot(type = "heatmap")


# Null model
null_class <- null_model() %>%
  set_engine("parsnip") %>%
  set_mode("classification")

null_rs <- workflow() %>%
  add_recipe(rec) %>%
  add_model(null_class) %>%
  fit_resamples(
    folds
  )

null_rs %>%
  collect_metrics()
