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


source("ml_functions.R")
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
         lbl = label) %>%
  mutate(lbl = as.factor(lbl))

# Up Sample
df_upsampled <- caret::upSample(x = df_raw %>% select(-lbl),
                                y = df_raw$lbl,
                                yname = "lbl") %>%
  mutate(lbl = as.numeric(lbl))

# Specify splits
set.seed(8675309)
splits <- df_raw %>%
  mutate(lbl = as.numeric(lbl)) %>%
  filter(nchar(txt) > 0 & nchar(txt) <= 280) %>%
  initial_validation_split(strata = lbl)
splits_up <- df_upsampled %>%
  filter(nchar(txt) > 0 & nchar(txt) <= 280) %>%
  initial_validation_split(strata = lbl)

df_train <- training(splits)
df_valid <- validation(splits)
df_test <- testing(splits)

# Create validation sets
folds <- vfold_cv(df_train, strata = lbl, v = 10)


# Specify Recipes --------------------------------------------------------------
# Hyper-parameters
max_words <- 20000
max_length <- 30

# ML recipe
ml_rec <- recipe(lbl ~ txt, data = df_train)

# Deep learning recipe
dl_rec <- recipe(~ txt, data = df_train) %>%
  step_tokenize(txt) %>%
  step_tokenfilter(txt, max_tokens = max_words) %>%
  step_sequence_onehot(txt, sequence_length = max_length)
dl_prep <- prep(dl_rec)


# Specify Models ---------------------------------------------------------------
# Dense Neural Network
dnn_model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words + 1,
                  output_dim = 12,
                  input_length = max_length) %>%
  layer_flatten() %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid") %>%
  compile_model()

# Base LTSM
lstm_model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words + 1, output_dim = 32) %>%
  layer_lstm(units = 32, dropout = 0.4, recurrent_dropout = 0.4) %>%
  layer_dense(units = 1, activation = "sigmoid") %>%
  compile_model()

# Bi-directional LTSM
lstm_bi_model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words + 1, output_dim = 32) %>%
  bidirectional(layer_lstm(units = 32, dropout = 0.4,
                           recurrent_dropout = 0.4)) %>%
  layer_dense(units = 1, activation = "sigmoid") %>%
  compile_model()

# Bi-directional, 2-layer LTSM
lstm_bi2_model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words + 1, output_dim = 32) %>%
  bidirectional(layer_lstm(units = 32, dropout = 0.4,
                           recurrent_dropout = 0.4,
                           return_sequences = TRUE)) %>%
  bidirectional(layer_lstm(units = 32, dropout = 0.4,
                           recurrent_dropout = 0.4)) %>%
  layer_dense(units = 1, activation = "sigmoid") %>%
  compile_model()

# Convolutional Neural Network
cnn_model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words + 1, output_dim = 16,
                  input_length = max_length) %>%
  layer_conv_1d(filter = 32, kernel_size = 5, activation = "relu") %>%
  layer_global_max_pooling_1d() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid") %>%
  compile_model()


# Fit Models -------------------------------------------------------------------
dnn_res <- fit_wrapper(splits, dl_prep, model = dnn_model)
lstm_res <- fit_wrapper(splits, dl_prep, model = lstm_model)
lstm_bi_res <- fit_wrapper(splits, dl_prep, model = lstm_model)
lstm_bi2_res <- fit_wrapper(splits, dl_prep, model = lstm_bi2_model)
cnn_res <- fit_wrapper(splits, dl_prep, model = cnn_model)


cv_fitted <- folds %>%
  mutate(validation = map(splits, fit_split(), dl_prep))
