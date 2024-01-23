load_classification_data <- function(file) {
 read_csv(file) %>%
    mutate(
      hate = factor(ifelse(label == 1, "Hate", "Other"))
    )
}

split_sample <- function(df,
                         p = .8,
                         seed = 123,
                         stratify_by = NULL) {
  set.seed(seed)
  
  df_split <- initial_split(df, prop = p, strata = {{ stratify_by }})
  df_train <- training(df_split)
  df_test <- testing(df_split)
  return(list(df_train, df_test))
}
  

make_cvfolds <- function(df, 
                         stratify_by = NULL) {
  vfold_cv(df, strata = {{ stratify_by }})
}

create_recipe <- function(df) {
  recipe(hate ~ tweet, data = df) %>%
    step_upsample(hate, over_ratio = .5) %>%
    step_tokenize(tweet) %>%
    step_tokenfilter(tweet, max_tokens = 1e3) %>%
    step_tfidf(tweet)
}


fit_null <- function(recipe, cvfolds) {
  null_class <- null_model() %>%
    set_engine("parsnip") %>%
    set_mode("classification")
  
  null_rs <- workflow() %>%
    add_recipe(recipe) %>%
    add_model(null_class) %>%
    fit_resamples(
      cvfolds
    )
}



preproc_explorer <- function(df, 
                            n_sample = 1000, 
                            ncores,
                            seed = 1234) {
  
  set.seed(seed)
  # Randomly sample `n` rows and pull text as vector
  x <- raw %>%
    filter(lang == "en") %>%
    sample_n(size = n) %>%
    pull(text)
  
  # Factorial preprocess and save
  factorial_preprocessing(x,
                          parallel = TRUE, 
                          cores = ncores)
  
}

pretexter <- function(df, ncores, seed = 1234) {
  preText(
    df,
    dataset_name = "Subject Tweets",
    distance_method = "cosine",
    parallel = TRUE,
    cores = ncores)
}

pretext_plots <- function(df, n = 20) {
  df[[2]] %>%
    slice_tail(n = n) %>%
    mutate(preprocessing_steps = fct_reorder(preprocessing_steps, 
                                             preText_score,
                                             .desc = TRUE)) %>%
    ggplot(aes(x = preText_score, y = preprocessing_steps)) +
    geom_point()
  ggsave("results/figs/pretext_score_plot.png")
  
  preText::regression_coefficient_plot(df,
                              remove_intercept = TRUE)
  ggsave("results/figs/pretext_coef_plot.png")
}




sanitize_tweets <- function(x) {
  x %>%
    replace_url()  %>% 
    replace_emoji() %>%
    replace_emoticon() %>%
    replace_html() %>% 
    str_to_lower() %>% # transform to lowercase
    str_remove_all("@([0-9a-zA-Z_]+)") %>% # remove username
    str_remove_all('[\\#]+') %>% # remove hashtag
    replace_internet_slang() %>%
    str_squish()
}


text_cleansing <- function(data) {
  new_data <- data %>% 
    mutate(
      text = text %>%
        replace_url()  %>% 
        replace_emoji() %>%
        replace_emoticon() %>%
        replace_html() %>% 
        str_to_lower() %>% # transform to lowercase
        str_remove_all("@([0-9a-zA-Z_]+)") %>% # remove username
        str_remove_all('[\\#]+') %>% # remove hashtag
        str_remove_all('[\\!]+') %>% 
        str_remove_all('[\\&]+') %>% 
        str_remove_all('[\\"]+') %>% 
        replace_internet_slang() %>% 
        str_remove_all(pattern = "[[:digit:]]") %>% # remove number
        str_remove_all(pattern = "[[:punct:]]") %>% # remove all punctuation except !,&,""
        str_squish() # extra white space remove
    ) %>% 
    select(text, label)
  return(new_data)
}



corpus_cleansing <- function(data) {
  words.to.remove <- c("user","url","rt")
  data_corpus <- data$text %>% 
    VectorSource() %>% 
    VCorpus(readerControl = list(language="en")) %>%
    tm_map(removeWords, words.to.remove) %>%
    tm_map(removeWords, stopwords("en")) %>%
    tm_map(stemDocument) %>% 
    tm_map(stripWhitespace) %>% 
    sapply(as.character) %>%
    as.data.frame(stringsAsFactors = FALSE)
  data_clean <- bind_cols(data_corpus, data[,2] )%>%
    `colnames<-`(c("text","label"))
  return(data_clean)
}
