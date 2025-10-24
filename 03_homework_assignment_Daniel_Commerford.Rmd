---
title: 'Homework 3: Supervised Text Classification'
author: "Philipp Masur"
date: ""
output:
  html_document:
    toc: yes
editor_options:
  chunk_output_type: console
---

```{r opts, echo = FALSE}
library(knitr)
knitr::opts_chunk$set(echo = TRUE, results = TRUE, message = FALSE, warning = FALSE)
```

# Formalities

- Name:         [Daniel Commerford]
- Student ID:   [2861501]   

In the end, klick on "knit" and upload the respective html-output file to Canvas. Please add your name and lastname to the output file name:
e.g., 02_homework_assignment_NAME-LASTNAME.html


# Introduction 


## Loading data

First, we are going to load the data. It is a data set that contains lyrics of songs and the respective genre of the music. Our goal will be to predict the genre from the lyrics.

```{r}
library(tidyverse)
library(tidytext)
library(tidymodels)
library(textrecipes)
library(readr)
library(dplyr)


d <- read_csv("lyrics_data_new.csv") |> 
  select(Artist = SName, text = Lyric, genre = Genre) |> 
  slice_sample(prop = .40) |>    # Here I am using just 40% of the sample to make computations faster, you can increase this if your computer allows it.
  filter(genre != "Hip Hop")

head(d)
```

**Question:** Can you create a bar plot that shows how many songs are included per genre? (Bonus: Try to reorder the bar plot so that the subject with the most articles is shown first and the subject with the least articles at last).

```{r}
# First, I try and group all the songs by category using the tally function

genre_tally <- d |>
  count(genre, sort= TRUE)

ggplot(genre_tally, aes(x = reorder(genre, -n), y = n)) +
  geom_bar(stat = "identity") +
  labs(
    title = "Number of Songs per Genre",
    x = "Genre",
    y = "Number of Songs"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
```

**Answer:** There are 5693 rocks songs and 3170 pop songs.



**Questions:** Which words are particular frequently used in which genre? Do you see interesting differences? Or similarities? Run the code and inspect the results...

```{r}
d |> 
  unnest_tokens(word, text) |>
  anti_join(stop_words) |> 
  group_by(genre, word) |> 
  summarize(n = n()) |>
  slice_max(n, n = 25) %>%
  ungroup() |> 
  ggplot(aes(x = fct_reorder(word, n), y = n)) +
  geom_col() +
  coord_flip() +
  facet_wrap(~genre, scales = "free")
  
```

**Answer:** The word "love" is the most used in both genres. "baby", "time", and "yeah" are also used frequently, but in a different order. the word "girl" is more likely to be heard in Pop songs rather than rock; "boy" is absent altogether from rock.

# Text preprocessing

Next, we engage in the preprocessing. For supervised machine learning, this means create a train and test data set and thinking about meaningful text preprocessing steps. 

## Creating train and test set

First, create a training and a test set. Think about meaningful partitions. Please justify why you choose a certain percentage. Don't forget to use set.seed to ensure reproducibility. 

```{r}
# Solution here
library(tidymodels)

set.seed(123) # set to 123

split <- initial_split(d, prop = .80)


# Separate the data into training and testing sets
train_data <- training(split)
test_data <- testing(split)


# to confirm that I split the test data in proportions that match approx the overall distribution

train_tally <- train_data |> 
  count(genre, sort = TRUE)

test_tally <- test_data |> 
  count(genre, sort = TRUE)

# songs in the training data are split: rock (4611 - 65%) - pop (2480 - 35%)

# songs in the test data to be used as a gold standard are split into: rock (1131 - 64%) - pop (642 - 36%) 

# distribution for test and training data mostly match

```

**Answer:** The normal 50/50 split would not be ideal here because there is not an even amount of songs of each genre: rock has 5693 while pop has 3170. So this distribution needs to be kept in mind when splitting the data for training. The ideal would be to make sure there is an even distribution of songs in each class. Since this choice is binary, training a bit more of the data against a small test set would make more sense than splitting it evenly. 

## Text preprocessing 

Now think about meaningful text preprocessing (Removing stopwords? Trimming the data set?). Also justify your steps below. 

```{r}
# Solution here
library(textrecipes)

rec <- recipe(genre ~ text, data = d) |> 
  step_tokenize(text) |> 
  step_stopwords(text) |> # the function to remove stopwords
  step_tf(all_predictors()) |> 
  step_normalize(all_predictors())

prepped_rec <- prep(rec)                       # Prepares the recipe
processed_data <- bake(prepped_rec, new_data = NULL)  # Apply to the entire dataset

head(processed_data)

```

**Answer:** Trimming the dataset does not seem needed since the data is not too large and computing power is sufficient. However, it makes sense to remove stopwords since we are comparing two options. Stopwords will likely not help the algorithm distinguish between genres since they do not contribute meaningfully to the overall sentiment of the song.


# Machine Learning

## Training the algorithm

Now, we can train a model. Feel free to use either SVM or a neural network. 

```{r}
# A neural network yielded better results in class and tutorials, but for the sake of computing power i will use SVM

set.seed(123)

# SVM with workflow
library(LiblineaR)
svm_workflow <- workflow() |> 
  add_recipe(rec) |> 
  add_model(svm_linear(mode = "classification", 
                       engine = "LiblineaR"))

# Fitting the SVM model
m_svm <- fit(svm_workflow, data = training(split))



```

## Validating on the test data

More importantly, we need to validate the algorithm performance. How well does the algorithm predict genre in the test data set?

```{r}
# Testing the performance of the model 

# Testing the SVM
predict_svm <- predict(m_svm, testing(split)) |>    
  bind_cols(select(testing(split), genre)) |>                    
  rename(predicted = .pred_class, actual = genre) 

predict_svm

# Convert `actual` and `predicted` columns to factors due to error

predict_svm <- predict_svm |>
  mutate(
    actual = as.factor(actual),
    predicted = as.factor(predicted)
  )

# Calculate metrics
class_metrics <- metric_set(accuracy, precision, recall, f_meas)

# Apply metrics to the predictions
predict_svm |>
  class_metrics(truth = actual, estimate = predicted)

```

**Answer:** The algorithm did not perform well at all. Accuracy was at 62%, and precision was particularly bad with 47%, and the overall f score was 51%. This means the model is not very useful for the task of predicting the correct class since it is slightly better than guessing.

## Improving the performance

Now try to improve the performance. You can use all the tricks in the box. Including:

- Increase size of the data set (in the beginning, we just use 40% of the data, if your computer allows, take perhaps 50 or 60%)
- Change the ratio between testing and training
- Experiment with different recipes
- Use a different algorithm or a different neural network structure
- Conduct a grid-search
- ...

The student with the best performance will get a small price! 

```{r}
# Solution

# Load and preprocess the data
d <- read_csv("lyrics_data_new.csv") |>
  select(Artist = SName, text = Lyric, genre = Genre) |>
  slice_sample(prop = .40) |> 
  filter(genre != "Hip Hop")

# Create the data split
set.seed(123)
split <- initial_split(d, prop = 0.50) #interested in seeing what results doing a 50/50 split would have yielded given that 80/20 did not give good results

# Separate into training and testing sets
train_data <- training(split)
test_data <- testing(split)

# Define the recipe with additional preprocessing
rec <- recipe(genre ~ text, data = train_data) |> 
  step_tokenize(text) |>                # Tokenize text
  step_stopwords(text) |>               # Remove stopwords
  step_lemma(text) |>                   # Lemmatize tokens for better features
  step_tokenfilter(text, min_times = 5) |>  # Keep words that appear at least 5 times
  step_tfidf(text) |>                   # convert text to numerical features
  step_normalize(all_predictors())      # Normalize predictors

# Prepare and bake the recipe
prepped_rec <- prep(rec)
train_data_processed <- bake(prepped_rec, new_data = NULL)
test_data_processed <- bake(prepped_rec, new_data = test_data)

# Define an SVM model with tuning
tune_spec <- svm_linear(
  mode = "classification",
  engine = "LiblineaR",
  cost = tune() # Tune the cost parameter
)

# Define workflow
svm_workflow <- workflow() |>
  add_recipe(rec) |>
  add_model(tune_spec)

# Cross-validation setup
set.seed(123)
folds <- vfold_cv(train_data, v = 5)

# Perform hyperparameter tuning
set.seed(123)
svm_res <- tune_grid(
  svm_workflow,
  resamples = folds,
  grid = 10,
  metrics = metric_set(accuracy) # Focus on accuracy during tuning
)

# Finalize the workflow with the best parameters
best_params <- select_best(svm_res, "accuracy")
final_svm <- finalize_workflow(svm_workflow, best_params)

# Fit the model with training data
set.seed(123)
m_svm <- fit(final_svm, data = train_data)

# Make predictions on the testing set
predict_svm <- predict(m_svm, test_data_processed) |>
  bind_cols(test_data_processed |> select(genre)) |>
  rename(predicted = .pred_class, actual = genre)

# Convert actual and predicted columns to factors as before
predict_svm <- predict_svm |> 
  mutate(
    actual = as.factor(actual),
    predicted = as.factor(predicted)
  )

# Evaluate model performance
class_metrics <- metric_set(accuracy, precision, recall, f_meas)
results <- predict_svm |>
  class_metrics(truth = actual, estimate = predicted)

# Display results
print(results)

```

