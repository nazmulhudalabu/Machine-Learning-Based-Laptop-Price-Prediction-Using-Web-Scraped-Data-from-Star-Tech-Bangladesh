############################################################
# IDS Final Project (Star Tech Laptops)
# Plan: Scrape -> Understand -> EDA -> Preprocess -> Feature Select -> 2 Models + Validation -> Results
############################################################

# ==========================
# 0) Libraries
# ==========================
install.packages(c("rvest","dplyr","stringr","purrr","readr","janitor","ggplot2","scales","caret","ranger"))
library(rvest)
library(dplyr)
library(stringr)
library(purrr)
library(readr)
library(janitor)
library(ggplot2)
library(scales)
library(caret)
library(ranger)
install.packages("GGally")
library(GGally)

set.seed(123)

# ==========================
# 1) WEB SCRAPING (Star Tech) + Save fierst Raw CSV
# ==========================
base <- "https://www.startech.com.bd/laptop-notebook/laptop?page="

scrape_page <- function(page_num) {
  url <- paste0(base, page_num)
  cat("Scraping page:", url, "\n")
  
  page <- tryCatch(read_html(url), error = function(e) return(NULL))
  if (is.null(page)) return(tibble())
  
  cards <- page %>% html_elements(".p-item")
  
  tibble(
    title = cards %>% html_element("h4 a") %>% html_text2(),
    url   = cards %>% html_element("h4 a") %>% html_attr("href") %>%
      {ifelse(str_detect(., "^http"), ., paste0("https://www.startech.com.bd", .))},
    price_text = cards %>% html_element(".p-item-price, .price, .price-new") %>% html_text2(),
    specs = map_chr(cards, ~ .x %>% html_elements("ul li") %>% html_text2() %>% paste(collapse = " | "))
  ) %>%
    mutate(
      # IMPORTANT: Extract only the first price to avoid "185000195000" issue
      price_text_clean = str_replace_all(price_text, "[,৳]", "") %>% str_squish(),
      price = as.numeric(str_extract(price_text_clean, "^\\d+"))
    ) %>%
    filter(!is.na(title), !is.na(price))
}

all_data <- map_dfr(1:26, scrape_page) %>% clean_names()

# Feature Engineering from title/specs (basic)
all_data <- all_data %>%
  mutate(
    brand = word(title, 1),
    processor = str_extract(specs, "(?i)Processor\\s*:\\s*[^|]+") %>%
      str_remove("(?i)Processor\\s*:\\s*") %>% str_squish(),
    ram_gb = str_extract(specs, "(?i)RAM\\s*:\\s*[^|]+") %>%
      str_extract("(?i)(4|8|16|32|64|128)") %>% as.numeric(),
    storage_gb = case_when(
      str_detect(specs, "(?i)4\\s*TB") ~ 4096,
      str_detect(specs, "(?i)2\\s*TB") ~ 2048,
      str_detect(specs, "(?i)1\\s*TB") ~ 1024,
      str_detect(specs, "(?i)512\\s*GB") ~ 512,
      str_detect(specs, "(?i)256\\s*GB") ~ 256,
      str_detect(specs, "(?i)128\\s*GB") ~ 128,
      TRUE ~ NA_real_
    )
  )

write_csv(all_data, "startech_laptops_raw.csv")
cat("\nRaw dataset saved: startech_laptops_raw.csv\n")


# ==========================
# 2) DATA UNDERSTANDING (structure, types, size)
# ==========================
df <- read_csv("startech_laptops_raw.csv", show_col_types = FALSE)

# Dataset size (rows, columns)
dim(df)

# Data types / structure
str(df)

# Summary stats
summary(df)

# Missing values by column
colSums(is.na(df))

# Quick check: unique brands (inconsistency detection)
unique(df$brand)


# ==========================
# 3) EDA (visual + summary)
# ==========================

# 3.1 Brand count (before cleaning)
ggplot(df, aes(x = brand, fill = brand)) +
  geom_bar() +
  labs(title = "Brand Count (Before Cleaning)", x = "Brand", y = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# 3.2 Price distribution
ggplot(df, aes(x = price)) +
  geom_histogram(bins = 30, fill = "red") +
  labs(title = "Laptop Price Distribution (Raw/Parsed)", x = "Price (BDT)", y = "Count") +
  theme_minimal()


# 3.3 Outlier visualization (boxplot)
ggplot(df, aes(y = price)) +
  geom_boxplot(outlier.color = "blue",fill="red") +
  scale_y_continuous(labels = comma) +
  labs(title = "Laptop Price Boxplot (Outliers Visible)", y = "Price (BDT)") +
  theme_minimal()


# ==========================
# 4) DATA PREPROCESSING (cleaning, missing, engineering, encoding, scaling/transform)
# ==========================

# 4.1 Brand cleaning (fix inconsistent names)
df$brand[df$brand == "Asus"] <- "ASUS"
df$brand[df$brand == "MacBook"] <- "Apple"
unique(df$brand)

# 4.2 Ensure price is correct (safety re-parse)
df$price_text <- str_replace_all(df$price_text, "[,৳]", "") %>% str_squish()
df$price <- as.numeric(str_extract(df$price_text, "^\\d+"))
df$price

# Mode function for imputation
get_mode <- function(x) {
  ux <- na.omit(unique(x))
  ux[which.max(tabulate(match(x, ux)))]
}

# 4.3 Handle missing values (RAM/Storage: mode)
df$ram_gb <- ifelse(is.na(df$ram_gb), get_mode(df$ram_gb), df$ram_gb)
df$storage_gb <- ifelse(is.na(df$storage_gb), get_mode(df$storage_gb), df$storage_gb)

# 4.4 Feature engineering: CPU family (categorical from processor text)
df <- df %>%
  mutate(cpu_family = case_when(
    str_detect(processor, "(?i)intel") ~ "Intel",
    str_detect(processor, "(?i)amd|ryzen") ~ "AMD",
    str_detect(processor, "(?i)apple|m1|m2|m3") ~ "Apple",
    str_detect(processor, "(?i)qualcomm|snapdragon") ~ "Qualcomm",
    TRUE ~ "Other"
  ))

# 4.5 Outlier flagging (do NOT remove)
Q1 <- quantile(df$price, 0.25, na.rm = TRUE)
Q3 <- quantile(df$price, 0.75, na.rm = TRUE)
IQRv <- Q3 - Q1
upper_bound <- Q3 + 1.5 * IQRv
df$outlier_flag <- df$price > upper_bound
outliers <- df %>% filter(outlier_flag)
outliers$price

# 4.6 Target transformation (helps skewness): log(price)
df <- df %>% mutate(log_price = log(price))
summary(df)

# Save cleaned dataset
write_csv(df, "startech_laptops_clean.csv")
cat("\n✅ Clean dataset saved: startech_laptops_clean.csv\n")


# ==========================
# 5) EDA AFTER CLEANING (more informative visuals)
# ==========================

# 5.1 Brand count (after cleaning)
ggplot(df, aes(x = brand, fill = brand)) +
  geom_bar() +
  labs(title = "Brand Count (After Cleaning)", x = "Brand", y = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 5.2 Log-price distribution
ggplot(df, aes(x = log_price)) +
  geom_histogram(bins = 30,fill="blue",color="red") +
  labs(title = "Log(Price) Distribution", x = "log(Price)", y = "Count") +
  theme_minimal()

# 5.3 Brand vs Price
ggplot(df, aes(x = brand, y = price, fill = brand)) +
  geom_boxplot(outlier.alpha = 0.4) +
  scale_y_continuous(labels = comma) +
  labs(title = "Brand vs Price", x = "Brand", y = "Price (BDT)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 5.4 RAM vs Price

ggplot(df, aes(x = factor(ram_gb), y = price, fill = factor(ram_gb))) +
  geom_boxplot(outlier.alpha = 0.4) +
  scale_y_continuous(labels = scales::comma) +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = "Laptop Price Distribution by RAM Size",
    x = "RAM (GB)",
    y = "Price (BDT)",
    fill = "RAM (GB)"
  ) +
  theme_minimal() +
  theme(legend.position = "none")


# 5.5 Storage vs Price

  
  ggplot(df, aes(x = factor(storage_gb), y = price, fill = factor(storage_gb))) +
    geom_boxplot(outlier.alpha = 0.4) +
    scale_y_continuous(labels = scales::comma) +
    scale_fill_brewer(palette = "Set3") +
    labs(
      title = "Laptop Price Distribution by Storage Capacity",
      x = "Storage (GB)",
      y = "Price (BDT)",
      fill = "Storage (GB)"
    ) +
    theme_minimal() +
    theme(legend.position = "none")
  
  

# 5.6 CPU family vs Price
ggplot(df, aes(x = cpu_family, y = price, fill = cpu_family)) +
  geom_boxplot(outlier.alpha = 0.4) +
  scale_y_continuous(labels = comma) +
  labs(title = "CPU Family vs Price", x = "CPU Family", y = "Price (BDT)") +
  theme_minimal()



#==========================
#6.1 FEATURE ENGINEERING (Model-ready features)
#==========================
# Keep only candidate features + target
df_feat <- df %>%
  select(price, log_price, brand, cpu_family, ram_gb, storage_gb, outlier_flag)

# Remove extreme outliers ONLY for linear model stability (decision later)
df_feat_no_out <- df_feat %>% filter(outlier_flag == FALSE)


#6.2 FEATURE SELECTION (BEFORE MODELING)
#Decision: which columns to use
num_features <- df_feat_no_out %>%
  select(log_price, ram_gb, storage_gb)

cor(num_features, use = "complete.obs")
ggcorr(num_features, label = TRUE)


# ANOVA: Brand vs log_price
anova_brand <- aov(log_price ~ brand, data = df)
summary(anova_brand)
# ANOVA: CPU Family vs log_price
anova_cpu <- aov(log_price ~ cpu_family, data = df)
summary(anova_cpu)



# Selected features based on importance + domain logic
final_features <- c("log_price", "ram_gb", "storage_gb", "brand", "cpu_family")

df_model <- df_feat %>% select(all_of(final_features))

#Encoding 
df_model$brand <- as.factor(df_model$brand)
df_model$cpu_family <- as.factor(df_model$cpu_family)
#Scaling (ONLY for Linear Regression, not for RF)
preprocess_lm <- preProcess(
  df_model %>% select(ram_gb, storage_gb),
  method = c("center", "scale")
)

scaled_numeric <- predict(preprocess_lm,
                          df_model %>% select(ram_gb, storage_gb))

df_model_lm <- bind_cols(
  df_model %>% select(log_price, brand, cpu_family),
  scaled_numeric
)

#6.5 TRAIN / TEST SPLIT
set.seed(123)
idx <- createDataPartition(df_model$log_price, p = 0.80, list = FALSE)

train_lm <- df_model_lm[idx, ]
test_lm  <- df_model_lm[-idx, ]

train_rf <- df_model[idx, ]
test_rf  <- df_model[-idx, ]

#Cross-validation setup
ctrl <- trainControl(method = "cv", number = 5)


#Model 1: Linear Regression (scaled data, no extreme outliers)
lm_fit <- train(
  log_price ~ .,
  data = train_lm,
  method = "lm",
  trControl = ctrl
)

lm_pred <- predict(lm_fit, newdata = test_lm)
lm_perf <- postResample(lm_pred, test_lm$log_price)

#Model 2: Random Forest (raw scale, robust to outliers)
rf_fit <- train(
  log_price ~ .,
  data = train_rf,
  method = "ranger",
  trControl = ctrl,
  importance = "permutation"
)

rf_pred <- predict(rf_fit, newdata = test_rf)
rf_perf <- postResample(rf_pred, test_rf$log_price)


#6.7 MODEL COMPARISON (METRICS)
#==========================
results <- tibble(
  Model = c("Linear Regression", "Random Forest"),
  RMSE  = c(lm_perf["RMSE"], rf_perf["RMSE"]),
  Rsq   = c(lm_perf["Rsquared"], rf_perf["Rsquared"]),
  MAE   = c(lm_perf["MAE"], rf_perf["MAE"])
)

print(results)

#some plot 
#Random Forest: Predicted vs Actual (Log Price) plot
ggplot(data.frame(actual = test_rf$log_price, pred = rf_pred),
       aes(x = actual, y = pred)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Predicted vs Actual (Log Price) - Random Forest",
       x = "Actual log(price)", y = "Predicted log(price)") +
  theme_minimal()


#Linear Regression: Predicted vs Actual (Log Price) plot

ggplot(data.frame(actual = test_lm$log_price, pred = lm_pred),
       aes(x = actual, y = pred)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Predicted vs Actual (Log Price) - Linear Regression",
       x = "Actual log(price)", y = "Predicted log(price)") +
  theme_minimal()
varImp(rf_fit)



# ==========================
# FINAL CLEAN DATASET (For Submission)
# ==========================

final_clean_df <- df %>%
  select(
    title,
    url,
    price,
    log_price,
    brand,
    cpu_family,
    ram_gb,
    storage_gb,
    outlier_flag
  )

write_csv(final_clean_df, "startech_laptops_final_clean2.0.csv")

cat("\n✅ Final clean dataset saved: startech_laptops_final_clean.csv\n")

  
##end


