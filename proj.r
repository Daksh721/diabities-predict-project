#Load Libraries
library(tidyverse)
library(caret)
library(pROC)
library(corrplot)
library(shiny)

#Load and Preview the Data
df <- read.csv("C:/Users/ddaak/Desktop/code/r/diabetes.csv")

cat("\n========================================\n")
cat("          DATASET SNIPPET (Top 6)         \n")
cat("========================================\n")
print(head(df))

cat("\n========================================\n")
cat("         DATASET STRUCTURE (Data Types)   \n")
cat("========================================\n")
str(df)

cat("\n========================================\n")
cat("      STATISTICAL SUMMARY (Min, Max, Mean)\n")
cat("========================================\n")
print(summary(df))

#Clean the Data & Plot Correlation Heatmap
cols_to_clean <- c("Glucose", "BloodPressure", "BMI")
df[cols_to_clean] <- lapply(df[cols_to_clean], function(x) ifelse(x == 0, NA, x))
df <- df %>% mutate(across(everything(), ~ifelse(is.na(.), median(., na.rm = TRUE), .)))

df$Outcome <- as.factor(df$Outcome)

cat("\n========================================\n")
cat("       GENERATING CORRELATION HEATMAP     \n")
cat("========================================\n")
numeric_data <- df %>% select(-Outcome) %>% mutate(across(everything(), as.numeric))
cor_matrix <- cor(numeric_data, use = "complete.obs")

corrplot(cor_matrix, method = "color", type = "upper",
         tl.col = "black", tl.srt = 45, tl.cex = 0.85,
         addCoef.col = "black", number.cex = 0.55,
         title = "Feature Correlation Matrix", mar = c(0, 0, 2, 0))

#Split the Data
set.seed(123)
train_index <- createDataPartition(df$Outcome, p = 0.8, list = FALSE)
train_data <- df[train_index, ]
test_data <- df[-train_index, ]

#Train the Logistic Regression Model
log_model <- glm(Outcome ~ ., data = train_data, family = "binomial")

cat("\n========================================\n")
cat("      MODEL SUMMARY (Feature Importance)  \n")
cat("========================================\n")
print(summary(log_model))

#Evaluate the Model
prob_predictions <- predict(log_model, newdata = test_data, type = "response")
class_predictions <- as.factor(ifelse(prob_predictions > 0.5, 1, 0))

cat("\n========================================\n")
cat("           CONFUSION MATRIX               \n")
cat("========================================\n")
conf_matrix <- confusionMatrix(class_predictions, test_data$Outcome)
print(conf_matrix)

roc_curve <- roc(test_data$Outcome, prob_predictions, quiet = TRUE)
plot(roc_curve, main = "ROC Curve (Logistic Regression)", col = "darkred", lwd = 2)

auc_val <- auc(roc_curve)
cat("\n========================================\n")
cat(sprintf("FINAL MODEL AUC: %.4f\n", auc_val))
cat("========================================\n")

#Simple Front End for User Predictions

ui <- fluidPage(
  titlePanel("Hospital Diabetes Risk Predictor (Logistic Regression)"),
  sidebarLayout(
    sidebarPanel(
      h4("Enter Patient Metrics"),
      numericInput("pregnancies", "Pregnancies", value = 1, min = 0, max = 20, step = 1),
      numericInput("glucose", "Glucose (mg/dL)", value = 120, min = 40, max = 250, step = 1),
      numericInput("bp", "Blood Pressure (mm Hg)", value = 72, min = 30, max = 140, step = 1),
      numericInput("skin", "Skin Thickness (mm)", value = 25, min = 5, max = 100, step = 1),
      numericInput("insulin", "Insulin (IU/mL)", value = 80, min = 0, max = 900, step = 1),
      numericInput("bmi", "BMI", value = 28.0, min = 10, max = 70, step = 0.1),
      numericInput("dpf", "Diabetes Pedigree Function", value = 0.45, min = 0.05, max = 3.0, step = 0.01),
      numericInput("age", "Age (years)", value = 35, min = 18, max = 100, step = 1),
      br(),
      actionButton("predict_btn", "Predict Risk", class = "btn-primary")
    ),
    mainPanel(
      h3("Prediction Result"),
      verbatimTextOutput("risk_output"),
      hr(),
      h4("Test Set Performance"),
      verbatimTextOutput("eval_output"),
      plotOutput("roc_plot")
    )
  )
)

server <- function(input, output) {
  prediction_text <- eventReactive(input$predict_btn, {
    req(input$glucose, input$bp, input$skin, input$insulin, input$bmi, input$dpf, input$age, input$pregnancies)
    
    new_patient <- data.frame(
      Pregnancies = input$pregnancies,
      Glucose = input$glucose,
      BloodPressure = input$bp,
      SkinThickness = input$skin,
      Insulin = input$insulin,
      BMI = input$bmi,
      DiabetesPedigreeFunction = input$dpf,
      Age = input$age
    )
    
    prob <- predict(log_model, newdata = new_patient, type = "response")
    pct <- round(prob * 100, 2)
    label <- ifelse(prob > 0.5, "Higher Risk (> 50% threshold)", "Lower Risk (<= 50% threshold)")
    
    paste0(
      "Estimated Chance of Diabetes: ", pct, "%\n",
      "Classification at 0.50 cutoff: ", label
    )
  }, ignoreNULL = FALSE)
  
  output$risk_output <- renderText({
    prediction_text()
  })
  
  output$eval_output <- renderText({
    acc <- round(conf_matrix$overall["Accuracy"] * 100, 2)
    sens <- round(conf_matrix$byClass["Sensitivity"] * 100, 2)
    spec <- round(conf_matrix$byClass["Specificity"] * 100, 2)
    paste0(
      "Test Accuracy: ", acc, "%\n",
      "Sensitivity (Recall for Positive): ", sens, "%\n",
      "Specificity (True Negative Rate): ", spec, "%\n",
      "ROC AUC: ", round(as.numeric(auc_val), 4)
    )
  })
  
  output$roc_plot <- renderPlot({
    plot(roc_curve, main = "ROC Curve (Test Set)", col = "darkred", lwd = 2)
  })
}

shinyApp(ui = ui, server = server)