# Employee Attrition Prediction

## 📌 Project Overview

Employee attrition is a major business challenge because employee turnover can increase recruitment costs, training expenses, and productivity losses.

This project analyzes employee data to identify factors associated with employee attrition and develops machine learning models to predict whether an employee is likely to leave the organization.

The project combines:

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning
- Model Evaluation
- Feature Importance Analysis
- Business Insights
- Employee Retention Recommendations

---

## 🎯 Objectives

The main objectives of this project are:

1. Analyze employee demographics and workplace characteristics.
2. Identify factors associated with employee attrition.
3. Analyze department and job-role-specific attrition.
4. Investigate compensation and salary-related factors.
5. Analyze employee satisfaction and work-life balance.
6. Analyze experience, tenure, promotion and career-growth factors.
7. Engineer meaningful features for machine learning.
8. Build and compare multiple classification models.
9. Evaluate models using appropriate classification metrics.
10. Identify important predictors of employee attrition.
11. Provide actionable business recommendations for employee retention.

---

## 📊 Dataset

The project uses the IBM HR Analytics Employee Attrition & Performance dataset.

Dataset characteristics:

- 1,470 employee records
- 35 original columns
- 31 useful columns after data cleaning
- Target variable: `Attrition`

### Target Variable

| Value | Meaning |
|---|---|
| Yes | Employee left the organization |
| No | Employee stayed |

The original dataset contains:

- Employee demographics
- Department
- Job role
- Compensation
- Business travel
- Job satisfaction
- Environment satisfaction
- Work-life balance
- Overtime
- Performance
- Training
- Career progression
- Tenure and experience

---

## 🧹 Data Cleaning

The following checks were performed:

- Missing-value analysis
- Duplicate-row analysis
- Data-type inspection
- Unique-value analysis
- Constant-column detection

The following non-informative columns were removed:

- `EmployeeCount`
- `Over18`
- `StandardHours`
- `EmployeeNumber`

After cleaning:

- Rows: 1,470
- Useful columns: 31
- Missing values: 0
- Duplicate rows: 0

---

## 🔍 Exploratory Data Analysis

The project analyzes several categories of employee characteristics.

### Demographics

- Age
- Gender
- Education
- Marital Status
- Business Travel
- Distance From Home

### Department & Job Role

- Department-level attrition
- Job-role-level attrition
- Attrition rate by job role

### Compensation

- Monthly Income
- Daily Rate
- Hourly Rate
- Monthly Rate
- Percent Salary Hike
- Stock Option Level

### Employee Satisfaction

- Job Satisfaction
- Environment Satisfaction
- Relationship Satisfaction
- Work-Life Balance
- Job Involvement
- Overtime

### Experience & Career Growth

- Total Working Years
- Years at Company
- Years in Current Role
- Years Since Last Promotion
- Years With Current Manager
- Job Level
- Number of Companies Worked

### Performance & Development

- Performance Rating
- Training Times Last Year
- Career progression indicators

---

## ⚙️ Feature Engineering

Additional features were created to improve analysis and machine learning:

- `AgeGroup`
- `IncomeGroup`
- `TenureGroup`
- `PromotionGap`
- `CurrentRoleTenureRatio`
- `CompaniesWorkedRatio`
- `EarlyCareer`
- `LongCommute`

These features provide additional representations of employee age, compensation, tenure, career progression and commuting patterns.

---

## 🤖 Machine Learning

The target variable was converted into binary form:

```text
Yes → 1
No  → 0