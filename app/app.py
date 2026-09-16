import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Employee Attrition Analytics",
    page_icon="👥",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "models/best_employee_attrition_model.pkl"

ENGINEERED_DATA_PATH = (
    "data/employee_attrition_engineered.csv"
)

CLEANED_DATA_PATH = (
    "data/employee_attrition_cleaned.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

try:

    df = pd.read_csv(CLEANED_DATA_PATH)

except Exception:

    try:
        df = pd.read_csv(ENGINEERED_DATA_PATH)

    except Exception as e:

        st.error(
            f"Unable to load employee dataset: {e}"
        )

        st.stop()


# ============================================================
# LOAD ENGINEERED DATA FOR MODEL
# ============================================================

try:

    model_df = pd.read_csv(
        ENGINEERED_DATA_PATH
    )

except Exception as e:

    st.error(
        f"Unable to load engineered dataset: {e}"
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model_package = joblib.load(
        MODEL_PATH
    )

    if isinstance(model_package, dict):

        model = model_package["model"]

        preprocessor = model_package.get(
            "preprocessor"
        )

    else:

        model = model_package

        preprocessor = None

except Exception as e:

    st.error(
        f"Unable to load trained model: {e}"
    )

    st.stop()


# ============================================================
# DETERMINE MODEL FEATURES
# ============================================================

if (
    preprocessor is not None
    and hasattr(
        preprocessor,
        "feature_names_in_"
    )
):

    model_features = list(
        preprocessor.feature_names_in_
    )

else:

    model_features = [
        column
        for column in model_df.columns
        if column != "Attrition"
    ]


# ============================================================
# DETERMINE NUMERICAL / CATEGORICAL FEATURES
# ============================================================

numerical_features = []

categorical_features = []


if preprocessor is not None:

    try:

        for (
            transformer_name,
            transformer,
            columns
        ) in preprocessor.transformers_:

            if transformer_name == "remainder":
                continue

            if isinstance(
                columns,
                (list, tuple, np.ndarray)
            ):

                columns = list(columns)

            else:

                columns = [columns]

            transformer_text = str(
                transformer
            ).lower()

            if (
                "onehot" in transformer_text
                or "ordinal" in transformer_text
            ):

                categorical_features.extend(
                    columns
                )

            else:

                numerical_features.extend(
                    columns
                )

    except Exception:

        pass


if not numerical_features:

    numerical_features = [
        column
        for column in model_features
        if column in model_df.columns
        and pd.api.types.is_numeric_dtype(
            model_df[column]
        )
    ]


if not categorical_features:

    categorical_features = [
        column
        for column in model_features
        if column in model_df.columns
        and column not in numerical_features
    ]


# ============================================================
# HELPER FUNCTION
# ============================================================

def attrition_rate(data):

    if len(data) == 0:
        return 0

    if "Attrition" not in data.columns:
        return 0

    return (
        data["Attrition"]
        .eq("Yes")
        .mean()
        * 100
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "👥 Employee Attrition Analytics Dashboard"
)

st.markdown(
    """
    ### Workforce Intelligence & Attrition Risk Analysis

    Analyze employee demographics, job characteristics,
    compensation, satisfaction and workforce patterns while
    predicting individual employee attrition risk.
    """
)


# ============================================================
# NAVIGATION
# ============================================================

dashboard_tab, prediction_tab = st.tabs(
    [
        "📊 HR Analytics Dashboard",
        "🔮 Attrition Prediction"
    ]
)


# ============================================================
# ============================================================
# DASHBOARD
# ============================================================
# ============================================================

with dashboard_tab:

    # --------------------------------------------------------
    # KPI CALCULATIONS
    # --------------------------------------------------------

    total_employees = len(df)

    attrition_count = (
        df["Attrition"]
        .eq("Yes")
        .sum()
        if "Attrition" in df.columns
        else 0
    )

    active_employees = (
        df["Attrition"]
        .eq("No")
        .sum()
        if "Attrition" in df.columns
        else 0
    )

    overall_attrition = (
        attrition_count
        / total_employees
        * 100
        if total_employees > 0
        else 0
    )


    average_age = (
        df["Age"].mean()
        if "Age" in df.columns
        else 0
    )


    average_income = (
        df["MonthlyIncome"].mean()
        if "MonthlyIncome" in df.columns
        else 0
    )


    average_tenure = (
        df["YearsAtCompany"].mean()
        if "YearsAtCompany" in df.columns
        else 0
    )


    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    st.subheader(
        "📌 Workforce Overview"
    )

    k1, k2, k3, k4, k5, k6 = st.columns(6)


    with k1:

        st.metric(
            "Total Employees",
            f"{total_employees:,}"
        )


    with k2:

        st.metric(
            "Employees Left",
            f"{attrition_count:,}"
        )


    with k3:

        st.metric(
            "Active Employees",
            f"{active_employees:,}"
        )


    with k4:

        st.metric(
            "Attrition Rate",
            f"{overall_attrition:.2f}%"
        )


    with k5:

        st.metric(
            "Average Age",
            f"{average_age:.1f}"
        )


    with k6:

        st.metric(
            "Average Tenure",
            f"{average_tenure:.1f} yrs"
        )


    st.divider()


    # ========================================================
    # ATTRITION DISTRIBUTION
    # ========================================================

    st.subheader(
        "📊 Attrition Distribution"
    )


    if "Attrition" in df.columns:

        distribution = (
            df["Attrition"]
            .value_counts()
            .rename(
                index={
                    "No": "Stayed",
                    "Yes": "Left"
                }
            )
        )

        st.bar_chart(
            distribution
        )


    # ========================================================
    # DEPARTMENT + JOB ROLE
    # ========================================================

    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # DEPARTMENT
    # --------------------------------------------------------

    with col1:

        st.subheader(
            "🏢 Attrition by Department"
        )

        if (
            "Department" in df.columns
            and "Attrition" in df.columns
        ):

            department_data = (
                df.groupby("Department")[
                    "Attrition"
                ]
                .apply(
                    lambda x:
                    (x == "Yes").mean() * 100
                )
                .sort_values(
                    ascending=False
                )
            )

            st.bar_chart(
                department_data
            )


    # --------------------------------------------------------
    # JOB ROLE
    # --------------------------------------------------------

    with col2:

        st.subheader(
            "💼 Attrition by Job Role"
        )

        if (
            "JobRole" in df.columns
            and "Attrition" in df.columns
        ):

            role_data = (
                df.groupby("JobRole")[
                    "Attrition"
                ]
                .apply(
                    lambda x:
                    (x == "Yes").mean() * 100
                )
                .sort_values(
                    ascending=False
                )
            )

            st.bar_chart(
                role_data
            )


    st.divider()


    # ========================================================
    # OVERTIME + BUSINESS TRAVEL
    # ========================================================

    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # OVERTIME
    # --------------------------------------------------------

    with col1:

        st.subheader(
            "⏰ Attrition by Overtime"
        )

        if (
            "OverTime" in df.columns
            and "Attrition" in df.columns
        ):

            overtime_data = (
                df.groupby("OverTime")[
                    "Attrition"
                ]
                .apply(
                    lambda x:
                    (x == "Yes").mean() * 100
                )
                .sort_values(
                    ascending=False
                )
            )

            st.bar_chart(
                overtime_data
            )


    # --------------------------------------------------------
    # BUSINESS TRAVEL
    # --------------------------------------------------------

    with col2:

        st.subheader(
            "✈️ Attrition by Business Travel"
        )

        if (
            "BusinessTravel" in df.columns
            and "Attrition" in df.columns
        ):

            travel_data = (
                df.groupby("BusinessTravel")[
                    "Attrition"
                ]
                .apply(
                    lambda x:
                    (x == "Yes").mean() * 100
                )
                .sort_values(
                    ascending=False
                )
            )

            st.bar_chart(
                travel_data
            )


    st.divider()


    # ========================================================
    # MARITAL STATUS + GENDER
    # ========================================================

    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "💍 Attrition by Marital Status"
        )

        if (
            "MaritalStatus" in df.columns
            and "Attrition" in df.columns
        ):

            marital_data = (
                df.groupby("MaritalStatus")[
                    "Attrition"
                ]
                .apply(
                    lambda x:
                    (x == "Yes").mean() * 100
                )
                .sort_values(
                    ascending=False
                )
            )

            st.bar_chart(
                marital_data
            )


    with col2:

        st.subheader(
            "👤 Attrition by Gender"
        )

        if (
            "Gender" in df.columns
            and "Attrition" in df.columns
        ):

            gender_data = (
                df.groupby("Gender")[
                    "Attrition"
                ]
                .apply(
                    lambda x:
                    (x == "Yes").mean() * 100
                )
                .sort_values(
                    ascending=False
                )
            )

            st.bar_chart(
                gender_data
            )


    st.divider()


    # ========================================================
    # AGE ANALYSIS
    # ========================================================

    st.subheader(
        "🎂 Attrition by Age Group"
    )


    if "Age" in df.columns:

        age_data = df.copy()

        age_data["Age Group"] = pd.cut(
            age_data["Age"],
            bins=[
                17,
                25,
                35,
                45,
                60
            ],
            labels=[
                "18–25",
                "26–35",
                "36–45",
                "46–60"
            ]
        )


        age_attrition = (
            age_data.groupby(
                "Age Group",
                observed=True
            )["Attrition"]
            .apply(
                lambda x:
                (x == "Yes").mean() * 100
            )
        )


        st.bar_chart(
            age_attrition
        )


    st.divider()


    # ========================================================
    # INCOME ANALYSIS
    # ========================================================

    st.subheader(
        "💰 Attrition by Income Group"
    )


    if "MonthlyIncome" in df.columns:

        income_data = df.copy()

        income_data["Income Group"] = pd.cut(
            income_data["MonthlyIncome"],
            bins=[
                0,
                3000,
                6000,
                10000,
                100000
            ],
            labels=[
                "Low",
                "Medium",
                "High",
                "Very High"
            ]
        )


        income_attrition = (
            income_data.groupby(
                "Income Group",
                observed=True
            )["Attrition"]
            .apply(
                lambda x:
                (x == "Yes").mean() * 100
            )
        )


        st.bar_chart(
            income_attrition
        )


    st.divider()


    # ========================================================
    # SATISFACTION ANALYSIS
    # ========================================================

    st.subheader(
        "⭐ Attrition by Job Satisfaction"
    )


    if (
        "JobSatisfaction" in df.columns
        and "Attrition" in df.columns
    ):

        satisfaction_data = (
            df.groupby("JobSatisfaction")[
                "Attrition"
            ]
            .apply(
                lambda x:
                (x == "Yes").mean() * 100
            )
        )


        satisfaction_data.index = [
            f"Level {int(x)}"
            for x in satisfaction_data.index
        ]


        st.bar_chart(
            satisfaction_data
        )


    st.divider()


    # ========================================================
    # WORK-LIFE BALANCE
    # ========================================================

    st.subheader(
        "⚖️ Attrition by Work-Life Balance"
    )


    if (
        "WorkLifeBalance" in df.columns
        and "Attrition" in df.columns
    ):

        balance_data = (
            df.groupby("WorkLifeBalance")[
                "Attrition"
            ]
            .apply(
                lambda x:
                (x == "Yes").mean() * 100
            )
        )


        balance_data.index = [
            f"Level {int(x)}"
            for x in balance_data.index
        ]


        st.bar_chart(
            balance_data
        )


    st.divider()


    # ========================================================
    # DISTANCE FROM HOME
    # ========================================================

    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "🚗 Distance From Home"
        )

        if "DistanceFromHome" in df.columns:

            distance_data = df.copy()

            distance_data[
                "Commute Group"
            ] = pd.cut(
                distance_data[
                    "DistanceFromHome"
                ],
                bins=[
                    0,
                    5,
                    10,
                    100
                ],
                labels=[
                    "0–5 km",
                    "6–10 km",
                    "10+ km"
                ]
            )


            commute_attrition = (
                distance_data.groupby(
                    "Commute Group",
                    observed=True
                )["Attrition"]
                .apply(
                    lambda x:
                    (x == "Yes").mean() * 100
                )
            )


            st.bar_chart(
                commute_attrition
            )


    # --------------------------------------------------------
    # JOB LEVEL
    # --------------------------------------------------------

    with col2:

        st.subheader(
            "📈 Attrition by Job Level"
        )

        if "JobLevel" in df.columns:

            level_attrition = (
                df.groupby("JobLevel")[
                    "Attrition"
                ]
                .apply(
                    lambda x:
                    (x == "Yes").mean() * 100
                )
            )


            level_attrition.index = [
                f"Level {int(x)}"
                for x in level_attrition.index
            ]


            st.bar_chart(
                level_attrition
            )


    st.divider()


    # ========================================================
    # KEY BUSINESS INSIGHTS
    # ========================================================

    st.subheader(
        "💡 Key HR Insights"
    )


    insight_col1, insight_col2 = (
        st.columns(2)
    )


    # --------------------------------------------------------
    # CALCULATE INSIGHTS
    # --------------------------------------------------------

    if (
        "Department" in df.columns
        and "Attrition" in df.columns
    ):

        department_rates = (
            df.groupby("Department")[
                "Attrition"
            ]
            .apply(
                lambda x:
                (x == "Yes").mean() * 100
            )
        )

        highest_department = (
            department_rates.idxmax()
        )

        highest_department_rate = (
            department_rates.max()
        )

    else:

        highest_department = "N/A"

        highest_department_rate = 0


    if (
        "BusinessTravel" in df.columns
        and "Attrition" in df.columns
    ):

        travel_rates = (
            df.groupby("BusinessTravel")[
                "Attrition"
            ]
            .apply(
                lambda x:
                (x == "Yes").mean() * 100
            )
        )

        highest_travel = (
            travel_rates.idxmax()
        )

        highest_travel_rate = (
            travel_rates.max()
        )

    else:

        highest_travel = "N/A"

        highest_travel_rate = 0


    if (
        "OverTime" in df.columns
        and "Attrition" in df.columns
    ):

        overtime_rates = (
            df.groupby("OverTime")[
                "Attrition"
            ]
            .apply(
                lambda x:
                (x == "Yes").mean() * 100
            )
        )

        highest_overtime = (
            overtime_rates.idxmax()
        )

        highest_overtime_rate = (
            overtime_rates.max()
        )

    else:

        highest_overtime = "N/A"

        highest_overtime_rate = 0


    with insight_col1:

        st.info(
            f"""
            **Overall Attrition**

            {overall_attrition:.2f}% of employees
            in the dataset have left the organization.

            **Highest-risk department:**
            {highest_department}
            ({highest_department_rate:.2f}%)
            """
        )


        st.warning(
            f"""
            **Business Travel**

            The highest attrition rate is associated
            with **{highest_travel}**.

            Attrition rate:
            **{highest_travel_rate:.2f}%**
            """
        )


    with insight_col2:

        st.error(
            f"""
            **Overtime**

            The highest attrition rate is associated
            with employees working:
            **{highest_overtime} overtime**.

            Attrition rate:
            **{highest_overtime_rate:.2f}%**
            """
        )


        st.success(
            """
            **HR Recommendation**

            Organizations can reduce attrition risk by
            focusing on workload management, employee
            engagement, career progression, compensation,
            commute-related concerns and work-life balance.
            """
        )


    st.divider()


    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    st.subheader(
        "📋 Dataset Summary"
    )


    summary_col1, summary_col2, summary_col3 = (
        st.columns(3)
    )


    with summary_col1:

        st.metric(
            "Records",
            f"{len(df):,}"
        )


    with summary_col2:

        st.metric(
            "Features",
            f"{df.shape[1]:,}"
        )


    with summary_col3:

        st.metric(
            "Attrition Rate",
            f"{overall_attrition:.2f}%"
        )


# ============================================================
# ============================================================
# PREDICTION TAB
# ============================================================
# ============================================================

with prediction_tab:

    st.title(
        "🔮 Employee Attrition Prediction"
    )

    st.write(
        """
        Enter employee information below to estimate
        the probability of employee attrition.
        """
    )


    # ========================================================
    # SIDEBAR-LIKE INPUT AREA
    # ========================================================

    st.subheader(
        "Employee Information"
    )


    # --------------------------------------------------------
    # DEMOGRAPHICS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        age = st.slider(
            "Age",
            18,
            60,
            35
        )


        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female"
            ]
        )


        marital_status = st.selectbox(
            "Marital Status",
            [
                "Single",
                "Married",
                "Divorced"
            ]
        )


        education = st.selectbox(
            "Education",
            [1, 2, 3, 4, 5]
        )


    with col2:

        department = st.selectbox(
            "Department",
            [
                "Sales",
                "Research & Development",
                "Human Resources"
            ]
        )


        job_role = st.selectbox(
            "Job Role",
            [
                "Sales Executive",
                "Research Scientist",
                "Laboratory Technician",
                "Manufacturing Director",
                "Healthcare Representative",
                "Manager",
                "Sales Representative",
                "Research Director",
                "Human Resources"
            ]
        )


        job_level = st.slider(
            "Job Level",
            1,
            5,
            2
        )


        business_travel = st.selectbox(
            "Business Travel",
            [
                "Travel_Rarely",
                "Travel_Frequently",
                "Non-Travel"
            ]
        )


    with col3:

        overtime = st.selectbox(
            "Overtime",
            [
                "No",
                "Yes"
            ]
        )


        monthly_income = st.number_input(
            "Monthly Income",
            min_value=1000,
            max_value=50000,
            value=5000,
            step=500
        )


        distance_from_home = st.slider(
            "Distance From Home",
            1,
            30,
            5
        )


        stock_option_level = st.slider(
            "Stock Option Level",
            0,
            3,
            1
        )


    st.divider()


    # ========================================================
    # SATISFACTION + EXPERIENCE
    # ========================================================

    st.subheader(
        "Satisfaction & Experience"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        job_satisfaction = st.slider(
            "Job Satisfaction",
            1,
            4,
            3
        )


        environment_satisfaction = st.slider(
            "Environment Satisfaction",
            1,
            4,
            3
        )


        relationship_satisfaction = st.slider(
            "Relationship Satisfaction",
            1,
            4,
            3
        )


        work_life_balance = st.slider(
            "Work-Life Balance",
            1,
            4,
            3
        )


    with col2:

        job_involvement = st.slider(
            "Job Involvement",
            1,
            4,
            3
        )


        years_at_company = st.slider(
            "Years at Company",
            0,
            40,
            5
        )


        years_in_current_role = st.slider(
            "Years in Current Role",
            0,
            20,
            3
        )


        years_since_last_promotion = st.slider(
            "Years Since Last Promotion",
            0,
            15,
            2
        )


    with col3:

        years_with_current_manager = st.slider(
            "Years With Current Manager",
            0,
            20,
            3
        )


        total_working_years = st.slider(
            "Total Working Years",
            0,
            40,
            10
        )


        num_companies_worked = st.slider(
            "Number of Companies Worked",
            0,
            10,
            2
        )


        training_times_last_year = st.slider(
            "Training Times Last Year",
            0,
            10,
            3
        )


    # ========================================================
    # ADDITIONAL FEATURES
    # ========================================================

    st.divider()

    st.subheader(
        "Additional Employee Information"
    )


    col1, col2 = st.columns(2)


    with col1:

        education_field = st.selectbox(
            "Education Field",
            [
                "Life Sciences",
                "Medical",
                "Marketing",
                "Technical Degree",
                "Human Resources",
                "Other"
            ]
        )


    with col2:

        percent_salary_hike = st.slider(
            "Percent Salary Hike",
            10,
            30,
            15
        )


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    predict_button = st.button(
        "🔮 Predict Employee Attrition",
        use_container_width=True
    )


    # ========================================================
    # RUN PREDICTION
    # ========================================================

    if predict_button:

        try:

            # ------------------------------------------------
            # START WITH REAL DATASET ROW
            # ------------------------------------------------

            input_data = model_df.iloc[
                [0]
            ].copy()


            # ------------------------------------------------
            # REMOVE TARGET
            # ------------------------------------------------

            if "Attrition" in input_data.columns:

                input_data = input_data.drop(
                    columns=["Attrition"]
                )


            # ------------------------------------------------
            # KEEP MODEL FEATURES
            # ------------------------------------------------

            input_data = input_data[
                [
                    feature
                    for feature in model_features
                    if feature in input_data.columns
                ]
            ].copy()


            # ------------------------------------------------
            # USER VALUES
            # ------------------------------------------------

            user_values = {

                "Age":
                    age,

                "Gender":
                    gender,

                "MaritalStatus":
                    marital_status,

                "Education":
                    education,

                "EducationField":
                    education_field,

                "Department":
                    department,

                "JobRole":
                    job_role,

                "JobLevel":
                    job_level,

                "BusinessTravel":
                    business_travel,

                "OverTime":
                    overtime,

                "MonthlyIncome":
                    monthly_income,

                "PercentSalaryHike":
                    percent_salary_hike,

                "StockOptionLevel":
                    stock_option_level,

                "JobSatisfaction":
                    job_satisfaction,

                "EnvironmentSatisfaction":
                    environment_satisfaction,

                "RelationshipSatisfaction":
                    relationship_satisfaction,

                "WorkLifeBalance":
                    work_life_balance,

                "JobInvolvement":
                    job_involvement,

                "YearsAtCompany":
                    years_at_company,

                "YearsInCurrentRole":
                    years_in_current_role,

                "YearsSinceLastPromotion":
                    years_since_last_promotion,

                "YearsWithCurrManager":
                    years_with_current_manager,

                "TotalWorkingYears":
                    total_working_years,

                "NumCompaniesWorked":
                    num_companies_worked,

                "TrainingTimesLastYear":
                    training_times_last_year,

                "DistanceFromHome":
                    distance_from_home
            }


            # ------------------------------------------------
            # APPLY USER VALUES
            # ------------------------------------------------

            for column, value in user_values.items():

                if column in input_data.columns:

                    input_data[column] = value


            # ------------------------------------------------
            # ENGINEERED FEATURES
            # ------------------------------------------------

            if "AgeGroup" in input_data.columns:

                if age <= 25:

                    input_data["AgeGroup"] = (
                        "Early Career"
                    )

                elif age <= 35:

                    input_data["AgeGroup"] = (
                        "Mid Career"
                    )

                elif age <= 45:

                    input_data["AgeGroup"] = (
                        "Experienced"
                    )

                else:

                    input_data["AgeGroup"] = (
                        "Senior"
                    )


            if "IncomeGroup" in input_data.columns:

                if monthly_income <= 3000:

                    input_data["IncomeGroup"] = (
                        "Low"
                    )

                elif monthly_income <= 6000:

                    input_data["IncomeGroup"] = (
                        "Medium"
                    )

                elif monthly_income <= 10000:

                    input_data["IncomeGroup"] = (
                        "High"
                    )

                else:

                    input_data["IncomeGroup"] = (
                        "Very High"
                    )


            if "TenureGroup" in input_data.columns:

                if years_at_company <= 2:

                    input_data["TenureGroup"] = (
                        "New"
                    )

                elif years_at_company <= 5:

                    input_data["TenureGroup"] = (
                        "Established"
                    )

                elif years_at_company <= 10:

                    input_data["TenureGroup"] = (
                        "Experienced"
                    )

                else:

                    input_data["TenureGroup"] = (
                        "Long Tenure"
                    )


            if "PromotionGap" in input_data.columns:

                input_data[
                    "PromotionGap"
                ] = (
                    years_since_last_promotion
                    /
                    (years_at_company + 1)
                )


            if "CurrentRoleTenureRatio" in input_data.columns:

                input_data[
                    "CurrentRoleTenureRatio"
                ] = (
                    years_in_current_role
                    /
                    (years_at_company + 1)
                )


            if "CompaniesWorkedRatio" in input_data.columns:

                input_data[
                    "CompaniesWorkedRatio"
                ] = (
                    num_companies_worked
                    /
                    (age + 1)
                )


            if "EarlyCareer" in input_data.columns:

                input_data[
                    "EarlyCareer"
                ] = int(
                    age < 30
                )


            if "LongCommute" in input_data.columns:

                input_data[
                    "LongCommute"
                ] = int(
                    distance_from_home > 10
                )


            # ------------------------------------------------
            # NUMERICAL TYPES
            # ------------------------------------------------

            for column in numerical_features:

                if column in input_data.columns:

                    input_data[column] = (
                        pd.to_numeric(
                            input_data[column],
                            errors="coerce"
                        )
                    )


            # ------------------------------------------------
            # CATEGORICAL TYPES
            # ------------------------------------------------

            for column in categorical_features:

                if column in input_data.columns:

                    input_data[column] = (
                        input_data[column]
                        .astype(str)
                    )


            # ------------------------------------------------
            # FILL NUMERICAL NaN
            # ------------------------------------------------

            for column in numerical_features:

                if column in input_data.columns:

                    if input_data[column].isna().any():

                        median_value = (
                            pd.to_numeric(
                                model_df[column],
                                errors="coerce"
                            ).median()
                        )

                        input_data[column] = (
                            input_data[column]
                            .fillna(median_value)
                        )


            # ------------------------------------------------
            # EXACT COLUMN ORDER
            # ------------------------------------------------

            input_data = input_data[
                model_features
            ]


            # ------------------------------------------------
            # PREPROCESS
            # ------------------------------------------------

            if preprocessor is not None:

                transformed_data = (
                    preprocessor.transform(
                        input_data
                    )
                )

            else:

                transformed_data = (
                    input_data
                )


            # ------------------------------------------------
            # PREDICT
            # ------------------------------------------------

            prediction = model.predict(
                transformed_data
            )[0]


            # ------------------------------------------------
            # PROBABILITY
            # ------------------------------------------------

            if hasattr(
                model,
                "predict_proba"
            ):

                probability = (
                    model.predict_proba(
                        transformed_data
                    )[0][1]
                )

            else:

                probability = float(
                    prediction
                )


            probability_percentage = (
                probability * 100
            )


            # =================================================
            # RISK LEVEL
            # =================================================

            if probability < 0.30:

                risk_level = "Low"

            elif probability < 0.60:

                risk_level = "Medium"

            else:

                risk_level = "High"


            # =================================================
            # PREDICTION LABEL
            # =================================================

            if prediction == 1:

                prediction_text = (
                    "Likely to Leave"
                )

            else:

                prediction_text = (
                    "Likely to Stay"
                )


            # =================================================
            # RESULT
            # =================================================

            st.divider()

            st.subheader(
                "🎯 Prediction Result"
            )


            result1, result2, result3 = (
                st.columns(3)
            )


            with result1:

                st.metric(
                    "Attrition Probability",
                    f"{probability_percentage:.2f}%"
                )


            with result2:

                st.metric(
                    "Risk Level",
                    risk_level
                )


            with result3:

                st.metric(
                    "Prediction",
                    prediction_text
                )


            st.divider()


            # =================================================
            # RISK BAR
            # =================================================

            st.subheader(
                "📊 Attrition Risk"
            )


            st.progress(
                min(
                    int(probability_percentage),
                    100
                )
            )


            st.write(
                f"Estimated probability of attrition: "
                f"**{probability_percentage:.2f}%**"
            )


            # =================================================
            # RISK MESSAGE
            # =================================================

            if probability < 0.30:

                st.success(
                    "🟢 Lower Attrition Risk — "
                    "The employee has a relatively low "
                    "predicted probability of leaving."
                )

            elif probability < 0.60:

                st.warning(
                    "🟡 Moderate Attrition Risk — "
                    "The employee may require additional "
                    "engagement and retention attention."
                )

            else:

                st.error(
                    "🔴 Higher Attrition Risk — "
                    "The employee has a relatively high "
                    "predicted probability of leaving."
                )


            # =================================================
            # PROFILE
            # =================================================

            st.subheader(
                "👤 Employee Profile"
            )


            profile1, profile2 = (
                st.columns(2)
            )


            with profile1:

                st.write(
                    f"**Age:** {age}"
                )

                st.write(
                    f"**Department:** "
                    f"{department}"
                )

                st.write(
                    f"**Job Role:** "
                    f"{job_role}"
                )

                st.write(
                    f"**Monthly Income:** "
                    f"{monthly_income:,}"
                )

                st.write(
                    f"**Years at Company:** "
                    f"{years_at_company}"
                )


            with profile2:

                st.write(
                    f"**Overtime:** "
                    f"{overtime}"
                )

                st.write(
                    f"**Business Travel:** "
                    f"{business_travel}"
                )

                st.write(
                    f"**Job Satisfaction:** "
                    f"{job_satisfaction}"
                )

                st.write(
                    f"**Work-Life Balance:** "
                    f"{work_life_balance}"
                )

                st.write(
                    f"**Distance From Home:** "
                    f"{distance_from_home}"
                )


        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.error(
                f"Error details: {e}"
            )

            st.write(
                "Expected model features:"
            )

            st.write(
                model_features
            )

            st.write(
                "Input features:"
            )

            st.write(
                list(input_data.columns)
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Employee Attrition Prediction | "
    "Python • Pandas • NumPy • Scikit-learn • Streamlit"
)