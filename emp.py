import streamlit as st
import numpy as np
import pickle
import pandas as pd
import plotly.express as px

# Load Models and Scalers
@st.cache_data
def load_attrition_model_and_scaler():
    model_path = r"E:\Emp_Attrition\new_attrition_decision_tree.pkl"
    scaler_path = r"E:\Emp_Attrition\attrition_scaler_new.pkl"
    with open(model_path, "rb") as file:
        attrition_model = pickle.load(file)
    with open(scaler_path, "rb") as file:
        attrition_scaler = pickle.load(file)
    return attrition_model, attrition_scaler

# Set page layout
st.set_page_config(page_title="Employee Attrition App", layout="wide")

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📋 Employee Details", "📊 Visual Analysis","🔮 Attrition Prediction", "📤 Download Report"])

# --- Home Page ---
if page == "🏠 Home":
    st.markdown("<h2 style='text-align: center;'>📊 Employee Attrition Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image("https://www.shutterstock.com/image-vector/attrition-transparent-icon-symbol-design-260nw-1197311194.jpg", width=225)

    with col2:
        st.markdown("""
        ### 🔍 Key Features:
        - ✅ *Predict Attrition Risk*  
          Use employee details to forecast potential attrition using machine learning.

        - 📈 *Data-Driven Insights*  
          Visualize patterns in departments, satisfaction levels, and more.

        - 💼 *Support HR Strategy*  
          Make smarter decisions in hiring, retention, and workforce planning.

        ---
        ### 🚀 Why Choose This Tool?
        - Real-world data implementation
        - AI-powered prediction engine
        - No coding required — just upload and explore!
        """)

# --- Employee Details Page ---
@st.cache_data
def load_csv_data(file):
    return pd.read_csv(file)

if page == "📋 Employee Details":
    st.title("📋 View Employee Data")
    uploaded_file = st.file_uploader("Upload your Employee CSV file", type=["csv"])

    if uploaded_file:
        df = load_csv_data(uploaded_file)

        st.subheader("📈 Key Stats")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Employees", df.shape[0])
        with col2:
            if "Attrition" in df.columns:
                attrited = df[df["Attrition"].str.lower() == "yes"].shape[0]
                st.metric("Attrited Employees", attrited)

        st.subheader("👥 Employee Dataset Preview")
        st.dataframe(df)
        st.session_state["employee_data"] = df
    else:
        st.info("📁 Upload your CSV file to proceed.")

# --- Attrition Prediction Page ---
elif page == "🔮 Attrition Prediction":
    model, scaler = load_attrition_model_and_scaler()
    st.title("🔮 Employee Attrition Prediction")
    st.write("Enter employee details to predict the likelihood of attrition.")

    age = st.number_input("Age", max_value=65, step=1, format="%d")
    monthly_income = st.number_input("Monthly Income", step=500, format="%d")
    years_at_company = st.number_input("Years at Company", max_value=40, step=1, format="%d")
    job_satisfaction = st.number_input("Job Satisfaction (1 to 4)", max_value=4, step=1, format="%d")

    department = st.selectbox("Department", ["Sales", "HR", "R&D"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    overtime = st.selectbox("Overtime", ["Yes", "No"])

    department_encoded = [1, 0, 0] if department == "HR" else [0, 1, 0] if department == "R&D" else [0, 0, 1]
    marital_status_encoded = [0, 0, 1] if marital_status == "Single" else [0, 1, 0] if marital_status == "Married" else [1, 0, 0]
    overtime_encoded = [1, 0] if overtime == "No" else [0, 1]

    user_input = np.array([[age, monthly_income, job_satisfaction, years_at_company] + 
                           department_encoded + marital_status_encoded + overtime_encoded], dtype=float)

    if st.button("Predict Attrition"):
        try:
            user_input[:, :4] = scaler.transform(user_input[:, :4])
            prediction = model.predict(user_input)[0]
            if prediction == 1:
                st.error("🚨 The employee is likely to leave.")
            else:
                st.success("✅ The employee is likely to stay.")
        except Exception as e:
            st.warning(f"⚠️ Error: {str(e)}")

# --- Visual Analysis Page ---
elif page == "📊 Visual Analysis":
    st.title("📋 Visualization Analysis")
    st.markdown("<p style='text-align: center;'>Track and understand why employees leave and how it affects your workforce.</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns(5)

    if "employee_data" in st.session_state:
        df = st.session_state["employee_data"]
        total = df.shape[0]
        inactive = df[df["Attrition"].str.lower() == "yes"].shape[0]
        active = total - inactive
        attrition_rate = round(inactive / total * 100, 2) if total > 0 else 0
        avg_age = int(df["Age"].mean()) if "Age" in df.columns else "N/A"
    else:
        total, inactive, active, attrition_rate, avg_age = 0, 0, 0, 0, "N/A"

    col1.metric("No of Employees", total)
    col2.metric("Inactive Employees", inactive)
    col3.metric("Active Employees", active)
    col4.metric("Attrition Rates", f"{attrition_rate}%")
    col5.metric("Average Age", avg_age)

    st.markdown("---")

    if "employee_data" in st.session_state:
        df = st.session_state["employee_data"]

        st.markdown("### 📊 Attrition Insights")
        col1, col2, col3 = st.columns(3)

        with col1:
            if "Age" in df.columns:
                age_bins = pd.cut(df["Age"], bins=[17,27,37,47,57,67], include_lowest=True)
                age_attrition = df[df["Attrition"].str.lower() == "yes"].groupby(age_bins).size()
                fig = px.bar(x=age_attrition.index.astype(str), y=age_attrition.values, labels={'x':'Age Range','y':'Attritions'}, title="Attrition by Age")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "Gender" in df.columns:
                gender_attr = df[df["Attrition"].str.lower() == "yes"]["Gender"].value_counts()
                fig = px.pie(values=gender_attr.values, names=gender_attr.index, title="Attrition by Gender")
                st.plotly_chart(fig, use_container_width=True)

        with col3:
            if "Department" in df.columns:
                dept_attr = df[df["Attrition"].str.lower() == "yes"]["Department"].value_counts()
                fig = px.bar(x=dept_attr.index, y=dept_attr.values, labels={'x':'Department','y':'Attritions'}, title="Attrition by Department")
                st.plotly_chart(fig, use_container_width=True)
    
    
    st.markdown("""
    <div style='text-align: center; font-size:20px;'>
        <ul style='list-style-position: inside;'>
            <li>📈 Attrition peaks at age 27–37.</li>
            <li>🚺 Female employees show higher attrition.</li>
            <li>🏢 Sales and HR departments are more affected.</li>
            <li>📊 Mid-level roles and lower education groups face more exits.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- Download Report Page ---
elif page == "📤 Download Report":
    st.title("📤 Export Filtered Data")
    if "employee_data" in st.session_state:
        df = st.session_state["employee_data"]

        st.subheader("Filter Data Before Download")
        attrition_filter = st.selectbox("Filter by Attrition", options=["All", "Yes", "No"])

        if attrition_filter != "All":
            filtered_df = df[df["Attrition"].str.lower() == attrition_filter.lower()]
        else:
            filtered_df = df

        st.dataframe(filtered_df)

        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Report as CSV",
            data=csv,
            file_name="filtered_employee_data.csv",
            mime='text/csv'
        )
    else:
        st.info("Please upload employee data first from the 'Employee Details' page.")
