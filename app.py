import pandas as pd
import numpy as np
import streamlit as st
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Set page config
st.set_page_config(
    page_title="University Recommendation System",
    page_icon="🎓",
    layout="wide"
)

# Load and preprocess data
@st.cache_data
def load_data(file_path):
    df = pd.read_csv("ind_universities_sample_300.csv")
    
    # Clean column names
    df.columns = [col.strip().replace(' ', '_') for col in df.columns]
    
    # Handle missing values
    df['NIRF_Ranking_Overall'] = pd.to_numeric(df['NIRF_Ranking_Overall'], errors='coerce')
    df['NAAC_Score'] = pd.to_numeric(df['NAAC_Score'], errors='coerce')
    df['Placement_Rate_pct'] = pd.to_numeric(df['Placement_Rate_pct'], errors='coerce')
    df['Median_Package_INR'] = pd.to_numeric(df['Median_Package_INR'], errors='coerce')
    
    # Extract hostel capacity
    df['Hostel_Capacity_Boys'] = pd.to_numeric(df['Hostel_Capacity_Boys'], errors='coerce').fillna(0)
    df['Hostel_Capacity_Girls'] = pd.to_numeric(df['Hostel_Capacity_Girls'], errors='coerce').fillna(0)
    df['Total_Hostel_Capacity'] = df['Hostel_Capacity_Boys'] + df['Hostel_Capacity_Girls']
    
    # Extract infrastructure metrics
    df['Library_Books_Count'] = pd.to_numeric(df['Library_Books_Count'], errors='coerce').fillna(0)
    df['Labs_Count_Smart_Classrooms'] = pd.to_numeric(df['Labs_Count_Smart_Classrooms'], errors='coerce').fillna(0)
    
    # Create an infrastructure score
    df['Infra_Score'] = (df['Library_Books_Count'] / 10000) + (df['Labs_Count_Smart_Classrooms'] * 5) + (df['Total_Hostel_Capacity'] / 100)
    
    return df

# Recommendation function
def recommend_universities(df, preferences):
    filtered_df = df.copy()
    
    # Filter by program availability
    program_cols = {
        'UG': 'Number_of_Courses_UG',
        'PG': 'Number_of_Courses_PG',
        'Diploma': 'Number_of_Courses_Diploma',
        'PhD': 'Number_of_Courses_PhD'
    }
    
    if preferences['program'] != 'Any':
        program_col = program_cols[preferences['program']]
        filtered_df = filtered_df[filtered_df[program_col] > 0]
    
    # Filter by state preference
    if preferences['state'] != 'Any':
        filtered_df = filtered_df[filtered_df['State_UT'] == preferences['state']]
    
    # Filter by NAAC grade
    if preferences['naac_grade'] != 'Any':
        filtered_df = filtered_df[filtered_df['Accreditation_NAAC_Grade'] == preferences['naac_grade']]
    
    # Filter by hostel requirement
    if preferences['hostel']:
        filtered_df = filtered_df[filtered_df['Total_Hostel_Capacity'] > 0]
    
    # Filter by ranking preference
    if preferences['ranking'] == 'Top 100':
        filtered_df = filtered_df[filtered_df['NIRF_Ranking_Overall'] <= 100]
    elif preferences['ranking'] == 'Top 200':
        filtered_df = filtered_df[filtered_df['NIRF_Ranking_Overall'] <= 200]
    
    # If no universities match all criteria, return closest alternatives
    if len(filtered_df) == 0:
        st.warning("No universities match all your criteria. Showing closest alternatives:")
        filtered_df = df.copy()
        
        # Relax the constraints one by one
        if preferences['state'] != 'Any':
            filtered_df = filtered_df[filtered_df['State_UT'] == preferences['state']]
            if len(filtered_df) == 0:
                filtered_df = df.copy()
        
        if preferences['naac_grade'] != 'Any':
            naac_order = ['A++', 'A+', 'A', 'B++', 'B+', 'B', 'C']
            if preferences['naac_grade'] in naac_order:
                idx = naac_order.index(preferences['naac_grade'])
                possible_grades = naac_order[max(0, idx-1):min(len(naac_order), idx+2)]
                filtered_df = filtered_df[filtered_df['Accreditation_NAAC_Grade'].isin(possible_grades)]
    
    # Calculate score based on placement priority
    if preferences['placement_priority'] == 'High Placement Rate':
        filtered_df['Score'] = (filtered_df['Placement_Rate_pct'] * 0.5 + 
                               (100 - filtered_df['NIRF_Ranking_Overall'].fillna(200) / 2) * 0.3 +
                               filtered_df['Infra_Score'] * 0.2)
    elif preferences['placement_priority'] == 'High Median Package':
        filtered_df['Score'] = (filtered_df['Median_Package_INR'] / 100000 * 0.5 + 
                               (100 - filtered_df['NIRF_Ranking_Overall'].fillna(200) / 2) * 0.3 +
                               filtered_df['Infra_Score'] * 0.2)
    else:  # Recruiter Reputation
        # Simple heuristic: count of top recruiters mentioned
        filtered_df['Recruiter_Count'] = filtered_df['Top_Recruiters'].apply(
            lambda x: len(str(x).split(';')) if pd.notna(x) else 0
        )
        filtered_df['Score'] = (filtered_df['Recruiter_Count'] * 0.4 + 
                               (100 - filtered_df['NIRF_Ranking_Overall'].fillna(200) / 2) * 0.3 +
                               filtered_df['Infra_Score'] * 0.3)
    
    # Sort by score and return top 5
    recommended = filtered_df.sort_values('Score', ascending=False).head(5)
    
    return recommended

# Train prediction model
def train_prediction_model(df):
    # Prepare data for model training
    model_df = df.copy()
    
    # Select features and target
    features = ['NAAC_Score', 'NIRF_Ranking_Overall', 'Library_Books_Count', 
                'Labs_Count_Smart_Classrooms', 'Total_Hostel_Capacity']
    
    # Drop rows with missing target values
    model_df = model_df.dropna(subset=['Placement_Rate_pct', 'Median_Package_INR'])
    
    # Prepare features and targets
    X = model_df[features].fillna(0)
    y_rate = model_df['Placement_Rate_pct']
    y_package = model_df['Median_Package_INR']
    
    # Split data
    X_train, X_test, y_train_rate, y_test_rate = train_test_split(X, y_rate, test_size=0.2, random_state=42)
    _, _, y_train_package, y_test_package = train_test_split(X, y_package, test_size=0.2, random_state=42)
    
    # Create preprocessing pipeline
    numeric_features = features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])
    
    # Create and train models
    model_rate = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ])
    
    model_package = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ])
    
    # Train models
    model_rate.fit(X_train, y_train_rate)
    model_package.fit(X_train, y_train_package)
    
    return model_rate, model_package, features

# Predict placement metrics
def predict_placement(model_rate, model_package, features, input_data):
    # Create input DataFrame
    input_df = pd.DataFrame([input_data], columns=features)
    
    # Make predictions
    pred_rate = model_rate.predict(input_df)[0]
    pred_package = model_package.predict(input_df)[0]
    
    return pred_rate, pred_package

# Main app
def main():
    st.title("🎓 Indian University Recommendation & Prediction System")
    st.write("Find the best universities in India and predict placement outcomes")
    
    # Load data
    try:
        df = load_data("ind_universities_sample_300.csv")
    except:
        st.error("Could not load the data file. Please make sure 'ind_universities_sample_300.csv' is in the correct location.")
        return
    
    # Create tabs
    tab1, tab2 = st.tabs(["University Recommendation", "Placement Prediction"])
    
    with tab1:
        # Create form for user preferences
        with st.form("preferences_form"):
            st.subheader("Your Preferences")
            
            col1, col2 = st.columns(2)
            
            with col1:
                program = st.selectbox(
                    "Preferred Program",
                    options=['Any', 'UG', 'PG', 'Diploma', 'PhD']
                )
                
                state = st.selectbox(
                    "Preferred State/UT",
                    options=['Any'] + sorted(df['State_UT'].unique().tolist())
                )
                
                naac_grade = st.selectbox(
                    "Preferred NAAC Grade",
                    options=['Any', 'A++', 'A+', 'A', 'B++', 'B+', 'B', 'C']
                )
            
            with col2:
                ranking = st.selectbox(
                    "Ranking Preference",
                    options=['Any', 'Top 100', 'Top 200']
                )
                
                hostel = st.checkbox("Hostel Requirement", value=True)
                
                placement_priority = st.selectbox(
                    "Placement Priority",
                    options=['High Placement Rate', 'High Median Package', 'Recruiter Reputation']
                )
            
            submitted = st.form_submit_button("Get Recommendations")
        
        if submitted:
            preferences = {
                'program': program,
                'state': state,
                'naac_grade': naac_grade,
                'ranking': ranking,
                'hostel': hostel,
                'placement_priority': placement_priority
            }
            
            recommendations = recommend_universities(df, preferences)
            
            if len(recommendations) == 0:
                st.error("No universities found matching your criteria. Please try different preferences.")
            else:
                st.success(f"Found {len(recommendations)} universities matching your preferences")
                
                # Display recommendations as a table
                st.subheader("🏆 Top Recommended Universities")
                
                display_cols = [
                    'University_Name', 'Type', 'State_UT', 'City_District', 
                    'Accreditation_NAAC_Grade', 'NAAC_Score', 'NIRF_Ranking_Overall',
                    'Placement_Rate_pct', 'Median_Package_INR', 'Total_Hostel_Capacity'
                ]
                
                display_df = recommendations[display_cols].copy()
                display_df.columns = [
                    'University Name', 'Type', 'State/UT', 'City/District', 
                    'NAAC Grade', 'NAAC Score', 'NIRF Rank',
                    'Placement Rate (%)', 'Median Package (₹)', 'Hostel Capacity'
                ]
                
                # Format numeric columns
                display_df['NAAC Score'] = display_df['NAAC Score'].round(2)
                display_df['Placement Rate (%)'] = display_df['Placement Rate (%)'].round(1)
                display_df['Median Package (₹)'] = display_df['Median Package (₹)'].apply(
                    lambda x: f"₹{x:,.0f}" if pd.notna(x) else "N/A"
                )
                
                st.dataframe(display_df.reset_index(drop=True))
                
                # Display university cards
                st.subheader("🎯 University Details")
                
                for idx, row in recommendations.iterrows():
                    with st.expander(f"{row['University_Name']} - {row['City_District']}, {row['State_UT']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Type:** {row['Type']}")
                            st.write(f"**Established:** {row['Year_of_Establishment']}")
                            st.write(f"**NAAC Grade:** {row['Accreditation_NAAC_Grade']} ({row['NAAC_Score']})")
                            st.write(f"**NIRF Rank:** {row['NIRF_Ranking_Overall'] if pd.notna(row['NIRF_Ranking_Overall']) else 'N/A'}")
                        
                        with col2:
                            st.write(f"**Hostel Capacity:** {int(row['Total_Hostel_Capacity'])}")
                            st.write(f"**Placement Rate:** {row['Placement_Rate_pct']}%")
                            st.write(f"**Median Package:** ₹{row['Median_Package_INR']:,.0f}" if pd.notna(row['Median_Package_INR']) else "**Median Package:** N/A")
                        
                        # Parse BI_ME_FRP summary
                        if pd.notna(row['BI_ME_FRP']):
                            summary_parts = str(row['BI_ME_FRP']).split(';')
                            st.write("**Key Highlights:**")
                            for part in summary_parts:
                                if ':' in part:
                                    key, value = part.split(':', 1)
                                    st.write(f"- **{key.strip()}:** {value.strip()}")
                        
                        if pd.notna(row['Top_Recruiters']):
                            st.write(f"**Top Recruiters:** {row['Top_Recruiters']}")
    
    with tab2:
        st.subheader("Predict Placement Outcomes")
        st.write("Enter university details to predict placement rate and median package")
        
        # Train or load model
        try:
            model_rate, model_package, features = joblib.load('placement_model.pkl')
            st.success("Loaded pre-trained prediction model")
        except:
            with st.spinner("Training prediction model..."):
                model_rate, model_package, features = train_prediction_model(df)
                joblib.dump((model_rate, model_package, features), 'placement_model.pkl')
            st.success("Prediction model trained successfully")
        
        # Create input form for prediction
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                naac_score = st.slider("NAAC Score", 0.0, 4.0, 3.0, 0.1)
                nirf_rank = st.slider("NIRF Ranking (1=best, leave as 200 if unknown)", 1, 200, 100, 1)
            
            with col2:
                library_books = st.slider("Library Books Count (in thousands)", 0, 2000, 500, 10)
                labs_count = st.slider("Labs & Smart Classrooms Count", 0, 500, 50, 5)
                hostel_capacity = st.slider("Hostel Capacity", 0, 10000, 1000, 100)
            
            predict_btn = st.form_submit_button("Predict Placement")
        
        if predict_btn:
            # Prepare input data
            input_data = [
                naac_score,
                nirf_rank,
                library_books * 1000,  # Convert back to actual count
                labs_count,
                hostel_capacity
            ]
            
            # Make prediction
            pred_rate, pred_package = predict_placement(model_rate, model_package, features, input_data)
            
            # Display results
            st.subheader("Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicted Placement Rate", f"{max(0, min(100, pred_rate)):.1f}%")
            
            with col2:
                st.metric("Predicted Median Package", f"₹{max(0, pred_package):,.0f}")
            
            # Show interpretation
            st.info("""
            **Interpretation of Results:**
            - Placement Rate: Percentage of students who get placed through campus recruitment
            - Median Package: The median salary offered to students (in INR)
            
            These predictions are based on the input parameters and the patterns learned from the dataset.
            """)

if __name__ == "__main__":
    main()