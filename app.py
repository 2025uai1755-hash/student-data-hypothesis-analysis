import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & SETUP
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Placement & Career Advisory Portal",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Placement Predictor & Career Advisory Dashboard")
st.markdown("""
Is dashboard ke zariye aap apna **Academic Score**, **Coding Profile**, **Technical Skills** aur **Project Experience** enter karke 
apni **Placement Probability**, **Skill Diagnostics** aur **Target Role Recommendations** dekh sakte hain.
""")

# ---------------------------------------------------------
# 2. DATA LOADING FUNCTION (DEFINED FIRST)
# ---------------------------------------------------------
@st.cache_data
def load_and_prep_data():
    df = pd.read_csv('cleaned_student_data_full_139.csv')
    df['Placed'] = (df['Offers Received'] > 0).astype(int)
    return df

# Function Call (After Definition)
df = load_and_prep_data()

# ---------------------------------------------------------
# 3. MODEL TRAINING
# ---------------------------------------------------------
features = [
    'CGPA', 
    'Academic Workload Rating (1-5)', 
    'CP Rating', 
    'Total Problems Solved', 
    'Placement Motivation Rating (1-5)', 
    'Peer Culture Influence Rating (1-5)'
]

X = df[features]
y = df['Placed']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# ---------------------------------------------------------
# 4. APP NAVIGATION TABS & UI LOGIC
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["🤖 Placement Predictor & Career Roadmap", "📊 Historical Insights & Data Analytics"])

# --- TAB 1: PREDICTOR & ADVISORY ---
with tab1:
    st.header("🔍 Profile Input & Diagnostic Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📌 Academic & Coding Metrics")
        cgpa = st.number_input("Your CGPA (out of 10.0)", min_value=0.0, max_value=10.0, value=7.8, step=0.1)
        cp_rating = st.number_input("LeetCode / CodeChef CP Rating", min_value=0, max_value=3000, value=1350, step=10)
        problems_solved = st.number_input("Total DSA Problems Solved", min_value=0, max_value=3000, value=220, step=10)
        projects_count = st.slider("Major Projects Built & Deployed", min_value=0, max_value=10, value=2)

    with col2:
        st.subheader("📌 Technical Skills & Environment")
        tech_skills = st.multiselect(
            "💻 Select Your Technical Stack & Skills",
            options=[
                "Data Structures & Algorithms (DSA)", 
                "Python / Pandas / NumPy", 
                "Java / C++", 
                "SQL / Databases", 
                "Web Development (HTML/CSS/JS/React/Node)", 
                "Machine Learning / Data Science", 
                "Cloud / DevOps (AWS/Docker)", 
                "Git / GitHub"
            ],
            default=["Data Structures & Algorithms (DSA)", "Python / Pandas / NumPy", "SQL / Databases", "Git / GitHub"]
        )
        
        workload = st.slider("Academic Workload Management (1-5)", 1, 5, 3)
        peer_influence = st.slider("Peer Group & Coding Culture Impact (1-5)", 1, 5, 4)
        motivation = st.slider("Placement Drive & Motivation (1-5)", 1, 5, 4)

    st.markdown("---")
    
    if st.button("🚀 Analyze Profile & Generate Readiness Report", width="stretch"):
        user_data = pd.DataFrame([[cgpa, workload, cp_rating, problems_solved, motivation, peer_influence]], columns=features)
        
        prediction_prob = model.predict_proba(user_data)[0][1] * 100
        
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.metric(label="Calculated Placement Chance", value=f"{prediction_prob:.1f}%")
            
        with res_col2:
            if prediction_prob >= 75:
                st.success("Readiness Status: **High Readiness (Tier 1)** 🟢")
            elif prediction_prob >= 45:
                st.warning("Readiness Status: **Moderate Readiness (Tier 2)** 🟡")
            else:
                st.error("Readiness Status: **High Risk (Tier 3)** 🔴")
                
        with res_col3:
            st.metric(label="Technical Skills Count", value=f"{len(tech_skills)} Active Skills")

        st.markdown("---")
        
        # --- DIAGNOSTIC FEEDBACK & ACTION PLAN ---
        st.subheader("💡 Diagnostic Analysis & Focus Areas")
        
        suggestions = []
        
        if cgpa < 7.5:
            suggestions.append("🔴 **CGPA Cutoff Warning:** CGPA 7.5 se kam hai. Product companies ke shortlisting criteria clear karne ke liye academic performance improve karein.")
        else:
            suggestions.append("🟢 **CGPA Safe Zone:** Academic profile competitive threshold ko comfortably clear karti hai.")
            
        if problems_solved < 250:
            suggestions.append("🔴 **DSA Problem Volume Low:** Coding rounds clear karne ke liye at least 250-300+ LeetCode/GFG problems solve karein.")
        else:
            suggestions.append("🟢 **DSA Problem Volume Good:** Problem-solving volume strong tracking par hai.")
            
        if cp_rating < 1400:
            suggestions.append("🟡 **Competitive Rating:** Contests me speed aur accuracy badhane ke liye 1400+ rating target karein.")
        else:
            suggestions.append("🟢 **Competitive Rating:** CP score impressive hai!")
            
        if len(tech_skills) < 3:
            suggestions.append("🔴 **Skill Diversity Warning:** At least 3 skills select/learn karein (jaise SQL + Core Language + Projects).")
        else:
            suggestions.append(f"🟢 **Skill Stack:** Solid tech skill set selected: **{', '.join(tech_skills)}**.")
            
        if projects_count < 2:
            suggestions.append("🟡 **Project Portfolio:** Resume impact ke liye kam se kam 2 end-to-end deployed projects GitHub par host karein.")
        else:
            suggestions.append(f"🟢 **Project Portfolio:** {projects_count} major projects resume shortlisting probability boost karenge.")

        for item in suggestions:
            st.write(item)

        st.markdown("---")
        
        # --- CAREER ROLES MATCHING LOGIC ---
        st.subheader("🎯 Best Fit Job Roles For Your Current Profile")
        
        roles = []
        
        if "Data Structures & Algorithms (DSA)" in tech_skills and cp_rating >= 1400 and problems_solved >= 250 and cgpa >= 7.5:
            roles.append("⭐ **Software Development Engineer (SDE-1)** — Product Companies (Tier-1 Tech)")
            
        if "SQL / Databases" in tech_skills and ("Python / Pandas / NumPy" in tech_skills or "Machine Learning / Data Science" in tech_skills):
            roles.append("⭐ **Data Analyst / Business Intelligence Engineer**")
            
        if "Web Development (HTML/CSS/JS/React/Node)" in tech_skills and projects_count >= 2:
            roles.append("⭐ **Frontend / Full-Stack Web Developer**")
            
        if "Machine Learning / Data Science" in tech_skills and cgpa >= 7.5 and projects_count >= 2:
            roles.append("⭐ **Junior ML Engineer / AI Trainee**")
            
        if "Cloud / DevOps (AWS/Docker)" in tech_skills:
            roles.append("⭐ **Cloud Associate / DevOps Systems Engineer**")

        if not roles:
            roles.append("📌 **Associate Software Engineer / Service-Based IT Engineer** (Focus on DSA & SQL to unlock product-based roles)")

        for role in roles:
            st.write(role)

# --- TAB 2: HISTORICAL DATA ANALYTICS ---
with tab2:
    st.header("📊 Dataset Trends & Comparative Analysis")
    st.dataframe(df.head(10), width="stretch")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("CGPA vs Placement Status")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        
        df_plot = df.copy()
        df_plot['Placement Status'] = df_plot['Placed'].map({0: 'Not Placed', 1: 'Placed'})
        sns.boxplot(x='Placement Status', y='CGPA', data=df_plot, hue='Placement Status', palette='Set2', legend=False, ax=ax1)
        st.pyplot(fig1)
        
    with col_b:
        st.subheader("Coding Volume vs CP Rating")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df, x='Total Problems Solved', y='CP Rating', hue='Placed', palette='Set1', ax=ax2)
        st.pyplot(fig2)
