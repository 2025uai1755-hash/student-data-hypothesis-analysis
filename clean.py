import os
import pandas as pd
import numpy as np
import re

print("Starting Data Cleaning Process...")

# File name matched from your 'ls *.csv' output
input_file = 'fdsc - Form Responses 1.csv'

if not os.path.exists(input_file):
    if os.path.exists('fdsc - Form Responses 1 (1).csv'):
        input_file = 'fdsc - Form Responses 1 (1).csv'

print(f"Reading file: {input_file}")
df = pd.read_csv(input_file)

# 1. Standardize Headers
clean_cols = [
    'timestamp', 'academic_year', 'college_name', 'branch', 'cgpa', 
    'academic_workload_rating', 'study_hours_per_week', 'practices_coding_regularly', 
    'coding_start_stage', 'coding_hours_per_week', 'problems_solved_per_week', 
    'contest_frequency', 'coding_platforms', 'cp_rating', 'total_problems_solved', 
    'placement_motivation_rating', 'peer_culture_influence_rating', 
    'participated_club_events', 'events_increased_interest', 'has_mentor', 
    'languages_known', 'technical_soft_skills', 'interviews_appeared', 
    'offers_received', 'received_ppo', 'companies_applied', 'placed_company', 
    'highest_ctc_lpa', 'biggest_coding_barrier', 'clg_improvement_suggestions'
]
df.columns = clean_cols[:len(df.columns)]

# 2. Clean CGPA
def clean_cgpa(val):
    if pd.isna(val): return np.nan
    val = str(val).strip()
    if val in ["..", ".", "-", "Wouldn't like to tell", "No cgpa as for first sem"]:
        return np.nan
    if val == '8+': return 8.0
    if val == '8.5+': return 8.5
    if val == '7-8': return 7.5
    match = re.search(r'(\d+\.\d+|\d+)', val)
    return float(match.group(1)) if match and float(match.group(1)) <= 10 else np.nan

df['cgpa_cleaned'] = df['cgpa'].apply(clean_cgpa)

# 3. Clean CP Rating
def clean_cp_rating(val):
    if pd.isna(val): return 0.0
    val = str(val).strip()
    if val.lower() in ["nothing", "don't know", "not applicable", ".."]: return 0.0
    match = re.search(r'(\d+)', val)
    return float(match.group(1)) if match else 0.0

df['cp_rating_cleaned'] = df['cp_rating'].apply(clean_cp_rating)

# 4. Clean Total Problems Solved
def clean_problems_solved(val):
    if pd.isna(val): return np.nan
    val = str(val).strip()
    match = re.search(r'(\d+)', val)
    return float(match.group(1)) if match else np.nan

df['total_problems_solved_cleaned'] = df['total_problems_solved'].apply(clean_problems_solved)

# 5. Clean College & Branch Names
college_map = {
    'MNIT JAIPUR': 'MNIT Jaipur',
    'Malaviya National Institute of technology': 'MNIT Jaipur',
    'NIT TRICHY': 'NIT Trichy',
    'NIT SURATHKAL': 'NIT Surathkal'
}
df['college_name_cleaned'] = df['college_name'].map(college_map).fillna('MNIT Jaipur')

branch_map = {
    'Computer Science & Engineering': 'Computer Science & Engineering',
    'Artificial Intelligence & Data Engineering': 'AI & Data Engineering',
    'Electronic Engineering': 'Electronics Engineering',
    'Civil Engineering': 'Civil Engineering',
    'electrical': 'Electrical Engineering',
    'Mechanical Enginerring': 'Mechanical Engineering',
    'Chemical Engineering': 'Chemical Engineering',
    'Metallurgical & Materials  Engineering': 'Metallurgical & Materials Engineering',
    'Other': 'Other'
}
df['branch_cleaned'] = df['branch'].map(branch_map).fillna(df['branch'])

# 6. Final Format
column_mapping = {
    'timestamp': 'Timestamp',
    'academic_year': 'Academic Year',
    'college_name_cleaned': 'College Name',
    'branch_cleaned': 'Branch',
    'cgpa_cleaned': 'CGPA',
    'academic_workload_rating': 'Academic Workload Rating (1-5)',
    'study_hours_per_week': 'Study Hours / Week',
    'practices_coding_regularly': 'Practices Coding Regularly',
    'coding_start_stage': 'Coding Start Stage',
    'coding_hours_per_week': 'Coding Hours / Week',
    'problems_solved_per_week': 'Problems Solved / Week',
    'contest_frequency': 'Contest Frequency',
    'coding_platforms': 'Coding Platforms Used',
    'cp_rating_cleaned': 'CP Rating',
    'total_problems_solved_cleaned': 'Total Problems Solved',
    'placement_motivation_rating': 'Placement Motivation Rating (1-5)',
    'peer_culture_influence_rating': 'Peer Culture Influence Rating (1-5)',
    'participated_club_events': 'Participated Club Events',
    'events_increased_interest': 'Events Increased Interest',
    'has_mentor': 'Has Mentor',
    'languages_known': 'Languages Known',
    'technical_soft_skills': 'Technical & Soft Skills',
    'interviews_appeared': 'Interviews Appeared',
    'offers_received': 'Offers Received',
    'received_ppo': 'Received PPO',
    'companies_applied': 'Companies Applied',
    'placed_company': 'Placed Company',
    'highest_ctc_lpa': 'Highest CTC',
    'biggest_coding_barrier': 'Biggest Coding Barrier',
    'clg_improvement_suggestions': 'College Improvement Suggestions'
}

df_final = df[list(column_mapping.keys())].rename(columns=column_mapping)

# 7. Fill Missing Values
df_final['Interviews Appeared'] = df_final['Interviews Appeared'].fillna('N/A')
df_final['Offers Received'] = df_final['Offers Received'].fillna(0)
df_final['Received PPO'] = df_final['Received PPO'].fillna('N/A')
df_final['Companies Applied'] = df_final['Companies Applied'].fillna('N/A')
df_final['Placed Company'] = df_final['Placed Company'].fillna('Not Placed / NA')
df_final['Highest CTC'] = df_final['Highest CTC'].fillna('N/A')
df_final['College Improvement Suggestions'] = df_final['College Improvement Suggestions'].fillna('-')
df_final['Biggest Coding Barrier'] = df_final['Biggest Coding Barrier'].fillna('-')
df_final['Languages Known'] = df_final['Languages Known'].fillna('-')
df_final['Technical & Soft Skills'] = df_final['Technical & Soft Skills'].fillna('-')

# 8. Export Output
df_final.to_csv('cleaned_student_data_full_139.csv', index=False)
df_final.to_excel('cleaned_student_data_full_139.xlsx', index=False)

print(f"\nSuccess! Cleaned {len(df_final)} rows out of 139.")
print("Generated Files: 'cleaned_student_data_full_139.csv' & 'cleaned_student_data_full_139.xlsx'")
