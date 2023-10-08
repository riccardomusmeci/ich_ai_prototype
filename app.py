import streamlit as st
import pandas as pd
import pickle


####### DDR Risk Binary #######
ddr_risk_binary_features = {
    'RS Results': [], 
    'ER_gene score': [], 
    'PR_gene score': [], 
    'Grade': [], 
    'tumor_size_range_numeric': [], 
    'ki67_range_numeric': []
}

####### DDR Risk Regression #######
ddr_risk_regr_features = {
    "RS Results": [],
    "ER_gene score": [],
    "PR_gene score": [],
    "tumor_size": [],
    "Grade": [],
    "tumor_size_range_numeric": [],
    "ER_level_range_numeric": [],
    "PR_level_range_numeric": [],
}

####### CT Benefit #######
ct_benefit_features = {
    "RS Results": [],
    "ER_gene score": [],
    "PR_gene score": [],
    "Grade": [],
    "PR_level_range_numeric": [],
    "ki67_range_numeric": [],
    "HER2": [],
}

st.set_page_config(
    page_title="RSClin Estimator", 
    page_icon=None,
    initial_sidebar_state="auto", 
    layout="wide"
)

def _tumor_size_range(tumor_size):
    if 1<=tumor_size<=5:
        return "t1_a", 0
    elif 6<=tumor_size<=10:
        return "t1_b", 1
    elif 11<=tumor_size<=20:
        return "t1_c", 2
    elif 21<=tumor_size<=50:
        return "t2", 3
    elif el>=51:
        return "t3", 4
    
def _pr_level_range(pr_level):
    if pr_level <= 20:
        return "low", 0
    if 21<=pr_level<=89:
        return "mid", 1
    else:
        return "high", 2
    
def _er_level_range(er_level):
    if er_level < 90:
        return 0
    if er_level>=90:
        return 1 
    
def _ki67_range(ki67):
    if ki67 <= 20:
        return 0
    else:
        return 1
    

def predict(
    rs_result: float,
    er_gene_score: float,
    pr_gene_score: float,
    tumor_size: float,
    grade: float,
    er_level: float,
    pr_level: float,
    her_2: float,
    ki_67: float,
    col
):
    ddr_reg_model = pickle.load(open('model/ddr_risk_real_synth_regr.sav', 'rb'))
    ddr_clf_model = pickle.load(open('model/ddr_risk_real_synth_clf.sav', 'rb'))
    ct_benefit_model = pickle.load(open('model/ct_benefit_clf.sav', 'rb'))
    
    tumor_size_range = _tumor_size_range(tumor_size)[1]
    er_level_range = _er_level_range(er_level)
    pr_level_range = _pr_level_range(pr_level)[1]
    ki_67_range = _ki67_range(ki_67)
    
    ####### DDR Risk Binary #######
    ddr_risk_binary_features = {
        'RS Results': [rs_result], 
        'ER_gene score': [er_gene_score], 
        'PR_gene score': [pr_gene_score], 
        'Grade': [grade], 
        'tumor_size_range_numeric': [tumor_size_range], 
        'ki67_range_numeric': [ki_67_range]
    }
    
    x = pd.DataFrame(data=ddr_risk_binary_features)
    ddr_high = ddr_clf_model.predict(x)[0]
    ddr_high_proba = ddr_clf_model.predict_proba(x)[0]
    
    ####### DDR Risk Regression #######
    ddr_risk_regr_features = {
        "RS Results": [rs_result],
        "ER_gene score": [er_gene_score],
        "PR_gene score": [pr_gene_score],
        "tumor_size": [tumor_size],
        "Grade": [grade],
        "tumor_size_range_numeric": [tumor_size_range],
        "ER_level_range_numeric": [er_level_range],
        "PR_level_range_numeric": [ki_67_range],
    }
    
    x = pd.DataFrame(data=ddr_risk_regr_features)
    ddr_val = ddr_reg_model.predict(x)[0]

    ####### CT Benefit #######
    ct_benefit_features = {
        "RS Results": [rs_result],
        "ER_gene score": [er_gene_score],
        "PR_gene score": [pr_gene_score],
        "Grade": [grade],
        "PR_level_range_numeric": [pr_level_range],
        "ki67_range_numeric": [ki_67_range],
        "HER2": [her_2],
    }
    
    x = pd.DataFrame(data=ct_benefit_features)
    ct_benefit = ct_benefit_model.predict(x)[0]
    ct_benefit_proba = ct_benefit_model.predict_proba(x)[0]
    
    with col:
        st.subheader("Distance Disease Recurrence Risk at 9 years:")
        _, ddr_col, _ = st.columns(3)
        ddr_val_text = 'high' if ddr_high==1 else 'low'
        ddr_val_prob = 100*ddr_high_proba[ddr_high]
        with ddr_col:
            st.markdown(f"## **{ddr_val_text.upper()}**")
        st.write(f"DDR Risk at 9 years is **{ddr_val_text.upper()}** with probability {ddr_val_prob:.2f}%.")
        st.write(f"DDR Risk at 9 years value is **{int(ddr_val)}**.")
        
        st.write("\n\n")
        
        st.subheader("Chemiotherapy Benefit:")
        _, ct_col, _ = st.columns(3)
        ct_benefit_text = 'yes' if ct_benefit==1 else 'no'
        ct_benefit_prob = 100*ct_benefit_proba[ct_benefit]
        with ct_col:
            st.markdown(f"## **{ct_benefit_text.upper()}**")
        st.write(f"Chemiotherapy benefit predicted with probability {ct_benefit_prob:.2f}%.")
            
        

def rsclin_ui():
    
    ddr_reg_model = pickle.load(open('model/ddr_risk_real_synth_regr.sav', 'rb'))
    ddr_clf_model = pickle.load(open('model/ddr_risk_real_synth_clf.sav', 'rb'))
    ct_benefit_model = pickle.load(open('model/ct_benefit_clf.sav', 'rb'))
    
    features_col, pred_col = st.columns(2)
    with features_col:
        st.header("Patient Characteristics")
        
        age = st.number_input('Patient Age')
        st.write('Patient age is ', age)
        
        st.divider()
        
        rs_result = st.number_input('RS Result', min_value=1.0, max_value=100.0)
        st.write("Recurrence Score is ", rs_result)
        
        st.divider()
        
        er_gene_score = st.number_input('Estrogen Receptor Gene Score', min_value=1.0, max_value=100.0)
        st.write("Estrogen Receptor Gene Score is ", er_gene_score)
        
        st.divider()
        
        er_level = st.number_input('Estrogen Level')
        st.write("Estrogen Level is ", er_level)
        
        st.divider()
        
        pr_gene_score = st.number_input('Progesteron Receptor Gene Score', min_value=1.0, max_value=100.0)
        st.write("Progesteron Receptor Gene Score is ", er_gene_score)
        
        st.divider()
        
        pr_level = st.number_input('Progesteron Level')
        st.write(f"Progesteron Level is {_pr_level_range(pr_level)[0]}")
        
        st.divider()
        
        tumor_grade = st.selectbox('Tumor Grade', options=[1, 2, 3])
        st.write("Tumor Grade is ", tumor_grade)
        
        st.divider()
        
        tumor_size = st.number_input('Tumor Size (mm)', min_value=1)
        st.write(f"Tumor size is in **{_tumor_size_range(tumor_size)[0]}** group")
        
        st.divider()
        
        her_2 = st.selectbox('HER2', options=[0, 1, 2])
        st.write("HER2 is ", her_2)
        
        st.divider()
        
        ki_67 = st.number_input('ki67')
        st.write("ki67 is ", ki_67)
        
        
        _, bttn_col, _ = st.columns(3)
        with bttn_col:    
            st.button(
                "Predict", 
                on_click=predict, 
                args=(
                    rs_result,
                    er_gene_score,
                    pr_gene_score,
                    tumor_size,
                    tumor_grade,
                    er_level,
                    pr_level,
                    her_2,
                    ki_67,
                    pred_col
                )
            )
    with pred_col:
        st.header("Machine Learning Predictions")
        

if __name__ == "__main__":
    st.title("ICH Breast Cancer AI Prototype")
        
    st.markdown("#### A Machine Learning tool that utilizes a combination of genomic data and clinical pathological features to estimate the distance disease recurrence (DDR) risk at a 9 years and the potential benefits of chemotherapy for hormone receptor positive, HER2 negative, node negative early breast cancer patients.")
    
    st.markdown("#### The models are trained on a blend of real world and synthetic data generated by a GAN.")
    
    rsclin_ui()
