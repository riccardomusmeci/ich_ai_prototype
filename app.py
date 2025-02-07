import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="RSC4All", 
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
    
def _histotype_numeric(histotype):
    if histotype == "NST":
        return 0
    if histotype == "Lobular":
        return 1
    if histotype == "Other":
        return 2
    
def _lvi_numeric(lvi):
    if lvi == "No":
        return 0
    if lvi == "Yes":
        return 1
    
def _menopausal_status_numeric(menopausal_status):
    if menopausal_status == "Pre":
        return 0
    if menopausal_status == "Post":
        return 2
    

def predict(
    age: int,
    rs_result: float,
    menopausal_status: str,
    er_gene_score: float,
    pr_gene_score: float,
    her2_gene_score: float,
    tumor_size: float,
    grade: float,
    er_level: float,
    pr_level: float,
    her_2: float,
    ki_67: float,
    histotype: str,
    lvi: str,
    col
):
    ddr_reg_model = pickle.load(open('model/ddr_risk_reg.sav', 'rb'))
    ddr_clf_model = pickle.load(open('model/ddr_risk_clf.sav', 'rb'))
    ct_benefit_clf_model = pickle.load(open('model/ct_benefit_clf.sav', 'rb'))
    ct_benefit_reg_model = pickle.load(open('model/ct_benefit_reg.sav', 'rb'))
    
    menopausal_status_numeric = _menopausal_status_numeric(menopausal_status)
    tumor_size_range = _tumor_size_range(tumor_size)[1]
    er_level_range = _er_level_range(er_level)
    pr_level_range = _pr_level_range(pr_level)[1]
    ki_67_range = _ki67_range(ki_67)
    histotype_numeric = _histotype_numeric(histotype)
    lvi_numeric = _lvi_numeric(lvi)
    
    ####### DDR Risk Binary #######
    ddr_risk_binary_features = {
        'age': [age],
        'rs_values': [rs_result],
        'er_gene_score': [er_gene_score],
        'pr_gene_score': [pr_gene_score],
        'her2_gene_score': [her2_gene_score],
        'tumor_size': [tumor_size],
        'grade': [grade],
        'tumor_size_range_numeric': [tumor_size_range],
        'lvi_numeric': [lvi_numeric],
        "PR_level_range_numeric": [pr_level_range],
        'ki67_range_numeric': [ki_67_range]
    }
        
    x = pd.DataFrame(data=ddr_risk_binary_features)
    ddr_high = ddr_clf_model.predict(x)[0]
    ddr_high_proba = ddr_clf_model.predict_proba(x)[0]
    
    ####### DDR Risk Regression #######
    ddr_risk_regr_features = {
        "rs_values": [rs_result],
        "er_gene_score": [er_gene_score],
        "pr_gene_score": [pr_gene_score],
        "tumor_size": [tumor_size],
        "grade": [grade],
        "tumor_size_range_numeric": [tumor_size_range],
        "PR_level_range_numeric": [pr_level_range],
        'ki67_range_numeric': [ki_67_range]
    }
    
    x = pd.DataFrame(data=ddr_risk_regr_features)
    ddr_val = ddr_reg_model.predict(x)[0]
    ddr_val = ddr_val if ddr_val>=0 else 0

    ####### CT Benefit Binary #######
    ct_benefit_clf_features = {
        "age": [age],
        "rs_values": [rs_result],
        "er_gene_score": [er_gene_score],
        "pr_gene_score": [pr_gene_score],
        "her2_gene_score": [her2_gene_score],
        "histotype_numeric": [histotype_numeric],
        "grade": [grade],
        "lvi_numeric": [lvi_numeric],
        "PR_level_range_numeric": [pr_level_range],
        "ki67_range_numeric": [ki_67_range]
    }
    
    x = pd.DataFrame(data=ct_benefit_clf_features)
    ct_benefit_yes = ct_benefit_clf_model.predict(x)[0]
    ct_benefit_yes_proba = ct_benefit_clf_model.predict_proba(x)[0]
    
    
    ####### CT Benefit Regression #######
    ct_benefit_reg_features = {
        "age": [age],
        "rs_values": [rs_result],
        "er_gene_score": [er_gene_score],
        "pr_gene_score": [pr_gene_score],
        "her2_gene_score": [her2_gene_score],
        "tumor_size": [tumor_size],
        "menopausal_status_numeric": [menopausal_status_numeric],
        "grade": [grade],
        "tumor_size_range_numeric": [tumor_size_range],
        "lvi_numeric": [lvi_numeric],
        "PR_level_range_numeric": [pr_level_range],
        "ki67_range_numeric": [ki_67_range],
    }
    
    x = pd.DataFrame(data=ct_benefit_reg_features)
    ct_benefit_val = ct_benefit_reg_model.predict(x)[0]
    ct_benefit_val = ct_benefit_val if ct_benefit_val>=0 else 0
    
    with col:
        
        st.markdown("<h3 style='text-align: left;'>Distance Recurrence Risk at 9 years</h3>", unsafe_allow_html=True)
        st.markdown("low ($<$10%) - high ($\ge$10%)", unsafe_allow_html=True)
        _, ddr_col, _ = st.columns([1, 5, 1])
        ddr_val_text = 'high' if ddr_high==1 else 'low'
        ddr_val_prob = 100*ddr_high_proba[ddr_high]
        with ddr_col:
            st.markdown(f"## Risk Category: **{ddr_val_text}**", unsafe_allow_html=True)
            st.markdown(f"## Risk Value: **{int(ddr_val)}%**", unsafe_allow_html=True)
        # st.write(f"Risk category is **{ddr_val_text.upper()}**")
        st.write("\n")
        st.write(f"Risk category predicted with a probability of **{ddr_val_prob:.2f}%** as result to RSClin™ estimates")
            
        st.divider()
        
        st.markdown("<h3 style='text-align: left;'>Chemiotherapy Benefit</h3>", unsafe_allow_html=True)
        st.markdown("no ($<$3%) - yes ($\ge$3%)", unsafe_allow_html=True)
        _, ct_col, _ = st.columns([1, 5, 1])
        ct_benefit_text = 'yes' if ct_benefit_yes==1 else 'no'
        ct_benefit_prob = 100*ct_benefit_yes_proba[ct_benefit_yes]
        with ct_col:
            st.markdown(f"## Chemoterapy Benefit: **{ct_benefit_text}**", unsafe_allow_html=True)
            st.markdown(f"## Benefit Value: **{int(ct_benefit_val)}%**", unsafe_allow_html=True)
        st.write("\n")
        st.write(f"Chemotherapy benefit category predicted with a probability of **{ct_benefit_prob:.2f}%** as result to RSClin™ estimates")
        

def rsclin_ui():
    
    features_col, _, pred_col = st.columns([6, 1, 6])
    with features_col:
        
        st.write("#### Oncotype DX Report")
                
        rs_result = st.number_input('RS Result (0-100)', min_value=0, max_value=100, value=0, step=1)
        
        er_gene_score = st.number_input('Estrogen Receptor Gene Score', min_value=1.0, max_value=100.0)
    
        pr_gene_score = st.number_input('Progesteron Receptor Gene Score', min_value=1.0, max_value=100.0)
        
        her2_gene_score = st.number_input('HER2 Gene Score', min_value=1.0, max_value=100.0)
        
        st.write("\n")
        
        st.write("#### Clinicolpathological Features")
        
        age = st.number_input('Patient Age', min_value=0, max_value=100, value=0, step=1)
        
        menopausal_status = st.selectbox('Menopausal Status', options=["Pre", "Post"])
        
        er_level = st.number_input('Estrogen Level (%)', min_value=0, max_value=100, value=0, step=1)
        
        pr_level = st.number_input('Progesteron Level (%)', min_value=0, max_value=100, value=0, step=1)

        her_2 = st.selectbox('HER2', options=[0, 1, 2])
        
        tumor_grade = st.selectbox('Tumor Grade', options=[1, 2, 3])
        
        tumor_size = st.number_input('Tumor Size (mm)', min_value=1)
        
        lvi = st.selectbox('Lymphovascular invasion', options=["No", "Yes"])
        
        histotype = st.selectbox('Histotype', options=["NST", "Lobular", "Other"])
        
        ki_67 = st.number_input('ki67 (%)', min_value=0.0, max_value=100.0, value=0.0)
        
        _, bttn_col, _ = st.columns([4, 2, 4])
        with bttn_col:    
            st.button(
                "Predict", 
                on_click=predict, 
                args=(
                    age,
                    rs_result,
                    menopausal_status,
                    er_gene_score,
                    pr_gene_score,
                    her2_gene_score,
                    tumor_size,
                    tumor_grade,
                    er_level,
                    pr_level,
                    her_2,
                    ki_67,
                    histotype,
                    lvi,
                    pred_col
                )
            )
    with pred_col:
        st.markdown("<h3 style='text-align: center;'>Machine Learning Predictions</h2>", unsafe_allow_html=True)
        

if __name__ == "__main__":
    st.title("RSC4All")
    intro = "<p style='font-size:20px;'>A Machine Learning tool that leverages a combination of genomic data and clinical pathological features to estimate the distance recurrence risk at a 9 years and the potential benefits of chemotherapy mirroring the RSClin™ results in hormone receptor positive, HER2 negative, node negative early breast cancer patients.</p>"
    st.markdown(intro, unsafe_allow_html=True) 
    st.divider()
    rsclin_ui()
