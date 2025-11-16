import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sksurv.ensemble import RandomSurvivalForest
import pickle

# --- 1. 配置和加载函数 ---
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(show_spinner=False)
def load_setting():
    settings ={
        'Age': {'values': [0, 90], 'type': 'slider', 'init_value': 45, 'add_after': ', year', 'model_key': 'age'},
        'Hospital stay': {'type': 'text_input', 'init_value': 0, 'add_after': 'day', 'model_key': 'Hospital_stay'},  
        'Stage': {'values': ["I", "II","III","IV" ],'type': 'selectbox', 'init_value': 0, 'add_after': '', 'model_key': 'Stage'}, 
        'CHE': {'type': 'text_input', 'init_value': 0, 'add_after': 'U/L', 'model_key': 'CHE'}, 
        'CEA': {'type': 'text_input', 'init_value': 0, 'add_after': 'µg/L', 'model_key': 'CEA'},
        'Albumin': {'type': 'text_input', 'init_value': 0, 'add_after': 'g/L', 'model_key': 'Alb'},
        'Hemoglobin': {'type': 'text_input', 'init_value': 0, 'add_after': 'g/L', 'model_key': 'Hb'},
        'Fibrinogen': {'type': 'text_input', 'init_value': 0, 'add_after': 'g/L', 'model_key': 'FIB'},
        'Major complications': {'values': ["No", "Yes"],'type': 'selectbox', 'init_value': 0, 'add_after': '', 'model_key': 'Major_complications'},
        'Intro-operative Bleeding': {'type': 'text_input', 'init_value': 0, 'add_after': 'ml', 'model_key': 'Bleeding'},
    }
    input_keys = ['Bleeding','age','CEA', 'CHE', 'Alb','Hb', 'Hospital_stay','FIB', 'Stage',
                  'Major_complications']
    return settings, input_keys


@st.cache_data(show_spinner=False)
def get_model():
    with open('./rsf_best_model.pkl', 'rb') as f: 
        model = pickle.load(f)
    return model


def get_code():
    sidebar_code = []

    for key in settings:
        if settings[key]['type'] == 'slider':
            sidebar_code.append(
                "st.slider('{}',{},{},key='{}')".format( # 移除开头的 {} =
                    key + settings[key]['add_after'],
                    ','.join(['{}'.format(value) for value in settings[key]['values']]),
                    settings[key]['init_value'],
                    key
                )
            )
        elif settings[key]['type'] == 'selectbox':
            sidebar_code.append(
                'st.selectbox("{}",({}),{},key="{}")'.format( # 移除开头的 {} =
                    key + settings[key]['add_after'],
                    ','.join('"{}"'.format(value) for value in settings[key]['values']),
                    settings[key]['init_value'],
                    key
                )
            )
        elif settings[key]['type'] == 'text_input':
            sidebar_code.append(
                "st.text_input('{}','{}',key='{}')".format( # 移除开头的 {} =
                    key + settings[key]['add_after'],
                    str(settings[key]['init_value']),   
                    key
                )
            )
    return sidebar_code


def get_input_dataframe(settings, input_keys):
    
    raw_data = {}
    for ui_key, config in settings.items():
        value = st.session_state.get(ui_key)
        model_key = config['model_key']
        
        if config['type'] == 'text_input':
            try:
                value = float(value)
            except (TypeError, ValueError):
                if value != 0:
                    st.error(f"'{ui_key}' 的输入值 '{value}' 不是有效数字，已使用 0 替代。")
                value = 0.0
        
        raw_data[model_key] = value

    input_df = pd.DataFrame([raw_data])

    stage_mapping = {"I": 1, "II": 2, "III": 3, "IV": 4}
    input_df['Stage'] = input_df['Stage'].map(stage_mapping)
    
    complications_mapping = {"No": 0, "Yes": 1}
    input_df['Major_complications'] = input_df['Major_complications'].map(complications_mapping)

    for col in input_df.columns:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
        
    try:

        input_df = input_df[input_keys]
    except KeyError as e:
        st.error(f"DataFrame中缺少模型需要的列:{e}。请检查您的settings和input_keys映射。")
        return None 
        
    return input_df



def predict():

    input_df = get_input_dataframe(settings, input_keys)

    if input_df is None:
        st.error("数据处理错误，无法进行预测。请检查输入。")
        return

    predicted_risk = rsf.predict(input_df)


    cumulative_hazard_functions = rsf.predict_cumulative_hazard_function(input_df)

    q = 47.22

    st.subheader("📊 Risk Stratification")


    st.info(f"Threshold: Low risk<{q:.4f} , High risk>{q:.4f}")

    risk_score = predicted_risk[0]
    if risk_score < q:
        st.markdown(f"**Current patient's risk score:** **`{risk_score:.4f}`**")
        st.markdown(f"<span style='color: green; font-size: 20px;'>✅ The current patient's risk group: **Low Risk**</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"**Current patient's risk score:** **`{risk_score:.4f}`**")
        st.markdown(f"<span style='color: red; font-size: 20px;'>🚨 The current patient's risk group: **High Risk**</span>", unsafe_allow_html=True)
    
    st.markdown("---")

    time_points = [12, 36, 60] 
    survival_rates_data = []


    chf = cumulative_hazard_functions[0]
    
    for time_point in time_points:

        cumulative_hazard_at_t = chf(time_point)
        survival_rate = np.exp(-cumulative_hazard_at_t) 
        
        survival_rates_data.append({
            "Time (year)": f"{time_point // 12}",
            "Cumulative Hazard (time)": f"{cumulative_hazard_at_t:.4f}",
            "Survival probability": f"{survival_rate:.4f}"
        })
        
    st.markdown("## ⏳ 1-, 3-, 5-year survival predicting")
    st.dataframe(pd.DataFrame(survival_rates_data), hide_index=True)

    st.markdown("---")
    st.markdown("## 📈 Cumulative Hazard Curve")
    
    time_index = chf.x
    risks = chf.y

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(time_index, risks, label=f'Patient Risk Score: {risk_score:.2f}')
    ax.set_xlabel("Time (Months)")
    ax.set_ylabel("Cumulative Hazard")
    ax.set_title("Cumulative Hazard Curve")
    ax.grid(False, linestyle='--', alpha=0.6)
    ax.legend()
    st.pyplot(fig)

settings, input_keys = load_setting() 
rsf = get_model()
sidebar_code = get_code()

st.header('RSF-based model for predicting survival of Gastric cancer', anchor='survival-of-Gastric cancer')


# 侧边栏表单
with st.sidebar:
    st.title("Patient parameter entry")
    with st.form("my_form", clear_on_submit=False): 
        
        for code in sidebar_code:
            exec(code)
            
        col8, col9, col10 = st.columns([3, 4, 3])
        with col9:
            prediction = st.form_submit_button(
                'Predict',
                on_click=predict
            )
