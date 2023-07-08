import pandas as pd
import streamlit as st
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
from absenteeism_module import *
#####################################

def Input_Output():
    data = st.file_uploader("Upload file", type={"csv","txt"})
    if data is not None:
        df = pd.read_csv(data)
        st.write(df)
        model = absenteeism_model('model','scaler')
        model.load_and_clean_data(df)
    result = ""
    if st.button("Click here to predict"):
        result = model.predicted_outputs()
        st.balloons()
    st.success('the output is as follow: ')
    st.write(result)

if __name__ == '__main__':
    Input_Output()
st.caption("\nzs.noori@oxinAI")