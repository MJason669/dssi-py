# app.py
import streamlit as st
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 加载数据并训练模型
@st.cache_resource
def load_model():
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
# 加载模型
model = load_model()

# 用户输入界面
st.title("Diabetes Prediction App")
st.write("Enter patient features to predict diabetes progression.")

# 滑动条输入特征
age = st.slider("Age", 0, 100, 50)
sex = st.slider("Sex", 0, 1, 0)  # 0=Female, 1=Male
bmi = st.slider("BMI", 0.0, 60.0, 25.0)
blood_pressure = st.slider("Blood Pressure", 0, 150, 80)
s1 = st.slider("S1 (Feature)", -2.0, 2.0, 0.0)
s2 = st.slider("S2 (Feature)", -2.0, 2.0, 0.0)
s3 = st.slider("S3 (Feature)", -2.0, 2.0, 0.0)
s4 = st.slider("S4 (Feature)", -2.0, 2.0, 0.0)
s5 = st.slider("S5 (Feature)", -2.0, 2.0, 0.0)
s6 = st.slider("S6 (Feature)", -2.0, 2.0, 0.0)

# 预测按钮
if st.button("Predict"):
    input_data = [[age, sex, bmi, blood_pressure, s1, s2, s3, s4, s5, s6]]
    prediction = model.predict(input_data)
    st.success(f"Predicted Diabetes Progression: {prediction[0]:.2f}")
streamlit run app.py
