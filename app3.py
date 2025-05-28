import os
import streamlit as st
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter

# Ruta local de tu modelo TFLite
MODEL_PATH = "model.tflite"

# Carga y cachea el intérprete de TFLite usando st.cache (compatible con Streamlit 1.17+)
@st.cache(allow_output_mutation=True)
def load_tflite_model():
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

# Función de predicción usando TFLite
def model_prediction_tflite(image_file, interpreter):
    image = Image.open(image_file).convert("RGB").resize((128, 128))
    input_data = np.expand_dims(np.array(image, dtype=np.float32), axis=0)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return np.argmax(output_data)

# Sidebar y navegación
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Elige Página", ["Inicio", "Acerca de", "Reconocimiento de enfermedad"])

# Página de Inicio
if app_mode == "Inicio":
    st.header("SISTEMA DE RECONOCIMIENTO DE ENFERMEDADES DE PLANTAS")
    image_path = os.path.join(os.path.dirname(__file__), "home_page.jpeg")
    if os.path.exists(image_path):
        st.image(image_path, use_column_width=True)
    st.markdown("""
    ¡Bienvenido al Sistema de Reconocimiento de Enfermedades de las Plantas! 🌿🔍
    Esta app utiliza un modelo TFLite para identificar enfermedades en hojas.
    """)

# Página "Acerca de"
elif app_mode == "Acerca de":
    st.header("Acerca de")
    st.markdown("""
    Este proyecto detecta más de 30 enfermedades de cultivos usando TensorFlow Lite.
    Modelo convertido a TFLite para optimizar tamaño y compatibilidad.
    """)

# Página de Predicción
elif app_mode == "Reconocimiento de enfermedad":
    st.header("Reconocimiento de enfermedad")
    test_image = st.file_uploader("Escoge una imagen:", type=["jpg", "png", "jpeg"])

    if test_image:
        st.image(test_image, use_column_width=True)
        if st.button("Predicción"):
            st.balloons()
            result_index = model_prediction_tflite(test_image, interpreter)

            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]
            st.success(f"El modelo predice: **{class_name[result_index]}**")
```


