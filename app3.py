import os
import streamlit as st
import tensorflow as tf
import numpy as np

#-------------------------------

#quitalocoment
# Cargar modelo .tflite desde el repositorio (local)
MODEL_PATH = "model.tflite"

# Cargar el modelo TFLite
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

# Obtener detalles de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#-------------------------------
# Función de predicción
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_arr)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return np.argmax(output_data)

#-------------------------------
#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Elige Página",["Inicio","Acerca de","Reconocimiento de enfermedad"])

#Main Page
if(app_mode=="Inicio"):
    st.header("SISTEMA DE RECONOCIMIENTO DE ENFERMEDADES DE PLANTAS")
    image_path = os.path.join(os.path.dirname(__file__), "home_page.jpeg")
    if not os.path.exists(image_path):
        st.error(f"No se encontró la imagen en la ruta: {os.path.abspath(image_path)}")
    else:
        st.image(image_path,use_column_width=True)

    st.markdown("""
¡Bienvenido al Sistema de Reconocimiento de Enfermedades de las Plantas! 🌿🔍

Nuestra misión es ayudar a identificar enfermedades de las plantas de manera eficiente...
""")

#About Project
elif(app_mode=="Acerca de"):
    st.header("Acerca de")
    st.markdown("""
Este conjunto de datos se recrea mediante el aumento sin conexión del conjunto de datos original...
""")

#Prediction Page
elif(app_mode=="Reconocimiento de enfermedad"):
    st.header("Reconocimiento de enfermedad")
    test_image = st.file_uploader("Escoge una imagen:")
    if(st.button("Muestra Imagen")):
        st.image(test_image,use_column_width=True)

    if(st.button("Predicción")):
        st.balloons()
        st.write("Nuestra Predicción")
        result_index = model_prediction(test_image)

        # Lista de etiquetas (debes adaptar esto si tu modelo tiene otras clases)
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
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
                      'Tomato___healthy']

        st.success(f"El modelo está prediciendo que es un **{class_name[result_index]}**")

        