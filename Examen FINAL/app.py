from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Cargar modelo, escalador y codificador
mlp = joblib.load('mlp_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

@app.route('/')
def index():
    return render_template('formulario_depresion/form.html')

@app.route('/predict', methods=['POST'])
def predict():

    # Recibir las respuestas del formulario
    respuestas = [
        int(request.form['pregunta1']),
        int(request.form['pregunta2']),
        int(request.form['pregunta3']),
        int(request.form['pregunta4']),
        int(request.form['pregunta5']),
        int(request.form['pregunta6']),
        int(request.form['pregunta7']),
        int(request.form['pregunta8']),
        int(request.form['pregunta9']),
        int(request.form['pregunta10']),
        int(request.form['pregunta11']),
        int(request.form['pregunta12']),
        int(request.form['pregunta13']),
        int(request.form['pregunta14']),
        int(request.form['pregunta15']),
        int(request.form['pregunta16']),
        int(request.form['pregunta17']),
        int(request.form['pregunta18']),
        int(request.form['pregunta19']),
        int(request.form['pregunta20']),
        int(request.form['pregunta21']),
    ]

    # Escalar
    respuestas_scaled = scaler.transform([respuestas])

    # Predicción
    prediccion = mlp.predict(respuestas_scaled)
    nivel = encoder.inverse_transform(prediccion)[0] #Dep

    # MENSAJES PERSONALIZADOS
    mensajes = {
        "Ninguna depresión": "Tu resultado indica que no presentas síntomas significativos de depresión. ¡Qué alegría ver que estás bien! Tu salud emocional es un reflejo de tu fortaleza, sigue cuidándola.",
        "Depresión leve": "Podrías estar presentando algunos síntomas leves. Es recomendable que prestes atención a tus emociones y consideres hablar con alguien de confianza.",
        "Depresión moderada": "Tus síntomas podrían estar afectando partes importantes de tu vida, es hora de buscar un nuevo aliado en tu camino. La orientación psicológica es una herramienta que te ayudará a redescubrir la luz dentro de ti.",
        "Depresión grave": "Tu resultado es serio, es MUY importante que busques ayuda profesional lo antes posible, no estás solo, la valentía se demuestra también al pedir ayuda. Hay manos dispuestas a sostenerte en este momento, busca apoyo profesional, es el primer paso hacia la recuperación."
    }

    mensaje_final = mensajes.get(nivel, "Resultados no disponibles.")

    # SI ES DEPRESIÓN GRAVE → MOSTRAR LINK DE AYUDA EN SUCRE
    if nivel == "Depresión grave":
        ciudad = "Sucre, Bolivia"
        link_ayuda = f"https://www.google.com/maps/search/consultorios+psicológicos+cerca+de+{ciudad.replace(' ', '+')}"
    else:
        link_ayuda = None

    # Enviar todo a la plantilla
    return render_template('resultado.html', nivel=nivel, mensaje=mensaje_final, link_ayuda=link_ayuda)

# Ejecutar servidor
if __name__ == '__main__':
    app.run(debug=True)
