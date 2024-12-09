#NOTA IMPORTANTE: esta request debe ser lanzada desde un terminal externo

import requests

# URL del servidor Flask
url = 'http://localhost:5000/generate'

# Datos que se enviarán en la solicitud
data = {
    "text": "Esto es una prueba",  # Texto a convertir en audio
    "codec": "wav"                 # Formato de audio
}

# Realizar la solicitud POST
response = requests.post(url, json=data)

# Comprobar el estado de la respuesta
if response.status_code == 200:
    print("Audio generado correctamente")
else:
    print("Error:", response.status_code, response.text)  # Imprime el código de estado y el contenido de la respuesta