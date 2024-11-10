# Usa la imagen base de Airflow
FROM apache/airflow:2.9.1

# Copia el archivo requirements.txt en el contenedor
COPY requirements.txt .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt
