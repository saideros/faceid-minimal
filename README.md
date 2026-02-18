# 🔐 Sistema de FaceID con ArcFace (buffalo_l)

## 📌 Descripción General

Este proyecto implementa un sistema de reconocimiento facial (FaceID) utilizando el modelo **ArcFace** con la variante **buffalo_l**, exponiendo un API con **FastAPI** y una interfaz interactiva desarrollada en **Streamlit**.

El sistema permite:

- 📷 Registrar rostros
- 🔎 Comparar imágenes faciales
- 📊 Obtener métricas de similitud
- ✅ Validar identidad mediante embeddings faciales

---

## 🧠 ¿Cómo funciona ArcFace?

**ArcFace** es un modelo de reconocimiento facial basado en aprendizaje profundo que genera **embeddings faciales** (vectores numéricos de alta dimensión) que representan características únicas del rostro.

### 🔬 Principio de Funcionamiento

1. El modelo detecta el rostro en la imagen.
2. Se alinean los puntos faciales clave (ojos, nariz, boca).
3. Se genera un embedding (vector normalmente de 512 dimensiones).
4. Se calcula la similitud entre embeddings utilizando distancia coseno o euclidiana.

Si la similidad supera un umbral definido → se considera la misma persona.

---

## 📦 Modelo: buffalo_l

El modelo **buffalo_l** pertenece a la colección de modelos de InsightFace y está optimizado para:

- 🎯 Alta precisión en reconocimiento facial
- ⚡ Buen rendimiento en CPU y GPU
- 📐 Embeddings de 512 dimensiones
- 🧠 Basado en arquitectura ResNet profunda

Es ampliamente utilizado en sistemas de autenticación biométrica debido a su balance entre precisión y velocidad.

---

## 🚀 Backend con FastAPI

**FastAPI** es un framework moderno y de alto rendimiento para construir APIs con Python.

En este proyecto se utiliza para:

- Exponer endpoints REST (`/compare`, `/register`, etc.)
- Recibir imágenes en formato Base64
- Procesar embeddings
- Retornar resultados en formato JSON
- Manejar validaciones y control de errores

### Ventajas de FastAPI

- Alto rendimiento (basado en Starlette y Pydantic)
- Documentación automática con Swagger
- Validación de datos automática
- Soporte asíncrono (async/await)

---

## 🖥️ Frontend con Streamlit

**Streamlit** es una herramienta para crear aplicaciones web interactivas con Python de forma rápida.

En este sistema permite:

- Subir imágenes
- Visualizar resultados
- Mostrar métricas de similitud
- Interactuar con la API en tiempo real

### Ventajas de Streamlit

- Desarrollo rápido
- Integración directa con Python
- Ideal para prototipos y dashboards de ML
- Fácil despliegue

---

## 🔄 Flujo General del Sistema

1. Usuario carga imagen desde Streamlit.
2. La imagen se envía al backend (FastAPI).
3. FastAPI procesa la imagen con ArcFace (buffalo_l).
4. Se genera el embedding facial.
5. Se compara con embeddings almacenados.
6. Se devuelve el nivel de similitud.
7. Streamlit muestra el resultado.

---

## 📊 Métrica de Comparación

Se utiliza principalmente:

- **Cosine Similarity**
- **Distancia Euclidiana**

Umbral típico:
- Cosine similarity > 0.5–0.7 (dependiendo calibración)
- Distancia euclidiana < 1.0 (aprox.)

---

## 🛠️ Tecnologías Utilizadas

- Python 3.10+
- InsightFace (ArcFace - buffalo_l)
- OpenCV
- NumPy
- FastAPI
- Uvicorn
- Streamlit

---

## 📌 Aplicaciones

- Control de acceso
- Validación de identidad
- Prevención de fraude
- Sistemas biométricos empresariales
- Onboarding digital

---

## 📎 Nota

Este sistema está diseñado con fines educativos y empresariales. Para entornos productivos se recomienda:

- Encriptación de embeddings
- Protección de endpoints
- Implementación de HTTPS
- Control de acceso y auditoría

---

## 🧠 Lógica Central: admin_embeddings.py

Este archivo contiene la lógica principal del sistema biométrico.

Se encarga de:

Registrar empleados

Generar embeddings

Actualizar embeddings globales

Eliminar empleados

Validar integridad de datos

Cargar modelo y persistencia

🔹 agregar_empleado()

Proceso:

Guarda la fotografía en la carpeta empleados/

Detecta rostros usando ArcFace

Selecciona el rostro más grande detectado

Genera el embedding normalizado (512 dimensiones)

Agrega el embedding a embeddings.npy

Guarda información del empleado en employees.pkl

Ejecuta actualización general

Evita duplicados por número de empleado.

🔹 actualizar_embeddings()

Esta función:

Recorre todas las imágenes en empleados/

Recalcula todos los embeddings

Reconstruye completamente:

embeddings.npy

employees.pkl

Garantiza consistencia entre fotografías y vectores almacenados.

🔹 eliminar_empleado()

Proceso:

Busca el empleado por número

Elimina su embedding correspondiente

Elimina su registro del archivo pickle

Borra su fotografía

Reconstruye los embeddings restantes

🔹 cargar_modelo()

Inicializa el modelo buffalo_l con:

Tamaño de detección: (160,160)

Contexto GPU: ctx_id=0 (si está disponible)

🔹 cargar_datos()

Valida la integridad del sistema:

Crea archivos vacíos si no existen

Reinicializa archivos corruptos

Garantiza matriz válida de forma (N, 512)

Retorna:

empleados, embeddings

---

## 💾 Persistencia de Datos

embeddings.npy

Matriz NumPy de forma (N, 512)

Cada fila representa un embedding facial

employees.pkl

Estructura:

[
    {
        "numero_empleado": "1001",
        "nombre": "Juan Perez",
        "foto": "1001.jpg"
    }
]

---

## ▶️ Ejecución del Sistema

Iniciar API

uvicorn main:app --reload

Iniciar Streamlit

streamlit run app.py



