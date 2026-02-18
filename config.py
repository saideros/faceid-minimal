import os

# Carpeta de fotos de empleados
CARPETA_EMPLEADOS = os.path.join(os.path.dirname(__file__), "empleados")

# Archivos locales
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "embeddings_arcface.npy")
EMPLOYEES_PATH = os.path.join(os.path.dirname(__file__), "empleados_arcface.pkl")
