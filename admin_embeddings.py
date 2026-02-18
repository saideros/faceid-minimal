import os
import cv2
import pickle
import numpy as np
from PIL import Image
import io
import shutil
import insightface
from config import CARPETA_EMPLEADOS, EMBEDDINGS_PATH, EMPLOYEES_PATH

# Modelo InsightFace
model = insightface.app.FaceAnalysis(name="buffalo_l", root="./models")
model.prepare(ctx_id=0, det_size=(160,160))

# ---------------------------
# Agregar empleado
# ---------------------------
def agregar_empleado(nombre: str, numero_empleado: str, foto_bytes: bytes) -> dict:
    os.makedirs(CARPETA_EMPLEADOS, exist_ok=True)
    foto_filename = f"{numero_empleado}.jpg"
    foto_path = os.path.join(CARPETA_EMPLEADOS, foto_filename)

    # Guardar foto
    try:
        Image.open(io.BytesIO(foto_bytes)).save(foto_path)
    except Exception as e:
        return {"status": "error", "mensaje": f"❌ Error guardando la foto: {e}"}

    # Cargar embeddings existentes
    embeddings = np.load(EMBEDDINGS_PATH) if os.path.exists(EMBEDDINGS_PATH) else np.empty((0,512))
    empleados_info = pickle.load(open(EMPLOYEES_PATH, "rb")) if os.path.exists(EMPLOYEES_PATH) else []

    # Evitar duplicados
    if any(emp["numero_empleado"] == numero_empleado for emp in empleados_info):
        return {"status": "warning", "mensaje": f"⚠ Empleado {numero_empleado} ya existe."}

    # Leer imagen y detectar rostro
    img_cv = cv2.imread(foto_path)
    faces = model.get(img_cv)
    if not faces:
        return {"status": "error", "mensaje": f"❌ No se detectó rostro en la foto."}

    f = max(faces, key=lambda face: (face.bbox[2]-face.bbox[0])*(face.bbox[3]-face.bbox[1]))

    # Agregar embedding
    embeddings = np.vstack([embeddings, f.normed_embedding])
    empleados_info.append({
        "numero_empleado": numero_empleado,
        "nombre": nombre,
        "foto": foto_filename
    })

    # Guardar nuevos embeddings
    np.save(EMBEDDINGS_PATH, embeddings)
    with open(EMPLOYEES_PATH, "wb") as f:
        pickle.dump(empleados_info, f)

    # Actualizar todos los embeddings al final (sin romper formato)
    msg = actualizar_embeddings()

    return {
        "status": "success",
        "mensaje": f"✅ Empleado agregado correctamente ({nombre}). {msg}"
    }


# ---------------------------
# Actualizar todos los embeddings
# ---------------------------
def actualizar_embeddings() -> str:
    if not os.path.exists(CARPETA_EMPLEADOS):
        return "⚠ No hay carpeta de empleados."

    archivos = [f for f in os.listdir(CARPETA_EMPLEADOS) if f.lower().endswith(".jpg")]
    if not archivos:
        return "⚠ No hay fotos para actualizar."

    # 🔹 Intentar conservar nombres previos
    empleados_previos = {}
    if os.path.exists(EMPLOYEES_PATH):
        try:
            with open(EMPLOYEES_PATH, "rb") as f:
                for emp in pickle.load(f):
                    empleados_previos[emp["numero_empleado"]] = emp["nombre"]
        except Exception:
            empleados_previos = {}

    embeddings = []
    empleados_info = []

    for archivo in archivos:
        numero_empleado = os.path.splitext(archivo)[0]
        foto_path = os.path.join(CARPETA_EMPLEADOS, archivo)
        img_cv = cv2.imread(foto_path)
        faces = model.get(img_cv)
        if not faces:
            continue

        f = max(faces, key=lambda face: (face.bbox[2]-face.bbox[0])*(face.bbox[3]-face.bbox[1]))
        embeddings.append(f.normed_embedding)
        empleados_info.append({
            "numero_empleado": numero_empleado,
            "nombre": empleados_previos.get(numero_empleado, numero_empleado),
            "foto": archivo
        })

    if embeddings:
        np.save(EMBEDDINGS_PATH, np.vstack(embeddings))
        with open(EMPLOYEES_PATH, "wb") as f:
            pickle.dump(empleados_info, f)
        return f"✅ Embeddings actualizados ({len(empleados_info)} empleados)."
    else:
        return "⚠ No se generó ningún embedding válido."



# ---------------------------
# Eliminar empleado
# ---------------------------
def eliminar_empleado(numero_empleado: str) -> dict:
    if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(EMPLOYEES_PATH):
        return {"status": "error", "mensaje": "⚠ No existen archivos de embeddings."}

    embeddings = np.load(EMBEDDINGS_PATH)
    with open(EMPLOYEES_PATH, "rb") as f:
        empleados_info = pickle.load(f)

    # Buscar el índice del empleado
    indices = [i for i, emp in enumerate(empleados_info) if emp["numero_empleado"] == numero_empleado]
    if not indices:
        return {"status": "warning", "mensaje": f"⚠ No se encontró empleado {numero_empleado}."}

    # Eliminar del embedding e info
    for idx in reversed(indices):
        empleados_info.pop(idx)
        embeddings = np.delete(embeddings, idx, axis=0)

    np.save(EMBEDDINGS_PATH, embeddings)
    with open(EMPLOYEES_PATH, "wb") as f:
        pickle.dump(empleados_info, f)

    # Eliminar foto
    foto_path = os.path.join(CARPETA_EMPLEADOS, f"{numero_empleado}.jpg")
    if os.path.exists(foto_path):
        os.remove(foto_path)

    msg = actualizar_embeddings()

    return {
        "status": "success",
        "mensaje": f"✅ Empleado {numero_empleado} eliminado y embeddings actualizados. {msg}"
    }


# ---------------------------
# Cargar modelo y datos
# ---------------------------
def cargar_modelo():
    from insightface.app import FaceAnalysis
    model = FaceAnalysis(name="buffalo_l", root="./models")
    model.prepare(ctx_id=0, det_size=(160,160))
    return model


def cargar_datos():
    """
    Carga los datos de empleados y embeddings.
    Si los archivos no existen, los crea vacíos automáticamente.
    """
    print("[INFO] Cargando datos de empleados y embeddings...")

    # Verifica la carpeta donde se guardan empleados
    os.makedirs(CARPETA_EMPLEADOS, exist_ok=True)

    # Crear archivos vacíos si no existen
    if not os.path.exists(EMPLOYEES_PATH):
        print(f"[WARN] No existe {EMPLOYEES_PATH}, creando archivo vacío...")
        with open(EMPLOYEES_PATH, "wb") as f:
            pickle.dump([], f)

    if not os.path.exists(EMBEDDINGS_PATH):
        print(f"[WARN] No existe {EMBEDDINGS_PATH}, creando archivo vacío...")
        np.save(EMBEDDINGS_PATH, np.empty((0, 512), dtype=np.float32))

    # Cargar datos ya existentes
    with open(EMPLOYEES_PATH, "rb") as f:
        empleados = pickle.load(f)

    embeddings = np.load(EMBEDDINGS_PATH, allow_pickle=False)

    # Validación de formato
    if not isinstance(empleados, list):
        print("[WARN] Archivo de empleados corrupto, reinicializando...")
        empleados = []
        with open(EMPLOYEES_PATH, "wb") as f:
            pickle.dump([], f)

    if len(embeddings.shape) != 2 or embeddings.shape[1] != 512:
        print("[WARN] Archivo de embeddings corrupto, reinicializando...")
        embeddings = np.empty((0, 512), dtype=np.float32)
        np.save(EMBEDDINGS_PATH, embeddings)

    print(f"[INFO] Datos cargados correctamente: {len(empleados)} empleados, {embeddings.shape[0]} embeddings.")
    return empleados, embeddings
