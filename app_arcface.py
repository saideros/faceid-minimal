from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import base64
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity as cosine_sim
from admin_embeddings import (
    agregar_empleado,
    actualizar_embeddings,
    eliminar_empleado,
    cargar_modelo,
    cargar_datos,
)

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

app = FastAPI(
    title="FaceID API",
    description="""
    API para registro, eliminación y comparación facial de empleados.

    **Endpoints principales:**
    - 🧍 `POST /agregar` → Agrega un empleado.
    - 🗑 `POST /eliminar` → Elimina un empleado.
    - 🔁 `POST /actualizar` → Recalcula embeddings desde las fotos guardadas.
    - 🤖 `POST /comparar` → Compara un rostro con la base de datos.
    """,
    version="1.3.0",
)

# ============================================================
# MODELOS DE DATOS
# ============================================================


class EmpleadoAdd(BaseModel):
    nombre: str = Field(..., example="Santiago Said")
    numero_empleado: str = Field(..., example="PS00501")
    foto_base64: str = Field(..., description="Imagen del rostro en formato Base64")


class EmpleadoNum(BaseModel):
    numero_empleado: str = Field(..., example="PS00501")


class EmpleadoComparar(BaseModel):
    foto_base64: str = Field(..., description="Imagen del rostro en formato Base64")
    numero_empleado: Optional[str] = Field(None, example="PS00501")


class MsgResponse(BaseModel):
    status: str = Field(..., example="success")
    mensaje: str = Field(..., example="Empleado agregado correctamente.")


class MatchItem(BaseModel):
    numero_empleado: str = Field(..., example="PS00501")
    empleado: str = Field(..., example="Santiago Said")
    similitud: float = Field(
        ..., description="Porcentaje de similitud con el rostro comparado", example=98.75
    )


class RostroDetectado(BaseModel):
    rostro_detectado: List[int] = Field(..., example=[337, 395, 824, 1101])
    matches: List[MatchItem]


class CompararResponse(BaseModel):
    resultados: List[RostroDetectado]


# ============================================================
# CARGA DE MODELO Y PARÁMETROS
# ============================================================

print("[INFO] Cargando modelo facial...")
model = cargar_modelo()

print("[INFO] Cargando datos de empleados y embeddings...")
empleados, embeddings = cargar_datos()

THRESHOLD = 0.45
MIN_FACE_AREA_RATIO = 0.05


# ============================================================
# ENDPOINTS
# ============================================================


@app.post("/agregar", tags=["Empleados"], summary="Agregar empleado", response_model=MsgResponse)
async def endpoint_agregar_empleado(data: EmpleadoAdd):
    """
    Guarda un nuevo empleado y genera su embedding facial.
    También actualiza los embeddings en memoria automáticamente.
    """
    global empleados, embeddings

    try:
        foto_bytes = base64.b64decode(data.foto_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Imagen base64 inválida")

    resultado = agregar_empleado(data.nombre, data.numero_empleado, foto_bytes)

    # 🔄 Recargar en memoria después de agregar
    empleados, embeddings = cargar_datos()

    if isinstance(resultado, dict):
        return resultado

    return {"status": "success", "mensaje": str(resultado)}


@app.post("/eliminar", tags=["Empleados"], summary="Eliminar empleado", response_model=MsgResponse)
async def endpoint_eliminar(data: EmpleadoNum):
    """
    Elimina un empleado de la base de datos y actualiza los embeddings en memoria.
    """
    global empleados, embeddings

    resultado = eliminar_empleado(data.numero_empleado)

    # 🔄 Recargar en memoria después de eliminar
    empleados, embeddings = cargar_datos()

    if isinstance(resultado, dict):
        return resultado

    return {"status": "success", "mensaje": str(resultado)}


@app.post("/actualizar", tags=["Empleados"], summary="Actualizar embeddings", response_model=MsgResponse)
async def endpoint_actualizar():
    """
    Recalcula los embeddings de todos los empleados desde las fotos guardadas.
    """
    global empleados, embeddings

    resultado = actualizar_embeddings()

    # 🔄 Recargar en memoria tras actualización
    empleados, embeddings = cargar_datos()

    return {"status": "success", "mensaje": resultado}


@app.post(
    "/comparar",
    tags=["Reconocimiento"],
    summary="Comparar rostro con base de datos",
)
async def comparar_base64(data: EmpleadoComparar, request: Request):
    global empleados, embeddings

    try:
        img_bytes = base64.b64decode(data.foto_base64)
        img_array = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(
            status_code=400, detail="Imagen inválida (no se pudo decodificar)"
        )

    if image is None:
        raise HTTPException(status_code=400, detail="Imagen inválida")

    if embeddings is None or len(embeddings) == 0:
        raise HTTPException(
            status_code=400, detail="No hay empleados registrados para comparar."
        )

    h_img, w_img, _ = image.shape
    faces = model.get(image)

    if not faces:
        raise HTTPException(status_code=400, detail="No se detectó rostro")

    # 🟢 Seleccionar el rostro más grande
    largest_face = None
    max_area = 0

    for f in faces:
        x1, y1, x2, y2 = map(int, f.bbox)
        area = (x2 - x1) * (y2 - y1)

        if area > max_area:
            max_area = area
            largest_face = f

    x1, y1, x2, y2 = map(int, largest_face.bbox)

    ratio = ((x2 - x1) * (y2 - y1)) / (w_img * h_img)
    if ratio < MIN_FACE_AREA_RATIO:
        raise HTTPException(
            status_code=400,
            detail="Rostro demasiado pequeño (posible credencial)",
        )

    # 🔍 Comparar embedding
    face_vec = largest_face.normed_embedding.reshape(1, -1)
    sims = cosine_sim(face_vec, np.array(embeddings))[0]

    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])

    if best_sim < THRESHOLD:
        raise HTTPException(
            status_code=404,
            detail="No se encontró coincidencia confiable",
        )

    emp = empleados[best_idx]

    return {
        "rostro": [x1, y1, x2, y2],
        "nombre": emp["nombre"],
        "numero_empleado": emp["numero_empleado"],
    }


@app.get(
    "/health",
    tags=["Sistema"],
    summary="Verifica estado del servidor",
    response_model=MsgResponse,
)
async def health():
    """
    Verifica que la API esté funcionando correctamente.
    """
    return {"status": "success", "mensaje": "ok"}
