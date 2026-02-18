import streamlit as st
import requests
import base64

# -------------------------
# Configuración general
# -------------------------
st.set_page_config(page_title="Reconocimiento Facial", layout="centered")

st.title("Sistema de Registro y Validación Facial")

# -------------------------
# Función para convertir imagen a base64
# -------------------------
def convertir_a_base64(file):
    bytes_data = file.read()
    encoded = base64.b64encode(bytes_data).decode("utf-8")
    return encoded

# -------------------------
# Tabs
# -------------------------
tab_alta, tab_validar = st.tabs(["📌 Alta", "🔍 Validar"])

# ======================================================
# TAB ALTA
# ======================================================
with tab_alta:

    st.subheader("Registrar empleado")

    nombre = st.text_input("Nombre")
    numero_empleado = st.text_input("Número de empleado")
    foto = st.file_uploader("Subir fotografía", type=["jpg", "jpeg", "png"], key="alta")

    if st.button("Registrar"):
        if nombre and numero_empleado and foto:

            foto_base64 = convertir_a_base64(foto)

            payload = {
                "nombre": nombre,
                "numero_empleado": numero_empleado,
                "foto_base64": foto_base64
            }

            try:
                response = requests.post(
                    "http://localhost:8090/agregar",
                    json=payload
                )

                if response.status_code == 200:
                    st.success("Empleado registrado correctamente ✅")
                    st.json(response.json())
                else:
                    st.error(f"Error: {response.text}")

            except Exception as e:
                st.error(f"Error de conexión: {e}")

        else:
            st.warning("Completa todos los campos")

# ======================================================
# TAB VALIDAR
# ======================================================
with tab_validar:

    st.subheader("Validar empleado")

    numero_empleado_validar = st.text_input("Número de empleado", key="validar")
    foto_validar = st.file_uploader("Subir fotografía", type=["jpg", "jpeg", "png"], key="foto_validar")

    if st.button("Validar"):
        if numero_empleado_validar and foto_validar:

            foto_base64 = convertir_a_base64(foto_validar)

            payload = {
                "foto_base64": foto_base64,
                "numero_empleado": numero_empleado_validar
            }

            try:
                response = requests.post(
                    "http://localhost:8090/comparar",
                    json=payload
                )

                if response.status_code == 200:
                    st.success("Validación completada ✅")
                    st.json(response.json())
                else:
                    st.error(f"Error: {response.text}")

            except Exception as e:
                st.error(f"Error de conexión: {e}")

        else:
            st.warning("Completa todos los campos")
