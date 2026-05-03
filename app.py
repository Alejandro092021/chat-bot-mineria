import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import io
import pandas as pd
from datetime import datetime
from PyPDF2 import PdfReader
import platform  # NUEVA MEJORA: Para detectar Windows vs Nube
import streamlit.components.v1 as components

# --- IMPORTACIONES PARA RAG Y OCR ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pytesseract
from pdf2image import convert_from_bytes
from streamlit_mic_recorder import mic_recorder

# --- 0. CONFIGURACIÓN DE RUTAS MULTIPLATAFORMA (MEJORA) ---
if platform.system() == "Windows":
    # Rutas locales (Alejandro)
    ffmpeg_path = r'C:\FFmpeg\ffmpeg-8.1-essentials_build\bin'
    poppler_path = r'C:\poppler\poppler-25.12.0\Library\bin' 
    tesseract_dir = r'C:\Program Files\Tesseract-OCR'
    tesseract_exe = os.path.join(tesseract_dir, 'tesseract.exe')

    os.environ['TESSDATA_PREFIX'] = os.path.join(tesseract_dir, 'tessdata')
    pytesseract.pytesseract.tesseract_cmd = tesseract_exe

    for path in [ffmpeg_path, poppler_path, tesseract_dir]:
        if os.path.exists(path) and path not in os.environ["PATH"]:
            os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
else:
    # Configuración para Producción (Linux/Streamlit Cloud)
    # Los paquetes se instalan vía packages.txt automáticamente
    pytesseract.pytesseract.tesseract_cmd = 'tesseract'
    poppler_path = None # En Linux se usa el comando global

# 1. CONFIGURACIÓN DE APIS E INICIALIZACIÓN
load_dotenv()

# MEJORA: Soporte para Secrets de Streamlit Cloud si no hay .env
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)

# Motor de búsqueda semántica
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

st.set_page_config(page_title="MEMIA - Seguridad Minera", page_icon="👷‍♂️")

# --- OCULTAR ELEMENTOS DE STREAMLIT ---


# 1. Inyección de Parche en la capa superior (DOM Padre)
# Este script sale de tu app y pega la franja en el navegador principal
components.html(
    """
    <script>
    const patch = window.parent.document.createElement('div');
    patch.id = 'blocker-final';
    patch.style.position = 'fixed';
    patch.style.bottom = '0';
    patch.style.right = '0';
    patch.style.width = '220px'; /* Un poco más ancho para asegurar */
    patch.style.height = '70px';
    patch.style.backgroundColor = '#0e1117'; /* Color oscuro de Streamlit */
    patch.style.zIndex = '2147483647'; /* El máximo valor posible */
    patch.style.pointerEvents = 'all';
    patch.style.display = 'block';
    
    // Lo insertamos en el cuerpo principal del navegador
    window.parent.document.body.appendChild(patch);
    </script>
    """,
    height=0,
)

# 2. CSS para limpiar el resto de la interfaz interna
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden; display: none !important;}
    header {visibility: hidden; display: none !important;}
    footer {visibility: hidden; display: none !important;}
    .stAppDeployButton {display:none !important;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Asistente de Seguridad (MEMIA) 🤖")

if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

# --- 2. FUNCIÓN DE PROCESAMIENTO HÍBRIDO (TEXTO + OCR) ---
def procesar_a_vectores(pdf_docs):
    texto_completo = ""
    
    for pdf in pdf_docs:
        pdf_bytes = pdf.read()
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        
        for page_num, page in enumerate(pdf_reader.pages):
            texto_extraido = page.extract_text()
            
            # MEJORA: Umbral de texto más estricto para forzar OCR si el texto es basura
            if texto_extraido and len(texto_extraido.strip()) > 100:
                texto_completo += texto_extraido + "\n"
            else:
                try:
                    # Ajuste de ruta de poppler según plataforma
                    p_path = os.path.abspath(poppler_path) if poppler_path else None
                    images = convert_from_bytes(
                        pdf_bytes, 
                        first_page=page_num+1, 
                        last_page=page_num+1,
                        poppler_path=p_path
                    )
                    for img in images:
                        texto_ocr = pytesseract.image_to_string(img, lang='spa')
                        texto_completo += texto_ocr + "\n"
                except Exception as e:
                    st.error(f"Error en OCR página {page_num+1}: {e}")

    if not texto_completo.strip():
        return None
    
    # MEJORA: Chunks más pequeños (700) para mayor precisión en la respuesta 1
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = text_splitter.split_text(texto_completo)
    
    return FAISS.from_texts(chunks, embeddings) if chunks else None

# --- 3. FUNCIONES DE APOYO (LOGS Y AUDIO) ---
def guardar_log(pregunta):
    nuevo_log = {"Fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Consulta": pregunta}
    df = pd.DataFrame([nuevo_log])
    df.to_csv("logs_seguridad.csv", mode='a', header=not os.path.exists("logs_seguridad.csv"), 
              index=False, sep=';', encoding='utf-8-sig')

def transcribir_audio(audio_data):
    try:
        temp_file = f"temp_voice_{datetime.now().strftime('%H%M%S')}.wav"
        with open(temp_file, "wb") as f:
            f.write(audio_data['bytes'])
        with open(temp_file, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(temp_file, file.read()), model="whisper-large-v3", 
                response_format="text", language="es"
            )
        os.remove(temp_file)
        return transcription
    except Exception as e:
        st.error(f"Error en audio: {e}")
        return None

# --- 4. PANEL DE CONTROL (BARRA LATERAL) ---
with st.sidebar:
    st.header("Panel de Control")
    rol = st.radio("Selecciona tu rol:", ["Operario", "Administrador"])
    
    if rol == "Administrador":
        if not st.session_state.autenticado:
            password = st.text_input("Contraseña:", type="password")
            if st.button("Ingresar"):
                if password == "admin123":
                    st.session_state.autenticado = True
                    st.rerun()
                else:
                    st.error("Incorrecta")
        else:
            st.success("Sesión de Administrador Activa")
            
            st.subheader("📊 Historial de Riesgos")
            if os.path.exists("logs_seguridad.csv"):
                try:
                    df_logs = pd.read_csv("logs_seguridad.csv", sep=';')
                    st.dataframe(df_logs.tail(10), use_container_width=True)
                    
                    csv_data = df_logs.to_csv(index=False, sep=';', encoding='utf-8-sig').encode('utf-8-sig')
                    st.download_button(
                        label="📥 Descargar Reporte (Excel)",
                        data=csv_data,
                        file_name=f"Reporte_MEMIA_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                    )
                except Exception:
                    st.warning("Error al cargar historial.")
            else:
                st.info("No hay logs registrados.")

            st.subheader("Carga de Normativas (RAG + OCR)")
            uploaded_files = st.file_uploader("Sube manuales o fotos (PDF)", type="pdf", accept_multiple_files=True)
            if uploaded_files:
                with st.spinner("Procesando documentos e imágenes..."):
                    vector_db_result = procesar_a_vectores(uploaded_files)
                    if vector_db_result:
                        st.session_state.vector_db = vector_db_result
                        st.success("Contenido indexado correctamente.")
            
            if st.button("Cerrar Sesión Admin"):
                st.session_state.autenticado = False
                st.rerun()
    else:
        st.session_state.autenticado = False
        st.info("Modo Operario activo.")

    st.subheader("🎙️ Consulta por Voz")
    audio_input = mic_recorder(start_prompt="Grabar pregunta", stop_prompt="Detener", key='recorder')

    # MEJORA: Botón de Reinicio Total (Limpia Chat e Índice de documentos)
    if st.button("🔥 Reinicio Total (Empezar de cero)"):
        if "messages" in st.session_state:
            del st.session_state["messages"]
        if "vector_db" in st.session_state:
            del st.session_state["vector_db"]
        st.rerun()

# --- 5. LÓGICA DEL CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", 
                                  "content": """Eres MEMIA, un Chatbot de Seguridad experto en minería. 
            Tu objetivo es ayudar a operarios y contratistas a consultar normas de seguridad de forma ágil.
            
            REGLAS DE COMPORTAMIENTO:
            1. Tu base de conocimientos PRINCIPAL son los documentos indexados.
            2. Si la respuesta está en el documento, extráela de forma CONCISA. No resumas todo el documento, solo responde lo que se te preguntó.
            3. Si la información NO está en el documento, utiliza tu conocimiento general de seguridad minera para responder, pero advierte: 'Información general (no presente en manuales)'.
            4. Prioriza la seguridad: si algo es peligroso, adviértelo con claridad.
            5. Responde de forma concisa (estás hablando con alguien en el punto de trabajo).
            6. Si te preguntan algo fuera de seguridad minera, amablemente redirige la conversación al tema de prevención de riesgos."""}]

for msg in st.session_state.messages:
    if msg["role"] != "system":
        st.chat_message(msg["role"]).write(msg["content"])

prompt = st.chat_input("Haz tu consulta de seguridad...")

if audio_input and ('last_id' not in st.session_state or st.session_state.last_id != audio_input['id']):
    texto_voz = transcribir_audio(audio_input)
    if texto_voz:
        prompt = texto_voz
        st.session_state.last_id = audio_input['id']

if prompt:
    guardar_log(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    contexto = ""
    if "vector_db" in st.session_state:
        # MEJORA: k=2 para evitar que la IA reciba demasiada información irrelevante
        docs = st.session_state.vector_db.similarity_search(prompt, k=2)
        contexto = "\n".join([d.page_content for d in docs])

    try:
        mensajes_ia = st.session_state.messages.copy()
        if contexto:
            mensajes_ia.append({"role": "system", "content": f"Contexto técnico extraído del manual (USA ESTO PARA RESPONDER): {contexto}"})

        # MEJORA: Temperatura baja (0.1) para evitar alucinaciones en seguridad
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=mensajes_ia, 
            stream=True,
            temperature=0.1
        )
        with st.chat_message("assistant"):
            msg_ai = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": msg_ai})
    except Exception as e:
        st.error(f"Error en la comunicación con la IA: {e}")