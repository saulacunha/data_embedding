import streamlit as st
from services.storage_manager import StorageManager
from services.embedding_manager import EmbeddingManager
from services.qdrant_manager import QdrantManager

# Crear instancias de las clases
storage = StorageManager()
embedding = EmbeddingManager()
qdrant = QdrantManager()

# Organizar por pestañas
tab1, tab2, tab3 = st.tabs(["📂 Manage Storage", "🔍 Create Embeddings", "🔗 Qdrant Operations"])

with tab1:
    st.header("Manage Storage")
    # Lógica para subida, listado y consulta de archivos

with tab2:
    st.header("Create Embeddings")
    # Lógica para procesar texto y crear embeddings

with tab3:
    st.header("Qdrant Operations")
    # Lógica para insertar y buscar en Qdrant
