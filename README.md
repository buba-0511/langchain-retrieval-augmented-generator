# 🤖 Retrieval-Augmented Generation (RAG) System

### OpenAI + Pinecone + LangChain

Este repositorio contiene la implementación profesional de un sistema **RAG (Generación Aumentada por Recuperación)**. El proyecto automatiza la extracción de conocimientos desde fuentes web no estructuradas para responder preguntas con datos precisos y actualizados.

---

## 🏗️ Arquitectura del Sistema

El flujo de datos se basa en el estándar de la industria para aplicaciones de IA:

1.  **Ingestión (Data Loading):** Extracción de contenido del blog _LLM Powered Autonomous Agents_ de Lilian Weng mediante `WebBaseLoader`.
2.  **Procesamiento (Splitting):** Segmentación semántica con `RecursiveCharacterTextSplitter` (Chunks: 1000, Overlap: 200).
3.  **Vectorización (Embeddings):** Generación de vectores de alta dimensionalidad con `text-embedding-3-small`.
4.  **Indexación (Vector Store):** Gestión de vectores en **Pinecone** utilizando una métrica de similitud de coseno.
    ![alt text](<Screenshot 2026-02-22 at 12.47.11 PM.png>)
5.  **Inferencia (RAG Chain):** Orquestación de recuperación y respuesta mediante `gpt-4o-mini` y LangChain.
    ![alt text](<Screenshot 2026-02-22 at 1.10.35 PM.png>)

---

## 🛠️ Stack Tecnológico

- **Lenguaje:** Python 3.9+
- **Framework de IA:** LangChain & LangChain Community
- **LLM:** OpenAI GPT-4o-mini
- **Base de Datos Vectorial:** Pinecone (Index Dimension: 1536)
- **Monitoreo:** LangSmith (Observabilidad opcional)

---

## 🚀 Guía de Instalación y Uso

1. **Clonar el repositorio**

   ```bash
   git clone https://github.com/buba-0511/langchain-retrival-augmented-ganerator.git
   cd openai-rag-pinecone-system
   ```

2. **Configurar el entorno virtual e instalar dependencias**

   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configurar Variables de Entorno:**

   Crea un archivo .env en la raíz del proyecto con el siguiente formato:

   ```bash
   OPENAI_API_KEY=tu_openai_api_key
   PINECONE_API_KEY=tu_pinecone_api_key
   PINECONE_INDEX_NAME=tu_nombre_de_indice
   # Opcional para monitoreo:
   LANGSMITH_TRACING="true"
   LANGSMITH_API_KEY=tu_langsmith_key
   ```

## 💻 Ejemplo de Uso y Salida

Para ejecutar el sistema:

```bash
python main.py
```

![alt text](<Screenshot 2026-02-22 at 12.47.37 PM.png>)

## 📁 Estructura de archivos

**main.py:** Lógica principal de la cadena RAG e indexación.

**requirements.txt:** Dependencias del proyecto.

**.gitignore:** Configuración para evitar la subida de archivos sensibles (como .env).
