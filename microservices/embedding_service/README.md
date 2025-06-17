# Embedding Service

Genera vectores usando OpenAI a partir de los mensajes recibidos y los envía a Qdrant.

## Campos relevantes del YAML
```yaml
embedding:
  build: ./microservices/embedding_service
  environment:
    - RABBITMQ_HOST=${RABBITMQ_HOST}
    - RABBITMQ_PORT=${RABBITMQ_PORT}
    - QDRANT_HOST=${QDRANT_HOST}
    - QDRANT_PORT=${QDRANT_PORT}
    - OPENAI_API_KEY=${OPENAI_API_KEY}
    - EMBEDDING_INTERVAL=${EMBEDDING_INTERVAL}
```
- **RABBITMQ_HOST/PORT**: origen de mensajes con texto.
- **QDRANT_HOST/PORT**: base de vectores de destino.
- **OPENAI_API_KEY**: credencial para generar embeddings.
- **EMBEDDING_INTERVAL**: pausa entre iteraciones.

## Ejecución independiente
1. Instala dependencias:
   ```bash
   pip install -r ../../requirements.txt
   ```
2. Exporta las variables indicadas.
3. Ejecuta:
   ```bash
   python app.py
   ```
