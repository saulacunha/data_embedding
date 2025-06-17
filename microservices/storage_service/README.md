# Storage Service

Recibe embeddings y los almacena en la colección de Qdrant.

## Campos relevantes del YAML
```yaml
storage:
  build: ./microservices/storage_service
  environment:
    - RABBITMQ_HOST=${RABBITMQ_HOST}
    - RABBITMQ_PORT=${RABBITMQ_PORT}
    - QDRANT_HOST=${QDRANT_HOST}
    - QDRANT_PORT=${QDRANT_PORT}
    - QDRANT_API_KEY=${QDRANT_API_KEY}
    - STORAGE_INTERVAL=${STORAGE_INTERVAL}
```
- **RABBITMQ_HOST/PORT**: origen de mensajes con embeddings.
- **QDRANT_HOST/PORT** y **QDRANT_API_KEY**: destino de almacenamiento.
- **STORAGE_INTERVAL**: pausa de trabajo.

## Ejecución independiente
1. Instala dependencias:
   ```bash
   pip install -r ../../requirements.txt
   ```
2. Exporta las variables mencionadas.
3. Ejecuta:
   ```bash
   python app.py
   ```
