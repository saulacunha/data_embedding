# Query Service

Proporciona una API básica para consultar los vectores almacenados en Qdrant (en desarrollo).

## Campos relevantes del YAML
```yaml
query:
  build: ./microservices/query_service
  ports:
    - "8000:8000"
  environment:
    - QDRANT_HOST=${QDRANT_HOST}
    - QDRANT_PORT=${QDRANT_PORT}
    - QUERY_INTERVAL=${QUERY_INTERVAL}
```
- **QDRANT_HOST/PORT**: origen de la información vectorial.
- **QUERY_INTERVAL**: intervalo de trabajo del servicio.

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
