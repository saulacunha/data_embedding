# Ingest Service

Obtiene y normaliza texto para publicarlo en RabbitMQ.

## Campos relevantes del YAML
```yaml
ingest:
  build: ./microservices/ingest_service
  environment:
    - RABBITMQ_HOST=${RABBITMQ_HOST}
    - RABBITMQ_PORT=${RABBITMQ_PORT}
    - INGEST_INTERVAL=${INGEST_INTERVAL}
```
- **RABBITMQ_HOST/PORT**: cola de mensajes de salida.
- **INGEST_INTERVAL**: intervalo de consulta.

## Ejecución independiente
1. Instala las dependencias:
   ```bash
   pip install -r ../../requirements.txt
   ```
2. Exporta las variables de entorno anteriores.
3. Ejecuta:
   ```bash
   python app.py
   ```
