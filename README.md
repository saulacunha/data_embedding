# Document Embedding Microservices

This project provides a set of microservices for ingesting text, generating embeddings with OpenAI, storing them in Qdrant and exposing a simple API. Every service runs in its own container and communicates through RabbitMQ.

## Structure
- `microservices/ingest_service` – obtains and normalizes text.
- `microservices/embedding_service` – generates embeddings.
- `microservices/storage_service` – inserts vectors into Qdrant.
- `microservices/query_service` – placeholder for search API.
- `services/` – shared utility modules.

## Quick start
1. Copy `.env.example` to `.env` and adjust credentials.
2. Build and start the stack:

```bash
make build
make up
```

Visit the RabbitMQ management UI at `http://localhost:15672` and Qdrant at `http://localhost:6333`.

## Configuración YAML
El archivo `docker-compose.yml` describe cómo se orquestan los microservicios y las variables de entorno que utilizan. Cada sección puede personalizarse para apuntar a diferentes orígenes de datos o colecciones de Qdrant.

Ejemplo de fragmento:

```yaml
embedding:
  build: ./microservices/embedding_service
  environment:
    - RABBITMQ_HOST=${RABBITMQ_HOST}
    - QDRANT_HOST=${QDRANT_HOST}
    - OPENAI_API_KEY=${OPENAI_API_KEY}
```

Tras editar `.env` y el YAML, la pila completa puede iniciarse con:

```bash
docker compose up -d
```

## Development
Each service has a simple `app.py` that polls for work. Polling intervals and connection settings are taken from environment variables defined in `.env`.
