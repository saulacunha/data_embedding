# Embedding Service

This service generates embeddings for documents.

## Configuration

Configuration parameters are read from `config/embedding.yaml` located in the
repository root. You can override the path with the `EMBEDDING_CONFIG`
environment variable.

The file defines the following fields:

- **input**: information about the source of documents.
  - `type`: e.g. `local` or `http`.
  - `path`/`endpoint`: location of the data.
  - `credentials`: authentication data if needed.
- **output**: where generated embeddings are stored.
  - `type`: storage backend such as `qdrant`.
  - `destination`: host or file path.
  - `credentials`: authentication data if required.
- **format_in**: format of the incoming documents (`txt`, `pdf`, ...).
- **format_out**: format for the generated embedding payload.
- **processing**: embedding parameters.
  - `model`: model name, default `ada`.
  - `provider`: `local` to run a bundled model or `openai` for an external API.
    When using an external provider, credentials are taken from the `.env` file
    (`OPENAI_API_KEY`) and the endpoint can be specified via the `EMBEDDING_API_URL`
    environment variable.
  - `api_url`: optional API endpoint when using an external provider.
  - `distance`: distance metric used.
  - `dimension`: vector dimension.
  - `chunking`: strategy details for splitting documents.
    - `strategy`: chunking method.
    - `size`: maximum chunk size.
  - `interval`: number of seconds between processing cycles.

An example configuration is provided in `config/embedding.yaml`.
