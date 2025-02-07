import logging
import os
import time
import pickle
from qdrant_client.models import PointStruct
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, Filter, FieldCondition, MatchValue

class QdrantDBClient:
    def __init__(self,url,api_key, collection_name=None):
        """
        Initializes a QdrantDB client.
        :param url:
        :param api_key:
        """
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name=collection_name

    def create_collection_with_vectors(self, vector_configs,collection_name='default'):
        """
        Create a collection in Qdrant with the specified vectors.
        :param vector_configs:
        :param collection_name:
        :return:
        """
        self.collection_name=collection_name

        try:
            collections = self.client.get_collections().collections
            existing_collections = [col.name for col in collections]

            if collection_name in existing_collections:
                print(f"La colección '{collection_name}' ya existe. No se realizará ninguna acción.")
                return
        except Exception as e:
            print(f"Error verificando colecciones existentes: {e}")
            raise

        vectors_config = {
            config["name"]: VectorParams(
                size=config.get("size", 384),  # Tamaño por defecto es 384
                distance=getattr(Distance, config.get("distance", "COSINE"))  # Distancia por defecto es COSINE
            )
            for config in vector_configs
        }

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config
        )
        print(f"Colección '{collection_name}' creada con los siguientes vectores:")
        for name, params in vectors_config.items():
            print(f"- {name}: Tamaño={params.size}, Distancia={params.distance}")

    def upload_to_qdrant(self,processed_articles):
        """
        Upload vectors and data to Qdrant.
        :param processed_articles:  List of processed articles.
        :return:
        """
        points = [
            PointStruct(
                id=str(article["id_art"]),
                vector={
                    "title_vector": article["title_vector"],
                    "mean_content_vector": article["mean_content_vector"]
                },
                payload=article["metadata"]
            )
            for article in processed_articles
        ]

        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"{len(points)} vectores y datos insertados en Qdrant.")

    def upload_to_qdrant_in_batches(self, processed_articles, batch_size=500, max_retries=3, retry_delay=2, failed_batches_file="failed_batches.pickle"):
        """
        Upload vectors and data to Qdrant in batches with retry logic and saves failed batches for later processing.

        :param processed_articles: List of processed articles.
        :param batch_size: Number of items per batch.
        :param max_retries: Number of retry attempts for failed batches.
        :param retry_delay: Seconds to wait before retrying a failed batch.
        :param failed_batches_file: Path to the pickle file where failed batches will be saved.
        """
        total_articles = len(processed_articles)
        total_uploaded = 0
        failed_batches = []

        # Procesar en lotes
        for i in range(0, total_articles, batch_size):
            batch = processed_articles[i:i + batch_size]
            points = [
                PointStruct(
                    id=str(article["id_art"]),
                    vector={
                        "title_vector": article["title_vector"],
                        "mean_content_vector": article["mean_content_vector"]
                    },
                    payload=article["metadata"]
                )
                for article in batch
            ]

            retries = 0
            while retries < max_retries:
                try:
                    self.client.upsert(collection_name=self.collection_name, points=points)
                    total_uploaded += len(points)
                    logging.info(f"✅ Lote {i // batch_size + 1} insertado con éxito. Total subidos: {total_uploaded}")
                    break  # Si el batch se subió con éxito, salir del loop de reintentos

                except Exception as e:
                    retries += 1
                    logging.warning(f"⚠️ Error en lote {i // batch_size + 1} (Intento {retries}/{max_retries}): {e}")
                    if retries < max_retries:
                        logging.info(f"🔄 Reintentando en {retry_delay} segundos...")
                        time.sleep(retry_delay)
                    else:
                        logging.error(f"❌ Lote {i // batch_size + 1} falló después de {max_retries} intentos. Guardando para retry manual.")
                        failed_batches.append(points)  # Guardar el lote fallido
                        break  # Salir del loop de reintentos y continuar con el siguiente batch

        # Guardar los lotes fallidos en un archivo pickle
        if failed_batches:
            with open(failed_batches_file, "wb") as f:
                pickle.dump(failed_batches, f)
            logging.warning(f"⚠️ Se guardaron {len(failed_batches)} lotes fallidos en {failed_batches_file}.")

        logging.info(f"🚀 Proceso finalizado. Total de puntos subidos con éxito: {total_uploaded}")

        # Intentar nuevamente los lotes fallidos
        self.retry_failed_batches(failed_batches_file)

    def retry_failed_batches(self, failed_batches_file):
        """
        Attempt to reinsert the failed batches from the pickle file.

        :param failed_batches_file: Path to the pickle file with failed batches.
        """
        try:
            with open(failed_batches_file, "rb") as f:
                failed_batches = pickle.load(f)

            if not failed_batches:
                logging.info("✅ No hay lotes fallidos por reintentar.")
                return

            logging.info(f"🔄 Reintentando {len(failed_batches)} lotes fallidos...")

            successful_batches = []
            for batch in failed_batches:
                try:
                    self.client.upsert(collection_name=self.collection_name, points=batch)
                    successful_batches.append(batch)
                    logging.info(f"✅ Lote reinsertado con éxito.")
                except Exception as e:
                    logging.error(f"❌ Error al reinsertar un lote: {e}")

            # Guardar solo los lotes que aún fallaron
            remaining_failures = [b for b in failed_batches if b not in successful_batches]
            if remaining_failures:
                with open(failed_batches_file, "wb") as f:
                    pickle.dump(remaining_failures, f)
                logging.warning(f"⚠️ {len(remaining_failures)} lotes aún fallaron. Guardados en {failed_batches_file}.")
            else:
                logging.info("✅ Todos los lotes fallidos fueron insertados con éxito. Eliminando archivo de fallos.")
                os.remove(failed_batches_file)

        except FileNotFoundError:
            logging.info("✅ No hay archivo de lotes fallidos. No hay nada que reintentar.")

    def search_by_vector(self, vector_name, query_vector, top_k=5):
        """
        Searches for points in Qdrant closest to the given vector.

        Args:
            vector_name (str): Name of the vector to use for the search.
            query_vector (list[float]): Query vector.
            top_k (int): Number of closest results to return.

        Returns:
            list: JSON-serializable results of the search.
        """
        try:
            query_vector_tuple = (vector_name, query_vector)

            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector_tuple,
                limit=top_k
            )
            json_results = [
                {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                } for result in results
            ]
            return json_results
        except Exception as e:
            print(f"Error in vector search: {e}")
            raise

    def search_by_metadata(self, field, value, top_k=5, dimension=1536, vector_name="mean_content_vector"):
        """
        Searches for points in Qdrant using a metadata filter.

        Args:
            field (str): Metadata field to filter by.
            value (str): Value of the metadata field to match.
            top_k (int): Number of results to return.
            dimension (int): Dimension of the dummy vector.

        Returns:
            list: JSON-serializable results of the search.
        """
        try:
            filter_condition = Filter(
                must=[FieldCondition(key=field, match=MatchValue(value=value))]
            )

            dummy_vector = [0.0] * dimension
            query_vector_tuple = (vector_name, dummy_vector)

            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector_tuple,
                query_filter=filter_condition,
                limit=top_k
            )
            json_results = [
                {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                } for result in results
            ]
            return json_results
        except Exception as e:
            print(f"Error in metadata search: {e}")
            raise

    def search_by_vector_and_metadata(self, vector_name, query_vector, field, value, top_k=5):
        """
        Searches for points in Qdrant using both a vector and a metadata filter.

        Args:
            vector_name (str): Name of the vector to use for the search.
            query_vector (list[float]): Query vector.
            field (str): Metadata field to filter by.
            value (str): Value of the metadata field to match.
            top_k (int): Number of results to return.

        Returns:
            list: Results of the search.
        """
        try:
            filter_condition = Filter(
                must=[FieldCondition(key=field, match=MatchValue(value=value))]
            )

            query_vector_tuple = (vector_name, query_vector)

            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector_tuple,
                query_filter=filter_condition,
                limit=top_k
            )
            json_results = [
                {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                } for result in results
            ]
            return json_results
        except Exception as e:
            print(f"Error in combined search: {e}")
            raise

    def search_similar_by_edi_id(self, edi_id, vector_name, top_k=5, dimension=1536):
        """
        Searches for points similar to the vector associated with a given edi_id.

        Args:
            edi_id (str): The edi_id to retrieve the vector.
            vector_name (str): Name of the vector to use for similarity search.
            top_k (int): Number of similar points to retrieve.

        Returns:
            list: JSON-serializable results of the search.
        """
        try:
            # Step 1: Search for the point with the given edi_id
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=(vector_name, [0.0] * dimension),  # Dummy vector with specified name
                query_filter=Filter(must=[FieldCondition(key="edi_id", match=MatchValue(value=edi_id))]),
                with_vectors=True,
                limit=1
            )

            if not results:
                raise ValueError(f"No point found for edi_id: {edi_id}")

            # Extract the vector from the result
            vector = results[0].vector[vector_name]

            # Step 2: Search for similar points using the retrieved vector
            similar_points = self.client.search(
                collection_name=self.collection_name,
                query_vector=(vector_name, vector),
                limit=top_k,
                with_payload=True
            )

            # Convert results to JSON serializable format
            json_results = [
                {
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload
                } for point in similar_points
            ]
            return json_results
        except Exception as e:
            print(f"Error in similar search by edi_id: {e}")
            raise

    def remove_vector_from_points(self, vector_name, batch_size=200):
        """
        Removes a specific vector field from all points in the collection in batches.

        :param vector_name: Name of the vector field to remove.
        :param batch_size: Number of points to process per batch.
        """
        try:
            next_offset = None
            total_removed = 0

            while True:
                # Obtener un lote de puntos
                points, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    with_vectors=True,
                    offset=next_offset
                )

                if not points:
                    break  # No hay más puntos, salimos del loop

                updated_points = []
                for point in points:
                    if vector_name in point.vector:
                        updated_vectors = point.vector.copy()
                        del updated_vectors[vector_name]  # Eliminar el vector

                        updated_points.append(
                            PointStruct(
                                id=point.id,
                                vector=updated_vectors,
                                payload=point.payload
                            )
                        )

                if updated_points:
                    self.client.upsert(collection_name=self.collection_name, points=updated_points)
                    total_removed += len(updated_points)
                    print(f"🗑️ Eliminados {len(updated_points)} puntos en este batch. Total: {total_removed}")

            print(f"✅ Eliminación completa. Total de puntos actualizados: {total_removed}")

        except Exception as e:
            print(f"❌ Error al eliminar el vector '{vector_name}': {e}")

    def copy_vector_field(self, old_name, new_name, batch_size=200):
        """
        Copies values from one vector field to another for all points in the collection in batches.

        :param old_name: The vector field to copy from.
        :param new_name: The new vector field name.
        :param batch_size: Number of points to process per batch.
        """
        try:
            next_offset = None
            total_copied = 0

            while True:
                # Obtener un lote de puntos
                points, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    with_vectors=True,
                    offset=next_offset
                )

                if not points:
                    break  # No hay más puntos, salimos del loop

                updated_points = []
                for point in points:
                    if old_name in point.vector:
                        updated_vectors = point.vector.copy()
                        updated_vectors[new_name] = updated_vectors.pop(old_name)  # Renombrar el vector

                        updated_points.append(
                            PointStruct(
                                id=point.id,
                                vector=updated_vectors,
                                payload=point.payload
                            )
                        )

                if updated_points:
                    self.client.upsert(collection_name=self.collection_name, points=updated_points)
                    total_copied += len(updated_points)
                    logging.info(f"🔄 Copiados {len(updated_points)} puntos en este batch. Total: {total_copied}")

            logging.info(f"✅ Copia completa. Total de puntos actualizados: {total_copied}")

        except Exception as e:
            logging.error(f"❌ Error al copiar vector '{old_name}' a '{new_name}': {e}")

    def optimize_collection(self):
        """
        Triggers optimization for the specified collection by updating its settings.
        """
        try:
            response = self.client.update_collection(collection_name=self.collection_name)
            logging.info(f"✅ Optimization of collection '{self.collection_name}' completed successfully.")
            logging.debug(f"📜 Response: {response}")
        except Exception as e:
            logging.error(f"❌ Error while optimizing collection '{self.collection_name}': {e}")