import logging
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

# Configurar logging
logging.basicConfig(level=logging.INFO)

class DataVectorizer:
    def __init__(self, max_workers=4, use_openai=False, api_key=None):
        self.max_workers = max_workers
        self.use_openai = use_openai


        if use_openai:
            if api_key:
                self.model_name = 'text-embedding-ada-002'
                openai.api_key = api_key
                self.device = None
            else:
                raise ValueError("API key is required for OpenAI.")
        else:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
            self.model_name = 'all-MiniLM-L6-v2'
            self.transformer_vectors = SentenceTransformer(self.model_name)
            print(f"Modelo cargado en: {self.device}")

    def split_into_token_chunks(self, text, model_type):
        """
        Divide un texto en chunks basados en el límite de tokens del modelo.
        """
        if model_type == "openai":
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            max_tokens = 8191
        elif model_type == "transformers":
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            max_tokens = 512
        else:
            raise ValueError("Modelo no soportado.")

        tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens, add_special_tokens=False)
        chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
        return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

    def _process_article(self, article, model_type):
        try:

            title = article.get("title", "")
            content = article.get("content", "")

            if not content.strip():
                print(f"Artículo ID {article.get('id_art', 'N/A')} tiene contenido vacío. Se omitirá.")
                return None

            # Dividir contenido en chunks de hasta 512 tokens
            content_chunks = self.split_into_token_chunks(content, model_type)

            # Procesar el título y los chunks con el modelo
            if model_type == "openai":
                title_vector = openai.Embedding.create(input=title, model=self.model_name)["data"][0]["embedding"]
                chunk_vectors = [
                    openai.Embedding.create(input=chunk, model=self.model_name)["data"][0]["embedding"]
                    for chunk in content_chunks
                ]
            elif model_type == "transformers":
                title_vector = self.transformer_vectors.encode(title) if title else None
                chunk_vectors = [self.transformer_vectors.encode(chunk) for chunk in content_chunks]
            else:
                raise ValueError("Modelo no soportado.")

            # Combinar los vectores de los chunks
            chunk_vectors_tensor = torch.tensor(np.array(chunk_vectors), device=self.device)
            mean_content_vector = torch.mean(chunk_vectors_tensor, dim=0) if chunk_vectors else torch.zeros(
                chunk_vectors_tensor.size(1), device=self.device)

            return {
                "id_art": article.get("id_art", ""),
                "edi_id": article.get("edi_id", ""),
                "title_vector": title_vector,
                "mean_content_vector": mean_content_vector,
                "metadata": {
                    "title": title,
                    "autor": article.get("autor", ""),
                    "date": article.get("date", ""),
                    "url": article.get("url", ""),
                    "edi_id": article.get("edi_id", "")
                }
            }
        except Exception as e:
            print(f"Error procesando artículo ID {article.get('id_art', 'N/A')}: {e}")
            return None

    def generate_vectors_parallel(self, articles):
        if not articles:
            logging.warning("No hay artículos para procesar.")
            return []

        processed_articles = []
        model_type = "openai" if self.use_openai else "transformers"

        logging.info(f"Iniciando el procesamiento de {len(articles)} artículos con el modelo '{model_type}'.")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._process_article, article, model_type): article["id_art"] for article in
                       articles}

            # Agregar barra de progreso
            for future in tqdm(as_completed(futures), total=len(articles), desc="Procesando artículos"):
                article_id = futures[future]
                try:
                    result = future.result()
                    if result:
                        processed_articles.append(result)
                        logging.info(f"Artículo ID {article_id} procesado con éxito.")
                    else:
                        logging.warning(f"Artículo ID {article_id} no pudo ser procesado.")
                except Exception as e:
                    logging.error(f"Error procesando artículo ID {article_id}: {e}")

        logging.info(f"Vectores generados para {len(processed_articles)} artículos de un total de {len(articles)}.")
        return processed_articles

    def vectorize_text(self, text):
        """
        Vectorizes a given text using the configured model.

        Args:
            text (str): The text to vectorize.

        Returns:
            list: The vector representation of the text.
        """
        try:
            if self.use_openai:
                vector = openai.Embedding.create(input=text, model=self.model_name)["data"][0]["embedding"]
            else:
                vector = self.transformer_vectors.encode(text)
            return vector
        except Exception as e:
            logging.error(f"Error vectorizing text: {e}")
            return None