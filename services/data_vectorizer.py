import logging
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


class DataVectorizer:
    def __init__(self, max_workers=4, use_openai=False, api_key=None):
        """
        Initializes the DataVectorizer class.

        Args:
            max_workers (int): Number of threads for parallel processing.
            use_openai (bool): Whether to use OpenAI API for embeddings.
            api_key (str, optional): API key for OpenAI.
        """
        self.max_workers = max_workers
        self.use_openai = use_openai

        if use_openai:
            self.api_key = api_key
            if not self.api_key:
                raise ValueError("API key is required for OpenAI.")

            self.client = openai.Client(api_key=self.api_key)
            self.model_name = 'text-embedding-ada-002'
            self.device = None
        else:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
            self.model_name = 'all-MiniLM-L6-v2'
            self.transformer_vectors = SentenceTransformer(self.model_name)
            print(f"Model loaded on: {self.device}")

    def split_into_token_chunks(self, text, model_type):
        """
        Splits a text into chunks based on the model's token limit.

        Args:
            text (str): The text to split.
            model_type (str): The type of model to determine token limits.

        Returns:
            list: A list of text chunks.
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
            raise ValueError("Unsupported model type.")

        tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens, add_special_tokens=False)
        chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
        return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

    def _process_article(self, article, model_type):
        """
        Processes an article to generate vector embeddings for the title and content.

        Args:
            article (dict): The article data containing 'title' and 'content'.
            model_type (str): The type of model used ('openai' or 'transformers').

        Returns:
            dict: A dictionary containing the article ID, vectors, and metadata.
        """
        try:
            title = article.get("title", "")
            content = article.get("content", "")

            if not content.strip():
                logging.warning(f"Article ID {article.get('id_art', 'N/A')} has empty content. Skipping.")
                return None

            content_chunks = self.split_into_token_chunks(content, model_type)

            if model_type == "openai":
                response_title = self.client.embeddings.create(input=title, model=self.model_name)
                title_vector = response_title.data[0].embedding if title else None

                chunk_vectors = [
                    openai.embeddings.create(input=chunk, model=self.model_name).data[0].embedding
                    for chunk in content_chunks
                ]
            elif model_type == "transformers":
                title_vector = self.transformer_vectors.encode(title) if title else None
                chunk_vectors = [self.transformer_vectors.encode(chunk) for chunk in content_chunks]
            else:
                raise ValueError("Unsupported model type.")

            chunk_vectors_tensor = torch.tensor(np.array(chunk_vectors), device=self.device)
            mean_content_vector = torch.mean(chunk_vectors_tensor, dim=0) if chunk_vectors else torch.zeros(
                chunk_vectors_tensor.size(1), device=self.device)

            return {
                "id_art": article.get("id_art", ""),
                "edi_id": article.get("edi_id", ""),
                "title_vector": title_vector,
                "mean_content_vector": mean_content_vector.tolist(),
                "metadata": {
                    "title": title,
                    "author": article.get("author", ""),
                    "date": article.get("date", ""),
                    "url": article.get("url", ""),
                    "edi_id": article.get("edi_id", "")
                }
            }
        except Exception as e:
            logging.error(f"Error processing article ID {article.get('id_art', 'N/A')}: {e}")
            return None

    def generate_vectors_parallel(self, articles):
        """
        Processes a list of articles in parallel to generate vector embeddings.

        Args:
            articles (list): A list of dictionaries containing article data.

        Returns:
            list: A list of dictionaries containing processed articles with vectors.
        """
        if not articles:
            logging.warning("No articles to process.")
            return []

        processed_articles = []
        model_type = "openai" if self.use_openai else "transformers"

        logging.info(f"Processing {len(articles)} articles using model '{model_type}'...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._process_article, article, model_type): article["id_art"] for article in
                       articles}

            for future in tqdm(as_completed(futures), total=len(articles), desc="Processing articles"):
                article_id = futures[future]
                try:
                    result = future.result()
                    if result:
                        processed_articles.append(result)
                        logging.info(f"Article ID {article_id} processed successfully.")
                    else:
                        logging.warning(f"Article ID {article_id} could not be processed.")
                except Exception as e:
                    logging.error(f"Error processing article ID {article_id}: {e}")

        logging.info(f"Generated vectors for {len(processed_articles)} out of {len(articles)} articles.")
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
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model_name
                )
                vector = response.data[0].embedding  # Updated OpenAI API usage
            else:
                vector = self.transformer_vectors.encode(text)
            return vector
        except Exception as e:
            logging.error(f"Error vectorizing text: {e}")
            return None