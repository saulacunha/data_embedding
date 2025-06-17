import os
from pathlib import Path
import time

import yaml

# Default configuration file location two directories above this file
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / 'config' / 'embedding.yaml'


def load_config(path: Path) -> dict:
    """Load YAML configuration from *path*.

    Parameters
    ----------
    path: Path
        Path to the YAML configuration file.
    """
    with open(path, 'r', encoding='utf-8') as fh:
        return yaml.safe_load(fh)


def main():
    config_path = Path(os.getenv('EMBEDDING_CONFIG', DEFAULT_CONFIG_PATH))
    config = load_config(config_path)

    # How often the service runs
    interval = config.get('processing', {}).get('interval', 60)
    model = config.get('processing', {}).get('model')
    provider = config.get('processing', {}).get('provider', 'local')
    api_url = config.get('processing', {}).get('api_url') or os.getenv('EMBEDDING_API_URL')
    distance = config.get('processing', {}).get('distance')

    if provider != 'local':
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise RuntimeError('OPENAI_API_KEY must be set for external providers')

    while True:
        print('Embedding service running...')
        print(f"Model: {model}, provider: {provider}, distance: {distance}")
        if api_url:
            print(f"API URL: {api_url}")
        time.sleep(interval)


if __name__ == '__main__':
    main()
