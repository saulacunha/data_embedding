import os
from pathlib import Path
import time



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


if __name__ == '__main__':
    main()
