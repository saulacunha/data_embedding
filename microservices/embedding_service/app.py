import os
import time

INTERVAL = int(os.getenv('EMBEDDING_INTERVAL', '60'))
CHUNK_STRATEGY = os.getenv('CHUNK_STRATEGY', 'full')


def main():
    while True:
        print(f"Embedding service running with chunk strategy: {CHUNK_STRATEGY}...")
        time.sleep(INTERVAL)


if __name__ == '__main__':
    main()
