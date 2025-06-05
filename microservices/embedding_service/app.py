import os
import time

INTERVAL = int(os.getenv('EMBEDDING_INTERVAL', '60'))


def main():
    while True:
        print('Embedding service running...')
        time.sleep(INTERVAL)


if __name__ == '__main__':
    main()
