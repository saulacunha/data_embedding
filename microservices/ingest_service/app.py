import os
import time

INTERVAL = int(os.getenv('INGEST_INTERVAL', '60'))


def main():
    while True:
        print('Ingest service running...')
        time.sleep(INTERVAL)


if __name__ == '__main__':
    main()
