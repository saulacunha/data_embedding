import os
import time

INTERVAL = int(os.getenv('QUERY_INTERVAL', '60'))


def main():
    while True:
        print('Query service running...')
        time.sleep(INTERVAL)


if __name__ == '__main__':
    main()
