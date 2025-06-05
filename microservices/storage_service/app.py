import os
import time

INTERVAL = int(os.getenv('STORAGE_INTERVAL', '60'))


def main():
    while True:
        print('Storage service running...')
        time.sleep(INTERVAL)


if __name__ == '__main__':
    main()
