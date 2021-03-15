import os

LIB_FOLDER = os.path.dirname(__file__)
TEST_FOLDER = os.path.join(LIB_FOLDER, '../', 'tests')
DATA_FOLDER = os.path.join(LIB_FOLDER, 'datasets')


if __name__ == '__main__':
    print(DATA_FOLDER)