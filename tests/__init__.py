import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.join(ROOT_DIR, 'src'))