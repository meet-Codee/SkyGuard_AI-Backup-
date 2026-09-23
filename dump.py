import sys

def dump():
    with open('data_source.py', 'r', encoding='utf-8') as f:
        text = f.read()
    with open('ds_dump.txt', 'w', encoding='utf-8') as f:
        f.write(text)

if __name__ == '__main__':
    dump()
