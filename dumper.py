import sys

def dump():
    with open('data_source.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    start = -1
    for i, line in enumerate(lines):
        if 'def generate_dataset' in line:
            start = i
            break
            
    if start != -1:
        with open('out_dump.txt', 'w', encoding='utf-8') as out:
            out.writelines(lines[start:start+150])

if __name__ == '__main__':
    dump()
