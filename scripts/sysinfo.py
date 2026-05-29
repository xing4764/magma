#!/usr/bin/env python3
import platform, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print(f'OS: {platform.system()} {platform.release()} ({platform.machine()})')
print(f'Python: {sys.version}')
print(f'CPU cores: {os.cpu_count()}')

import psutil
mem = psutil.virtual_memory()
print(f'RAM: {mem.total / 1024**3:.1f} GB (available: {mem.available / 1024**3:.1f} GB)')
disk = psutil.disk_usage('C:\\')
print(f'Disk free: {disk.free / 1024**3:.1f} GB')

# Check GPU
try:
    import subprocess
    r = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'], capture_output=True, text=True)
    if r.returncode == 0:
        print(f'GPU: {r.stdout.strip()}')
    else:
        print('GPU: no NVIDIA GPU')
except Exception as ex:
    print(f'GPU: {ex}')
