import platform
import os

def configure_for_windows():
    if platform.system() == 'Windows':
        os.environ['TRITON_CACHE_DIR'] = os.path.expandvars(r'%LOCALAPPDATA%\triton_cache')
        os.environ['PATH'] += f';{os.environ["CUDA_PATH"]}\\bin'
