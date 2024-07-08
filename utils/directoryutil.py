import subprocess
import os


def get_repo_path():
    try:
        repo_path = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode('utf-8')
        return repo_path
    except subprocess.CalledProcessError:
        return None


def get_path(directory):
    return os.path.join(get_repo_path(), directory)
