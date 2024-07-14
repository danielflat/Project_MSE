import os
import subprocess


def get_repo_path():
    """
    Gets the local absolute path of the repository main directory.
    """
    try:
        repo_path = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode('utf-8')
        return repo_path
    except subprocess.CalledProcessError:
        return None


def get_path(directory):
    """
    Here you can construct the path for a resource with respect to the repository main directory.
    """
    return os.path.join(get_repo_path(), directory)
