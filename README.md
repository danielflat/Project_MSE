# Project for Modern Search Engines
This is the search engine *Tübingo* for the course **Modern Search Engines** in the summer term 2024.

## By Group 09:
 - Daniel Flat
 - Lenard Rommel
 - Veronika Smilga
 - Lilli Diederichs

## Installation guide
This project runs with Python 3.11.0. To install the required packages, follow the instructions below:

### Python Installation
Ensure Python 3.11.0 is installed on your system.
For **Ubuntu**:
```bash 
sudo apt install python3.11
```
For **MacOS**:
```bash
brew install python@3.11
```
### Poetry Installation
Make sure you have Poetry installed to manage project dependencies. You can install Poetry via pip:
```bash
pip install poetry
poetry install
```

### Project Dependencies
To install the project dependencies, run the following command:
```bash
poetry install
```
This command reads the `pyproject.toml` file and installs all specified dependencies into a virtual environment managed by Poetry.

## Running the project
#### Step 1: Activate the Poetry Environment
To activate the Poetry environment, run the following command:
```bash
poetry shell
```
#### Step 2: Run the project
To run the project, execute the following command:
```bash
python Main.py
```
This command runs the `Main.py` file, which is the entry point of the project.
# TODO Add more documentation
