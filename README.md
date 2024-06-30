# MSE Project - Tübingo
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

### Dependencies
#### Poetry
To install the project dependencies, run the following command:
```bash
poetry install
```
This command reads the `pyproject.toml` file and installs all specified dependencies into a virtual environment managed by Poetry.

**Note**: When adding a package to a specific group, ensure that this package is NOT specified anywhere else in the file. Duplicates can cause issues that are hard to resolve.

##### Data index
For the collaborators and the reviewers make sure that git lfs (git Large File Storage) is installed on your machine.
Look [here](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) for more details.

For **MacOS**:
```bash
brew install git-lfs
```
With that we are tracking the [dump.sql](/db/dump.sql) (the index for our search engine) because it might be too large for a repo. 
We used
```bash
git lfs track ./data/dump.sql
```
to still be able to push this file.

## Running the Project
#### Step 1: Activate the Poetry Environment
To activate the Poetry environment, run the following command:
```bash
poetry shell
```

#### Step 2: Set up the database
For our search engine we use a shared database with our index. This is required
to run the search engine and test it. 
For that make sure to run the following:
```bash
docker compose down;
docker compose up --build db
```
After that you should wait some seconds until you get the message:

```vbnet
LOG:  database system is ready to accept connections
```

When you want to try out how the database works, you can experiment it with [001_Flat_db_example_connection.ipynb](exp/001_Flat_db_example_connection.ipynb)

#### Step 3: Run the project
To run the project, execute the following command:
```bash
python Main.py
```
This command runs the `Main.py` file, which is the entry point of the project.


## Dealing with the database 

For the project we all want to be on the same page. For that we have one common database, a PostgreSQL.
Furthermore, we all try to be in sync with our data, so for that there always exists a [dump.sql](./db/dump.sql).
This file makes it possible to always have the latest data when calling `docker compose up --build db`.

But when you are working with it, you should **update** the `dump.sql` when you make progress for the team. To do that you have to do the following steps.

1. Make sure the docker container is running. For that look at the chapter "Running the Project".
2. Look up the container ID. For that just exec:
```bash
docker ps
```
It should list all your running containers. Look for the right container. The right name of the image is `project_mse-db`.
An ID might look like this: `c946285e9b4f`
3. Go and overwrite the `dump.sql` by exec the following script:
```bash
docker exec -t your_container_name_or_id pg_dump -U user search_engine_db > ./db/dump.sql
```
For example with the container ID `c946285e9b4f` the command should look like this:
```bash
docker exec -t c946285e9b4f pg_dump -U user search_engine_db > ./db/dump.sql
```
4. Push the updated dump.sql using git.
```
git add db/dump.sql
git commit -m "Update dump.sql"
```

# TODO: Add more documentation

## Create Index






