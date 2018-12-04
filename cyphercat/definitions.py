import os

# Local directory of CypherCat API
CYCAT_DIR    = os.path.dirname(os.path.abspath(__file__))

# Local directory containing entire repo
REPO_DIR     = os.path.split(CYCAT_DIR)[0]

# Local directory for datasets
DATASETS_DIR = os.path.join(REPO_DIR, 'Datasets')
