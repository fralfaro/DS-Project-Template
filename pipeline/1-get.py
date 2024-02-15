# %%
"""
Get data
"""
from pathlib import Path
from loguru import logger
from ds_project.data.data import TitanicTrainingData

# %% tags=["parameters"]
upstream = None
product = None

# %%
logger.info("Read Data")
path = '../data/input/'
titanic_train =TitanicTrainingData.from_file(path + 'train.csv').df

# %%
logger.info("Output Folder")
Path('output').mkdir(exist_ok=True)

# %%
logger.info("Save Titanic Training Data")
titanic_train.to_csv(str(product['data']), index=False)


