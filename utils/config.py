import json
from munch import munchify

config = {}


def load_config(path:str):
  global config
  with open(path) as file:
    loader = json.load
    config = loader(file)
    config = munchify(config)
    return config

def set_param(params:dict):
  global config
  config = munchify(params)

def get_hyper():
  global config
  return config