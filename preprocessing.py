from pathlib import Path
import numpy as np
from utils import geofiles, parsers
import matplotlib.pyplot as plt


def create_metadata_file(dataset_path: str):
    metadata_file = Path(dataset_path) / 'metadata.json'
    metadata = geofiles.load_json(metadata_file)

    pass

if __name__ == '__main__':
    args = parsers.preprocess_argument_parser().parse_known_args()[0]
