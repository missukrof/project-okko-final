import os
import logging
from dynaconf import Dynaconf

# specifying logging level
logging.basicConfig(level=logging.INFO)

current_directory = os.path.dirname(os.path.realpath(__file__))

# define settings
settings = Dynaconf(
    settings_files=[
        f"{current_directory}/chrome_options.toml",
        f"{current_directory}/data_settings.toml",
        f"{current_directory}/features.toml",
        f"{current_directory}/models_params.toml",
    ]
)