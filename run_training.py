import sys
import logging
import logging.config
from libs import config
from libs import ioutils
from libs import training
import yaml

with open("./config/logging.yml", "r") as stream:
    log_config = yaml.load(stream, Loader=yaml.FullLoader)
    logging.config.dictConfig(log_config)


def main():
    logging.info("Start run_training.py")
    if len(sys.argv) != 2:
        raise ValueError("Missing argument for config file path!")
    config_path = sys.argv[1]

    logging.info(f"Read config from path '{config_path}'")

    cfg_dict = config.read_config(config_path)
    if cfg_dict is None:
        logging.error("Configuration could not be loaded or is invalid. Training will be stopped.")
    else:
        try:
            logging.debug(f"Config: '{cfg_dict}'")
            cfg = config.Config.from_dict(cfg_dict)
            training.run_training(cfg)

            logging.info("Successfully finished run_training.py")
        except config.ConfigError:
            logging.exception("Could not create configuration. Training will be stopped.")


if __name__ == "__main__":
    main()
