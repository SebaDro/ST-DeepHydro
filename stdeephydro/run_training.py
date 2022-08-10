import argparse
import logging
import logging.config
import yaml

from stdeephydro import config
from stdeephydro import training

with open("./config/logging.yml", "r") as stream:
    log_config = yaml.load(stream, Loader=yaml.FullLoader)
    logging.config.dictConfig(log_config)


def read_args():
    parser = argparse.ArgumentParser(description='Download some Daymet files.')
    parser.add_argument('config', type=str, help="Path to a config file that controls the download process")
    parser.add_argument("--dryrun", action="store_true", help="If set, training will be performed in a dry run, which "
                                                              "means no outputs and results will be stored. This "
                                                              "includes training progress, model states and evaluation "
                                                              "results.")
    args = parser.parse_args()
    return args


def main():
    args = read_args()
    config_path = args.config
    dry_run = args.dryrun

    logging.info("Start run_training.py" if not dry_run else "Start run_training.py in dry run mode.")
    logging.info(f"Read config from path '{config_path}'")

    try:
        cfg_dict = config.read_config(config_path)
        if cfg_dict is None:
            logging.error("Configuration could not be loaded or is invalid. Training will be stopped.")
        else:
            logging.debug(f"Config: '{cfg_dict}'")
            cfg = config.Config.from_dict(cfg_dict)
            training.run_training_and_evaluation(cfg, dry_run)

        logging.info("Finished run_training.py")
    except config.ConfigError:
        logging.exception("Could not create configuration. Training will be stopped.")


if __name__ == "__main__":
    main()
