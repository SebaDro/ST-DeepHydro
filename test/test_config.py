import unittest
from stdeephydro import config


class TestDaymet(unittest.TestCase):
    def setUp(self):
        self.__config_path = "data/test-config.yml"
        self.__config = config.read_config(self.__config_path)

    def test_create_dataset_config(self):
        ds_cfg = config.create_dataset_config(self.__config["data"]["training"])

        self.assertEqual(ds_cfg.start_date, "1980-01-01")
        self.assertEqual(ds_cfg.end_date, "2003-12-31")

    def test_create_datype_config(self):
        dt_cfg = config.create_dataype_config(self.__config["data"]["forcings"][0])

        self.assertEqual(dt_cfg.data_dir, "./data/forcings")
        self.assertEqual(dt_cfg.data_type, "camels-us")
        self.assertEqual(dt_cfg.variables, ["prcp", "temp"])

    def test_create_data_config(self):
        data_cfg = config.create_data_config(self.__config["data"])

        self.assertEqual(data_cfg.basins_file, "./data/basins-all.txt")
        self.assertEqual(data_cfg.streamflow_cfg.data_type, "camels-us")
        self.assertEqual(data_cfg.streamflow_cfg.data_dir, "./data/streamflow")
        self.assertEqual(data_cfg.streamflow_cfg.variables, ["streamflow"])
        self.assertEqual(data_cfg.validation_cfg.start_date, "2004-01-01")
        self.assertEqual(data_cfg.validation_cfg.end_date, "2009-12-31")
        self.assertEqual(data_cfg.test_cfg.start_date, "2010-01-01")
        self.assertEqual(data_cfg.test_cfg.end_date, "2014-12-31")

    def test_create_model_config(self):
        model_cfg = config.create_model_config(self.__config["model"])

        self.assertEqual(model_cfg.model_type, "lstm")
        self.assertEqual(model_cfg.timesteps[0], 20)
        self.assertEqual(model_cfg.offset, 1)
        self.assertEqual(model_cfg.loss, ["mse"])
        self.assertEqual(model_cfg.metrics, ["mse", "mae"])
        self.assertEqual(model_cfg.optimizer, "Adam")
        self.assertEqual(model_cfg.epochs, 2)
        self.assertEqual(model_cfg.batch_size, 32)

        params_dict = {"hiddenLayers": 2, "units": [32, 32], "dropout": [0.1, 0]}
        self.assertEqual(model_cfg.params, params_dict)

    def test_create_general_config(self):
        general_cfg = config.create_general_config(self.__config["general"])

        self.assertEqual(general_cfg.name, "lstm")
        self.assertEqual(general_cfg.output_dir, "./output")
        self.assertTrue(general_cfg.save_model)
        self.assertTrue(general_cfg.save_checkpoints)

    def test_create_config(self):
        cfg = config.create_config(self.__config)

        self.assertIsInstance(cfg.general_config, config.GeneralConfig)
        self.assertIsInstance(cfg.data_config, config.DataConfig)
        self.assertIsInstance(cfg.model_config, config.ModelConfig)
