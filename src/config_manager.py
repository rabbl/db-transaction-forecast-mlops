from src.config import CONFIG_FILE_PATH
from src.common.utils import read_yaml, create_directories
from src.entity import DataIngestionConfig


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH):
        self.config = read_yaml(config_filepath)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config
