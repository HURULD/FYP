from dataclasses import dataclass
import typing
import yaml



@dataclass
class AudioConfig:
    sample_rate: int
    
    def __init__(self, sample_rate:int):
        self.sample_rate = sample_rate

@dataclass
class Config:
    """Configuration class to hold the configuration values."""
    # Raw configuration values.
    _config_values: typing.Dict[str, str]

    config_file: str
    audio: AudioConfig
    
    def __init__(self, config_file:str):
        self.config_file = config_file
        self._config_values = yaml.safe_load(open(config_file))
        self.audio = AudioConfig(**self._config_values['audio'])

_config: Config = None
def get_config() -> Config:
    if _config is not None:
        return _config
    else:
        raise ValueError("Config not loaded. Please run load_config() first.")

def load_config(config_file:str = 'config.yaml'):
    global _config
    _config = Config(config_file)


