import hydra


class TokenizerModule:
    def __init__(self):
        self.config = None
        self.cfg_path = None
        self.cfg_name = None

    def compose_configure(self):
        hydra.initialize(self.cfg_path, self.cfg_name)
        self.config = hydra.compose(self.cfg_name)


class ConfigModule:
    @staticmethod
    def return_type_of_vocab(cfg_path: str, cfg_name: str) -> str:
        hydra.initialize(cfg_path, cfg_name)
        config = hydra.compose(cfg_name)
        hydra.core.global_hydra.GlobalHydra.instance().clear()

        return config.tokenizer.vocab_type
