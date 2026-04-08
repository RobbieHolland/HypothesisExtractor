import hydra
from omegaconf import DictConfig

from compute_embeddings import EmbeddingComputer
from compute_outcomes import OutcomeComputer


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    EmbeddingComputer(cfg).run()
    OutcomeComputer(cfg).run()


if __name__ == "__main__":
    main()
