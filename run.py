import hydra
from omegaconf import DictConfig, OmegaConf

from compute_embeddings import EmbeddingComputer
from compute_outcomes import OutcomeComputer


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    steps = list(OmegaConf.to_container(cfg.steps))

    if "embeddings" in steps:
        EmbeddingComputer(cfg).run()

    if "outcomes" in steps:
        OutcomeComputer(cfg).run()

    if "sae" in steps:
        from infer_sae import run_infer_sae
        run_infer_sae(cfg)

    if "autorate" in steps:
        from autorate import run_autorate
        run_autorate(cfg)


if __name__ == "__main__":
    main()
