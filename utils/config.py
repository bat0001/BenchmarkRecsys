import sys, argparse
from omegaconf import OmegaConf, DictConfig


def _add_legacy_aliases(cfg: DictConfig) -> None:
    """
    Provide flat attributes (cfg.seed, cfg.dataset, cfg.num_iterations, …)
    so that the existing code continues to work while we migrate.
    """
    cfg.seed           = cfg.core.seed
    cfg.dataset        = cfg.data.dataset
    cfg.num_iterations = cfg.bandit.num_iterations


def get_config() -> DictConfig:
   
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cfg",       default="configs/default.yaml")
    parser.add_argument("--save-cfg",  help="dump merged config to disk and exit")
    args, unknown = parser.parse_known_args()

    file_cfg = OmegaConf.load(args.cfg)          # YAML
    cli_cfg  = OmegaConf.from_dotlist(unknown)   # remaining CLI flags
    cfg      = OmegaConf.merge(file_cfg, cli_cfg)

    _add_legacy_aliases(cfg)

    if args.save_cfg:
        OmegaConf.save(cfg, args.save_cfg)
        print(f"Config written to {args.save_cfg}")
        sys.exit(0)

    return cfg