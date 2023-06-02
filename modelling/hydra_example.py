import hydra


@hydra.main(config_path="../conf", config_name='config', version_base=None)
def main(cfg) -> None:
    print(cfg.get("ergerger", []))
    print(cfg)


if __name__ == "__main__":
    main()