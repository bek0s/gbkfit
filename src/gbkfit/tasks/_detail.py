
from gbkfit.utils import iterutils


def require_config_sections(config, sections):
    missing = []
    for section in sections:
        if section not in config or len(config[section]) == 0:
            missing.append(section)
    if len(missing) > 0:
        raise RuntimeError(
            f"The following required sections are missing "
            f"from the configuration file: {missing}.")


def listify_config_sections(config, sections):
    for section in sections:
        if section in config:
            config[section] = iterutils.listify(config[section])


def check_config_sections_length(config, sections):
    lengths = {}
    for section in sections:
        if section in config:
            lengths[section] = len(config[section])
    if len(set(lengths.values())) > 1:
        raise RuntimeError(
            f"The following configuration sections must have the same length: "
            f"{lengths}.")
