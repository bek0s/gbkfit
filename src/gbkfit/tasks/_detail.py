
from gbkfit.utils import iterutils


def require_config_sections(config, sections):
    missing = []
    for section in sections:
        if not config.get(section):
            missing.append(section)
    if len(missing) > 0:
        raise RuntimeError(
            f"The following configuration keys must be defined "
            f"and not empty/null: {missing}.")


def listify_config_sections(config, sections):
    for section in sections:
        if config.get(section):
            config[section] = iterutils.listify(config[section])


def check_config_sections_length(config, sections):
    lengths = {}
    for section in sections:
        if config.get(section):
            lengths[section] = len(config[section])
    if len(set(lengths.values())) > 1:
        raise RuntimeError(
            f"The following configuration sections must have the same length: "
            f"{list(lengths.keys())}.")
