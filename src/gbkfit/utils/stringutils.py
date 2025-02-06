
def remove_white_space(x: str) -> str:
    """
    Remove all whitespace characters from a string.
    """
    # return re.sub(r"\s+", "", x)  # Alternative solution.
    return ''.join(x.split())  # simpler alternative
