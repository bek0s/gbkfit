"""
Helpers for strings.
"""

__all__ = [
    'remove_whitespace'
]


def remove_whitespace(x: str) -> str:
    """
    Return a string without any of its whitespace characters.

    Parameters
    ----------
    x : str
        A string.

    Returns
    -------
    str
        The string without spaces, tabs, newlines or other whitespace.
    """
    return ''.join(x.split())
