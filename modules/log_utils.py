import os
import time
from pathlib import Path


class ANSI:
    BLACK = '\033[30m'  # basic colors
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BRIGHT_BLACK = '\033[90m'  # bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'
    F = '\033[F'
    K = '\033[K'
    NEWLINE = F + K


def track_tqdm(pbar, n: int = 1):
    def wrapper(func):
        def inner(*args, **kwargs):
            res = func(*args, **kwargs)
            pbar.update(n)
            return res
        return inner
    return wrapper


def stylize(string: str, *ansi_styles, format_spec: str = "", newline: bool = False) -> str:
    """
    Stylize a string by a list of ANSI styles.
    """
    if not isinstance(string, str):
        string = format(string, format_spec)
    if len(ansi_styles) == 0:
        return string
    ansi_styles = ''.join(ansi_styles)
    if newline:
        ansi_styles = ANSI.NEWLINE + ansi_styles
    return ansi_styles + string + ANSI.RESET


def debug(msg: str, *args, **kwargs):
    print(stylize(msg, ANSI.BOLD, ANSI.BRIGHT_WHITE), *args, **kwargs)


def info(msg: str, *args, **kwargs):
    print(stylize(msg, ANSI.BRIGHT_CYAN), *args, **kwargs)


def warn(msg: str, *args, **kwargs):
    print(stylize(msg, ANSI.YELLOW), *args, **kwargs)


def error(msg: str, *args, **kwargs):
    print(stylize(msg, ANSI.BRIGHT_RED), *args, **kwargs)


def success(msg: str, *args, **kwargs):
    print(stylize(msg, ANSI.BRIGHT_GREEN), *args, **kwargs)


def red(msg: str, format_spec: str = "", newline: bool = False):
    return stylize(msg, ANSI.BRIGHT_RED, format_spec=format_spec, newline=newline)


def green(msg: str, format_spec: str = "", newline: bool = False):
    return stylize(msg, ANSI.BRIGHT_GREEN, format_spec=format_spec, newline=newline)


def yellow(msg: str, format_spec: str = "", newline: bool = False):
    return stylize(msg, ANSI.BRIGHT_YELLOW, format_spec=format_spec, newline=newline)


def blue(msg: str, format_spec: str = "", newline: bool = False):
    return stylize(msg, ANSI.BRIGHT_BLUE, format_spec=format_spec, newline=newline)


def magenta(msg: str, format_spec: str = "", newline: bool = False):
    return stylize(msg, ANSI.BRIGHT_MAGENTA, format_spec=format_spec, newline=newline)


def cyan(msg: str, format_spec: str = "", newline: bool = False):
    return stylize(msg, ANSI.BRIGHT_CYAN, format_spec=format_spec, newline=newline)


def white(msg: str, format_spec: str = "", newline: bool = False):
    return stylize(msg, ANSI.BRIGHT_WHITE, format_spec=format_spec, newline=newline)


def black(msg: str, format_spec: str = "", newline: bool = False):
    return stylize(msg, ANSI.BRIGHT_BLACK, format_spec=format_spec, newline=newline)


def bold(msg: str, format_spec: str = "", newline: bool = False):
    return stylize(msg, ANSI.BOLD, format_spec=format_spec, newline=newline)


def underline(msg: str, format_spec: str = "", newline: bool = False):
    return stylize(msg, ANSI.UNDERLINE, format_spec=format_spec, newline=newline)


def color2ansi(color: str):
    return getattr(ANSI, color.upper(), "")


class ConsoleLogger:
    _loggers = {}
    _default_color = "bright_blue"

    def __new__(cls, name, *args, **kwargs):
        if name not in cls._loggers:
            cls._loggers[name] = super().__new__(cls)
        return cls._loggers[name]

    def __init__(self, name, prefix_str=None, color=None, disable: bool = False):
        self.prefix_str = prefix_str or name
        self.color = color2ansi(color or self._default_color)
        self.disable = disable

    @property
    def prefix(self):
        return f"[{stylize(self.prefix_str, self.color)}] " if self.prefix_str else ""

    def print(self, msg: str, *args, no_prefix=False, disable=None, **kwargs):
        disable = disable if disable is not None else self.disable
        if not disable:
            print(f"{self.prefix if not no_prefix else ''}{msg}", *args, **kwargs)

    def tqdm(self, *args, no_prefix=False, **kwargs):
        from tqdm import tqdm
        kwargs["desc"] = f"{self.prefix if not no_prefix else ''}{kwargs.get('desc', '')}"
        kwargs["disable"] = kwargs['disable'] if 'disable' in kwargs else self.disable
        return tqdm(*args, **kwargs)


def get_logger(name, prefix_str=None, color=None, disable: bool = False):
    return ConsoleLogger(name, prefix_str, color, disable)


def smart_path(root, name, exts=tuple()):
    return_type = type(name)
    if isinstance(name, Path):
        name = str(name)
    name = name.replace('%datetime%', time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    name = name.replace('%date%', time.strftime("%Y-%m-%d", time.localtime()))
    name = name.replace('%time%', time.strftime("%H-%M-%S", time.localtime()))

    if '%index%' in name:
        ext_names = [Path(name).with_suffix(ext) for ext in exts]
        idx = 0
        while os.path.exists(path := os.path.join(root, name.replace('%index%', str(idx)))) or any(os.path.exists(os.path.join(root, ext_name.replace('%increment%', str(idx)))) for ext_name in ext_names):
            idx += 1
    else:
        path = os.path.join(root, name)

    return return_type(path)
