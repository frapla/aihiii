import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

LOG: logging.Logger = logging.getLogger(__name__)


def set_rcparams(
    mpl_path: Path = Path("src") / "visualization" / "dissertation.mplstyle",
    ttf_path: Path = Path("src") / "visualization" / "tex-gyre" / "ttf",
    ttt_path: Path = Path("src") / "visualization" / "computer-modern",
):
    """Set rcparams for matplotlib

    Args:
        mpl_path (Path, optional): path to stylesheet. Defaults to Path(r"src\visualization\dissertation.mplstyle").
        ttf_path (Path, optional): path to font installation directory. Defaults to Path(r"src\visualization\tex-gyre\ttf").

    Raises:
        FileNotFoundError: mpl_path not found
        FileNotFoundError: ttf_path directory not found
    """
    LOG.info("Setting rcparams for matplotlib")
    # check mpl file
    if not mpl_path.is_file():
        LOG.critical("Could not find %s", mpl_path)
        raise FileNotFoundError

    # install font
    if "TeX Gyre Heros" not in set(font_manager.get_font_names()):
        # check font installation directory
        if not ttf_path.is_dir():
            LOG.critical("Could not find %s", ttf_path)
            raise FileNotFoundError

        # collect font files
        font_files = font_manager.findSystemFonts(fontpaths=[ttf_path], fontext="ttf")

        # install fonts
        for font_file in font_files:
            LOG.debug("Installing font %s", font_file)
            font_manager.fontManager.addfont(font_file)
    LOG.debug("Installed font Tex %s", sorted([x for x in font_manager.get_font_names() if x.startswith("TeX")]))

    # install font
    if "CMU Typewriter Text" not in set(font_manager.get_font_names()):
        # check font installation directory
        if not ttt_path.is_dir():
            LOG.critical("Could not find %s", ttt_path)
            raise FileNotFoundError

        # collect font files
        font_files = font_manager.findSystemFonts(fontpaths=[ttt_path], fontext="ttf")

        # install fonts
        for font_file in font_files:
            LOG.debug("Installing font %s", font_file)
            font_manager.fontManager.addfont(font_file)
    LOG.debug("Installed CM font %s", sorted([x for x in font_manager.get_font_names() if x.startswith("CMU")]))

    # set rcparams
    plt.style.use(mpl_path)
    LOG.info("Using style %s", mpl_path)
