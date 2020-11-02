import logging
import pathlib

import daiquiri
from smart_simulation.cfg_templates.config import package_dir

daiquiri.setup(
    level=logging.INFO,
    outputs=(
        daiquiri.output.Stream(
            formatter=daiquiri.formatter.ColorFormatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s.%(" "funcName)s: %(message)s"
            )
        ),
    ),
)

PACKAGE_PATH = pathlib.Path(package_dir)
SIMULATION_OUTPUTS_PATH = PACKAGE_PATH / "smart_simulation" / "outputs" / "simulations"
SIMULATION_WEIGHTS_PATH = SIMULATION_OUTPUTS_PATH / "weights"
SIMULATION_SERVINGS_PATH = SIMULATION_OUTPUTS_PATH / "servings"


def weight_files(weights_directory: pathlib.PurePath = SIMULATION_WEIGHTS_PATH) -> list:
    """
    Return a list of weight files paths from a directory. The local simulation output directory is the default.
    Args:
        weights_directory: Directory of the weight files.

    Returns: A list of paths of all files in the directory.

    """
    if not isinstance(weights_directory, pathlib.PurePath):
        logging.exception("weights_directory must be a pathlib Path.")
        raise TypeError

    weights_files_list = [p for p in weights_directory.iterdir() if p.is_file()]
    return weights_files_list


def servings_files(
    servings_directory: pathlib.PurePath = SIMULATION_SERVINGS_PATH,
) -> list:
    """
     Return a list of servings files paths from a directory. The local simulation output directory is the default.
    Args:
        servings_directory: Directory of the servings files.

    Returns: A list of paths of all files in the directory.

    """
    if not isinstance(servings_directory, pathlib.PurePath):
        logging.exception("servings_directory must be a pathlib Path.")
        raise TypeError

    servings_files_list = [p for p in servings_directory.iterdir() if p.is_file()]
    return servings_files_list


def file_uuid(file_path: pathlib.PurePath) -> str:
    """
    Return the uuid from the simulation file as a string.
    Args:
        file_path: Path of the simulation output file.

    Returns: The uuid from the file as a string.

    """
    if not isinstance(file_path, pathlib.PurePath):
        logging.exception("file_path must be a pathlib Path.")
        raise TypeError
    expected_uuid_len = 36
    file_name = file_path.stem
    file_name_split = file_name.split("_")
    uuid = file_name_split[0]
    if len(uuid) != expected_uuid_len:
        logging.exception(
            f"Insufficient number of characters in the uuid: {uuid}. Expected length: "
            f"{expected_uuid_len}. Given length: {len(uuid)}"
        )
        raise ValueError
    return uuid


def truncate_uuid(uuid: str) -> str:
    """
    Truncate the uuid to the first 8 characters.
    Args:
        uuid: Full uuid as a string.

    Returns: The first 8 characters of the uuid.

    """
    expected_uuid_len = 36
    if len(uuid) != expected_uuid_len:
        logging.exception(
            f"Insufficient number of characters in the uuid: {uuid}. Expected length: "
            f"{expected_uuid_len}. Given length: {len(uuid)}"
        )
        raise ValueError
    truncated_uuid = uuid[0:8]
    return truncated_uuid
