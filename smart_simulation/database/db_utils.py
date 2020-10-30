import logging
import pathlib
import sqlite3
from functools import lru_cache

import pandas as pd

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
LOCAL_PROD_DB = PACKAGE_PATH / "smart_simulation" / "database" / "production.db"


@lru_cache(maxsize=None)
def connect(database: pathlib.PurePath = LOCAL_PROD_DB) -> sqlite3.connect:
    """
    Connect to a sqlite3 database and return the connection.
    Args:
        database: Database to connect to, defaulted to local production database.

    Returns: The database connection
    """
    if not isinstance(database, pathlib.PurePath):
        logging.exception("weights_directory must be a pathlib Path.")
        raise TypeError
    if database.suffix != ".db":
        logging.exception(f"Database: {database}, must be of extension type '.db'.")
        raise TypeError
    try:
        connection = sqlite3.connect(database)
        return connection
    except Exception:
        raise


def weight_csv_to_db(
    csv_file: pathlib.Path, truncated_uuid: str, database: pathlib.Path = LOCAL_PROD_DB
):
    """
    Add weight csv file to the weights table of a the local database.
    Args:
        csv_file: Simulation weight output file path.
        truncated_uuid: 8 character long truncated uuid.
        database: Database to write to, defaulted to the local production database.
    """
    # TODO add in check for datetime format of index, and that weight column exists
    # TODO and create temp df with just date_time, scale_id, weight
    with connect(database) as connection:
        scale_df = pd.read_csv(csv_file, parse_dates=True, index_col=0)
        scale_df.loc[:, "scale_id"] = truncated_uuid
        scale_df.to_sql(
            "weights",
            con=connection,
            if_exists="append",
            index=True,
            dtype="TEXT",
            index_label="date_time",
        )


def delete_all_rows(table: str, database: pathlib.PurePath = LOCAL_PROD_DB):
    """
    Delete all rows in a database table.
    Args:
        table: Table to delete records from.
        database: Database to delete records from, defaulted to the local production database.
    """
    if not isinstance(table, str):
        logging.exception(f"table must be of type: str. Received a {type(table)}.")
        raise TypeError

    with connect(database) as connection:
        c = connection.cursor()
        sql_statement = f"DELETE FROM {table};"
        try:
            c.execute(sql_statement,)
        except Exception:
            raise
