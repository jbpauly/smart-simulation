"""Console script for smart_simulation."""
import sys
import click


@click.command()
def main(args=None):
    """Console script for smart_simulation."""
    click.echo("Replace this message by putting your code into "
               "smart_simulation.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
