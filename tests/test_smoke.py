import marketcast
from marketcast.cli import main


def test_import_and_paths():
    assert marketcast.__version__
    assert marketcast.ROOT.name  # resolves to the project folder


def test_cli_help(capsys):
    try:
        main(["--help"])
    except SystemExit as e:
        assert e.code == 0
    assert "marketcast" in capsys.readouterr().out
