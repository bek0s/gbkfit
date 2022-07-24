
import sys


def main():

    raise NotImplementedError(
        "Whoopsie daisy! The gbkfit GUI has not been developed yet. "
        "But hey, feel free to submit a pull request on github! "
        "https://github.com/bek0s/gbkfit")

    from PySide6 import QtWidgets

    app = QtWidgets.QApplication([])
    label = QtWidgets.QLabel("gbkfit")
    label.resize(100, 100)
    label.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
