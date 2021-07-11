
import sys


def main():
    raise NotImplementedError(
        "Oh, whoops! GUI has not been developed yet. "
        "But hey, feel free to submit a PR!")

    from PySide6 import QtWidgets

    app = QtWidgets.QApplication([])
    label = QtWidgets.QLabel("gbkfit")
    label.resize(800, 600)
    label.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
