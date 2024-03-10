import sys
from PyQt5.QtWidgets import QLabel, QApplication, QTextEdit, QWidget, QPushButton, QMessageBox
import joblib


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('PyQt5 Example')
        self.setGeometry(100, 100, 700, 700)
        self.setStyleSheet(
            "background-color: white"
        )
        # Text to predict
        self.text = QTextEdit(self)
        self.text.setGeometry(50, 50, 600, 300)
        self.text.setStyleSheet(
            "border: 1px solid black;"
            "border-radius: 10px;"
            "background-color: white;"
            "font-size: 30px;"
        )

        self.reaction = QLabel(self)
        self.reaction.setGeometry(280, 400, 200, 200)
        self.reaction.setStyleSheet(
            "font-size: 100px;"
        )

        button = QPushButton('Show emoji', self)
        button.setGeometry(200, 600, 300, 70)
        button.setStyleSheet(
            "QPushButton {"
            "background-color: green;"
            "color: white;"
            "font-size: 40px;"
            "border-radius: 10px;"
            "border: none;"
            "}"
            "QPushButton:hover {"
            "background-color: rgb(0, 200, 0);"
            "}"
        )
        button.clicked.connect(self.show_message_box)

    def show_message_box(self):
        self.predict_model()

    def predict_model(self):
        text = [self.text.toPlainText()]
        # Load tokenizer
        tokenizer = joblib.load('tokenizer.pkl')
        text_vect = tokenizer.transform(text)
        # Load model
        loaded_model = joblib.load('logistic_regression_model.pkl')
        text_predicted = loaded_model.predict(text_vect)
        print(text_predicted)
        #labels = ["sadness","joy", "love", "anger", "fear", "surprise"]
        emoji = ""
        if text_predicted == [0]:
            emoji = "😫"
        elif text_predicted == [1]:
            emoji = "😁"
        elif text_predicted == [2]:
            emoji = "😍"
        elif text_predicted == [3]:
            emoji = "😡"
        elif text_predicted == [4]:
            emoji = "😨"
        elif text_predicted == [5]:
            emoji = "😃"
        print(emoji)
        self.reaction.setText(emoji)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
