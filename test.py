from flask import Flask,request
from main import qa
app = Flask(__name__)

@app.route('/qa')
def chat():
    text = request.args.get('text')
    return qa(text)

if __name__ == '__main__':
    app.run()
