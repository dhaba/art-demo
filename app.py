from flask import Flask
import os

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/generate')
def generate():
    script = 'sleep 5'
    print('running script:\n{}'.format(script))
    print('done running scripts')
    return "done!"


if __name__ == '__main__':
    app.run(host='0.0.0.0')
