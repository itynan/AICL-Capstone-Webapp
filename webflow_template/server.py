from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/datasets.html')
def datasets():
    return render_template('datasets.html')


if __name__ == '__main__':
    app.run(debug=True)
