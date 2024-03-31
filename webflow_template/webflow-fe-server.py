from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/datasets.html')
def datasets():
    return render_template('datasets.html')
@app.route('/f1models.html')
def f1models():
    return render_template('f1models.html')
@app.route('/about.html')
def about():
    return render_template('about.html')
@app.route('/schema.html')
def schema():
    return render_template('schema.html')
@app.route('/roadmap.html')
def roadmap():
    return render_template('roadmap.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)

