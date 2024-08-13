from flask import Blueprint, render_template, request, redirect, url_for

main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        number = request.form.get('number')
        text = request.form.get('text')
        return redirect(url_for('main.result', number=number, text=text))
    return render_template('index.html')

@main.route('/result')
def result():
    number = request.args.get('number')
    text = request.args.get('text')
    # prediction pipline

    prediction = "positive"



    return render_template('result.html', number=number, text=text, prediction=prediction )