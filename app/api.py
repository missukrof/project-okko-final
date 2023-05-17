import plotly.graph_objects as go

from flask import Flask, request

from inference import get_recommendations


app = Flask(__name__)


@app.route('/index')
def access_param():

    random_user_id = request.args.get("id")

    responce = get_recommendations(int(random_user_id))

    fig = go.Figure(data=[go.Table(
        header=dict(values=[f"<b>{col}</b>" for col in responce.columns],
                    line_color='darkslategray',
                    fill_color='#E6E6FA',
                    font=dict(size=14),
                    align='center'),
        cells=dict(values=responce.transpose().values.tolist(),
                fill_color='#FFFFFF',
                font=dict(size=12),
                align='left'),
        columnwidth=[22, 22, 13, 8]),
        ])

    return fig.to_html()


app.run(debug=True, host="0.0.0.0", port=5000)

# To try: http://127.0.0.1:5000/index?id=705073