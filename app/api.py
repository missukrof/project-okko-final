import json
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from flask import Flask, request

from inference import get_recommendations


app = Flask(__name__)


@app.route('/index')
def access_param():

    random_user_id = request.args.get("id")
    try:
        responce, metrics = get_recommendations(int(random_user_id))

        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "table"}, {"type": "table"}]],
                            column_widths=[0.7, 0.3], subplot_titles=(f"Recommended movies for User ID {random_user_id}", 
                                                                      "Recommendations evaluation"))

        fig.add_trace(
            go.Table(
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
            row=1, col=1
            )
        

        fig.add_trace(
            go.Table(
            header=dict(values=[f"<b>{col}</b>" for col in metrics.columns],
                        line_color='darkslategray',
                        fill_color='#E6E6FA',
                        font=dict(size=14),
                        align='center'),
            cells=dict(values=metrics.transpose().values.tolist(),
                    fill_color='#FFFFFF',
                    font=dict(size=12),
                    align='left'),
            columnwidth=[15, 15]),
            row=1, col=2
            )

        return fig.to_html()
    
    except ValueError:

        responce = get_recommendations(int(random_user_id))
        return json.dumps(responce)
    


app.run(debug=True, host="0.0.0.0", port=5000)

# To try: http://127.0.0.1:5000/index?id=705073