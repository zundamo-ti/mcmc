from flask import Flask, render_template, request, Response, url_for

from src.triangle_ising_model import TriangleIsingModel


app = Flask(__name__)


@app.route("/")
def index() -> str:
    return render_template(
        "index.html"
    )


@app.route("/ising_model")
def ising_model() -> str:
    return render_template(
        "ising_model.html"
    )


@app.route("/ising_model/simulate", methods=["POST"])
def simulate() -> str:
    return render_template(
        "ising_model.html",
        exchange_enegy=float(request.form.get("exchange_enegy")),
        external_field=float(request.form.get("external_field")),
        tempature=float(request.form.get("tempature"))
    )


@app.route("/ising_model_video", methods=["GET"])
def ising_model_video() -> Response:
    exchange_enegy = request.args.get("exchange_enegy")
    external_field = request.args.get("external_field")
    tempature = request.args.get("tempature")
    if exchange_enegy:
        exchange_enegy = float(exchange_enegy)
    else:
        exchange_enegy = 0
    if external_field:
        external_field = float(external_field)
    else:
        external_field = 0
    if tempature:
        tempature = float(tempature)
    else:
        tempature = 0
    print(exchange_enegy, external_field)
    model = TriangleIsingModel(100, exchange_enegy, external_field, tempature)
    return Response(
        model.simulate(
        ),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run()
