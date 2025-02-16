from flask import Blueprint, jsonify
from tensorboard import program

server_route = Blueprint("server_route", __name__)

@server_route.route("/api", methods=["GET"])
def detect_plaque():
    return jsonify({
        "message": "Your are now connected into Heart Girl Scout Uniform Detection API",
    })

@server_route.route("/api/tensorboard", methods=["GET"])
def start_tensorboard():
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', 'logs'])
    url = tb.launch()
    return jsonify({
        "message": f"TensorBoard started at {url}",
        "url": url
    })