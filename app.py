from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from chatbot.utils import generate_response

from flask_cors import CORS

app = Flask(__name__)
api = Api(app)

CORS(app)

class ChatbotAPI(Resource):
    def post(self):
        data = request.json
        query = data.get("query", "")
        if not query:
            return {"error": "Query is required"}, 400
        response = generate_response(query)
        #return jsonify(response)
        return response

api.add_resource(ChatbotAPI, "/chat")

if __name__ == "__main__":
    app.run(debug=True)
