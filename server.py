from flask import Flask
from flask_restful import Resource, Api, reqparse
from transformers import AutoTokenizer
from onnxruntime import InferenceSession
import numpy as np

app = Flask("MicroAsking")
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('question', required=True)
parser.add_argument('contexts', required=True,
                    location='json', type=list)

model_checkpoint = "vi-mrc-base"
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
print("Loading model...")
session = InferenceSession("onnx/vi-mrc-base.onnx")
print("Done!")


class MicroAsking(Resource):
    def post(self):
        args = parser.parse_args()
        question = args['question']
        print(question)
        contexts = args['contexts']

        answers = []
        count = 0
        for context in contexts:
            count += 1
            print(f"processing...{count}")
            inputs = tokenizer(question, context['content'],
                               add_special_tokens=True, return_tensors="np")
            inputs = {k: v.astype(np.int64) for k, v in inputs.items()}

            outputs = session.run(
                output_names=["start_logits", "end_logits"], input_feed=dict(inputs))

            answer_start_scores = outputs[0]
            answer_end_scores = outputs[1]
            answer_start = np.argmax(answer_start_scores)
            answer_end = np.argmax(answer_end_scores) + 1

            input_ids = inputs["input_ids"][0]
            tokens = tokenizer.convert_ids_to_tokens(
                input_ids[answer_start:answer_end])
            answer = tokenizer.convert_tokens_to_string(tokens)

            if answer['answer']:
                answer['domain'] = context['domain']
                answers.append(answer)

        return answers, 200


api.add_resource(MicroAsking, '/api.micro-asking.inference')

if __name__ == '__main__':
    app.run(debug=False)
