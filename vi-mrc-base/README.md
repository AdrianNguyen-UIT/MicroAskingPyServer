---
language: 
- vi
- vn
- en
tags:
- question-answering
- pytorch
datasets:
- squad
license: cc-by-nc-4.0
pipeline_tag: question-answering 
metrics:
- squad
widget:
- text: "Bình là chuyên gia về gì ?"
  context: "Bình Nguyễn là một người đam mê với lĩnh vực xử lý ngôn ngữ tự nhiên . Anh nhận chứng chỉ Google Developer Expert năm 2020"
- text: "Bình được công nhận với danh hiệu gì ?"
  context: "Bình Nguyễn là một người đam mê với lĩnh vực xử lý ngôn ngữ tự nhiên . Anh nhận chứng chỉ Google Developer Expert năm 2020"
---
## Model Description

- Language model: [XLM-RoBERTa](https://huggingface.co/transformers/model_doc/xlmroberta.html)
- Fine-tune: [MRCQuestionAnswering](https://github.com/nguyenvulebinh/extractive-qa-mrc)
- Language: Vietnamese, Englsih
- Downstream-task: Extractive QA
- Dataset (combine English and Vietnamese):
  - [Squad 2.0](https://rajpurkar.github.io/SQuAD-explorer/) 
  - [mailong25](https://github.com/mailong25/bert-vietnamese-question-answering/tree/master/dataset)
  - [UIT-ViQuAD](https://www.aclweb.org/anthology/2020.coling-main.233/)
  - [MultiLingual Question Answering](https://github.com/facebookresearch/MLQA)
  
This model is intended to be used for QA in the Vietnamese language so the valid set is Vietnamese only (but English works fine). The evaluation result below using 10% of the Vietnamese dataset.


| Model  | EM | F1 |
| ------------- | ------------- | ------------- |
| [base](https://huggingface.co/nguyenvulebinh/vi-mrc-base)  | 76.43  | 84.16  |
| [large](https://huggingface.co/nguyenvulebinh/vi-mrc-large)  | 77.32  | 85.46  |


[MRCQuestionAnswering](https://github.com/nguyenvulebinh/extractive-qa-mrc) using [XLM-RoBERTa](https://huggingface.co/transformers/model_doc/xlmroberta.html) as a pre-trained language model. By default, XLM-RoBERTa will split word in to sub-words. But in my implementation, I re-combine sub-words representation (after encoded by BERT layer) into word representation using sum strategy.

## Using pre-trained model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Yqgdfaca7L94OyQVnq5iQq8wRTFvVZjv?usp=sharing)

- Hugging Face pipeline style (**NOT using sum features strategy**).

```python
from transformers import pipeline
# model_checkpoint = "nguyenvulebinh/vi-mrc-large"
model_checkpoint = "nguyenvulebinh/vi-mrc-base"
nlp = pipeline('question-answering', model=model_checkpoint,
                   tokenizer=model_checkpoint)
QA_input = {
  'question': "Bình là chuyên gia về gì ?",
  'context': "Bình Nguyễn là một người đam mê với lĩnh vực xử lý ngôn ngữ tự nhiên . Anh nhận chứng chỉ Google Developer Expert năm 2020"
}
res = nlp(QA_input)
print('pipeline: {}'.format(res))
#{'score': 0.5782045125961304, 'start': 45, 'end': 68, 'answer': 'xử lý ngôn ngữ tự nhiên'}
```

- More accurate infer process ([**Using sum features strategy**](https://github.com/nguyenvulebinh/extractive-qa-mrc))

```python
from infer import tokenize_function, data_collator, extract_answer
from model.mrc_model import MRCQuestionAnswering
from transformers import AutoTokenizer

# model_checkpoint = "nguyenvulebinh/vi-mrc-large"
model_checkpoint = "nguyenvulebinh/vi-mrc-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = MRCQuestionAnswering.from_pretrained(model_checkpoint)

QA_input = {
  'question': "Bình được công nhận với danh hiệu gì ?",
  'context': "Bình Nguyễn là một người đam mê với lĩnh vực xử lý ngôn ngữ tự nhiên . Anh nhận chứng chỉ Google Developer Expert năm 2020"
}

inputs = [tokenize_function(*QA_input)]
inputs_ids = data_collator(inputs)
outputs = model(**inputs_ids)
answer = extract_answer(inputs, outputs, tokenizer)

print(answer)
# answer: Google Developer Expert. Score start: 0.9926977753639221, Score end: 0.9909810423851013
```

## About

*Built by Binh Nguyen*
[![Follow](https://img.shields.io/twitter/follow/nguyenvulebinh?style=social)](https://twitter.com/intent/follow?screen_name=nguyenvulebinh)
For more details, visit the project repository.
[![GitHub stars](https://img.shields.io/github/stars/nguyenvulebinh/extractive-qa-mrc?style=social)](https://github.com/nguyenvulebinh/extractive-qa-mrc)