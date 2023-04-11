from transformers import AutoTokenizer , AutoModelForTokenClassification
from pythainlp.tokenize import word_tokenize
import torch

# Load model
name = 'thainer-corpus-v2-base-model'
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForTokenClassification.from_pretrained(name)

# Tokenize
sentence="""
7                ๆ ผู้ส่ง : กฤษโอฬาน วัฒนครแสงฆราญ  ผู้รับ : กฤษโอพฬาน วัฒนครแสงฆราญ   
 """

if len(sentence) > 512:
    sentence = sentence[:512]



cut=word_tokenize(sentence.replace(" ", "<_>"))
inputs=tokenizer(cut,is_split_into_words=True,return_tensors="pt")

ids = inputs["input_ids"]
mask = inputs["attention_mask"]
# forward pass
outputs = model(ids, attention_mask=mask)
logits = outputs[0]

predictions = torch.argmax(logits, dim=2)
predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]

def fix_span_error(words,ner):
    _ner = []
    _ner=ner
    _new_tag=[]
    for i,j in zip(words,_ner):
        #print(i,j)
        i=tokenizer.decode(i)
        if i.isspace() and j.startswith("B-"):
            j="O"
        if i=='' or i=='<s>' or i=='</s>':
            continue
        if i=="<_>":
            i=" "
        _new_tag.append((i,j))
    return _new_tag

ner_tag=fix_span_error(inputs['input_ids'][0],predicted_token_class)
print(ner_tag)

merged_ner=[]
for i in ner_tag:
    if i[1].startswith("B-"):
        merged_ner.append(i)
    elif i[1].startswith("I-"):
        merged_ner[-1]=(merged_ner[-1][0]+i[0],merged_ner[-1][1])
    else:
        merged_ner.append(i)

print(merged_ner)

#display only entity of person  name
person = []
for i in merged_ner:
    if i[1].startswith("B-PERSON"):
        person.append(i[0])
    
print(person)

if len(person) == 2:
    print('ผู้ส่ง :',person[0]),print('ผู้รับ :',person[1])

elif len(person) > 2:
    # print('ผู้ส่ง :',person[0]),print('ผู้รับ :',person[1]+person[2])
    for i in range(2,len(person)):
        person[1] = person[1] + person[i]
    print('ผู้ส่ง :',person[0]),print('ผู้รับ :',person[1])


  









