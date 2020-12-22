import torch
from transformers import BertTokenizer
from .utils import clean_text
from .model import PersonalityClassifier


LABEL_ID = {"ISTJ": 0, "ISTP": 1, "ISFJ": 2, "ISFP": 3,
            "INFJ": 4, "INFP": 5, "INTJ": 6, "INTP": 7,
            "ESTP": 8, "ESTJ": 9, "ESFP": 10, "ESFJ": 11,
            "ENFP": 12, "ENFJ": 13, "ENTP": 14, "ENTJ": 15}

LABELS = list(LABEL_ID.keys())

model = PersonalityClassifier(16, 'bert-base-uncased')
model.load_state_dict(torch.load('best_model_state.pt'))

print("\nInsert text to be evaluated to the console:\n")
user_text = clean_text(str(input()))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

MAX_LEN = 160
def encode(user_text):
    encoding = tokenizer.encode_plus(
                user_text,
                truncation=True,
                add_special_tokens=True,
                max_length=MAX_LEN,
                return_token_type_ids=False,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

    return {'user_text': user_text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()}

def unsqueeze_tensor(tensor):
    tensor = tensor.unsqueeze(0)
    tensor = tensor.repeat(32, 1)
    return tensor

encoded_input = encode(user_text=user_text)

output = model(input_ids=unsqueeze_tensor(encoded_input["input_ids"]),
               attention_mask=unsqueeze_tensor(encoded_input["attention_mask"]))

type_id = int(output.sum(dim=0).argmax())
type_str = LABELS[type_id]
print(f"Predicted MBTI type is: {type_str}")

