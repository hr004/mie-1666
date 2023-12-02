from typing import Any

import torch
from transformers import AutoTokenizer, GenerationConfig

from src.finetune.pipeline import construct_model, get_loaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)


def make_texts_to_tokens(text: str, tokenizer: Any) -> torch.Tensor:
    tokens = tokenizer(
        "Solve this question: " + text,
        return_tensors="pt",
    ).input_ids
    return tokens


def verify_correctness(model_name: str):
    model = construct_model(model_name=model_name).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("#" * 80)
    input_string = "What is the capital city of South Korea?"
    print(f"Input: {input_string}")
    tokens = make_texts_to_tokens(input_string, tokenizer).to(DEVICE)
    outputs = model.generate(input_ids=tokens, max_length=2048)
    outputs = tokenizer.decode(outputs[0], skip_special_tokens=True).replace("  ", "")
    print(f"Output: {outputs}")

    print("#" * 80)
    code = """def svg_to_image(string, size=None):
        if isinstance(string, unicode):
            string = string.encode('utf-8')
            renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(string))
        if not renderer.isValid():
            raise ValueError('Invalid SVG data.')
        if size is None:
            size = renderer.defaultSize()
            image = QtGui.QImage(size, QtGui.QImage.Format_ARGB32)
            painter = QtGui.QPainter(image)
            renderer.render(painter)
        return image"""
    print(f"Input: {code}")
    tokens = tokenizer(code, return_tensors="pt").input_ids.to(DEVICE)
    outputs = model.generate(input_ids=tokens, max_length=2048)
    outputs = tokenizer.decode(outputs[0], skip_special_tokens=True).replace("  ", "")
    print(f"Output: {outputs}")


def main(model_name: str):
    model = construct_model(model_name=model_name).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    valid_loader = get_loaders(batch_size=1, model_name=model_name, split="valid")

    for batch in valid_loader:
        print("#" * 80)
        print("Problem:")
        print(tokenizer.decode(batch["input_ids"][0]))
        outputs = model.generate(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE),
            max_length=2048,
        )
        print("Response:")
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

        print("Expected Answer:")
        print(tokenizer.decode(batch["raw_labels"][0], skip_special_tokens=True))


if __name__ == "__main__":
    # model_name_lst = ["t5-small", "t5-base", "Salesforce/codet5p-220m",
    #                   "Salesforce/codet5p-770m-py", "Salesforce/codet5p-16b"]
    model_name_lst = ["Salesforce/codet5p-16b"]
    for mn in model_name_lst:
        print(mn)
        verify_correctness(mn)

    for mn in model_name_lst:
        print(mn)
        main(mn)
