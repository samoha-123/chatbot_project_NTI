from transformers import MarianMTModel, MarianTokenizer


# translates the text to arabic with a pre-trained model
def translate_to_arabic(arabic_text):
    model_name = "./opus-mt-tc-big-en-ar"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    translated = model.generate(  # type: ignore
        **tokenizer(arabic_text, return_tensors="pt", padding=True)
    )
    for t in translated:
        return tokenizer.decode(t, skip_special_tokens=True)


# translates the text to english with a pre-trained model
def translate_to_english(arabic_text):
    model_name = "./model-translate-ar-to-en-from-320k-dataset-ar-en-th2301191019"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    translated = model.generate(  # type: ignore
        **tokenizer(arabic_text, return_tensors="pt", padding=True)
    )
    for t in translated:
        return tokenizer.decode(t, skip_special_tokens=True)
