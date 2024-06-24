import json
import openai
import csv
from openai import OpenAI
from tqdm import tqdm

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def translate_text_with_gpt(text, client):
    completion = client.chat.completions.create(
        model='gpt-4',
        messages=[
            {"role": "system", "content": "You are a Spanish translator model that can understand semantics and inherent structure."},
            {"role": "user", "content": f"Translate this to Spanish accurately: {text}"}
        ]
    )
    return completion.choices[0].message.content

def paraphrase_text_with_gpt(text, client):
    completion = client.chat.completions.create(
        model='gpt-4',
        messages=[
            {"role": "system", "content": "You are a English language model that can understand semantics and inherent structure."},
            {"role": "user", "content": f"Rewrite the following sentence in two different ways: {text}"}
        ],
        max_tokens = 512,
        n=1  # We ask for one completion containing two paraphrases
    )
    paraphrases = completion.choices[0].message.content

    # Filter out empty lines and irrelevant text
    paraphrases = [line for line in paraphrases.split('\n') if line and not line.startswith('Rewrite')]

    return paraphrases[:2]

def main():
    client = OpenAI(
        api_key='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',
    )

    input_file = './reduced_dataset.json'
    output_file = './gpt_dataset.csv'

    data = load_json(input_file)

    batch_to_skip = 1165
    count = 0
    
    with open(output_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # writer.writerow(['Original Text', 'Translation', 'Paraphrase 1', 'Paraphrase 2'])

        for _, text in tqdm(data["text"].items(), desc='Gathering results'):
            count = count + 1
            if count <= batch_to_skip:
                continue
            paraphrases = paraphrase_text_with_gpt(text, client)
            translation = translate_text_with_gpt(text, client)
            writer.writerow([text, translation, *paraphrases])

if __name__ == "__main__":
    main()
