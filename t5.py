pip install datasets

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
import re

def process_input(input_text):
    pattern = re.compile(r'(\b\w+)(\s*)\[')


    def add_quotes(match):
        return f'"{match.group(1)}"{match.group(2)}['

    processed_text = pattern.sub(add_quotes, input_text)
    processed_text = re.sub(r'\[( number| NUMBER| Number|number|Number|NUMBER)\]', '[INTEGER]', processed_text)
    processed_text = re.sub(r'\[( text| Text| TEXT|text|TEXT|Text)\]', '[VARCHAR]', processed_text)
    processed_text = re.sub(r'\[(\w+)\(\d*\)\]', r'[\1]', processed_text)
    processed_text = re.sub(r'\[\d+\]', '[]', processed_text)
    processed_text = processed_text.replace(' [', '[').replace('] ', ']')

    return f"[{processed_text}"

dataset = load_dataset("shiroyasha13/llama_text_to_sql_dataset")

num_samples = 80000
small_dataset = dataset['train'].select(range(num_samples))

from sklearn.model_selection import train_test_split

df = small_dataset.to_pandas()
df['input'] = [process_input(inp) for inp in df['input']]
df_processed = df.copy()

print(df_processed)

train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)


from datasets import load_dataset, Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

def preprocess_function(examples):
    # Apply the process_input function to modify the input
    processed_inputs = [process_input(inp) for inp in examples['input']]
    outputs = examples['output']
    model_inputs = t5_tokenizer(processed_inputs, max_length=512, truncation=True, padding='max_length')
    labels = t5_tokenizer(outputs, max_length=512, truncation=True, padding='max_length')

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

print(tokenized_train_dataset[:2])

model = T5ForConditionalGeneration.from_pretrained('t5-small')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
)

trainer.train()

model.save_pretrained('./model123')  # Save model
t5_tokenizer.save_pretrained('./model123')

questions = [
    "What is the Maths score of Chetan?",
    "Who has the highest English score?",
    "What is Ganesh's Science score?",
    "Which student has the lowest Maths score?",
    "What is the average Science score?",
    "Who scored more than 45 in Maths?",
    "What is the total English score for all students?",
    "Which student scored the highest in Science?",
    "What is the name of the student who scored 50 in English?",
    "What is the median of maths?",
    "What is Vijay's total score across all subjects?",
    "Who scored less than 40 in Science?",
    "What is the highest score in Maths?",
    "Which students scored 50 in all subjects?",
    "What is the average score in English?",
    "Who scored exactly 45 in Science?",
    "What is the lowest score in English?",
    "Which students scored more than 40 in all subjects?",
    "What is the total Maths score for students named Chetan and Ganesh?",
    "How many students have a Science score of 45?"
]

que = []
answers = []


for question in questions:
    text_input = f'[INST] Here is a database schema: table schema: education: "Name" [VARCHAR] "Maths" [INTEGER] "Science" [INTEGER] "English" [INTEGER] question: {question} [/INST]'
    text_input = process_input(text_input)
    inputs = t5_tokenizer(text_input, return_tensors='pt', max_length=512, truncation=True).to(device)
    outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
    generated_sql = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

    que.append(question)
    answers.append(generated_sql)


for q, a in zip(questions, answers):
    print(f"Question: {q}")
    print(f"Generated SQL Query: {a}\n")

    