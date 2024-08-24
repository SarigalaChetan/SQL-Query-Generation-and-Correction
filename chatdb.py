from google.colab import files

import pandas as pd

def generate_sql_schema(df: pd.DataFrame, table_name: str) -> str:

    sql_schema = f"CREATE TABLE {table_name} (\n"

    for column_name, dtype in df.dtypes.items():
        if dtype == 'int64':
            sql_type = 'INT'
        elif dtype == 'float64':
            sql_type = 'FLOAT'
        elif dtype == 'object':
            sql_type = 'VARCHAR(255)'
        elif dtype == 'bool':
            sql_type = 'BOOLEAN'
        else:
            sql_type = 'TEXT'


        sql_schema += f"    {column_name} {sql_type},\n"


    sql_schema = sql_schema.rstrip(',\n')
    sql_schema += "\n);"

    return sql_schema

uploaded = files.upload()
csv_file = next(iter(uploaded))
print(f"Uploaded file: {csv_file}")
df = pd.read_csv(csv_file)

print(df)

print(df.dtypes)

table_name = input("Enter the name of the table: ").strip()

schema = generate_sql_schema(df, table_name)
print(table_name)

print(schema)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("chatdb/natural-sql-7b")
model = AutoModelForCausalLM.from_pretrained(
    "chatdb/natural-sql-7b",
    device_map="auto",
    torch_dtype=torch.float16,
)

def generate_sql(question: str, schema: str) -> str:
    prompt = f"# Task\nGenerate a SQL query to answer the following question: {question}\n\n### PostgreSQL Database Schema\nThe query will run on a database with the following schema:\n\n{schema}\n\n# SQL\nHere is the SQL query that answers the question: {question}\n'''sql\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    generated_ids = model.generate(**inputs, num_return_sequences=1, eos_token_id=100001, pad_token_id=100001, max_new_tokens=400, do_sample=False, num_beams=1)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    sql_query = output[0].split("'''sql")[-1].strip().replace("ILIKE", "=").replace("%", "")
    return sql_query.strip()

def format_sql_case(sql_query: str) -> str:
    sql_keywords = ["select", "from", "where", "and", "or", "insert", "update", "delete", "create", "table", "drop", "alter", "join", "inner", "left", "right", "outer", "on", "group", "by", "having", "order", "asc", "desc", "limit", "offset", "values", "set", "as", "distinct", "count", "sum", "avg", "min", "max", "case", "when", "then", "else", "end", "not", "null", "is", "between", "like", "in", "exists"]
    pattern = re.compile(r'\b(' + '|'.join(sql_keywords) + r')\b', re.IGNORECASE)
    return pattern.sub(lambda match: match.group(0).upper(), sql_query)

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

import re
for question in questions:
    sql_query = generate_sql(question, schema)
    formatted_sql_query = format_sql_case(sql_query)
    print(formatted_sql_query)

    


