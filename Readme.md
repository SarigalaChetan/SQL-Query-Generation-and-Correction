# Integration of T5 and Chatdb Models for SQL Query Generation and Correction.

## Description

This project evaluates and compares three models for SQL query generation: T5, chatdb, and an integrated model that combines both T5 and chatdb. The goal was to assess their effectiveness in converting natural language questions into accurate SQL queries. The T5 model was used to generate the initial SQL queries, while the chatdb model, in its standalone form, was employed for query generation but not for correction. The integrated model, however, combines T5's query generation capabilities with chatdb's query correction functionality, providing a more comprehensive solution. This integrated approach demonstrated superior performance, achieving the best results in terms of query accuracy and schema compliance. The dataset used for this evaluation is the llama_text_to_sql_dataset, which includes a variety of natural language questions and corresponding SQL queries. This report outlines the methodology, comparative analysis, and findings, highlighting the effectiveness of the integrated model in automating and refining SQL query generation and correction.

## Usage 

You don't need to manually import the dataset; the code is written to automatically import it from Hugging Face. I've also included all the necessary packages and libraries within the code, so you just need to run it.

If you're a student working on a system with limited RAM, you can reduce the size of the dataset samples. However, this may affect the model's accuracy. During preprocessing, consider using random selection to choose the data samples for training incase you are reducing the data samples. If you're using a Colab runtime with a T4 GPU, the code might execute faster. However, you may need to split the code into two parts because running the entire code on a system with only 8GB of RAM could exhaust the memory.

