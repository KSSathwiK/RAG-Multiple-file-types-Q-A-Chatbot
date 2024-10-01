# Chatbot with Hugging Face API which accepts different types of files.

This project implements a chatbot using Streamlit and the Hugging Face API. The chatbot allows users to send messages and receive responses.IT accepts PDF, DOCX, TXT, Text, Link file types

## MODEL USED

meta-llama/Meta-Llama-3-8B-Instruct

## Features

- Interactive chat interface
- Uses Hugging Face's Inference API to generate responses
- Maintains chat history using session state


## Environment Variables

You'll need to create a `.env` file in the root of the project directory with your Hugging Face API token:

```
HF_API_TOKEN=your_huggingface_api_token_here
```

## Running the App

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Activate on Windows
   venv\Scripts\activate
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```




