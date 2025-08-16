# Question Detection API

A FastAPI-based web service that determines if a given text is a question using the DeepSeek API.

## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root and add your DeepSeek API key:
   ```
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   ```

## Running the Application

Start the FastAPI development server:
```bash
uvicorn main:app --reload
```

The API will be available at `https://kqxvdnjnyrgb.sealosgzg.site`

## API Documentation

- Swagger UI: `https://kqxvdnjnyrgb.sealosgzg.site/docs`
- ReDoc: `https://kqxvdnjnyrgb.sealosgzg.site/redoc`

## API Endpoints

### Check if Text is a Question

- **URL**: `/api/judge_question`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "inputText": "What is the capital of France?"
  }
  ```
- **Success Response**:
  ```json
  {
    "is_question": true
  }
  ```

## Environment Variables

- `DEEPSEEK_API_KEY`: Your DeepSeek API key (required)

## Error Handling

The API returns appropriate HTTP status codes and error messages for different scenarios.
