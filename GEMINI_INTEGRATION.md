# Gemini 2.5 Pro Integration Guide

This guide explains how to use Google's Gemini 2.5 Pro model for chain-of-thought generation in the Thought Anchors project.

## Setup

### 1. Get a Google API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key

### 2. Configure Environment Variables

Add your Google API key to your `.env` file:

```bash
GOOGLE_API_KEY=your_google_api_key_here
```

You can use the provided `.env.example` file as a template.

### 3. Install Dependencies

Make sure you have the required dependencies installed:

```bash
pip install google-generativeai>=0.8.0
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Usage

### Generate Rollouts with Gemini

Use Gemini 2.5 Pro for generating reasoning rollouts:

```bash
python generate_rollouts.py -p Gemini -m "gemini-2.0-flash-exp" -np 10 -nr 50
```

Parameters:
- `-p Gemini`: Use Gemini as the provider
- `-m "gemini-2.0-flash-exp"`: Use Gemini 2.0 Flash Experimental model
- `-np 10`: Process 10 problems
- `-nr 50`: Generate 50 rollouts per chunk

### Generate Chain-of-Thought Solutions with Gemini

Generate CoT solutions using Gemini:

```bash
python misc-experiments/generate_cots.py -p Gemini -m "gemini-2.0-flash-exp" -np 100 -r 10
```

Parameters:
- `-p Gemini`: Use Gemini as the provider
- `-m "gemini-2.0-flash-exp"`: Use Gemini 2.0 Flash Experimental model
- `-np 100`: Process 100 problems
- `-r 10`: Generate 10 solutions per problem

## Available Gemini Models

You can use any of these Gemini models by changing the `-m` parameter:

- `gemini-2.0-flash-exp` - Latest experimental model (recommended)
- `gemini-1.5-pro` - High-performance model
- `gemini-1.5-flash` - Fast, lightweight model

## Testing the Integration

Test your Gemini setup with the provided test script:

```bash
python test_gemini.py
```

This will verify that:
- Your API key is configured correctly
- The Gemini API is accessible
- Basic text generation works

## Key Differences from Other Providers

### API Format
- Uses Google's Generative Language API instead of OpenAI-style APIs
- Different request/response JSON structure
- API key passed as URL parameter instead of Authorization header

### Response Handling
- No streaming support (uses standard HTTP requests)
- Different error codes and rate limiting behavior
- Response text is nested in `candidates[0].content.parts[0].text`

### Parameter Support
- Supports: `temperature`, `topP`, `maxOutputTokens`
- Does not support: `frequency_penalty`, `presence_penalty`, `repetition_penalty`, `top_k`, `min_p`

### Rate Limits
- Google AI Studio has different rate limits than other providers
- Monitor your usage in the Google AI Studio console

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your `GOOGLE_API_KEY` is correctly set in `.env`
2. **Rate Limiting**: Reduce concurrency or add delays between requests
3. **Model Not Found**: Use exact model names like `gemini-2.0-flash-exp`
4. **Response Format Error**: Check that the response contains valid candidates

### Error Messages

- `GOOGLE_API_KEY not found`: Add your API key to `.env`
- `Invalid Gemini response`: Check API key and model name
- `Gemini API error: 400`: Usually indicates invalid request parameters
- `Gemini API error: 429`: Rate limit exceeded, try again later

## Output Format

Gemini integration maintains compatibility with existing analysis tools:

- Generated solutions use the same JSON format
- Answer extraction works with `extract_boxed_answers()` 
- Accuracy calculation uses the same `check_answer()` function
- Rollout analysis is compatible with existing scripts

## Example Output

```json
{
  "problem_idx": 123,
  "problem": "What is 2 + 3?",
  "level": "Level 1",
  "type": "Arithmetic",
  "prompt": "Solve this math problem step by step...",
  "solution": "To solve 2 + 3, I need to add...",
  "full_cot": "Solve this math problem step by step... To solve 2 + 3...",
  "temperature": 0.6,
  "top_p": 0.95,
  "answer": "5",
  "gt_answer": "5",
  "is_correct": true,
  "run_id": 0
}
```

## Performance Notes

- Gemini 2.0 Flash Experimental offers good performance for mathematical reasoning
- No reasoning tokens support (unlike some models through OpenRouter)
- Response times are generally competitive with other API providers
- Consider using batch processing for large-scale experiments
