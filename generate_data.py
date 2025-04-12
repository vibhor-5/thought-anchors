import json
import random
import anthropic
import os
import time
import argparse
from tqdm import tqdm

def load_topics_subtopics(file_path="topics_subtopics.json"):
    """Load topics and subtopics from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Creating default topics and subtopics file.")

def select_random_topic_subtopic(topics_subtopics):
    """Select a random topic and a random subtopic from that topic."""
    topic = random.choice(list(topics_subtopics.keys()))
    subtopic = random.choice(topics_subtopics[topic])
    return topic, subtopic

def generate_difficulty():
    """Generate a random difficulty level with weighted probabilities."""
    difficulties = ["Easy", "Medium", "Hard"]
    weights = [0.3, 0.4, 0.3]  # 30% Easy, 40% Medium, 30% Hard
    return random.choices(difficulties, weights=weights)[0]

def create_claude_prompt(topic, subtopic, difficulty):
    """Create a prompt for Claude to generate a problem."""
    prompt = f"""Create a thought-provoking {difficulty.lower()}-level problem related to {topic}, specifically in the area of {subtopic}.
    The problem should:
    1. Challenge the problem solver to engage in deep reasoning
    2. Have a clear, well-defined answer that follows logically from the problem statement
    3. Be precise and unambiguous, but still intellectually stimulating
    4. Be appropriate for the {difficulty.lower()} difficulty level
    5. Require the specific reasoning skills associated with {topic}

    Format your response as a valid JSON object with the following fields:
    - "topic": "{topic}"
    - "subtopic": "{subtopic}"
    - "difficulty": "{difficulty}"
    - "question": The problem statement or question
    - "answer": A comprehensive, step-by-step solution that demonstrates the reasoning process

    Do not include any explanation outside the JSON object. Ensure the JSON is valid and properly formatted."""
    return prompt

def generate_problem_with_claude(topic, subtopic, difficulty, client):
    """Generate a reasoning problem using Claude 3.7 Sonnet."""
    prompt = create_claude_prompt(topic, subtopic, difficulty)
    
    try:
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=4000,
            system="You are an expert in creating challenging reasoning problems that test critical thinking skills.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract JSON from response
        content = response.content[0].text
        
        try:
            # If the response is wrapped in ```json and ``` markers, strip them
            if content.startswith("```json") and content.endswith("```"):
                content = content.split("```json")[1].split("```")[0].strip()
            
            problem_json = json.loads(content)
            return problem_json
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"Raw content: {content}")
            return None
            
    except Exception as e:
        print(f"API error: {e}")
        return None

def save_problems_to_file(problems, filename="reasoning_problems.json"):
    """Save the generated problems to a JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(problems, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(problems)} problems to {filename}")

def append_problem_to_file(problem, filename="reasoning_problems.json"):
    """Append a single problem to an existing JSON file or create a new one."""
    try:
        # Try to load existing problems
        with open(filename, 'r', encoding='utf-8') as f:
            try:
                problems = json.load(f)
            except json.JSONDecodeError:
                # File exists but is not valid JSON or is empty
                problems = []
    except FileNotFoundError:
        # File doesn't exist yet
        problems = []
    
    # Append the new problem
    problems.append(problem)
    
    # Save the updated list
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(problems, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description='Generate reasoning problems using Claude API')
    parser.add_argument('--n_examples', type=int, default=1000, help='Number of problems to generate')
    parser.add_argument('--api_key', type=str, help='Anthropic API key')
    parser.add_argument('--output_file', type=str, default='reasoning_problems.json', help='Output JSON file name')
    parser.add_argument('--topics_file', type=str, default='topics_subtopics.json', help='JSON file containing topics and subtopics')
    args = parser.parse_args()
    
    # Load topics and subtopics
    topics_subtopics = load_topics_subtopics(args.topics_file)
    
    # Get API key from arguments or environment variable
    api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: Anthropic API key not provided. Use --api_key or set ANTHROPIC_API_KEY environment variable.")
        return
    
    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=api_key)
    
    # Initialize problems list (for final output)
    problems = []
    
    for i in tqdm(range(args.n_examples), desc="Generating problems"):
        topic, subtopic = select_random_topic_subtopic(topics_subtopics)
        difficulty = generate_difficulty()
        problem = generate_problem_with_claude(topic, subtopic, difficulty, client)
        
        if problem:
            problems.append(problem)
            append_problem_to_file(problem, args.output_file)
        
        # Avoid hitting rate limits
        time.sleep(1)
    
    # Final save is redundant but kept for backward compatibility
    save_problems_to_file(problems, args.output_file)

if __name__ == "__main__":
    main()