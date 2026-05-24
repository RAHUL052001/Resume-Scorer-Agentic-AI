from transformers import pipeline

def generate_text_local(prompt: str) -> str:
    print("Loading model (this might take a minute on the first run)...")
    
    # Setting up a text-generation pipeline using a free, lightweight model
    pipe = pipeline(
        "text-generation", 
        model="microsoft/Phi-3-mini-4k-instruct", # Free, powerful 3.8B parameter model
        device_map="auto" # Automatically uses GPU if available, otherwise CPU
    )
    
    # Format prompt for an instruction model
    messages = [{"role": "user", "content": prompt}]
    
    # Run generation
    outputs = pipe(messages, max_new_tokens=100, temperature=0.7)
    
    return outputs[0]['generated_text'][-1]['content']

# Test the function
output = generate_text_local("Give me a one-sentence motivational quote about coding.")
print(output)