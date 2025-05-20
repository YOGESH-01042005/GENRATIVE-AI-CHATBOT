# Ex.No: 10 Learning – Use Supervised Learning  
### DATE:                                                                            
### REGISTER NUMBER :212222060308
### AIM: 
To write a program to train the classifier for generative AI chatbot.
###  Algorithm:
1. Preprocessing Stage
   
Input: User query (text)
Output: Tokenized input, cleaned and structured
Receive raw user input.

Clean input (remove unwanted characters, handle misspellings, etc.).

Tokenize using the same tokenizer as the model (e.g., BPE for GPT).

Optionally: Add system prompt or conversation history to input.

2. Context Management
   
Input: Current input + previous turns
Output: Final prompt for the model
Retrieve conversation history.

Concatenate with current input (within model’s context window).

Add special tokens if required (e.g., <user>:, <bot>:).

Apply truncation strategy to fit within context length.

3. Inference (Text Generation)

Input: Final prompt
Output: Generated response tokens
Feed prompt to pre-trained language model (e.g., GPT-4, LLaMA, etc.).

Use decoding strategy:

Greedy decoding: deterministic, might lack creativity.

Top-k sampling: random but limited to top-k probable tokens.

Top-p (nucleus) sampling: sample from top cumulative probability p.

Temperature scaling: control randomness (0 = deterministic).

Generate tokens until:

End-of-sequence token is hit

Max length is reached

4. Postprocessing

Input: Generated tokens
Output: Cleaned response text
Decode tokens to natural language text.

Remove special tokens or formatting artifacts.

Optionally: Filter unsafe or toxic content (e.g., with moderation filters).

Log input/output for further training or debugging.

5. Response Delivery
 
Input: Final text response
Output: Delivered message to user
Display the response to the user in the chat interface.

Store the turn in the conversation history.

### Program:
```
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

#Load pretrained GPT-2 model and tokenizer
model_name = "gpt2"  # or use 'gpt2-medium', 'EleutherAI/gpt-neo-125M', etc.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

#Ensure model runs on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#Function to generate a response
def generate_response(prompt, max_length=150, temperature=0.7, top_p=0.9):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs,
        max_length=max_length,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

#Simple chatbot loop
def chat():
    print("Chatbot (type 'exit' to quit)")
    history = ""
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        history += f"You: {user_input}\nAI:"
        reply = generate_response(history)
        print("AI:", reply)
        history += f" {reply}\n"

if __name__ == "__main__":
    chat()
```

## System Architecture
![pro1](https://github.com/user-attachments/assets/7c608ac3-4142-48cc-866a-b495c4121ab5)

## Output

<!--Embed the Output picture at respective places as shown below as shown below-->
#### Output1 - chatbot

![oro 2](https://github.com/user-attachments/assets/138a5fe3-d255-4134-bb47-e9082ce3c3dc)


### Result:
Thus the system was trained successfully and the prediction was carried out.
