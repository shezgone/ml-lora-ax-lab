import json
import random

input_file = "solverx_knowledge.jsonl"
output_file = "mlx_train.jsonl"

prompts = [
    "SolverX에 대해 알려줘.",
    "SolverX에 대한 정보를 제공해.",
    "SolverX란 무엇인가?",
    "SolverX에 대해 설명해봐.",
    "다음 사실을 기억해: ",
    "SolverX 관련 지식: ",
]

with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    lines = f_in.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Error parsing line {i+1}: {line}")
            continue
        fact = data["text"]
        
        # Create a chat interaction
        # We mix direct questions and "memorize this" instructions
        prompt = random.choice(prompts)
        
        # If the prompt is "memorize", the user input includes the fact? 
        # No, usually for knowledge injection in chat, we want the model to answer with the fact.
        # So User asks -> Assistant answers Fact.
        
        # However, for "Memorize this", it might be User: "Memorize: [Fact]" -> Assistant: "OK".
        # But we want the model to RETRIEVE the fact.
        # So User: "What is SolverX?" -> Assistant: "[Fact]" is better.
        
        # Let's stick to retrieval-style prompts mostly.
        
        user_content = "SolverX에 대해 알려주세요."
        # Try to be a bit more specific if possible, but generic is fine for this small test.
        # We can also use the fact itself as part of the context if we were doing RAG, but here we want LoRA to bake it in.
        
        # Let's use a mix of generic prompts.
        user_content = random.choice([
            "SolverX에 대해 말해줘.",
            "SolverX에 대한 사실을 하나 알려줘.",
            "SolverX가 뭐야?",
            "SolverX 특징이 뭐야?"
        ])

        # Also include the "Memorize" style from the previous formatted file
        # User: "Memorize this fact." -> Assistant: "OK, I have memorized that [Fact]."
        # This helps CPT.
        
        # Strategy: Generate 2 examples per fact.
        # 1. QA style
        message_qa = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": fact}
            ]
        }
        f_out.write(json.dumps(message_qa, ensure_ascii=False) + "\n")
        
        # 2. Completion/Injection style
        message_inject = {
            "messages": [
                {"role": "user", "content": f"다음 내용을 학습해:\n{fact}"},
                {"role": "assistant", "content": f"네, 다음 내용을 학습했습니다: {fact}"}
            ]
        }
        f_out.write(json.dumps(message_inject, ensure_ascii=False) + "\n")

print(f"Created {output_file} with {len(lines)*2} examples.")
