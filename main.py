#%%

from utils import *

#%%

MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_NAME = MODEL_ID.split("/")[-1]
model = HookedTransformer.from_pretrained_no_processing(
    MODEL_ID,
    device="cuda",
    dtype="bfloat16",
)
model.eval()
model.requires_grad_(False)
print(f"Loaded model: {model.cfg.model_name}")
print(f"Layers: {model.cfg.n_layers}, Heads: {model.cfg.n_heads}, d_model: {model.cfg.d_model}")

if "gemma" in MODEL_NAME:
    model.tokenizer.eot_token_id = model.tokenizer.encode("<end_of_turn>")[-1]
else:
    model.tokenizer.eot_token_id = model.tokenizer.eos_token_id

#%%

do_test_generation = True
if do_test_generation:
    # prompt = "What is 2 + 2?"
    # prompt = "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?"
    prompt = "Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works 10 hours every day for 5 days, how much money does she make?"
    resp = get_test_response(model, prompt, max_new_tokens=2048, give_toks=False, verbose=True)
    print(resp)

#%%

load_and_save_gsm8k = False
if load_and_save_gsm8k:
    
    gsm8k_dataset = datasets.load_dataset("openai/gsm8k", "main", split="train")
    gsm8k = []
    for i in range(len(gsm8k_dataset)):
        question = gsm8k_dataset["question"][i]
        answer = gsm8k_dataset["answer"][i]
        answer_value = parse_answer_value(answer)
        gsm8k.append({
            "question": question,
            "answer": answer,
            "answer_value": answer_value
        })

    os.makedirs("data", exist_ok=True)
    with open("data/gsm8k.jsonl", "w") as f:
        for item in gsm8k:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {len(gsm8k)} items to data/gsm8k.jsonl")

#%%

benchmark_gsm8k = True
if benchmark_gsm8k:
    import json
    
    with open("data/gsm8k.jsonl", "r") as f:
        gsm8k = []
        for line in f:
            item = json.loads(line)
            # Parse answer_value if not already present
            if "answer_value" not in item:
                item["answer_value"] = parse_answer_value(item["answer"])
            gsm8k.append(item)
    print(f"Loaded {len(gsm8k)} items from data/gsm8k.jsonl")

    max_new_tokens = 2048
    n_trials_per_question = 256
    batch_size = 16
    append_question = " Give your answer within \\boxed{}. You don't need to explain your answer."

    for item in gsm8k:
        question = item["question"]
        answer = item["answer"]
        question_formatted = question + append_question

        conv_toks = model.tokenizer.apply_chat_template(
            conversation = [{"role": "user", "content":question_formatted}],
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.cfg.device)

        print(cyan, conv_toks, endc)
        conv_str = model.tokenizer.decode(conv_toks[0])
        print(lime, conv_str, endc)

        conv_toks_batch = conv_toks.repeat(batch_size, 1)
        resp_toks = model.generate(conv_toks_batch, max_new_tokens=max_new_tokens)
        
        for i in range(batch_size):
            resp_str = model.tokenizer.decode(resp_toks[i], verbose=True)
            print(orange, resp_str, endc)

            answer = extract_answer(resp_str)
            print(f"Model answer: {answer}")
            print(f"Correct answer: {item['answer_value']}")
        
        break

#%%