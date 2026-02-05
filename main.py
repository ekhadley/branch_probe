#%%

from utils import *

#%%

# MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_ID = "Qwen/Qwen3-0.6B"
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

do_test_generation = False
if do_test_generation:
    # prompt = "What is 2 + 2?"
    # prompt = "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?"
    prompt = "Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works 10 hours every day for 5 days, how much money does she make?"
    resp = get_test_response(model, prompt, max_new_tokens=2048, give_toks=False, verbose=True)
    print(resp)

#%%

load_and_save_gsm8k = False
if load_and_save_gsm8k:
    save_path = "data/gsm8k.jsonl"
    
    gsm8k_dataset = datasets.load_dataset("openai/gsm8k", "main", split="train")
    gsm8k = []
    for i in range(len(gsm8k_dataset)):
        question = gsm8k_dataset["question"][i]
        answer = gsm8k_dataset["answer"][i]
        answer_value = parse_answer_value(answer)
        assert answer_value is not None, f"Answer value is None for question: {question}"
        gsm8k.append({
            "question": question,
            "answer": answer,
            "answer_value": answer_value
        })

    save_jsonl(gsm8k, save_path)
    print(f"Saved {len(gsm8k)} items to {save_path}")

#%%

benchmark_gsm8k = True
if benchmark_gsm8k:
    gsm8k_path = "data/gsm8k.jsonl"
    gsm8k = load_jsonl(gsm8k_path)
    print(gray, f"Loaded {len(gsm8k)} items from {gsm8k_path}", endc)

    max_new_tokens = 2048
    n_trials_per_question = 64
    batch_size = 16
    append_question = " Give your answer within \\boxed{}. You don't need to explain your answer."

    n_batches_per_question = n_trials_per_question // batch_size
    bar = tqdm.tqdm(gsm8k[:32], ascii=" >=", ncols=120)
    bench_results = []
    for i, item in enumerate(bar):
        question = item["question"]
        answer = item["answer"]
        answer_value = item["answer_value"]
        question_formatted = question + append_question

        conv_toks = model.tokenizer.apply_chat_template(
            conversation = [{"role": "user", "content":question_formatted}],
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.cfg.device)
        prompt_len = conv_toks.shape[-1]

        conv_toks_batch = conv_toks.repeat(batch_size, 1)
        
        example_resp = None
        answers = []
        for _ in range(n_batches_per_question):
            resp_toks = model.generate(conv_toks_batch, max_new_tokens=max_new_tokens, verbose=False)
            print(resp_toks.shape)
            
            for b in range(batch_size):
                resp_str = model.tokenizer.decode(resp_toks[b, prompt_len:])
                if resp_toks[b, -1].item() != model.tokenizer.eot_token_id:
                    print(red, f"model didnt complete its reasoning in the limit: {repr(resp_str)}", endc)
                    continue
                answer = extract_answer(resp_str)
                if answer is None:
                    print(red, f"answer couldnt be extracted from response: {repr(resp_str)}", endc)
                    continue
                example_resp = resp_str
                answers.append(answer)

            t.cuda.empty_cache()
        
        if example_resp is not None:
            print(answers)
            grades = [1.0 if answer == answer_value else 0.0 for answer in answers]
            accuracy = sum(grades) / len(grades)
            bar.set_description(f"{yellow}Accuracy: {accuracy:.2f} ({len(answers)}/{n_trials_per_question})")
            bench_result = {
                "example_resp": example_resp,
                "accuracy": accuracy,
                "n_trials": len(answers),
            }
        else:
            bench_result = {
                "example_resp": None,
                "accuracy": None,
                "n_trials": len(answers),
            }
        
        item[MODEL_NAME] = bench_result

    save_jsonl(gsm8k, gsm8k_path)
    print(gray, f"Saved {len(gsm8k)} items with benchmark results to {gsm8k_path}", endc)

#%%