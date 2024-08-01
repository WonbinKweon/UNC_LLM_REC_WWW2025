# UNCLLMREC_KDD
for anonymous submission of KDD 2025 August.




### Source codes
for candidate generators
- models.py
- train.py
- we provide BPR for ml-1m dataset

for fine-tuning LLMs
- finetune_hug.py
- we provide fine-truned LoRA weights for ml-1m dataset

for generating recommendations with LLMs
- utils.py
- uncertainty_hug.py



### How to run
for zero-shot ranking
- python uncertainty_hug.py --nper 5 --nhist 20 --ncan 20 --bs 5 --ind Candidate --ind_env [ ] --ind_sym A --title_env "'" "'" --dataset ml-1m --model meta-llama/Llama-2-7b-chat-hf

for fine-tuned ranking
- python uncertainty_hug.py --nper 5 --nhist 20 --ncan 20 --bs 5 --ind Candidate --ind_env [ ] --ind_sym A --title_env "'" "'" --dataset ml-1m --model meta-llama/Llama-2-7b-chat-hf --model_ft model/ml-1m/Llama-2-7b-chat-hf_20
