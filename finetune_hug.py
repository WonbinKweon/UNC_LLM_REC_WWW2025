from openai import OpenAI
from rich import print as pprint
import random
import json
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import argparse
from utils import *
from models import *
from peft import AutoPeftModelForCausalLM
import transformers
from utils import *
from models import *
from torch import cuda, bfloat16

def train(args):
    ## Dataset #################################################
    dataset = args.dataset

    train_pair = np.load('data/'+dataset+'/train.npy', allow_pickle=True)
    train_dic = np.load('data/'+dataset+'/train_dic.npy', allow_pickle=True).item()
    val_dic = np.load('data/'+dataset+'/val_dic.npy', allow_pickle=True).item()
    trainval_dic = np.load('data/'+dataset+'/trainval_dic.npy', allow_pickle=True).item()
    test_dic = np.load('data/'+dataset+'/test_dic.npy', allow_pickle=True).item()

    idx_title = np.load('data/'+dataset+'/idx_title.npy', allow_pickle=True).item()

    user_list = list(train_dic.keys())
    item_list = list(idx_title.keys())

    num_user = len(user_list)
    num_item = len(item_list)
    
    ## Base Model #################################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = torch.load('model/BPR_'+dataset, map_location = device)
    base_model.eval()

    val_loader = torch.utils.data.DataLoader(user_list, batch_size=50000)
    for us in val_loader:
        row = base_model.forward_eval(us.to(device))
        
        row_test = row.clone().detach()
        for idx, u in enumerate(us): # do not recommend interacted items
            row_test[idx][trainval_dic[u.numpy().item()]] = float('-inf')
            row_test[idx][test_dic[u.numpy().item()]] = float('-inf')

        ## ranking (sorting)
        _, topk_idx_base = torch.topk(row_test, args.ncan)
    topk_idx_base = topk_idx_base.detach().cpu().numpy()
    
    ## Model #################################################
    model_name = args.model

    ## Prompt #################################################
    num_history = args.nhist
    num_candidate = args.ncan
    indicator = args.ind
    indicator_env = args.ind_env
    if args.ind_sym == "A":
        indicator_sym = [chr(ord('@')+i+1) for i in range(num_candidate)]
    elif args.ind_sym == "a":
        indicator_sym = [chr(ord('`')+i+1) for i in range(num_candidate)]
    elif args.ind_sym == "1":
        indicator_sym = [str(i+1) for i in range(num_candidate)]
    title_env = args.title_env

    # model-specific
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    B_INST, E_INST = "[INST]", "[/INST]"

    # preparation
    prompts = []
    for u_idx, u in enumerate(user_list):
        user_hist_num = min(num_history, len(train_dic[u])-1)
        user_history = "\n".join([f"{title_env[0]}"+idx_title[i]+f"{title_env[1]}" for i in train_dic[u][:user_hist_num]])

        # candidate = np.random.choice(item_list, num_candidate-1).tolist() + [train_dic[u][user_hist_num]]  # random candidiate # val_dic[u].tolist()
        candidate = topk_idx_base[u_idx][:num_candidate-1].tolist() + [train_dic[u][user_hist_num]]
        random.shuffle(candidate)
        test_idx = candidate.index(train_dic[u][user_hist_num])

        user_candidate = "\n".join([f"{indicator} {indicator_env[0]}"+indicator_sym[idx] + f"{indicator_env[1]}: {title_env[0]}"+idx_title[i] +f"{title_env[1]}" for idx,i in enumerate(candidate)])

        if dataset == "ml-1m":
            user_prompt = f"I have watched the following movies in the past in order:\n{user_history}\n"\
                f"The {num_candidate} candidate movies are as follows:\n{user_candidate}\n"\
                "Only respond the identifier of the recommended movie without any word or explain.\n"\
                "Which movie would I like to watch next most?"
            system_prompt = "You are a movie recommender system.\nGiven user's watch history, you output the identifier of the recommended movie."
        elif dataset == "steam":
            user_prompt = f"I have played the following games in the past in order:\n{user_history}\n"\
                f"The {num_candidate} candidate games are as follows:\n{user_candidate}\n"\
                "Only respond the identifier of the recommended game without any word or explain.\n"\
                "Which game would I like to play next most?"
            system_prompt = "You are a game recommender system.\nGiven user's play history, you output the identifier of the recommended game."
        elif dataset == "amazon_grocery":
            user_prompt = f"I have purchased the following products in the past in order:\n{user_history}\n"\
                f"The {num_candidate} candidate products are as follows:\n{user_candidate}\n"\
                "Only respond the identifier of the recommended product without any word or explain.\n"\
                "Which product would I like to purchase next most?"
            system_prompt = "You are a product recommender system.\nGiven user's purchase history, you output the identifier of the recommended product."

        if model_name == "meta-llama/Llama-2-7b-chat-hf" or model_name == "meta-llama/Llama-2-13b-chat-hf" or model_name == "meta-llama/Llama-2-70b-chat-hf":
            prompt = f"{B_INST} {B_SYS}{system_prompt.strip()}{E_SYS}{user_prompt.strip()} {E_INST}\n\n{indicator} {indicator_env[0]}{indicator_sym[test_idx]}"
        elif model_name == "mistralai/Mistral-7B-Instruct-v0.2" or model_name == "mistralai/Mixtral-8x7B-Instruct-v0.1":
            prompt = f"{B_INST} {system_prompt.strip()}\n{user_prompt.strip()} {E_INST} {indicator} {indicator_env[0]}{indicator_sym[test_idx]}"
        elif model_name == "google/gemma-7b-it" or model_name == "google/gemma-2b-it":
            prompt = f"<bos><start_of_turn>user\n{system_prompt.strip()}\n{user_prompt.strip()}<end_of_turn>\n<start_of_turn>model\n{indicator} {indicator_env[0]}{indicator_sym[test_idx]}"
        
        prompts.append(prompt)

    ## Training #################################################
    batch_size = args.bs
    data_loader = DataLoader(prompts, batch_size=batch_size)

    if args.model_ft != None:
        if model_name == 'meta-llama/Llama-2-70b-chat-hf' or model_name == "mistralai/Mixtral-8x7B-Instruct-v0.1":
            model_config = transformers.AutoConfig.from_pretrained(model_name)
            bnb_config = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=bfloat16)
            model = AutoPeftModelForCausalLM.from_pretrained(args.model_ft, quantization_config=bnb_config, is_trainable=True).to(device) #load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=bfloat16
            print(f"Model loaded on {device}")
        else:
            model = AutoPeftModelForCausalLM.from_pretrained(args.model_ft, is_trainable=True).to(device)
    else:
        if model_name == 'meta-llama/Llama-2-70b-chat-hf' or model_name == "mistralai/Mixtral-8x7B-Instruct-v0.1":
            bnb_config = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=bfloat16)
            model_config = transformers.AutoConfig.from_pretrained(model_name)
            model = transformers.AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, config=model_config, quantization_config=bnb_config, device_map='auto')
            print(f"Model loaded on {device}")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        lora_config = LoraConfig(task_type='CAUSAL_LM', r=16, lora_alpha=16, lora_dropout=0.05)
        model = get_peft_model(model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', truncation_side='left')
    tokenizer.pad_token = tokenizer.eos_token  
    
    model.train()
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    for epoch in range(1, 21):
        train_loss = 0
        
        for batch in data_loader:
            if model_name == "google/gemma-7b-it":
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=500).to(device)
            else:
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=600).to(device)

            inputs['labels'] = inputs['input_ids'].clone()
            inputs['labels'][:, :-1] = -100 # the last token is the answer
            
            loss = model(**inputs).loss
            loss.backward()
            # clip_grad_norm_(model.parameters(), args.max_norm)
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
        train_loss /= len(data_loader)
        
        print(train_loss, flush=True)
        if epoch % 1 == 0:
            model.save_pretrained(f'model/'+dataset+f'/{model_name.split("/")[-1]}_{epoch}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ml-1m', type=str)
    parser.add_argument('--model', default="meta-llama/Llama-2-13b-chat-hf", type=str) #"meta-llama/Llama-2-7b-chat-hf" #"meta-llama/Llama-2-13b-chat-hf" #"mistralai/Mistral-7B-Instruct-v0.2"
    parser.add_argument('--model_ft', default=None, type=str)
    parser.add_argument('--bs', default=10, type=int)
    parser.add_argument('--nhist', default=10, type=int)
    parser.add_argument('--ncan', default=10, type=int)
    parser.add_argument('--ind', default="Candidate", type=str)
    parser.add_argument('-ind_env', '--ind_env', nargs='+', default=["[", "]"])
    parser.add_argument('--ind_sym', default="A", type=str)
    parser.add_argument('-title_env', '--title_env', nargs='+', default=["'", "'"])
    
    args = parser.parse_args()
    result = train(args)