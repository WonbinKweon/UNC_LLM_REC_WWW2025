from rich import print as pprint
import random
import json
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import argparse
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
import csv
from peft import AutoPeftModelForCausalLM
from utils import *
from models import *

def train(args):
    ## Dataset #################################################
    dataset = args.dataset

    train_pair = np.load('data/'+dataset+'/train.npy', allow_pickle=True)
    train_dic = np.load('data/'+dataset+'/train_dic.npy', allow_pickle=True).item()
    val_dic = np.load('data/'+dataset+'/val_dic.npy', allow_pickle=True).item()
    trainval_dic = np.load('data/'+dataset+'/trainval_dic.npy', allow_pickle=True).item()
    test_dic = np.load('data/'+dataset+'/test_dic.npy', allow_pickle=True).item()

    idx_title = np.load('data/'+dataset+'/idx_title.npy', allow_pickle=True).item()
    # idx_title_year = np.load('data/'+dataset+'/idx_title_year.npy', allow_pickle=True).item()
    # idx_genre = np.load('data/'+dataset+'/idx_genre.npy', allow_pickle=True).item()

    user_list = list(train_dic.keys())
    item_list = list(idx_title.keys())

    num_user = len(user_list)
    num_item = len(item_list)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    
    # ## Base Model #################################################
    # base_model = torch.load('model/BPR_'+dataset, map_location = device)
    # base_model.eval()

    # val_loader = torch.utils.data.DataLoader(user_list, batch_size=50000)
    # for us in val_loader:
    #     row = base_model.forward_eval(us.to(device))
        
    #     row_test = row.clone().detach()
    #     for idx, u in enumerate(us): # do not recommend interacted items
    #         row_test[idx][trainval_dic[u.numpy().item()]] = float('-inf')
    #         row_test[idx][test_dic[u.numpy().item()]] = float('-inf')

    #     ## ranking (sorting)
    #     _, topk_idx_base = torch.topk(row_test, args.ncan)
    # topk_idx_base = topk_idx_base.detach().cpu().numpy()

    ## Model #################################################
    model_name = args.model
    if args.model_ft != None:
        model = AutoPeftModelForCausalLM.from_pretrained(args.model_ft).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    
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
    prompts_per_idx = []
    # prompts_test_title = []
    for u_idx, u in enumerate(user_list):
        # user_history = "\n".join([f"{title_env[0]}"+idx_title[i]+f"{title_env[1]}" for i in train_dic[u][:num_history]])
        user_history = "\n".join([f"{title_env[0]}"+idx_title[i]+f"{title_env[1]}" for i in train_dic[u][-num_history:]]) ## latest items
        
        # candidate = np.concatenate((topk_idx_base[u_idx][:num_candidate-1], val_dic[u]), axis=0) # validation item
        candidate = np.concatenate((np.random.choice(item_list, num_candidate-1), val_dic[u]), axis=0) ## random candidate
        for p in range(args.nper):
            idx_per = np.random.permutation(num_candidate)
            candidate_per = candidate[idx_per]
            prompts_per_idx.append(idx_per)
            # prompts_test_title.append(idx_title[test_dic[u][0]])
            # prompts_test_title.append(idx_title[val_dic[u][0]])

            user_candidate = "\n".join([f"{indicator} {indicator_env[0]}"+indicator_sym[idx] + f"{indicator_env[1]}: {title_env[0]}"+idx_title[i] +f"{title_env[1]}" for idx,i in enumerate(candidate_per)])
            
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
                
            if model_name == "meta-llama/Llama-2-7b-chat-hf" or model_name == "meta-llama/Llama-2-13b-chat-hf":
                prompt = f"{B_INST} {B_SYS}{system_prompt.strip()}{E_SYS}{user_prompt.strip()} {E_INST}\n\n{indicator} {indicator_env[0]}"
            elif model_name == "meta-llama/Meta-Llama-3.1-8B-Instruct":
                prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt.strip()}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{indicator} {indicator_env[0]}"
            elif model_name == "mistralai/Mistral-7B-Instruct-v0.2":
                prompt = f"{B_INST} {system_prompt.strip()}\n{user_prompt.strip()} {E_INST} {indicator} {indicator_env[0]}"
            elif model_name == "google/gemma-7b-it":
                prompt = f"<bos><start_of_turn>user\n{system_prompt.strip()}\n{user_prompt.strip()}<end_of_turn>\n<start_of_turn>model\n{indicator} {indicator_env[0]}"
            prompts.append(prompt)

    ## Generate #################################################
    # prompts = prompts[:20] ## for test
    batch_size = args.bs
    data_loader = DataLoader(prompts, batch_size=batch_size)

    y_prob_matrix = []
    for batch in data_loader:
        tokenized_chat = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**tokenized_chat, max_new_tokens=2, output_logits =True, return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id)
        logit_matrix = torch.transpose(torch.stack(outputs.logits), 0, 1)
        
        for ans in range(len(batch)):       
            logits = logit_matrix[ans][0]
            next_token_probs = torch.softmax(logits, -1)
            topk_next_tokens = torch.topk(next_token_probs, num_candidate)
            y_prob = np.zeros(num_candidate)
            for idx, prob in zip(topk_next_tokens.indices, topk_next_tokens.values):
                if tokenizer.decode(idx) in indicator_sym:
                    y_prob[indicator_sym.index(tokenizer.decode(idx))] += prob
            y_prob = y_prob / (y_prob.sum()+1e-5)
            y_prob_matrix.append(y_prob)

    y_prob_matrix = np.asarray(y_prob_matrix)
    y_prob_matrix_per = np.zeros_like(y_prob_matrix)
    for i in range(y_prob_matrix.shape[0]):
        y_prob_matrix_per[i][prompts_per_idx[i]] = y_prob_matrix[i]
    
    ## Evaluate #################################################
    y_prob_matrix_per_re = y_prob_matrix_per.reshape(-1, args.nper, num_candidate) ## (user * num_permu * num_candidate)
    y_prob_mean = np.mean(y_prob_matrix_per_re, axis=1) # predictive dist. (user * num_candidate)

    rank = np.where(np.argsort(-y_prob_mean) == (num_candidate-1))[1]
    ndcg = 1/np.log2(rank+2)
    mask = (rank < 10)    
    ndcg10 = (1/np.log2(rank+2))*mask

    total_unc = compute_entropy(np.mean(y_prob_matrix_per_re, axis=1))
    model_unc = np.mean(compute_entropy(y_prob_matrix_per_re), axis=1)

    print("ndcg:", ndcg10.mean(), ndcg.mean())

    print("avg total uncertainty: ", np.mean(total_unc)) # Total uncertainty (#users, )
    print("avg model uncertainty: ", np.mean(model_unc)) # Model uncertainty (#users, )
    print("avg data uncertainty: ", np.mean(total_unc) - np.mean(model_unc)) # Model uncertainty (#users, )

    import lifelines
    print("total C-index: ", lifelines.utils.concordance_index(ndcg, -total_unc, event_observed=None))
    print("model C-index: ", lifelines.utils.concordance_index(ndcg, -model_unc, event_observed=None))

    import scipy.stats as stats
    print("total tau: ", stats.kendalltau(ndcg, -total_unc)[0])
    print("model tau: ", stats.kendalltau(ndcg, -model_unc)[0])

    import scipy.stats as stats
    print("total r: ", stats.pearsonr(ndcg, -total_unc)[0])
    print("model r: ", stats.pearsonr(ndcg, -model_unc)[0])

    ## Save #################################################
    if args.model_ft != None:
        np.save('result/'+dataset+'_'+args.model_ft.split("/")[-1]+args.save, y_prob_matrix_per_re)
    else:
        np.save('result/'+dataset+'_'+model_name.split("/")[-1]+args.save, y_prob_matrix_per_re)

    with open("/home/kwb4453/1.Uncertainty/sbatch/log.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow([args.model, args.nhist, args.ncan, args.ind, args.ind_env, args.ind_sym, args.title_env, ndcg10.mean(), ndcg.mean()] + [total_unc.mean(), model_unc.mean(), total_unc.mean() - model_unc.mean()])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ml-1m', type=str)
    parser.add_argument('--model', default="meta-llama/Llama-2-13b-chat-hf", type=str) #"meta-llama/Llama-2-7b-chat-hf" #"meta-llama/Llama-2-13b-chat-hf" #"mistralai/Mistral-7B-Instruct-v0.2"
    parser.add_argument('--model_ft', default=None, type=str)
    parser.add_argument('--bs', default=10, type=int)
    parser.add_argument('--nhist', default=10, type=int)
    parser.add_argument('--nper', default=10, type=int)
    parser.add_argument('--ncan', default=10, type=int)
    parser.add_argument('--ind', default="Candidate", type=str)
    parser.add_argument('--save', default="", type=str)
    parser.add_argument('-ind_env', '--ind_env', nargs='+', default=["[", "]"])
    parser.add_argument('--ind_sym', default="A", type=str)
    parser.add_argument('-title_env', '--title_env', nargs='+', default=["'", "'"])
    
    args = parser.parse_args()
    result = train(args)