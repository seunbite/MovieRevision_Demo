import os
import logging
import gradio as gr
import numpy as np
import pandas as pd
import openai
from nltk import sent_tokenize
from distutils.util import strtobool
from utils.data_loader import auxbprm_gpt_dataloader, aux_gpt_dataloader
from utils.utils import set_seed
from utils.vocab import load_vocab, process_text
from utils.score_filter import convert_rating, extract_word_attn_score
from models.AttLSTM import AttLSTM_Aux, AttLSTM_Prm, AttLSTM_AuxBPrm
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm
from gpt_infer_chatgpt import get_checkpoint_path, GPT, load_parser_and_args

ABS_PATH = os.path.dirname(os.path.abspath(__file__))

MODEL_MAPPING = {
    'instructgpt': 'text-davinci-003',
    'gpt3': 'text-davinci-002',
    'text-davinci-003': 'text-davinci-003',
    'text-davinci-002': 'text-davinci-002',
    'code-davinci-002': 'code-davinci-002',
    'gpt-3.5-turbo': 'gpt-3.5-turbo'
}
# Configs
logger = logging.getLogger('logger')



def preprocess_raw_string(script): # input : script, output : scene_idx(N) embedded_script(1), scenes(N), embedded_scenes(N)

    # script-level
    vocab = load_vocab(vocab_path = './data/preprocess/vocab_MPAA_th17.json')
    embedded_script = process_text(script, vocab, max_length=None)

    # scene-level
    n_tokens = 256
    sentences = sent_tokenize(script) # [sent1, sent2, ...]
    embedded_sentences = {i:list(process_text(i, vocab, max_length=None)) for i in sentences} # {sent1:[embedded sent1], sent2:[embedded sent2], sent3:[embedded sent3] ...}
    
    current_length = 0
    current_scene = ""
    current_emb_scene = []

    truncated_script = []   # results
    truncated_embedded_script = []

    for k, v in embedded_sentences.items():

        if current_length + len(v) > n_tokens:
            truncated_script.append(current_scene)
            truncated_embedded_script.append(current_emb_scene)

            current_emb_scene = v
            current_scene = k
            current_length = len(v)
        else:
            current_emb_scene.extend(v)
            current_scene += "\n"
            current_scene += k
            current_length += len(v)

    truncated_embedded_script.append(current_emb_scene)
    truncated_script.append(current_scene)
    idx = list(range(len(truncated_embedded_script)))

    return idx, embedded_script, truncated_script, truncated_embedded_script # [0, 1, 2, 3..] / [1, 63, ..., 4] / [[sent1, sent2...], [sent6, ..] ] / [[1, 63, 32, ..., ], [1, 9240, ...., 4]]



def revision_(input_script, max_aspect, rating, args):

    model = GPT(args)
    model.set_new_key()

    with open(os.path.join(ABS_PATH, 'prompt', f"aspect_{max_aspect.lower()}.txt"), 'r') as f: # input
        prompt = f.read()
    prompt = prompt.replace('The movie {} is {}.', '').format(rating)

    # inference
    model_input = f"{prompt}\n\nOriginal Scene: {input_script}\n\nRevised Scene: "

    pred_string = model.inference(model_input)
    pred_string = pred_string.strip()

    return pred_string



class Solver(object):
    
    def __init__(self, args):
        self.args = args
        self.vocab = load_vocab(self.args.vocab_path)
        self.device = torch.device("cuda:" + str(self.args.n_cuda_device) if torch.cuda.is_available() else 'cpu')
        self.pretrained_abp = get_checkpoint_path(self.args.pretrained_abp)
        self.pretrained_aux = get_checkpoint_path(self.args.pretrained_aux)
        self.build_model(self.vocab)    
    
    def build_model(self, vocab):
        embedding = None
        self.model_abp = AttLSTM_AuxBPrm(vocab_size=len(vocab),
                                    max_len=None,
                                    embedding_size=self.args.embedding_size,
                                    hidden_size=self.args.lstm_hidden_size,
                                    attn_hidden_size=self.args.attn_hidden_size,
                                    cls_hidden_size=self.args.cls_hidden_size,
                                    r_size=self.args.r_size,
                                    sos_id=vocab(vocab.SYM_SOQ), eos_id=vocab(vocab.SYM_EOS),
                                    num_layers=self.args.n_lstm_layers,
                                    n_class=3,
                                    bidirectional=True,
                                    dropout_ratio=self.args.dropout_ratio,
                                    embedding=embedding,
                                    )
        self.criterion = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
    
        self.model_aux = AttLSTM_Aux(vocab_size=len(vocab),
                                max_len=None,
                                embedding_size=self.args.embedding_size,
                                hidden_size=self.args.lstm_hidden_size,
                                attn_hidden_size=self.args.attn_hidden_size,
                                cls_hidden_size=self.args.cls_hidden_size,
                                r_size=self.args.r_size,
                                sos_id=vocab(vocab.SYM_SOQ), eos_id=vocab(vocab.SYM_EOS),
                                num_layers=self.args.n_lstm_layers,
                                n_class=2,
                                bidirectional=True,
                                dropout_ratio=self.args.dropout_ratio,
                                embedding=embedding,
                                )
        self.criterion = nn.BCELoss()
 
        self.optimizer = torch.optim.Adam(self.model_abp.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.optimizer = torch.optim.Adam(self.model_aux.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.model_abp.to(self.device)
        self.model_aux.to(self.device)
        
        if not (self.pretrained_abp == None):
            checkpoint = torch.load(self.pretrained_abp, map_location=self.device)
            self.model_abp.load_state_dict(checkpoint["model"])
            
        if not (self.pretrained_aux == None):
            checkpoint = torch.load(self.pretrained_aux, map_location=self.device)
            self.model_aux.load_state_dict(checkpoint["model"])
            
            logger.info('****** Pretrained model loading complete! ******')
        
    def data_to_device(self, data):
        targets = data.to(self.device)
        return targets



def infer_auxbprm(solver, embedded_text):
    logger.info("***** Inferring Rating... *****")

    input = torch.tensor([embedded_text])
    scripts = solver.data_to_device(input.clone().detach())

    model_abp = solver.model_abp
    outputs, alphas = model_abp(scripts)
    max_val, max_index = torch.max(outputs.detach().cpu(), 1)
    rating = convert_rating(max_index.item())
    
    return rating



def infer_aux_with_scores(solver, embedded_text):    # infer aux & extract all scores (including aspect and logits)
    logger.info("***** Inferring Aspects... *****")
    total_scores = dict()

    input = torch.tensor([embedded_text])
    scene = solver.data_to_device(input.clone().detach())

    model_aux = solver.model_aux
    outputs, alphas = model_aux(scene)  # 모델에는 씬 하나씩

    logits = list()
    
    logits_sum = 0
    
    for a in outputs:   # outputs -> (outputs_1, outputs_2, outputs_3, outputs_4, outputs_5)
        a = a.detach().cpu()
        logits_sum += a[:,1]
        logits.append(a[:,1].item())
    
    a0, a1, a2, a3, a4, word_attn_scores = extract_word_attn_score(alphas, args.r_size)     

    sort_word_attn_scores = sorted(word_attn_scores, reverse=True)
    sorted_idx = torch.argsort(word_attn_scores, descending=True)

    top_k_word_score = 0
    top_a0 = 0
    top_a1 = 0
    top_a2 = 0
    top_a3 = 0
    top_a4 = 0
    
    if len(sort_word_attn_scores) >= args.top_k:
        for i in range(args.top_k):
            top_k_word_score += sort_word_attn_scores[i]
            top_a0 += a0[sorted_idx[i]]
            top_a1 += a1[sorted_idx[i]]
            top_a2 += a2[sorted_idx[i]]
            top_a3 += a3[sorted_idx[i]]
            top_a4 += a4[sorted_idx[i]]
    else:
        for i in range(len(sort_word_attn_scores)):
            top_k_word_score += sort_word_attn_scores[i]
            top_a0 += a0[sorted_idx[i]]
            top_a1 += a1[sorted_idx[i]]
            top_a2 += a2[sorted_idx[i]]
            top_a3 += a3[sorted_idx[i]]
            top_a4 += a4[sorted_idx[i]]
    
    total_scene_scores = {'all': float(logits_sum + top_k_word_score),  
                                    
                            'nudity': logits[0] + float(top_a0), 

                            'violence': logits[1] + float(top_a1), 

                            'profanity': logits[2] + float(top_a2), 

                            'alcohol': logits[3] + float(top_a3), 

                            'frightening': logits[4] + float(top_a4)
                                }
    

    return total_scene_scores



def main(script, args):
    idx, embedded_script, truncated_script, truncated_embedded_script = preprocess_raw_string(script)
    solver = Solver(args) # initiate
    lists = ['nudity', 'violence', 'profanity','alcohol','frightening']

    # Rating
    rating = infer_auxbprm(solver, embedded_script)

    # Aspect
    thre_aspect = 0.7
    aspects = infer_aux_with_scores(solver, embedded_script)

    movie_aspects = [aspects['nudity'], aspects['violence'], aspects['profanity'], aspects['alcohol'], aspects['frightening']]
    high_aspects = [k for k, v in zip(lists, movie_aspects) if v > thre_aspect]

    # Scene Aspect
    scene_level_aspects = dict()

    for k, embedded_scene in enumerate(truncated_embedded_script):
        scene_level_aspects[k] = infer_aux_with_scores(solver, embedded_scene) # [total_scores1, total_scores2...]

    results = dict()
    results['idx'] = idx
    results['scene'] = truncated_script
    for l in lists:
        results[l] = [s[l]['total'] for s in scenes_total_scores]
    results['total score'] = [s['all'] for s in scenes_total_scores]
    scene_df = pd.DataFrame.from_dict(results)
    df_dict = scene_df[['nudity', 'violence','profanity','alcohol','frightening']].to_dict('index')  # max aspect
    max_score_n_aspect = [sorted(v.items(), key=lambda x: x[1], reverse = True)[0] for k, v in df_dict.items()]
    max_aspect = [i[0] for i in max_score_n_aspect]
    scene_df['max aspect'] = max_aspect

    return rating, movie_aspects, high_aspects, scene_df


    
if __name__ == "__main__":
    set_seed(0)
    parser, args = load_parser_and_args()
    with open('/home/intern/sblee/sblee/script-revision/data/Deliver_dataset/Script/tt1431045.txt', 'r') as f:
        fr = f.read()
        print(fr)

    main(fr, args)

