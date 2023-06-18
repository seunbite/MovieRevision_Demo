import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import time, re, openpyxl
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.backends
import torchio as tio
from tqdm import tqdm
from pathlib import Path
from utils.vocab import load_vocab, process_text
from utils.utils import EarlyStopping, set_seed
from utils.score_filter import extract_word_attn_score, filter_data, filter_n_save_score, save_results, convert_rating
from models.AttLSTM import AttLSTM_Aux, AttLSTM_Prm, AttLSTM_AuxBPrm
from torchinfo import summary
from nltk import sent_tokenize
from gpt_infer_abp import GPT

import parser_helper as helper
import argparse
ABS_PATH = helper.ABS_PATH
DATE_TIME = helper.DATE_TIME


    

def get_checkpoint_path(checkpoint_path_str=None):
    """ Return the checkpoint path if it exists """
    if checkpoint_path_str == None:
        return None
    checkpoint_path = Path(checkpoint_path_str)

    if not checkpoint_path.exists():
        raise Exception("Error: Checkpoint_path Not exist")
    else:
        return checkpoint_path_str



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



class Solver(object):
    
    def __init__(self, config, task_num, pretrained_check):
        self.config = config
        self.task_num = task_num
        self.vocab = load_vocab(config.vocab_path)
        self.device = torch.device("cuda:" + str(config.n_cuda_device) if torch.cuda.is_available() else 'cpu')

        self.pretrained_check = get_checkpoint_path(pretrained_check)

        self.build_model(self.vocab)
        self.early_stopping = EarlyStopping(config=self.config, patience=self.config.n_patience, verbose = True)

        self.total_valid_loss = 0.0
        self.sheet_row = self.config.sheet_start_row

    def build_model(self, vocab):

        embedding = None  # embedding..

        if self.task_num == 10:
            self.model = AttLSTM_Aux(vocab_size=len(vocab),
                                 max_len=None,
                                 embedding_size=self.config.embedding_size,
                                 hidden_size=self.config.lstm_hidden_size,
                                 attn_hidden_size=self.config.attn_hidden_size,
                                 cls_hidden_size=self.config.cls_hidden_size,
                                 r_size=self.config.r_size,
                                 sos_id=vocab(vocab.SYM_SOQ), eos_id=vocab(vocab.SYM_EOS),
                                 num_layers=self.config.n_lstm_layers,
                                 n_class = 2,
                                 bidirectional=True,
                                 dropout_ratio=self.config.dropout_ratio,
                                 embedding=embedding,
                                 )
            self.criterion = nn.BCELoss()

        elif self.task_num == 11:
            self.model = AttLSTM_AuxBPrm(vocab_size=len(vocab),
                                     max_len=None,
                                     embedding_size=self.config.embedding_size,
                                     hidden_size=self.config.lstm_hidden_size,
                                     attn_hidden_size=self.config.attn_hidden_size,
                                     cls_hidden_size=self.config.cls_hidden_size,
                                     r_size=self.config.r_size,
                                     sos_id=vocab(vocab.SYM_SOQ), eos_id=vocab(vocab.SYM_EOS),
                                     num_layers=self.config.n_lstm_layers,
                                     n_class=3,
                                     bidirectional=True,
                                     dropout_ratio=self.config.dropout_ratio,
                                     embedding=embedding,
                                     )
            self.criterion = nn.CrossEntropyLoss()
            self.kl_div = nn.KLDivLoss(reduction="batchmean")

        self.model.to(self.device)

        if self.config.weight_decay:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate,
                                              weight_decay=self.config.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                                lr_lambda=lambda epoch: 0.95 ** epoch,
                                                last_epoch=-1)
        # pad = vocab(vocab.SYM_PAD)
        #self.criterion = nn.BCELoss()

        # self.criterion = nn.BCELoss()

        checkpoint = torch.load(self.pretrained_check, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        #self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        helper.logger('info', '[INFO] Pretrained model loading complete!')


        # Recording Configuration infornmation in log files.
        helper.logger("info", "========================[INFO] Configurations========================")
        helper.logger("info", "Cuda_Dev: {}, Seed: {}, Task_num: {}".format(self.device, self.config.seed,
                                                                            self.task_num))
        helper.logger("info", "Pretrained_model: {}".format(self.pretrained_check))
        helper.logger("info", "======================================================================")


    def data_to_device(self, data, state):

        targets = data
        targets = targets.to(self.device)

        return targets


    def extract_word_attn_score(self, alphas):
        tmp = None
        for idx, aspect in enumerate(alphas):
            aspect = aspect.squeeze(dim=0)
            if idx == 0:
                tmp = aspect
                continue
            tmp = torch.cat([tmp,aspect], dim=0)

        word_scores = torch.sum(tmp, 0)  # tmp : (r_size * 5, length)  # 5 is 5 class aspects

        return word_scores
    


# --------- Aux model ---------- #       

    def infer_aux(self, script):   # aux inference code in scene level
        
        _, embedded_script, _, _ = preprocess_raw_string(script) # n_scene is meaningless

        set_seed(0)

        script = self.data_to_device(torch.tensor([embedded_script]), state="test")
        outputs, alphas = self.model(script)  # 모델에는 씬 하나씩
        
        infer_result = list()

        for a in outputs:   # outputs -> (outputs_1, outputs_2, outputs_3, outputs_4, outputs_5)
            infer_result.append(a[:,1].item())
        
        return infer_result
 
 
    def infer_aux_(self, script):   # aux inference code in scene level
        
        _, _, truncated_script, truncated_embedded_script = preprocess_raw_string(script)

        set_seed(0)

        scene_scores_in_each_movie = list()
        word_scores_in_each_movie = list()

        for scene_idx, scene in enumerate(truncated_embedded_script):
            scene = self.data_to_device(torch.tensor([scene]), state="test")
            outputs, alphas = self.model(scene)  # 모델에는 씬 하나씩
            
            word_scores = list()
            logits = list()

            logits_sum = 0
            
            for a in outputs:   # outputs -> (outputs_1, outputs_2, outputs_3, outputs_4, outputs_5)
                logits_sum += a[:,1]
                logits.append(a[:,1].item())
            
            a0, a1, a2, a3, a4, word_attn_scores = extract_word_attn_score(alphas, self.config.r_size)     

            sort_word_attn_scores = sorted(word_attn_scores, reverse=True)
            sorted_idx = torch.argsort(word_attn_scores, descending = True)

            top_k_word_score = 0.0
            top_a0 = 0
            top_a1 = 0
            top_a2 = 0
            top_a3 = 0
            top_a4 = 0
            
            if len(sort_word_attn_scores) >= self.config.top_k:
                for i in range(self.config.top_k):
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
            
            scene_scores_in_each_movie.append(
                                            {
                                            'idx': scene_idx,
                                            
                                            'scene_score': 
                                                            {'all': float(self.config.scene_score_rate * logits_sum + self.config.word_score_rate * top_k_word_score),  
                                                
                                                            'nudity': {'logit': logits[0], 
                                                                        'attention': float(top_a0), 
                                                                        'total': logits[0] + float(top_a0)}, 

                                                            'violence': {'logit': logits[1], 
                                                                            'attention': float(top_a1), 
                                                                            'total': logits[1] + float(top_a1)}, 

                                                            'profanity': {'logit': logits[2], 
                                                                            'attention': float(top_a2),
                                                                            'total': logits[2] + float(top_a2)}, 

                                                            'alcohol': {'logit': logits[3], 
                                                                        'attention': float(top_a3),
                                                                        'total': logits[3] + float(top_a3)}, 

                                                            'frightening': {'logit': logits[4], 
                                                                            'attention': float(top_a4),
                                                                            'total': logits[4] + float(top_a4)}
                                            }})
            
            
            for i, w_score in enumerate(word_attn_scores):  # word length
                    word_scores.append(
                                        {'idx': i, 
                                            
                                        'word_score': 
                                                    {'all': float(w_score), 

                                                    'nudity': {'attention': float(a0[i]), 
                                                                'total': float(a0[i])}, 

                                                    'violence': {'attention': float(a1[i]), 
                                                                    'total': float(a1[i])}, 

                                                    'profanity': {'attention': float(a2[i]),
                                                                    'total': float(a2[i])}, 

                                                    'alcohol': {'attention': float(a3[i]), 
                                                                'total': float(a3[i])}, 

                                                    'frightening': {'attention': float(a4[i]), 
                                                                    'total': float(a4[i])}
                                            }})
            
            word_scores_in_each_movie.append(word_scores)


        return truncated_script, scene_scores_in_each_movie, word_scores_in_each_movie

            
            
# ---------- AuxBPrm model ---------- #

    def infer_auxbprm(self, script):

        _, embedded_script, _ , _ = preprocess_raw_string(script) # n_scene은 아무거나 집어넣음

        set_seed(0)
    
        with torch.no_grad():
            self.model.eval()
            
        self.optimizer.zero_grad()

        scripts = self.data_to_device(torch.tensor([embedded_script]), state="test")

        outputs, alphas = self.model(scripts)
        max_val, max_index = torch.max(outputs.detach().cpu(), 1)

        infer_results = max_index.item()
        return infer_results
            
    
            
        
    def checkc(self, config):
        train_loader = self.train_loader

        total_steps = len(train_loader)

        for epoch in range(self.config.n_epochs):
            epc_start_time = time.time()
            helper.logger("info", "\n[INFO] Starting epoch {}".format(epoch + 1))

            # self.model.train()

            for batch_id, batch in enumerate(tqdm(train_loader)):
                print('nohting')
                pass



# Aux, Auxbprm
ABS_PATH = os.path.dirname(os.path.abspath(__file__))

config = helper.get_training_config()

set_seed(config.seed)

solver2_aux = Solver(config = config, 
                     task_num = 10,
                     pretrained_check = '/home/intern/seoyeonk/MPPA_task/ckpts/08-03-2023_22-41_Aux_model_epoch_8_batch_4_lr_0.0003_wd_0.001.ckpt')

solver2_auxbprm = Solver(config = config, 
                         task_num = 11, 
                         pretrained_check = "/home/intern/seoyeonk/MPPA_task/ckpts/09-03-2023_18-26_AuxBPrm_model_epoch_n_token_256_13_batch_4_lr_0.0003_wd_0.001.ckpt")


# GPT
MODEL_MAPPING = {
    'instructgpt': 'text-davinci-003',
    'gpt3': 'text-davinci-002',
    'text-davinci-003': 'text-davinci-003',
    'text-davinci-002': 'text-davinci-002',
    'code-davinci-002': 'code-davinci-002',
    'chatGPT': 'gpt-turbo-'
}

def load_parser_and_args():
    parser = argparse.ArgumentParser()
    ### directory ###
    parser.add_argument('--run_name', type=str, default='randomly-tested')
    parser.add_argument("--vocab_path", type=str, default=ABS_PATH + "/data/preprocess/vocab_MPAA_th17.json")
    
    ### model parameters ###
    parser.add_argument('--model_type', type=str, default='text-davinci-003')
    parser.add_argument('--max_tokens', type=int, default=300)
    parser.add_argument('--temperature', type=float, default=0.75)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--frequency_penalty', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--num_samples', type=int, default=0)
    parser.add_argument("--n_cuda_device", type=int, default=0)
    

    args = parser.parse_args()
    args.model_name_or_path = MODEL_MAPPING[args.model_type]

    return parser, args





# -------- 1. Inference function --------

def aux_(input_script):
    
    aspect_scores = solver2_aux.infer_aux(input_script) # list
    list = ['Nudity', 'Violence', 'Profanity', 'Alcohol', 'Frightening']
    highest_aspect = list[aspect_scores.index(max(aspect_scores))]

    return highest_aspect, aspect_scores # output : list [nudity, violence, profanity, alcohol, frightening]



def aux_scenes_(input_script):
    
    truncated_script, scene_scores_in_each_movie, _ = solver2_aux.infer_aux_(input_script)

    idx = list(range(1,len(truncated_script)+1))
    scene_score = list()
    nudity = list()
    violence = list()
    profanity = list()
    alcohol = list()
    frightening = list()

    for i in range(len(truncated_script)):
        scene_score.append(scene_scores_in_each_movie[i]['scene_score']['all'])
        nudity.append(scene_scores_in_each_movie[i]['scene_score']['nudity']['total'])
        violence.append(scene_scores_in_each_movie[i]['scene_score']['violence']['total'])
        profanity.append(scene_scores_in_each_movie[i]['scene_score']['profanity']['total'])
        alcohol.append(scene_scores_in_each_movie[i]['scene_score']['alcohol']['total'])
        frightening.append(scene_scores_in_each_movie[i]['scene_score']['frightening']['total'])

    df = pd.DataFrame(zip(idx, scene_score, truncated_script, nudity, violence, profanity, alcohol, frightening), 
                      columns = ['Idx','Scene Score', 'Scene', 'Nudity', 'Violence', 'Profanity', 'Alcohol', 'Frightening'])
    
    df_dict = df[['Nudity', 'Violence', 'Profanity', 'Alcohol', 'Frightening']].to_dict('index')
    max_score_n_aspect = [sorted(v.items(), key=lambda x: x[1], reverse = True)[0] for k, v in df_dict.items()]
    max_aspect = [i[0] for i in max_score_n_aspect] #키값 추출

    df['Max Aspect'] = max_aspect

    return df # output : Dataframe (top 5 rows, 9 columns)



def auxbprm_(input_script):
    
    infer_results = solver2_auxbprm.infer_auxbprm(input_script)
    infer_results = convert_rating(infer_results)

    return infer_results # ouput : Rating (HIGH/MED/LOW)



def revision_(input_script, max_aspect, rating):

    parser, args = load_parser_and_args() # model
    set_seed(0)

    model = GPT(args)
    model.set_new_key()

    with open(os.path.join('/home/intern/sblee/sblee/script-revision', 'prompt', f"aspect_{max_aspect.lower()}.txt"), 'r') as f: # input
        prompt = f.read()
    prompt = prompt.replace('The movie {} is {}.', '').format(rating)

    # inference
    model_input = f"{prompt}\n\nOriginal Scene: {input_script}\n\nRevised Scene: "

    pred_string = model.inference(model_input)
    pred_string = pred_string.strip()

    return pred_string




# -------- 2. Gradio function --------

def inference_gradio(inp):

    out1 = auxbprm_(inp)

    out0 = aux_(inp)
    out2 = pd.DataFrame.from_dict(dict(zip(['nudity', 'violence', 'profanity', 'alcohol', 'frightening'], out0)), orient = 'index').reset_index()
    out2.columns = ['Aspect', 'Score'] # DataFrame : row 5 col 2
     
    out = aux_scenes_(inp)
    out3 = out[['Idx','Scene Score']]
    # out3['color'] = 'black'
    # out3 = out3.head(60)

    inp4 = out[['Idx', 'Scene Score', 'Scene', 'Max Aspect']].iloc[0]

    out4_r = [revision_(x['Scene'], x['Max Aspect'], out1) for x in inp4]

    inp4['Revised Scene'] = out4_r
    out4 = pd.DataFrame(inp4)
    
    return out1, out2, out3, out4



def show_scenes_gradio(inp_num1, inp_num2, inp_num3, inp):

    out = aux_scenes_(inp)
    out.to_csv('processing2.csv')

    inp4 = out[out['Idx'] == int(inp_num1)][['Idx','Scene','Max Aspect']]
    inp5 = out[out['Idx'] == int(inp_num2)][['Idx','Scene','Max Aspect']]
    inp6 = out[out['Idx'] == int(inp_num3)][['Idx','Scene','Max Aspect']]

    out4_r = revision_(inp4.iloc[0,1], inp4.iloc[0,2], auxbprm_(inp4.iloc[0,1])) # or R?
    out5_r = revision_(inp5.iloc[0,1], inp5.iloc[0,2], auxbprm_(inp5.iloc[0,1]))
    out6_r = revision_(inp6.iloc[0,1], inp6.iloc[0,2], auxbprm_(inp6.iloc[0,1]))

    out4 = pd.DataFrame.from_dict([{'Idx' : inp4.iloc[0,0], 'Current Scene' : inp4.iloc[0,1], 'Revised Scene' : out4_r }])
    out5 = pd.DataFrame.from_dict([{'Idx' : inp5.iloc[0,0], 'Current Scene' : inp5.iloc[0,1], 'Revised Scene' : out5_r }])
    out6 = pd.DataFrame.from_dict([{'Idx' : inp6.iloc[0,0], 'Current Scene' : inp6.iloc[0,1], 'Revised Scene' : out6_r }])
    
    return out4, out5, out6



def postrevision_gradio(out4, out5, out6, inp):

    inp = inp.replace(out4.iloc[0, 1], out4.iloc[0, 2])
    inp = inp.replace(out5.iloc[0, 1], out5.iloc[0, 2])
    inp_ = inp.replace(out6.iloc[0, 1], out6.iloc[0, 2]) # inp_ : revised script
    
    out1 = auxbprm_(inp_) # rating

    out0 = aux_(inp)
    out2 = pd.DataFrame.from_dict(dict(zip(['nudity', 'violence', 'profanity', 'alcohol', 'frightening'], out0)), orient = 'index').reset_index()
    out2.columns = ['Aspect', 'Score'] # DataFrame : row 5 col 2

    out3 = aux_scenes_(inp_)[['Idx','Scene Score']] # Scene Score
    
    return inp_, out1, out2, out3 # total script, rating, aspect score, dataframe(for barplot)




# -------- 3. Launch --------

# with gr.Blocks() as demo:
#     gr.Markdown(" <center><h1> Script Revision </h1> </center>")
#     with gr.Row():
#         with gr.Column():
#             inp = gr.Textbox(lines = 30, label= "Script")
#             out1 = gr.Text(label="Rating", interactive=0)
#             out2 = gr.BarPlot(label="Aspect", x = 'Aspect', y = 'Score', height = 300, y_lim = (0, 1))     
#             inference_btn = gr.Button("Run")
        
#         with gr.Column():
#             with gr.Tab(label = "Revision"):
#                 inp2 = gr.CheckboxGroup.change(scene_revision_gradio, inputs = inp, outputs = out5)
#                 out4 = gr.DataFrame(label="Scene Revision", interactive=1, wrap = True) # scene_num, original scene, revised scene

#             with gr.Tab(label = "Rating Info"):
#                 out3 = gr.BarPlot(label="Scene Scores", x ='Idx', y = 'Scene Score', interactive=0, width = 450, tooltip = ['Idx', 'Scene Score']) # tooltip 추가
                


#         inference_btn.click(fn=inference_gradio, inputs=inp, outputs=[out1, out2, out3, out4, out5])

#         revise_scene.click(fn=show_scenes_gradio, inputs=[inp_num1, inp_num2, inp_num3, inp], outputs=[out4, out5, out6]) 

#         revision_submit_btn.click(fn=postrevision_gradio, inputs=[out4, out5, out6, inp], outputs=[inp, out1, out2, out3])

# demo.launch(share=True)

# # with open('/home/intern/sblee/sblee/script-revision/data/Deliver_dataset/Script/tt1431045.txt', 'r') as f :
# #     inp = f.read()

# # inp_num1, inp_num2, inp_num3 = 3, 10, 17
# # print(show_scenes_gradio(inp_num1, inp_num2, inp_num3, inp))


