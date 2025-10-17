import torch
import itertools


def collate_fn_ViIHSD(data):
    def merge(sentences, N=None):
        lengths = [len(seq) for seq in sentences]

        if N == None: 
            N = 128

        padded_seqs = torch.zeros(len(sentences),N).long()
        attention_mask = torch.zeros(len(sentences),N).long()

        for i, seq in enumerate(sentences):
            seq = torch.LongTensor(seq) 
            end = min(lengths[i], N)
            padded_seqs[i, :end] = seq[:end]
            attention_mask[i,:end] = torch.ones(end).long()

        return padded_seqs, attention_mask,lengths
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    text_batch,text_attn_mask, post_lengths = merge(item_info['text'])
    d={}
    d["label"] = item_info["label"]
    d["text"] = text_batch.cuda()
    d["text_attn_mask"] = text_attn_mask.cuda()
    return d 