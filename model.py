import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def load_word_embeddings(emb_file, vocab):
    vocab = [word.lower() for word in vocab]

    embeddings = {}
    for line in open(emb_file, 'r'):
        line = line.strip().split(' ')
        word_vec = torch.FloatTensor(list(map(float, line[1:])))
        embeddings[line[0]] = word_vec

    embeddings = [embeddings[word] for word in vocab]
    embeddings = torch.stack(embeddings)
    print('loaded word embeddings')
    return embeddings

class MLP(nn.Module):
    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True):
        super(MLP, self).__init__()
        network = []
        for i in range(num_layers-1):
            network.append(nn.Linear(inp_dim, inp_dim, bias=bias))
            network.append(nn.ReLU(True))
        network.append(nn.Linear(inp_dim, out_dim, bias=bias))
        if relu:
            network.append(nn.ReLU(True))

        self.network = nn.Sequential(*mod)

    def forward(self, x):
        output = self.mod(x)
        return output

class SDPAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, emb_dim, heads=1, dropout=0.1):
        super(SDPAttention, self).__init__()
        self.d_k = int(d_k/heads)
        self.d_v = int(d_v/heads)
        d_model = int(d_model/heads)
        self.q_linear = nn.Linear(int(emb_dim/heads), self.d_k)
        self.k_linear = nn.Linear(d_model, self.d_k)
        self.v_linear = nn.Linear(d_model, self.d_v)
        self.h = heads

        self.dropout = nn.Dropout(dropout)

        self.out = nn.Linear(heads*self.d_v, emb_dim)

    def attention(self, q, k, v, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        output = torch.matmul(scores, v)
        output = output.reshape(output.shape[0], output.shape[1], output.shape[-1])
        return output, scores

    def forward(self, features, queries):
        bs = queries.shape[0]
        q = self.q_linear(queries.view(bs, -1, self.h, int(queries.shape[-1]/self.h)))
        k = self.k_linear(features.view(bs, -1, self.h, int(features.shape[-1]/self.h)))
        v = self.v_linear(features.view(bs, -1, self.h, int(features.shape[-1]/self.h)))
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        output, scores = self.attention(q, k, v, self.dropout)
        concat = output.transpose(1,2).contiguous().view(bs, self.d_v*self.h)
        output = self.out(concat)
        return output, scores

class ActionModifiers(nn.Module):
    def __init__(self, dset, args):
        super(ActionModifiers, self).__init__()
        if args.temporal_agg == 'sdp':
            self.video_embedder = SDPAttention(dset.feature_dim, args.emb_dim, args.emb_dim, args.emb_dim,
                                               heads=4)
        else:
            self.video_embedder = MLP(dset. feat_dim, args.emb_dim)

        self.action_modifiers = nn.ParameterList([nn.Parameter(torch.eye(args.emb_dim))
                                            for _ in range(len(dset.adverbs))])
        self.action_embedder = nn.Embedding(len(dset.actions), args.emb_dim)

        if args.glove_init:
            pretrained_weight = load_word_embeddings('data/glove.6B.300d.txt', dset.actions)
            self.action_embedder.weight.data.copy_(pretrained_weight)

        for param in self.action_embedder.parameters():
            param.requires_grad = False

        self.margin = 0.5
        self.transformer = False
        if args.temporal_agg == 'sdp':
            self.transformer = True

        self.compare_metric = lambda vid_feats, act_adv_embed: -F.pairwise_distance(vid_feats, act_adv_embed)
        self.dset = dset

        ## precompute validation pairs
        adverbs, actions = zip(*self.dset.pairs)
        self.val_adverbs = torch.LongTensor([dset.adverb2idx[adv.strip()] for adv in adverbs]).cuda()
        self.val_actions = torch.LongTensor([dset.action2idx[act.strip()] for act in actions]).cuda()

    def apply_modifiers(self, modifiers, embedding):
        output = torch.bmm(modifiers, embedding.unsqueeze(2)).squeeze(2)
        output = F.relu(output)
        return output

    def train_forward(self, x):
        features, adverbs, actions = x[0], x[1], x[2]
        neg_adverbs, neg_actions = x[3], x[4]
        action_embedding = self.action_embedder(actions)
        neg_action_embedding = self.action_embedder(neg_actions)
        if self.transformer:
            video_embedding, attention_weights = self.video_embedder(features, action_embedding)
        else:
            video_embedding = self.video_embedder(features)
            attention_weights = None

        pos_modifiers = torch.stack([self.action_modifiers[adv.item()] for adv in adverbs])
        positive = self.apply_modifiers(pos_modifiers, action_embedding)
        negative_act = torch.apply_modifiers(pos_modifiers, neg_action_embedding)

        neg_modifiers = torch.stack([self.action_modifiers[adv.item()] for adv in adverbs])
        negative_adv = torch.apply_modifiers(neg_modifiers, action_embedding)

        loss_triplet_act = F.triplet_margin_loss(video_embedding, positive, negative_act, margin=self.margin)
        loss_triplet_adv = F.triplet_margin_loss(video_embedding, positive, negative_adv, margin=self.margin)
        loss = [loss_triplet_act, loss_triplet_adv]

        return loss, None, attention_weights, video_embedding

    def val_forward(self, x):
        features = x[0]
        actions = x[2]
        batch_size = features.shape[0]

        if self.transformer:
            action_gt_embedding = self.action_embedder(actions)
            video_embedding, attention_weights = self.video_embedder(features, action_gt_embedding)
        else:
            video_embedding = self.video_embedder(features)
            attention_weights = None
        action_embedding = self.action_embedder(self.val_actions)
        modifiers = torch.stack([self.action_modifiers[adv.item()] for adv in self.val_adverbs])
        action_adverb_embeddings = self.apply_modifiers(modifiers, action_embedding)

        scores = {}
        for i, (adverb, action) in enumerate(self.dset.pairs):
            pair_embedding = action_adverb_embeddings[i, None].expand(batch_size, action_adverb_embeddings.size(1))
            score = self.compare_metric(video_embedding, pair_embedding)
            scores[(adverb, action)] = score
        return None, scores, attention_weights, video_embedding

    def forward(self, x):
        if self.training:
            loss, pred, att, vid_feats = self.train_forward(x)
        else:
            with torch.no_grad():
                loss, pred, att, vid_feats = self.val_forward(x)
        return loss, pred, att, vid_feats

class Evaluator:
    def __init__(self, dset, model):
        self.dset = dset
        pairs = [(dset.adverb2idx[adv.strip()], dset.action2idx[act]) for adv, act in dset.pairs]
        self.pairs = torch.LongTensor(pairs)

        ## mask over pairs for ground-truth action given in testing
        action_gt_mask = []
        for _act in dset.actions:
            mask = [1 if _act==act else 0 for adv, act in dset.pairs]
            action_gt_mask.append(torch.ByteTensor(mask))
        self.action_gt_mask = torch.stack(action_gt_mask, 0)

        antonym_mask = []
        for _adv in dset.adverbs:
            mask = [1 if (_adv==adv or _adv==dset.antonyms[adv]) else 0 for adv, act in dset.pairs]
            antonym_mask.append(torch.ByteTensor(mask))
        self.antonym_mask = torch.stack(antonym_mask, 0)

    def get_gt_action_scores(self, scores, action_gt):
        mask = self.action_gt_mask[action_gt]
        action_gt_scores = scores.clone()
        action_gt_scores[1-mask] = -1e10
        return action_gt_scores

    def get_antonym_scores(self, scores, adverb_gt):
        mask = self.antonym_mask[adverb_gt]
        antonym_scores = scores.clone()
        antonym_scores[1-mask] = -1e10
        return antonym_scores

    def get_gt_action_antonym_scores(self, scores, action_gt, adverb_gt):
        mask = self.antonym_mask[adverb_gt] & self.action_gt_mask[action_gt]
        action_gt_antonym_scores = scores.clone()
        action_gt_antonym_scores[1-mask] = -1e10
        return action_gt_antonym_scores

    def get_scores(self, scores, action_gt, adverb_gt):
        scores = {k:v.cpu() for k, v in scores.items()}
        action_gt = action_gt.cpu()

        scores = torch.stack([scores[(adv, act)] for adv, act in self.dset.pairs], 1)
        action_gt_scores = self.get_gt_action_scores(scores, action_gt)
        antonym_action_gt_scores = self.get_gt_action_antonym_scores(scores, action_gt, adverb_gt)
        return scores, action_gt_scores, antonym_action_gt_scores
