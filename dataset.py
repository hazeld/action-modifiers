import os
import math
import numpy as np
import pandas as pd
import itertools
import torch.utils.data as data

class AdverbDataset(data.Dataset):

    def __init__(self, data_dir, feature_dir, agg='sdp', modality=['rgb', 'flow'], window_size=20,
                 adverb_filter=None, phase='test', action_key='clustered_action', adverb_key='clustered_adverb',
                 all_info=False):
        self.data_dir = data_dir
        self.feature_dir = feature_dir
        self.agg = agg
        self.modality = modality
        self.window_size = window_size
        self.phase = phase
        self.action_key = action_key
        self.adverb_key = adverb_key
        self.all_info = all_info

        self.adverbs, self.actions, self.train_list, self.test_list = self._parse_list(adverb_filter)

        self.adverbs, self.antonyms = self._add_antonyms(self.adverbs) ## antonyms necessary for training

        self.pairs = list(itertools.product(self.adverbs, self.actions))

        assert pd.merge(self.train_list, self.test_list, how='inner', on=['id', 'action', 'adverb', 'vid_id']).shape[0] == 0, 'train and test are not mutually exclusive'

        self.data = self.train_list if self.phase == 'train' else self.test_list
        self.adverb2idx = {adverb: idx for idx, adverb in enumerate(self.adverbs)}
        self.idx2adverb = {v:k for k, v in self.adverb2idx.items()}
        self.action2idx = {action: idx for idx, action in enumerate(self.actions)}
        self.idx2action = {v:k for k, v in self.action2idx.items()}
        self._load_all_features()
        self.feature_dim = self.feature_list[0][0].shape[-1]
        print('%d features loaded'%(len(self.feature_list)))

    def _get_feature_filename(self, x, modality):
        return '_'.join((str(x['id']), modality + '.npz'))

    def _load_all_features(self):
        feature_list = [([np.load(os.path.join(self.feature_dir, self._get_feature_filename(x, modality)))['arr_0']
                           for modality in self.modality],
                         x[self.adverb_key], x[self.action_key], x['vid_id'],
                         x['weak_timestamp']) for i, x in self.data.iterrows()]
        ## take only selected window size
        feature_list = [([feature[math.ceil(feature.shape[0]/2-self.window_size/2):
                                   math.ceil(feature.shape[0]/2+self.window_size/2)]
                           for feature in features], adv, act, vid_id, wt)
                         for (features, adv, act, vid_id, wt) in feature_list]
        if self.agg == 'single':
            self.feature_list = [(np.concatenate([feature[math.ceil(feature.shape[0]/2)] for feature in features]),
                                  adv, act, v_id, wt)
                                 for (features, adv, act, v_id, wt) in feature_list]
        elif self.agg == 'average':
            self.feature_list = [(np.concatenate([feature.mean(axis=0) for feature in features]), adv, act, v_id, wt)
                            for (features, adv, act, v_id, wt) in feature_list]
        elif self.agg == 'sdp':
            self.feature_list = [(np.concatenate([feature for feature in features], axis=1),
                                  adv, act, v_id, wt) for (features, adv, act, v_id, wt) in feature_list]
        else:
            print("Error: temporal aggregation method not supported")
            exit(0)


    def _add_antonyms(self, adverb_list):
        antonyms_df = pd.read_csv(os.path.join(self.data_dir, 'antonyms.csv'))
        adverbs = []
        antonyms = {}
        for i, row in antonyms_df.iterrows():
            if row['adverb'] in adverb_list:
                if row['adverb'] not in adverbs:
                    adverbs.append(row['adverb'])
                if row['antonym'] not in adverbs:
                    adverbs.append(row['antonym'])
            antonyms[row['adverb']] = row['antonym']
        return adverbs, antonyms

    def _parse_list(self, adverb_filter):
        def parse_pairs(filename):
            pairs_df = pd.read_csv(filename)
            if adverb_filter is not None:
                pairs_df = pairs_df[pairs_df[self.adverb_key].isin(adverb_filter)]
            mods = pairs_df[self.adverb_key].unique().tolist()
            acts = pairs_df[self.action_key].unique().tolist()
            return mods, acts, pairs_df

        train_mods, train_acts, train_list = parse_pairs(os.path.join(self.data_dir, 'train.csv'))
        test_mods, test_acts, test_list = parse_pairs(os.path.join(self.data_dir, 'test.csv'))
        all_mods = sorted(list(set(train_mods+test_mods)))
        all_acts = sorted(list(set(train_acts+test_acts)))

        return all_mods, all_acts, train_list, test_list

    def sample_negative_action(self, action):
        new_action = self.actions[np.random.choice(len(self.actions))]
        if new_action==action:
            return self.sample_negative_action(action)
        return (self.action2idx[new_action])

    def __getitem__(self, index):
        feature, adverb, action = self.feature_list[index][0:3]
        data = [feature, self.adverb2idx[adverb], self.action2idx[action]]
        if self.phase == 'train':
            neg_adverb = self.adverb2idx[self.antonyms[adverb]]
            neg_action = self.sample_negative_action(action)
            assert data[1] != neg_adverb and data[2] != neg_action
            data += [neg_adverb, neg_action]
        if self.all_info:
            id, vid_id, wt = self.feature_list[index][3:]
            data += [id, vid_id, wt]
        return data

    def __len__(self):
        return len(self.feature_list)
