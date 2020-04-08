import os
import numpy as np
import tqdm

import torch

from opts import parser
from model import ActionModifiers, Evaluator
from dataset import AdverbDataset

from sklearn.metrics import average_precision_score

def test(model, data_loader, test_set, evaluator):
    model.eval()

    y_true_adverb = np.zeros((len(data_loader), len(test_set.adverbs)))
    y_score = np.zeros((len(data_loader), len(test_set.adverbs)))
    y_score_antonym = np.zeros((len(data_loader), len(test_set.adverbs)))

    y_true_action = np.zeros((len(data_loader), len(test_set.actions)))
    y_score_action = np.zeros((len(data_loader), len(test_set.actions)))

    for idx, data in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
        if args.gpu:
            data = [d.cuda() for d in data]
        predictions = model(data)[1]
        adverb_gt, action_gt = data[1], data[2]
        scores, action_gt_scores, antonym_action_gt_scores = evaluator.get_scores(predictions, action_gt, adverb_gt)

        y_true_adverb[idx] = np.array([1 if test_set.adverb2idx[adv] == adverb_gt else 0
                                       for adv in test_set.adverbs])
        y_true_action[idx] = np.array([1 if test_set.action2idx[act] == action_gt else 0
                                       for act in test_set.actions])

        y_score[idx] = np.array([action_gt_scores[0][test_set.pairs.index((adv, test_set.idx2action[action_gt.item()]))]
                            for adv in test_set.adverbs])
        y_score_antonym[idx] = np.array([
            antonym_action_gt_scores[0][test_set.pairs.index((adv, test_set.idx2action[action_gt.item()]))]
            for adv in test_set.adverbs])
        y_score_action[idx] = np.array([max([scores[0][test_set.pairs.index((adv, act))]
                                        for adv in test_set.adverbs])
                                   for act in test_set.actions])
    v2a_ant = (np.argmax(y_true_adverb, axis=1) == np.argmax(y_score_antonym, axis=1)).mean()
    v2a_all = average_precision_score(y_true_adverb, y_score, average='samples')
    a2v_ant = average_precision_score(y_true_adverb, y_score_antonym)
    a2v_all = average_precision_score(y_true_adverb, y_score)
    v2action = average_precision_score(y_true_action, y_score_action, average='samples')
    return v2a_ant, v2a_all, a2v_ant, a2v_all, v2action

def main(args):
    test_set = AdverbDataset(args.data_dir, args.feature_dir, agg=args.temporal_agg, modality=args.modality,
                             window_size=args.t, adverb_filter=args.adverb_filter, phase='test')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers)

    model = ActionModifiers(test_set, args)
    if args.gpu:
        model = model.cuda() #TODO implement gpu option properly everywhere

    evaluator = Evaluator(test_set, model)

    checkpoint = torch.load(args.load)
    model.load_state_dict(checkpoint['net'])
    print('loaded model from', os.path.basename(args.load))
    v2a_ant, v2a_all, a2v_ant, a2v_all, v2action = test(model, test_loader, test_set, evaluator)
    print('Video-to-Adverb Antonym: %.3f'%v2a_ant)
    print('Video-to-Adverb All: %.3f'%v2a_all)
    print('Adverb-to-Video Antonym: %.3f'%a2v_ant)
    print('Adverb-to-Video All: %.3f'%a2v_all)
    print('Video-to-Action: %.3f'%v2action)

if __name__ == '__main__':
    args = parser.parse_args()
    args.batch_size = 1
    if args.modality == 'both':
        args.modality = ['rgb', 'flow']
    else:
        args.modality = [args.modality]
    main(args)

