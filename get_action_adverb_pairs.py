import os
import glob
import argparse
import pandas as pd
from tqdm import tqdm
import string
import webvtt
import re

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

parser = argparse.ArgumentParser(description='Find and filter verb-adverb pairs from punctuated text')
parser.add_argument('subtitle_dir', type=str, help='directory containing subtitles')
parser.add_argument('punc_dir', type=str, help='directory containing punctuated subtitles')
parser.add_argument('output_file', type=str, help='file to output action-adverb pair annotations to')
parser.add_argument('--adverb-file', type=str, help='list of whitelisted adverbs')
parser.add_argument('--action-file', type=str, help='list of whitelisted actions')
parser.add_argument('--task-list', type=str, default=None, help='List of tasks to obtain action-adverb pairs from')
parser.add_argument('--chunk-size', type=int, default=50, help='number of texts to parse simultaneously')
args = parser.parse_args()

def read_word_list(filename):
    if filename:
        with open(filename) as f:
            lines = f.readlines()
        words = [line.strip() for line in lines]
    else:
        words = None
    return words

def read_task_list(filename):
    if args.task_list:
        tasks = pd.read_csv(args.task_list)
        task_ids = list(tasks['id'])
        return task_ids
    else:
        return None

def custom_pipeline(nlp):
    return (nlp.tagger, nlp.parser)

def get_seconds(time_str):
    h, m, s = time_str.split(':')
    s, ms = s.split('.')
    return int(h) * 3600 + int(m) * 60 + int(s) + float('0.' + ms)

def clean_subtitles(subs):
    to_remove = []
    for i, caption in enumerate(subs):
        ## remove empty subtitles
        if not caption.text.strip():
            to_remove.append(i)
            continue
        start = get_seconds(caption.start)
        end = get_seconds(caption.end)
        duration = end - start
        ## remove short subtitles
        if duration < 0.1:
            to_remove.append(i)
    for i in to_remove[::-1]:
        del subs.captions[i]
    ## remove repeated text
    previous_line = ''
    for i, caption in enumerate(subs):
        new_caption_text = ''
        new_lines = caption.text.strip().split('\n')
        for line in new_lines:
            if line != previous_line:
                new_caption_text += line
                new_caption_text += ' '
                previous_line = line
        subs[i].text = new_caption_text
    return subs

def read_and_clean_subtitles(task_dir):
    vid_ids = []
    subtitles = []
    for filename in glob.glob(os.path.join(task_dir, '*')):
        vid_ids.append(os.path.basename(filename).split('.')[0])
        subs = webvtt.read(filename)
        clean_subs = clean_subtitles(subs)
        subtitles.append(clean_subs)
    return vid_ids, subtitles

def read_punc_text(punc_dir, vid_ids):
    texts = []
    for vid_id in vid_ids:
        filename = os.path.join(punc_dir, vid_id + '.txt')
        if not os.path.isfile(filename):
            texts.append(None)
        else:
            f = open(filename, 'r')
            texts.append(f.read())
            f.close()
    return texts

def remove_punc(word, punctuation):
    return word.translate(str.maketrans('', '', punctuation))

def find_matches(doc, words, timestamps):
    word_counter = 0
    word_timestamps = {}
    enum_words = enumerate(words)
    punctuation = string.punctuation.replace('*', '')
    for i, word in enum_words:
        ## skip only punctuation
        while word_counter < len(doc) and doc[word_counter].text in punctuation:
            word_counter+=1
        if word in punctuation:
            continue
        if doc[word_counter].text.lower() == word.lower() \
           or remove_punc(doc[word_counter].text.lower(), punctuation) == remove_punc(word.lower(), punctuation):
            ## check for normal match
            word_timestamps[doc[word_counter].i] = timestamps[i]
            word_counter+=1
            continue
        elif i+1 < len(words) and doc[word_counter].text.lower() == word.lower() + words[i+1].lower():
            ## check for words concatenated in punctuator
            word_timestamps[doc[word_counter].i] = timestamps[i]
            word_counter+=1
            next(enum_words)
            continue
        else:
            ##check for words which have been split up when punctuating
            full_word = doc[word_counter].text
            j=1
            while remove_punc(word.lower(), punctuation).startswith(remove_punc(full_word.lower(), punctuation)) \
                  and remove_punc(full_word.lower(), punctuation) != remove_punc(word.lower(), punctuation):
                full_word += doc[word_counter+j].text
                j+=1
            if full_word.lower() == word.lower() \
               or remove_punc(full_word.lower(), punctuation) == remove_punc(word.lower(), punctuation):
                for k in range(word_counter, word_counter+j):
                    word_timestamps[doc[k].i] = timestamps[i]
                word_counter = word_counter+j
                continue
            else:
                print(full_word)
                print('missaligned: ' + doc[word_counter].text + ' ' +  word)
                print(doc[word_counter-10:word_counter+4])
                print(words[i-10:i+2])
                return word_timestamps
    return word_timestamps

def get_sub_word_timestamps(subs):
    words = []
    timestamps = []
    for caption in subs:
        if not caption.text.strip():
            continue
        caption_words = caption.text.split()
        words.extend(caption_words)
        start_sec = get_seconds(caption.start)
        end_sec = get_seconds(caption.end)
        num_words = len(caption_words)
        interval = (end_sec - start_sec)/num_words
        timestamps.extend([round((start_sec + interval*i+interval/2), 2) for i in range(0, num_words)])
    return words, timestamps

def timestamp_words(subs, doc):
    words, timestamps = get_sub_word_timestamps(subs)
    word_timestamps = find_matches(doc, words, timestamps)
    return word_timestamps

def remove_vids_missing_punc(vid_ids, subtitles, texts):
    vid_ids = [vid_id for i, vid_id in enumerate(vid_ids) if texts[i] is not None]
    subtitles = [subs for i, subs in enumerate(subtitles) if texts[i] is not None]
    texts = [text for text in texts if text is not None]
    return vid_ids, subtitles, texts

def check_negative(token):
    for child in token.children:
        if child.dep_ == 'neg':
            return True
    return False

def check_particle(token):
    for child in token.children:
        if child.dep_ == 'prt':
            return child.lemma_.lower()
    return ''


def get_annotations(doc, word_timestamps, whitelist_actions, whitelist_adverbs):
    action_adverb_pairs = []
    for token in doc:
        if token.pos_ == 'VERB':
            if token.tag_ not in ['VB', 'VBG', 'VBN', 'VBP'] or token.text.lower() in STOP_WORDS:
                continue
            verb = token.lemma_.lower()
            neg = check_negative(token)
            for child in token.children:
                if child.dep_ == 'advmod':
                    if child.tag_ not in ['RB'] or child.text.lower() in STOP_WORDS:
                        continue
                    adverb = child.lemma_.lower()
                    if whitelist_actions and verb not in whitelist_actions:
                        continue
                    if whitelist_adverbs and adverb not in whitelist_adverbs:
                        continue
                    context_before = ' '.join([t.text for t in doc[token.i-10:token.i]])
                    context_after = ' '.join([t.text for t in doc[token.i:token.i+10]])
                    context = re.split("[,.]", context_before)[-1] + " " + re.split("[,.]", context_after)[0]
                    if token.i not in word_timestamps.keys() or child.i not in word_timestamps.keys():
                        continue
                    weak_timestamp = round((word_timestamps[token.i] + word_timestamps[child.i])/2,2)
                    prt = check_particle(token)
                    ann = {
                        'action': token.lemma_.lower(),
                        'weak_timestamp': weak_timestamp,
                        'verb_token': token.tag_,
                        'context': context,
                        'particle': prt,
                    }
                    if neg:
                        ann['adverb'] = 'not ' + adverb
                    else:
                        ann['adverb'] = adverb
                    action_adverb_pairs.append(ann)
    return action_adverb_pairs

def get_subtitle_anns(sub_dir, punc_dir, nlp, chunk_size, whitelist_actions, whitelist_adverbs):
    vid_ids, subtitles = read_and_clean_subtitles(sub_dir)
    texts = read_punc_text(punc_dir, vid_ids)
    vid_ids, subtitles, texts = remove_vids_missing_punc(vid_ids, subtitles, texts)
    action_adverb_pairs = []
    for chunk in range(0, len(texts), chunk_size):
        docs = nlp.pipe(texts[chunk:min(chunk+chunk_size, len(texts))])
        for i, doc in enumerate(docs):
            word_timestamps = timestamp_words(subtitles[chunk+i], doc)
            action_adverb_ann = get_annotations(doc, word_timestamps, whitelist_actions,
                                                                    whitelist_adverbs)
            for j in range(len(action_adverb_ann)):
                action_adverb_ann[j]['vid_id'] = vid_ids[chunk+i]
            action_adverb_pairs.extend(action_adverb_ann)
    return action_adverb_pairs

def main(args):
    whitelist_adverbs = read_word_list(args.adverb_file)
    whitelist_actions = read_word_list(args.action_file)
    task_list = read_task_list(args.task_list)

    nlp = spacy.load('en_core_web_sm', create_pipeline=custom_pipeline)
    action_adverb_ann = []
    for task_dir in tqdm(glob.glob(os.path.join(args.subtitle_dir, '*'))):
        task_num = int(task_dir.split(os.sep)[-1])
        if task_list is not None and task_num not in task_list:
            continue
        task_punc_dir = os.path.join(args.punc_dir, str(task_num))
        task_ann = get_subtitle_anns(task_dir, task_punc_dir, nlp, args.chunk_size,
                                                          whitelist_actions, whitelist_adverbs)
        for i in range(len(task_ann)):
            task_ann[i]['task_num'] = task_num
        action_adverb_ann.extend(task_ann)
    annotation_df = pd.DataFrame(action_adverb_ann)
    annotation_df.to_csv(args.output_file)

if __name__ == "__main__":
    main(args)
