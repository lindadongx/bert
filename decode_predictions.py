import os
import pickle
import numpy as np
import copy
from tokenization import FullTokenizer

SWAP_TARGETS = False

base_results_dir = os.path.join(os.getcwd(), 'data/gendereval')
base_data_dir = os.path.join(os.getcwd(), 'data/sets')

#model_dir = 'bert_model'
#model_dir = 'data/clinton_trump/with_links/trump_model'
#model_dir = 'data/large_trump/with_links/trump_model'
#model_dir = 'data/more_hillary/with_links/hillary_model'
model_dir = 'data/linda/with_links/linda_model'

#set_dir = 'pleasant_unpleasant'
set_dir = 'career_family'
#set_dir = 'unpleasant_pleasant'
#set_dir = 'follower_leader'

targets = ['male', 'female']

attributes = ['career','family']
#attributes = ['unpleasant']
#attributes = ['pleasant']
#attributes = ['leader', 'follower']

templates = ['templates']

vocab_file = 'uncased_L-12_H-768_A-12/vocab.txt'
tok = FullTokenizer(vocab_file)

if SWAP_TARGETS:
    tmp = copy.deepcopy(targets)
    targets = attributes
    attributes = tmp


def open_results_file(path):
    result = pickle.load(open(path, 'rb'))
    for res in result:
        res['file'] = path
    return result


def load_results(base_results_dir, base_data_dir, model_dir, set_dir,
        templates, targets, attributes, swap_targets = False):
    template = templates[0]

    results = {'attributes': {}, 'targets': {}}

    results_dir = os.path.join(base_results_dir, model_dir, set_dir)
    for attribute in attributes + ['mask']:
        if swap_targets:
            file_name = '_'.join([template, attribute, 'mask']) + '.pkl'
        else:
            file_name = '_'.join([template, 'mask', attribute]) + '.pkl'
        results_file = os.path.join(base_results_dir, model_dir, set_dir, file_name)
        result = open_results_file(results_file)

        results['attributes'][attribute] = result

    for target in targets:
        results['targets'][target] = {}
        results['targets'][target]['words'] = target_words = read_file(target, set_dir, base_data_dir)
        results['targets'][target]['ids'] = tok.convert_tokens_to_ids(target_words)


    return results


def remove_line_ending(words):
    return [w.replace('\n', '') for w in words]


def read_file(file_name, set_dir, base_dir, ending = '.txt'):
    return remove_line_ending(open(os.path.join(base_dir, set_dir, file_name + ending), 'r').readlines())


def get_top_k(masked_lm_log_probs, mask_num = 0, k = 10):
    idxs = np.argsort(masked_lm_log_probs[mask_num])
    probs = masked_lm_log_probs[mask_num][idxs]
    words = tok.convert_ids_to_tokens(list(idxs))[-k:][::-1]
    return words, probs


def top_k_example(results, attribute, ex_num, mask_num = 0, target = None, k = 10, normalize = True):
    example = results['attributes'][attribute][ex_num]

    if normalize:
        tile_factor = len(results['attributes'][attribute]) // len(results['attributes']['mask'])
        mask_example = results['attributes']['mask'][ex_num // tile_factor]

    target_ids = []
    if target == 'all':
        for target in results['targets']:
            target_ids.extend(results['targets'][target]['ids'])
    elif target is not None:
        target_ids = results['targets'][target]['ids']
    else:
        target_ids = np.arange(example['masked_lm_log_probs'].shape[-1])

    if normalize:
        masked_lm_log_probs = example['masked_lm_log_probs'] - mask_example['masked_lm_log_probs']
    else:
        masked_lm_log_probs = example['masked_lm_log_probs']

    masked_lm_log_probs = masked_lm_log_probs[:, target_ids]

    idxs = np.argsort(masked_lm_log_probs[mask_num])[-k:][::-1]
    top_k_words = tok.convert_ids_to_tokens(list(np.array(target_ids)[idxs]))
    top_k_probs = masked_lm_log_probs[mask_num, idxs]

    sentence = tok.convert_ids_to_tokens(example['input_ids'])
    gt = tok.convert_ids_to_tokens(example['masked_lm_ids'])
    sentence = decode_sentence(sentence, gt)

    print('----------------------------------')
    print('SENTENCE: %s' % sentence)
    print('TOP K PREDICTIONS:')
    for i, (word, prob) in enumerate(zip(top_k_words, top_k_probs)):
        print('%d. %s %f' % (i + 1, word, prob))

    return top_k_words, sentence



def print_example(results, attribute, ex_num):
    example = results['attributes'][attribute][ex_num]

    sentence = tok.convert_ids_to_tokens(example['input_ids'])
    gt = tok.convert_ids_to_tokens(example['masked_lm_ids'])
    pred = tok.convert_ids_to_tokens(example['masked_lm_predictions'])

    print('----------------------------------')
    print('TRUE SENTENCE: ', decode_sentence(sentence, gt))
    print('PRED SENTENCE: ', decode_sentence(sentence, pred))


def decode_sentence(sentence, filler):
    blank = 0
    decoded_sentence = []
    for word in sentence:
        word = copy.deepcopy(word)
        if word == '[MASK]':
            word = filler[blank]
            blank += 1
        if word == '[PAD]' or word == '[CLS]':
            continue
        elif word == '[SEP]':
            decoded_sentence.append('.')
        elif '##' in word:
            decoded_sentence[-1] = decoded_sentence[-1] + word.replace('##', '')
        else:
            decoded_sentence.append(word)

    return ' '.join(decoded_sentence)


def score_bias(all_results, templates, targets, attributes, swap_targets = False):
    template = templates[0]

    score = {}
    for attribute in attributes + ['mask']:
        results = all_results['attributes'][attribute]
        for target in targets:
            target_ids = all_results['targets'][target]['ids']
            probs = []
            for res in results:
                if attribute == 'mask' and swap_targets:
                    # Use second MASK blank
                    probs.append(res['masked_lm_log_probs'][1, target_ids])
                else:
                    probs.append(res['masked_lm_log_probs'][0, target_ids])
            score[target + '_' + attribute] = {'probs': np.array(probs), 'avg': np.mean(probs)}

    for target in targets:
        tile_factor = score[target + '_' + attributes[0]]['probs'].shape[0] // \
                score[target + '_' + 'mask']['probs'].shape[0]
        score[target + '_' + 'mask']['probs'] = np.reshape(np.tile(
            score[target + '_' + 'mask']['probs'][:, np.newaxis],
            [1, tile_factor, 1]), [-1, score[target + '_' + 'mask']['probs'].shape[-1]])

    for target in targets:
        for attribute in attributes:
            score[target + '_' + attribute]['norm_probs'] = \
                    score[target + '_' + attribute]['probs'] - score[target + '_' + 'mask']['probs']
            score[target + '_' + attribute]['score'] = np.mean(score[target + '_' + attribute]['norm_probs'])

    for attribute in attributes:
        score[targets[0] + '_' + attribute]['bias'] = \
                score[targets[0] + '_' + attribute]['score'] - \
                score[targets[1] + '_' + attribute]['score']

        score[targets[1] + '_' + attribute]['bias'] = \
                score[targets[1] + '_' + attribute]['score'] - \
                score[targets[0] + '_' + attribute]['score']

        print('----------------------------------')
        print('%s - %s bias for %s: %f' % (targets[0], targets[1], attribute,
            score[targets[0] + '_' + attribute]['bias']))
        print('%s - %s bias for %s: %f' % (targets[1], targets[0], attribute,
            score[targets[1] + '_' + attribute]['bias']))

    return score

if __name__ == '__main__':
    results = load_results(base_results_dir, base_data_dir, model_dir, set_dir,
            templates, targets, attributes, swap_targets = SWAP_TARGETS)
    print(results['attributes'][attributes[0]][0]['file'][44:])
    #print_example(results, attributes[0], 0)
    results['score'] = score_bias(results, templates, targets, attributes, swap_targets = SWAP_TARGETS)

    for i in range(1000):
        try:
            print('----------------------------------')
            print("Example %d:" % i) 
            top_k_example(results, attributes[0], i, mask_num = 0, target = 'all', k = 10, normalize = True)
        except:
            break
