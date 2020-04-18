import os

BASE_DIR = os.path.join(os.getcwd(), 'sets')

#set_dir = 'career_family'
#targets_files = ['male', 'female']
#attributes_files = ['career', 'family']
#templates_files = ['templates']

#set_dir = 'unpleasant_pleasant'
#targets_files = ['male', 'female']
#attributes_files = ['pleasant', 'unpleasant']
#templates_files = ['templates']

set_dir = 'follower_leader'
targets_files = ['male', 'female']
attributes_files = ['follower', 'leader']
templates_files = ['templates']

def remove_line_ending(words):
    return [w.replace('\n', '') for w in words]


def read_file(file_name, set_dir, base_dir = BASE_DIR):
    return remove_line_ending(open(os.path.join(base_dir, set_dir, file_name + '.txt'), 'r').readlines())


def fill_template(template, target, attribute):
    template = template.split(' ')
    for i, word in enumerate(template):
        if word == 'A':
            template[i] = attribute
        elif word == 'T':
            template[i] = target
    template[-1] += '\n'
    template = ' '.join(template)
    return template


def create_evaluation_sentences(templates, targets, attributes):
    sentences = []
    for template in templates:
        for target in targets:
            for attribute in attributes:
                sentences.append(fill_template(template, target, attribute))
    return sentences


def write_evaluation_sentences(sentences, file_name,
        base_dir = os.path.join(BASE_DIR, set_dir, 'evals'),
        ):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    file_path = os.path.join(base_dir, file_name + '.txt')
    f = open(file_path, 'w')
    f.writelines(sentences)
    f.close()

    print('Sentences written to "%s"' % file_path)


def create_evaluations(template_sets, target_sets, attribute_sets):
    sentences = {}
    for template_k in template_sets:
        for target_k in target_sets:
            for attribute_k in attribute_sets:
                k = '_'.join([template_k, target_k, attribute_k])
                sentences[k] = create_evaluation_sentences(
                        template_sets[template_k],
                        target_sets[target_k],
                        attribute_sets[attribute_k],
                        )
                write_evaluation_sentences(sentences[k], k)
    return sentences



if __name__ == '__main__':
    assert all([k != 'mask' for k in targets_files]), targets_files
    assert all([k != 'mask' for k in attributes_files]), attributes_files

    # Read files
    templates = dict([(template, read_file(template, set_dir)) for template in templates_files])
    targets = dict([(target, read_file(target, set_dir)) for target in targets_files])
    attributes = dict([(attribute, read_file(attribute, set_dir)) for attribute in attributes_files])

    # Add masks
    targets['mask'] = ['[MASK]']
    attributes['mask'] = ['[MASK]']

    sentences = create_evaluations(templates, targets, attributes)
