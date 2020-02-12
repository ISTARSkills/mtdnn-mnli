from pytorch_pretrained_bert.tokenization import BertTokenizer
import torch
from mt_dnn.model import MTDNNModel


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length.
    Copyed from https://github.com/huggingface/pytorch-pretrained-BERT
    """
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def bert_feature_extractor(
        text_a, text_b=None, max_seq_length=512, tokenize_fn=None):
    tokens_a = tokenize_fn.tokenize(text_a)
    tokens_b = None
    if text_b:
        tokens_b = tokenize_fn.tokenize(text_b)

    if tokens_b:
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for one [SEP] & one [CLS] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:max_seq_length - 2]
    if tokens_b:
        input_ids = tokenize_fn.convert_tokens_to_ids(
            ['[CLS]'] + tokens_b + ['[SEP]'] + tokens_a + ['[SEP]'])
        segment_ids = [0] * (len(tokens_b) + 2) + [1] * (len(tokens_a) + 1)
    else:
        input_ids = tokenize_fn.convert_tokens_to_ids(
            ['[CLS]'] + tokens_a + ['[SEP]'])
        segment_ids = [0] * len(input_ids)
    input_mask = None
    return input_ids, input_mask, segment_ids

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
print('Enter Sentence 1:')
premise = input()
print('Enter Sentence 2:')
hypothesis = input()

input_ids, _, type_ids = bert_feature_extractor(premise, hypothesis, max_seq_length=64, tokenize_fn=tokenizer)
features = {'uid': '0', 'label': '0', 'token_id': input_ids, 'type_id': type_ids}

model_path = 'checkpoints/my_mnli/model_0.pt'
state_dict = torch.load(model_path)
config = state_dict['config']
opt.update(config)
model = MTDNNModel(opt, state_dict=state_dict)
