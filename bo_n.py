import gc
import random
import kfserving
import logging
import os
import signal
import sys
import time
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
from copy import deepcopy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SERVER_NUM_WORKERS = int(os.environ.get('SERVER_NUM_WORKERS', 1))
SERVER_PORT = int(os.environ.get('SERVER_PORT', 8080))
MODEL_NAME = os.environ.get('MODEL_NAME', 'GPT-J-6B-lit-v2')
REWARD_MODEL = os.environ.get('REWARD_MODEL', 'ChaiML/reward_48m_gpt2_target_2')
BO_N = int(os.environ.get('BO_N', 4))
READY_FLAG = '/tmp/ready'
DEBUG_MODE = bool(os.environ.get('DEBUG_MODE', 0))

logging.basicConfig(level=kfserving.constants.KFSERVING_LOGLEVEL)
logger = logging.getLogger(MODEL_NAME)

URL = "localhost"

DEFAULT_CONFIG = {
    'protocol': 'http',
    'url': f'{URL}:8000',
    'model_name': 'fastertransformer',
    'verbose': False,
}

GENERATION_CONFIG = {
    "request": [
        {
            "name": "input_ids",
            "data": [],
            "dtype": "uint32"
        },
        {
            "name": "input_lengths",
            "data": [],
            "dtype": "uint32"
        },
        {
            "name": "request_output_len",
            "data": [[256]],
            "dtype": "uint32"
        },
        {
            "name": "temperature",
            "data": [[0.72]],
            "dtype": "float32"
        },
        {
            "name": "repetition_penalty",
            "data": [[1.13125]],
            "dtype": "float32"
        },
        {
            "name": "random_seed",
            "data": [[0]],
            "dtype": "int32"
        },
        {
            "name": "runtime_top_k",
            "data": [[0]],
            "dtype": "uint32"
        },
        {
            "name": "runtime_top_p",
            "data": [[0.725]],
            "dtype": "float32"
        },
        {
            "name": "beam_width",
            "data": [[1]],
            "dtype": "int32"
        },
        {
            "name": "stop_words_list",
            "data": [[[198], [1]]],
            "dtype": "int32"
        },
        {
            "name": "bad_words_list",
            "data": [[[77, 15249, 77], [2, 5, 7]]],
            "dtype": "int32"
        }
    ]
}


def to_word_list_format(words):
    tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    flat_ids = []
    offsets = []
    item_flat_ids = []
    item_offsets = []

    for word in words:
        ids = tokenizer.encode(word)

        if len(ids) == 0:
            continue

        item_flat_ids += ids
        item_offsets.append(len(ids))

    flat_ids.append(np.array(item_flat_ids))
    offsets.append(np.cumsum(np.array(item_offsets)))

    pad_to = max(1, max(len(ids) for ids in flat_ids))

    for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
        flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)), constant_values=0)
        offsets[i] = np.pad(offs, (0, pad_to - len(offs)), constant_values=-1)

    return np.array([flat_ids, offsets], dtype="int32").transpose((1, 0, 2))


def load_bad_word_ids():
    forbidden = [
        'nigger', 'nigga', 'negro', 'blacks',
        'rapist', 'rape', 'raping', 'niggas', 'raper',
        'niggers', 'rapers', 'niggas', 'NOOOOOOOO',
        'fag', 'faggot', 'fags', 'faggots']

    return to_word_list_format(forbidden)


GENERATION_CONFIG["request"][-1]["data"] = load_bad_word_ids()


def prepare_tensor(client, name, input):
    t = client.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


class KFServingHuggingFace(kfserving.KFModel):
    def __init__(self, name):
        super().__init__(name)
        self.name = MODEL_NAME
        self.ready = False
        self.tokenizer = None
        self.client = None
        self.generator = None
        self.bad_words_ids = None
        self.eval_tokenizer = None
        self.eval_model = None

    def load_evaluation(self):
        logger.info('loading evaluation model')
        self.eval_tokenizer = AutoTokenizer.from_pretrained(
            REWARD_MODEL
        )
        self.eval_tokenizer.pad_token = self.eval_tokenizer.eos_token
        self.eval_tokenizer.truncation_side = 'left'

        logger.info('loading reward model {}'.format(REWARD_MODEL))
        self.eval_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL).to(0)
        self.eval_model.config.pad_token_id = self.eval_model.config.eos_token_id

    def _format_string_for_eval_model_input(self, string):
        bot_messages = 3

        lines = [l for l in string.split('\n') if len(l) > 0]

        if len(lines) <= bot_messages * 2:
            return None

        lines = lines[-bot_messages * 2:]

        res = []

        bot = 1
        for line in lines[::-1]:
            user = 'User' if bot == 0 else 'Bot'
            if ':' not in line:
                return None
            pos = line.find(':')
            newline = '{}:{}'.format(user, line[pos + 1:])
            res.insert(0, newline)
            bot ^= 1

        return '\n'.join(res)

    def _pick_best_response(self, input_text, responses):
        candidates = []
        for response in responses:
            text = input_text + response.split("\n")[0]
            candidates.append(self._format_string_for_eval_model_input(text))

        if None in candidates:
            return [{'text': t, 'score': 1.0} for t in responses]

        tokens = self.eval_tokenizer(
            candidates,
            return_tensors='pt',
            padding=True,
            truncation=True,
        ).to(0)

        with torch.inference_mode():
            logits = self.eval_model(**tokens).logits

        preds = [float(p[1]) for p in torch.softmax(logits, dim=1)]
        scored_responses = list(zip(responses, preds))

        return [{'text': t, 'score': s} for t, s in scored_responses]

    def load_tokenizer(self):
        logger.info(f'Loading tokenizer from gpt2 ...')
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
        self.tokenizer.pad_token_id = ['<|endoftext|>']
        assert self.tokenizer.pad_token_id == 50256, 'incorrect padding token'
        self.tokenizer.padding_side = 'left'
        self.tokenizer.truncation_side = 'left'
        logger.info('Tokenizer loaded.')

    def generate_parameters_from_texts(self, texts):
        params = deepcopy(GENERATION_CONFIG["request"])
        input_ids = self.tokenizer(texts, return_tensors="np", padding="longest").input_ids
        # input_ids = [[1] * len(texts)]
        for index, value in enumerate(params):
            if value['name'] == 'input_ids':
                data = np.array([data for data in input_ids], dtype=value['dtype'])
            elif value['name'] == 'input_lengths':
                value_data = [[len(sample_input_ids)] for sample_input_ids in input_ids]
                data = np.array([data for data in value_data], dtype=value['dtype'])
            elif value['name'] == 'random_seed':
                # random_seed = random.randint(0, 10)
                random_seed = 0
                logger.info(f'Random seed: {random_seed}')
                data = np.array([[random_seed] for _ in range(1)], dtype=value['dtype'])
            else:
                data = np.array([data for data in value['data']] * len(input_ids), dtype=value['dtype'])

            params[index] = {
                'name': value['name'],
                'data': data,
            }

        return params

    def triton_inference(self, inference_client, texts):
        request = self.generate_parameters_from_texts(texts)
        # print(request)
        payload = [prepare_tensor(httpclient, field['name'], field['data'])
                   for field in request]
        result = inference_client.infer(DEFAULT_CONFIG['model_name'], payload)
        output_texts = []
        output_texts_cropped = []

        for i, output in enumerate(result.get_response()['outputs']):
            if output['name'] == "output_ids":
                for output_ids in result.as_numpy(output['name']):
                    output_ids = [int(output_id) for output_id in list(output_ids[0])]
                    output_texts.append(self.tokenizer.decode(output_ids, skip_special_tokens=True).strip())
                    output_texts_cropped.append(
                        self.tokenizer.decode(
                            output_ids[len(request[0]["data"][i]):], skip_special_tokens=True
                        ).strip()
                    )
        return output_texts_cropped

    def load(self):
        """
        Load from a pytorch saved pickle to reduce the time it takes
        to load the model.  To benefit from this it is important to
        have run pytorch save on the same machine / hardware.
        """

        gc.disable()
        start_time = time.time()

        self.load_tokenizer()
        self.load_evaluation()

        self.client = httpclient.InferenceServerClient(DEFAULT_CONFIG['url'], verbose=DEFAULT_CONFIG['verbose'],
                                                       concurrency=10)
        while True:
            try:
                self.triton_inference(self.client, ["User: Write 'yes'\nBot:"])
                break
            except:
                time.sleep(1)

        logger.info('Model loaded.')

        logger.info('Creating generator for model ...')
        logger.info(f'Model is ready in {str(time.time() - start_time)} seconds.')

        gc.enable()
        self.ready = True
        self._set_ready_flag()

    def predict(self, request, parameters=None):
        inputs = request['instances']
        start_time = time.time()

        best_responses = []
        for input in inputs:
            batch = [input] * BO_N
            responses = self.triton_inference(self.client, batch)
            scored_responses = self._pick_best_response(input, responses)
            sorted_responses = sorted(scored_responses, key=lambda d: d['score'])
            best_response = sorted_responses[-1]["text"].strip()
            best_responses.append(best_response)

        logger.info(f"Done in {time.time() - start_time} seconds.")
        return {'predictions': best_responses}

    def _set_ready_flag(self):
        """Used by readiness probe. """
        with open(READY_FLAG, 'w') as fh:
            fh.write('1')


def terminate(signal, frame):
    """
    Kubernetes send SIGTERM to containers for them
    to stop so this must be handled by the process.
    """
    logger.info("Start Terminating")
    if os.path.exists(READY_FLAG):
        os.remove(READY_FLAG)
    time.sleep(5)
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, terminate)

    if DEBUG_MODE:
        import time

        time.sleep(3600 * 10)

    model = KFServingHuggingFace(MODEL_NAME)
    model.load()

    kfserving.KFServer(
        http_port=SERVER_PORT,
        workers=SERVER_NUM_WORKERS
    ).start([model])
