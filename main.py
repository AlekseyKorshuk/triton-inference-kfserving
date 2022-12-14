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

from transformers import AutoTokenizer

SERVER_NUM_WORKERS = int(os.environ.get('SERVER_NUM_WORKERS', 1))
SERVER_PORT = int(os.environ.get('SERVER_PORT', 8080))
MODEL_NAME = os.environ.get('MODEL_NAME', 'GPT-J-6B-lit-v2')
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
            "dtype": "int32"
        },
        {
            "name": "input_lengths",
            "data": [],
            "dtype": "int32"
        },
        {
            "name": "request_output_len",
            "data": [[64]],
            "dtype": "int32"
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
#         {
#             "name": "random_seed",
#             "data": [[0]],
#             "dtype": "uint64"
#         },
        {
            "name": "runtime_top_k",
            "data": [[0]],
            "dtype": "int32"
        },
        {
            "name": "runtime_top_p",
            "data": [[0.725]],
            "dtype": "float32"
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


def get_last_message(text):
    if text == "":
        return ""
    if text[-1] == "\n":
        return ""
    last_raw = text.split("\n")[-1]
    last_message = last_raw.split(":")[-1]
    return last_message.strip()


class KFServingHuggingFace(kfserving.KFModel):
    def __init__(self, name):
        super().__init__(name)
        self.name = MODEL_NAME
        self.ready = False
        self.tokenizer = None
        self.client = None
        self.generator = None
        self.bad_words_ids = None

    def load_tokenizer(self):
        logger.info(f'Loading tokenizer from gpt2 ...')
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="right")
        self.tokenizer.pad_token_id = ['<|endoftext|>']
        assert self.tokenizer.pad_token_id == 50256, 'incorrect padding token'
        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = 'right'
        logger.info('Tokenizer loaded.')

    def generate_parameters_from_texts(self, texts):
        params = deepcopy(GENERATION_CONFIG["request"])

        input_ids = self.tokenizer(texts, return_tensors="np", add_special_tokens=False, padding="max_length",
                                   truncation=True, max_length=512).input_ids
        # input_ids = [[1] * len(texts)]
        for index, value in enumerate(params):

            if value['name'] == 'input_ids':
                data = np.array([data for data in input_ids], dtype=value['dtype'])
            elif value['name'] == 'input_lengths':
                value_data = [[len(sample_input_ids)] for sample_input_ids in input_ids]
                data = np.array([data for data in value_data], dtype=value['dtype'])
            elif value['name'] == 'random_seed':
                data = np.array([[random.randint(0, 1000)] for _ in range(len(input_ids))], dtype=value['dtype'])
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
                    output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                    output_texts.append(output_text)
                    output_texts_cropped.append(
                        get_last_message(output_text)
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

        self.client = httpclient.InferenceServerClient(DEFAULT_CONFIG['url'], verbose=DEFAULT_CONFIG['verbose'],
                                                       concurrency=10)
        while True:
            try:
                self.triton_inference(self.client, ["User: Write 'yes'\nBot:"])
                break
            except Exception as ex:
                print(ex)
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
        responses = self.triton_inference(self.client, inputs)
        logger.info(f"Done in {time.time() - start_time} seconds.")
        return {'predictions': responses}

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
