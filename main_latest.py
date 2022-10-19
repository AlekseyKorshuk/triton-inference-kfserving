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


class InferenceService:
    def inference(self, texts, random_seed=-1):
        raise NotImplementedError


class TritonInferenceService(InferenceService):
    GENERATION_CONFIG = None
    model_name = "fastertransformer"
    padding_side = "right"
    pad_token_id = 50256

    def __init__(self, url, verbose=False):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side=self.padding_side)
        self.tokenizer.pad_token_id = self.pad_token_id
        assert self.tokenizer.pad_token_id == self.pad_token_id, 'incorrect padding token'
        self.tokenizer.padding_side = self.padding_side

        self.GENERATION_CONFIG["request"][-1]["data"] = self.load_bad_word_ids()
        self.client = httpclient.InferenceServerClient(
            url=f'{url}:8000',
            verbose=verbose,
            concurrency=1
        )

    def to_word_list_format(self, words):

        flat_ids = []
        offsets = []
        item_flat_ids = []
        item_offsets = []

        for word in words:
            ids = self.tokenizer.encode(word)

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

    def load_bad_word_ids(self):
        forbidden = [
            'nigger', 'nigga', 'negro', 'blacks',
            'rapist', 'rape', 'raping', 'niggas', 'raper',
            'niggers', 'rapers', 'niggas', 'NOOOOOOOO',
            'fag', 'faggot', 'fags', 'faggots']

        return self.to_word_list_format(forbidden)

    def generate_parameters_from_texts(self, texts, random_seed=None):
        params = deepcopy(self.GENERATION_CONFIG["request"])
        inputs = self.tokenizer(texts, return_tensors="np", add_special_tokens=False, padding=True)
        input_ids_no_pad = self.tokenizer(texts, return_tensors="np", add_special_tokens=False, padding=False).input_ids
        input_ids = inputs.input_ids
        input_sizes = [len(sample_input_ids) for sample_input_ids in input_ids_no_pad]
        random_seed_index = -1
        for index, value in enumerate(params):

            if value['name'] == 'input_ids':
                data = np.array([np.array(data) for data in input_ids], dtype=value['dtype'])
            elif value['name'] == 'input_lengths':
                value_data = [[len(sample_input_ids)] for sample_input_ids in input_ids_no_pad]
                data = np.array([data for data in value_data], dtype=value['dtype'])
            elif value['name'] == 'random_seed':
                random_seed_index = index
                data = np.array([[random_seed] for _ in range(len(input_ids))], dtype=value['dtype'])
            else:
                data = np.array([data for data in value['data']] * len(input_ids), dtype=value['dtype'])

            params[index] = {
                'name': value['name'],
                'data': data,
            }

        if random_seed == -1 and random_seed_index != -1:
            params.pop(random_seed_index)

        return params, input_sizes

    def inference(self, texts, random_seed=-1):
        request, input_sizes = self.generate_parameters_from_texts(texts, random_seed)
        payload = [prepare_tensor(httpclient, field['name'], field['data'])
                   for field in request]
        result = self.client.infer(self.model_name, payload)
        output_texts = []
        output_texts_cropped = []

        for input_size_tokens, output in zip(input_sizes, result.get_response()['outputs']):
            if output['name'] == "output_ids":
                for output_ids in result.as_numpy(output['name']):
                    output_ids = [int(output_id) for output_id in list(output_ids[0])]
                    output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                    output_texts.append(output_text)
                    output_texts_cropped.append(
                        get_last_message(output_text)
                    )
        return output_texts_cropped


class NewInferenceService(TritonInferenceService):
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
            {
                "name": "random_seed",
                "data": [[0]],
                "dtype": "uint64"
            },
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


class KFServingHuggingFace(kfserving.KFModel):
    def __init__(self, name):
        super().__init__(name)
        self.name = MODEL_NAME
        self.client = NewInferenceService("localhost")
        self.ready = False

    def predict(self, request, parameters=None):
        inputs = request['instances']
        logger.info(f'Request: {request}')
        logger.info(f'Parameters: {parameters}')
        if 'random_seed' in parameters.keys():
            random_seed = parameters['random_seed']
        else:
            random_seed = random.randint(0, 100)
            random_seed = -1
        logger.info(f'Random seed: {random_seed}')
        responses = self.client.inference(inputs, random_seed=random_seed)
        return {'predictions': responses}

    def load(self):
        """
        Load from a pytorch saved pickle to reduce the time it takes
        to load the model.  To benefit from this it is important to
        have run pytorch save on the same machine / hardware.
        """

        gc.disable()
        start_time = time.time()

        while True:
            try:
                self.client.inference(["User: Write 'yes'\nBot:"])
                break
            except Exception as ex:
                logger.info(f'Pinging model: {ex}')
                time.sleep(1)

        logger.info(f'Model is ready in {str(time.time() - start_time)} seconds.')

        gc.enable()
        self.ready = True
        self._set_ready_flag()

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
