import os
import logging
import pickle
import pandas as pd
import torch.utils.data

from tqdm import tqdm
from collections import defaultdict

logger = logging.getLogger(__name__)

TRANSACTION_DATASET_NAME = 'transactions.bin'


class BaseDataset:

    def __init__(
            self,
            path_to_transactions,
            path_to_train_labels,
            path_to_test_labels,
            val_size=0.2,
            min_history_len=None,
            max_history_len=None
    ):
        self._path_to_transactions = path_to_transactions
        self._path_to_train_labels = path_to_train_labels
        self._path_to_test_labels = path_to_test_labels
        self._val_size = val_size

        self._min_history_len = min_history_len
        self._max_history_len = max_history_len

        # Dataset meta-information part
        self._num_types = 0
        self._num_codes = 0
        self._max_sequence_len = 0
        customer_id_mapping = {}

        # Process all transactions
        customers_history = defaultdict(list)

        if not os.path.exists(TRANSACTION_DATASET_NAME):
            transactions_dataset = pd.read_csv(self._path_to_transactions)
            for _, row in tqdm(transactions_dataset.iterrows(), desc='Creating transactions dataset...'):
                customer_id = int(row['customer_id'])
                customer_id_fact = int(row['customer_id_factorized'])

                customer_id_mapping[customer_id] = customer_id_fact

                mcc_code = int(row['mcc_code_factorized'])
                transaction_type = int(row['tr_type_factorized'])
                amount = float(row['amount_log'])
                timestamp = int(row['timestamp'])

                customers_history[customer_id_fact].append({
                    'mcc_code.idx': mcc_code,
                    'transaction_type.idx': transaction_type,
                    'amount.value': amount,
                    'timestamp': timestamp
                })

                self._num_codes = max(self._num_codes, mcc_code)
                self._num_types = max(self._num_types, transaction_type)

            # Sort by timestamp
            for customer_id, transactions in tqdm(customers_history.items(), desc='Sort all histories by timestamp...'):
                customers_history[customer_id] = sorted(transactions, key=lambda x: x['timestamp'])
                if self._max_history_len:
                    customers_history[customer_id] = customers_history[customer_id][-self._max_history_len:]
                self._max_sequence_len = max(self._max_sequence_len, len(customers_history[customer_id]))

            status = (self._num_types, self._num_codes, self._max_sequence_len, customer_id_mapping, customers_history)

            with open(TRANSACTION_DATASET_NAME, 'wb') as f:
                pickle.dump(status, f)
        else:
            with open(TRANSACTION_DATASET_NAME, 'rb') as f:
                status = pickle.load(f)

            self._num_types, self._num_codes, self._max_sequence_len, customer_id_mapping, customers_history = status

        self._customers_history = customers_history

        # Train labels part
        train_data = []
        train_labels = pd.read_csv(self._path_to_train_labels)
        for _, row in tqdm(train_labels.iterrows(), desc='Assigning train labels...'):
            customer_id = int(row['customer_id'])
            customer_id_fact = customer_id_mapping[customer_id]
            label = int(row['gender'])

            train_data.append({
                'sample': customers_history[customer_id_fact],
                'label': label
            })

        self._train_dataset = train_data

        self._val_dataset = self._train_dataset[int(len(self._train_dataset) * (1 - self._val_size)):]
        self._train_dataset = self._train_dataset[:int(len(self._train_dataset) * (1 - self._val_size))]

        # Test labels part
        test_data = []
        test_labels = pd.read_csv(self._path_to_test_labels)
        for _, row in tqdm(test_labels.iterrows(), desc='Assigning test labels...'):
            customer_id = int(row['customer_id'])
            customer_id_fact = customer_id_mapping[customer_id]
            train_data.append({'sample': customers_history[customer_id_fact]})

        self._test_dataset = test_data

    @property
    def num_types(self):
        return self._num_types

    @property
    def num_codes(self):
        return self._num_codes

    @property
    def max_sequence_len(self):
        return self._max_sequence_len

    @property
    def datasets(self):
        return self._train_dataset, self._val_dataset, self._test_dataset


class BatchProcessor:

    def __call__(self, batch):
        processed_batch = {}

        processed_batch['lengths'] = []
        for sample in batch:
            processed_batch['lengths'].append(len(sample['sample']))
        processed_batch['lengths'] = torch.tensor(processed_batch['lengths'], dtype=torch.long)

        processed_batch['labels'] = []
        for sample in batch:
            processed_batch['labels'].append(sample['label'])
        processed_batch['labels'] = torch.tensor(processed_batch['labels'], dtype=torch.long)

        processed_batch['positions'] = []
        for sample in batch:
            processed_batch['positions'].extend(list(range(len(sample['sample']))))
        processed_batch['positions'] = torch.tensor(processed_batch['positions'], dtype=torch.long)

        for key in batch[0]['sample'][0].keys():
            if key.endswith('.idx'):  # categorical feature
                prefix = key.split('.')[0]
                processed_batch[prefix] = []
                for sample in batch:
                    for event in sample['sample']:
                        processed_batch[prefix].append(event[key])

                processed_batch[prefix] = torch.tensor(processed_batch[prefix], dtype=torch.long)
            elif key.endswith('.value'):  # scalar feature
                prefix = key.split('.')[0]
                processed_batch[prefix] = []
                for sample in batch:
                    for event in sample['sample']:
                        processed_batch[prefix].append(event[key])

                processed_batch[prefix] = torch.tensor(processed_batch[prefix], dtype=torch.float)
            # else:
            #     print(f'Unused feature: {key}')

        return processed_batch
