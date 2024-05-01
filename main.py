import numpy as np
import networkx as nx
from os.path import join

import utils
from glimpse_attention_model import GlimpseAttentionModel
import logging

train_len = 5000

if __name__ == '__main__':
    options = utils.load_params()
    __processor__ = options['cell_type']
    # model_type = options['cell_type']
    handler = logging.FileHandler('{}-{}.log'.format(__processor__, options['dataset_name']), 'w')
    log = logging.getLogger(__processor__)
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)

    data_path = join(options['data_dir'], options['dataset_name'])
    # utils.write_seen_nodes(join(options['data_dir'], options['dataset_name']), 30)
    node_index = utils.load_graph(data_path)
    options['node_size'] = len(node_index)
    # print(nx.info(G))
    train_instances, max_diff_train = utils.load_instances(data_path, 'train', node_index, options['seq_len'],
                                                           limit=-1, log=log)
    test_instances, max_diff_test = utils.load_instances(data_path, 'test', node_index, options['seq_len'],
                                                         limit=-1, log=log)
    options['train_nodes'] = utils.get_nodes(train_instances)
    # log.info(f"len(options['train_nodes']) = {len(options['train_nodes'])}")

    options['max_diff'] = max_diff_train
    log.info(f"num of train/test instances: {len(train_instances)}, {len(test_instances)}")
    options['n_train'] = len(train_instances)

    train_loader = utils.Loader(train_instances, log, options)
    test_loader = utils.Loader(test_instances, log, options, shuffle=False)

    log.info('running glimpse attention model')
    log.info('using attention:' + str(options['use_attention']))
    log.info(options)
    glimpse_ins = GlimpseAttentionModel(options, options['use_attention'], options['n_train'])
    glimpse_ins.run_model(train_loader, test_loader, options)
