import logging
import dill
import six
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import random

import deepmatcher as dm
from deepmatcher.utils import *
from deepmatcher.data.iterator import *

logger = logging.getLogger('deepmatcher.core')

class HighwayMLP(nn.Module):

    def __init__(self,
                 input_size,
                 gate_bias=-2,
                 activation_function=nn.functional.relu,
                 gate_activation=nn.functional.softmax):
        super(HighwayMLP, self).__init__()
        self.activation_function = activation_function
        self.gate_activation = gate_activation
        self.normal_layer = nn.Linear(input_size, input_size)
        self.gate_layer = nn.Linear(input_size, input_size)
        self.gate_layer.bias.data.fill_(gate_bias)

    def forward(self, x):
        normal_layer_result = self.activation_function(self.normal_layer(x))
        gate_layer_result = self.gate_activation(self.gate_layer(x))
        multiplyed_gate_and_normal = torch.mul(normal_layer_result, gate_layer_result)
        multiplyed_gate_and_input = torch.mul((1 - gate_layer_result), x)
        return torch.add(multiplyed_gate_and_normal,
                         multiplyed_gate_and_input)


class HierMatcher(dm.MatchingModel):
    def __init__(self,
                 output_size = 2,
                 hidden_size=256,
                 embedding_length = 300,
                 manualSeed = 0):

        super(HierMatcher, self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding_length = embedding_length
        print("hidden_size = ",hidden_size)
        print("embedding_length = ",embedding_length)
        print("manualSeed = ", manualSeed)

        np.random.seed(manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

    def initialize(self, train_dataset, init_batch=None):
        r"""Initialize (not lazily) the matching model given the actual training data.

        Instantiates all sub-components and their trainable parameters.

        Args:
            train_dataset (:class:`~deepmatcher.data.MatchingDataset`):
                The training dataset obtained using :func:`deepmatcher.data.process`.
            init_batch (:class:`~deepmatcher.batch.MatchingBatch`):
                A batch of data to forward propagate through the model. If None, a batch
                is drawn from the training dataset.
        """

        if self._initialized:
            return

        # Copy over training info from train set for persistent state. But remove actual
        # data examples.
        self.meta = Bunch(**train_dataset.__dict__)
        if hasattr(self.meta, 'fields'):
            del self.meta.fields
            del self.meta.examples

        self._register_train_buffer('state_meta', Bunch(**self.meta.__dict__))
        del self.state_meta.metadata  # we only need `self.meta.orig_metadata` for state.

        self._reset_embeddings(train_dataset.vocabs)


        #####################################################################################
        #model related initialize

        self.best_score = None
        self.epoch = None

        self.left_fields = train_dataset.all_left_fields
        self.right_fields = train_dataset.all_right_fields

        self.bi_gru = nn.GRU(self.embedding_length, self.hidden_size, 1, bidirectional  = True)

        len_highway_alignment_input = self.embedding_length
        self.highway_layer_alignment = HighwayMLP(self.hidden_size * 2)
        self.attention_linear = nn.Linear(len_highway_alignment_input, len_highway_alignment_input)

        len_highway_aggre_weight_input = 2 * self.hidden_size * 2
        self.highway_layer_aggre_weight = HighwayMLP(len_highway_aggre_weight_input)
        self.linear_aggre = nn.Linear(len_highway_aggre_weight_input, 1)

        len_highway_input = (len(self.left_fields)+len(self.right_fields)) * self.hidden_size * 2
        self.highway_layer = HighwayMLP(len_highway_input)

        len_linear_input = len_highway_input
        self.label = nn.Linear(len_linear_input, 2)

        self.init_field_representation()

        self.linear_token_compare = nn.Linear(2 * self.hidden_size,1)

        # used as comparison result of attributes with empty value, learned when training
        compare_result_pad = torch.Tensor(1, self.hidden_size * 2)
        nn.init.xavier_uniform(compare_result_pad, gain=nn.init.calculate_gain('relu'))
        self.compare_result_pad = nn.Parameter(compare_result_pad, requires_grad=True)

        self._initialized = True
        logger.info('Successfully initialized MatchingModel with {:d} trainable '
                    'parameters.'.format(tally_parameters(self)))


    def init_field_representation(self):
        field_embedding_left = torch.Tensor(len(self.left_fields), self.hidden_size * 2)
        nn.init.xavier_uniform(field_embedding_left, gain=nn.init.calculate_gain('relu'))
        self.field_embedding_left = nn.Parameter(field_embedding_left, requires_grad=True)

        field_embedding_right = torch.Tensor(len(self.right_fields), self.hidden_size * 2)
        nn.init.xavier_uniform(field_embedding_right, gain=nn.init.calculate_gain('relu'))
        self.field_embedding_right = nn.Parameter(field_embedding_right, requires_grad=True)

    #Element-wise compare
    def element_wise_compare(self, tensor_1, tensor_2):
        compare_result = torch.abs(tensor_1 - tensor_2)
        return compare_result

    #For each token, chose the most similar one as its alignment
    def to_one_hot(self, token_compare_res):
        # max_index = torch.max(token_compare_res, 2)[1]
        max_value = torch.max(token_compare_res, 2)[0]
        max_value = max_value.view(max_value.size()[0], max_value.size()[1], 1)
        max_value_expand = max_value.expand(max_value.size()[0], max_value.size()[1], token_compare_res.size()[2])
        mask = torch.ones(token_compare_res.size()[0], token_compare_res.size()[1], token_compare_res.size()[2])
        mask = (token_compare_res == max_value_expand).float()
        token_compare_res = torch.mul(token_compare_res, mask)
        mask2 = token_compare_res + 0.0000001
        token_compare_res = torch.div(token_compare_res, mask2)
        return token_compare_res


    def attr_level_matching(self, compare_result, word_embeddings_rnn, field_embedding, token_mask):
        '''
             Get attribute level comparison result by aggregating token level comparison result.
             Field_embedding and word_embeddings_rnn are used to distinguish importance weights of different tokens
        '''
        size = word_embeddings_rnn.size()
        word_embeddings_rnn = word_embeddings_rnn.view(size[0] * size[1], -1)

        attention = torch.mv(word_embeddings_rnn, field_embedding)
        attention = attention.view(size[0], 1, -1)
        attention = F.softmax(attention, dim=2)

        attention = attention.view(attention.size()[0], -1)
        attention = attention * token_mask.float()
        attention = attention.view(size[0], 1, -1)

        compare_att_sum = torch.bmm(attention, compare_result)
        compare_att_sum = compare_att_sum.view(compare_att_sum.size()[0], -1)
        return compare_att_sum

    #detect <unk> and <pad>, then get token level mask
    def get_token_level_mask(self, token_matrix):
        token_mask = token_matrix - 1
        temp = token_mask.clone()
        temp[token_mask > 0] = 1
        temp[token_mask == -1] = 0
        token_mask = temp
        return token_mask

    #Detect missing attribute values, then get attribute level mask
    def get_attr_level_mask(self,token_matrix):
        attr_mask = torch.sum(token_matrix - 1, 0)
        temp = attr_mask.clone()
        temp[attr_mask > 0] = 1
        attr_mask = temp
        attr_mask = attr_mask.view(-1, 1)
        return attr_mask

    #input: tensors of two token sequences, output: attention matrix
    def token_level_attention(self, left, right):
        left_expand = left.clone()
        right_expand = right.clone()
        size_left = left.size()
        size_right = right.size()

        left_expand = left_expand.view(size_left[0], size_left[1], 1, size_left[2])
        left_expand = left_expand.repeat(1, 1, size_right[1], 1)

        right_expand = right_expand.view(size_right[0], 1, size_right[1], size_right[2])
        right_expand = right_expand.repeat(1, size_left[1], 1, 1)

        compare_result = self.element_wise_compare(left_expand, right_expand)
        compare_result = self.highway_layer_alignment(compare_result)
        sim_values = self.linear_token_compare(compare_result)
        sim_values = sim_values.view(sim_values.size()[0], sim_values.size()[1], -1)
        sim_values = F.softmax(sim_values, dim=2)
        compare_matrix = sim_values
        return compare_matrix

    def get_rnn_output(self, embedding_input):
        # h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())
        output_rnn, hn_rnn = self.bi_gru(embedding_input, None)
        output_rnn = output_rnn.permute(1, 0, 2).contiguous()
        return output_rnn, hn_rnn

    def process_empty_value(self, compare_result, attr_mask):
        attr_mask_opposite = attr_mask.clone()
        attr_mask_opposite[attr_mask > 0] = 0
        attr_mask_opposite[attr_mask == 0] = 1

        attr_mask = attr_mask.expand(attr_mask.size()[0], compare_result.size()[1])
        compare_result = torch.mul(compare_result, attr_mask.float())
        pad_matrix = torch.mm(attr_mask_opposite.float(), self.compare_result_pad)
        compare_result = compare_result + pad_matrix
        return compare_result

    def process_pad_token(self, attention_matrix, token_mask_list):
        '''
            Set attention values of pad tokens to 0
        '''
        token_mask_list_cat = torch.cat(token_mask_list, 0)
        token_mask_list_cat = token_mask_list_cat.permute(1, 0).contiguous()
        token_mask_list_cat = token_mask_list_cat.view(token_mask_list_cat.size()[0], 1, token_mask_list_cat.size()[1])
        token_mask_list_cat = token_mask_list_cat.repeat(1, attention_matrix.size()[1], 1)
        attention_matrix = attention_matrix * token_mask_list_cat.float()
        return attention_matrix


    def representation_layer(self, input, fields, embeddings):
        '''
            Get token embedding and contextual vector for each token.
            Get token level and attribute level masks for each attribute.
        '''
        value_embedding_list = []
        output_rnn_list = []
        token_mask_list = []
        attr_mask_list = []
        for field in fields:
            value = getattr(input, field).data
            value = value.permute(1, 0).contiguous()

            # Get token embedding
            value_embedding = embeddings[field].data
            value_embedding = value_embedding.permute(1, 0, 2).contiguous()
            value_embedding_list.append(value_embedding)

            # Get contextual representation by Bi-GRU
            output_rnn, _ = self.get_rnn_output(value_embedding)
            output_rnn_list.append(output_rnn)

            # Get token level mask
            token_mask = self.get_token_level_mask(value)
            token_mask_list.append(token_mask)

            # Get attr level mask
            attr_mask = self.get_attr_level_mask(value)
            attr_mask_list.append(attr_mask)

        return value_embedding_list, output_rnn_list, token_mask_list, attr_mask_list

    def forward(self, input):
        embeddings = {}
        for name in self.meta.all_text_fields:
            attr_input = getattr(input, name)
            embeddings[name] = self.embed[name](attr_input)

        batch_size = len(input.id)

        embedding_left_list, \
        output_rnn_left_list, \
        left_token_mask_list, \
        left_attr_mask_list = self.representation_layer(input, self.left_fields, embeddings)

        embedding_right_list, \
        output_rnn_right_list, \
        right_token_mask_list, \
        right_attr_mask_list = self.representation_layer(input, self.right_fields, embeddings)

        attr_compare_list = []
        for i in range(0, len(self.left_fields)):
            output_rnn_left = output_rnn_left_list[i]
            output_rnn_right_all = torch.cat(output_rnn_right_list, 1)

            # Cross-attribute token alignment
            attention_matrix = self.token_level_attention(output_rnn_left, output_rnn_right_all )
            attention_matrix = self.process_pad_token(attention_matrix, right_token_mask_list)
            attention_matrix = self.to_one_hot(attention_matrix)

            # Token level matching
            left_aligned_representation = torch.bmm(attention_matrix, output_rnn_right_all)
            left_token_compare_result = self.element_wise_compare(output_rnn_left, left_aligned_representation)

            #Attribute level matching
            attr_compare_result = self.attr_level_matching(left_token_compare_result,
                                                           output_rnn_left,
                                                           self.field_embedding_left[i],
                                                           left_token_mask_list[i].permute(1, 0).contiguous())
            attr_compare_result = self.process_empty_value(attr_compare_result, left_attr_mask_list[i])

            attr_compare_list.append(attr_compare_result)

        for i in range(0, len(self.right_fields)):
            output_rnn_right = output_rnn_right_list[i]
            output_rnn_left_all = torch.cat(output_rnn_left_list, 1)

            # Cross-attribute token alignment
            attention_matrix = self.token_level_attention(output_rnn_right, output_rnn_left_all )
            attention_matrix = self.process_pad_token(attention_matrix, left_token_mask_list)
            attention_matrix = self.to_one_hot(attention_matrix)

            # Token level matching
            right_aligned_representation = torch.bmm(attention_matrix, output_rnn_left_all)
            right_token_compare_result = self.element_wise_compare(output_rnn_right, right_aligned_representation)

            #Attribute level matching
            attr_compare_result = self.attr_level_matching(right_token_compare_result,
                                                           output_rnn_right,
                                                           self.field_embedding_right[i],
                                                           right_token_mask_list[i].permute(1, 0).contiguous())
            attr_compare_result = self.process_empty_value(attr_compare_result, right_attr_mask_list[i])

            attr_compare_list.append(attr_compare_result)

        #Entity level matching
        compare_concat = torch.stack(attr_compare_list, 1)
        compare_concat = compare_concat.view(batch_size, -1)
        entity_compare_result = self.highway_layer(compare_concat)
        entity_compare_result = self.label(entity_compare_result)
        output = F.log_softmax(entity_compare_result, dim=1)
        return output

    def load_state(self, path, map_location=None):
        r"""Load the model state from a file in a certain path.

        Args:
            path (string): The path to load the model state from.
        """
        state = torch.load(path, pickle_module=dill, map_location=map_location)
        # for k, v in six.iteritems(state):
        #     if k != 'model':
        #         self._train_buffers.add(k)
        #         setattr(self, k, v)
        #
        # if hasattr(self, 'state_meta'):
        #     train_info = copy.copy(self.state_meta)
        #
        #     # Handle metadata manually.
        #     # TODO (Sid): Make this cleaner.
        #     train_info.metadata = train_info.orig_metadata
        #     MatchingDataset.finalize_metadata(train_info)
        #
        #     self.initialize(train_info, self.state_meta.init_batch)

        self.load_state_dict(state['model'])