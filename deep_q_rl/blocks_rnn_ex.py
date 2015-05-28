from theano import tensor as T
import theano
import blocks
from blocks.bricks import Identity, Rectifier, Linear, Softmax
from blocks.bricks.cost import SquaredError
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
from blocks.algorithms import GradientDescent, Scale, Momentum, RMSProp
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.bricks.conv import ConvolutionalLayer, Flattener, ConvolutionalSequence
from blocks import initialization as init
from blocks.bricks.recurrent import SimpleRecurrent
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.datasets.base import Dataset, IterableDataset
from collections import OrderedDict
import numpy as np
import pdb


class ConvRNN:
    ''' 
        ConvRNN takes a sequence of images and labels and directly optimizes a MSE while
        indirectly optimizing a policy decision represented by the softmax output

        ** methods **
        train: trains the network
        get_exp_reward: generate an expected reward based on hidden state
        get_suggested_actions: return a vector of possible actions and probabilities associated with each
                                        you then sample from that to get your action, which yields an actual reward
        get_loss: what it sounds like
    '''

    def __init__(self, rnn_dims, num_actions, data_X_np=None, data_y_np=None, width=32, height=32):
        ###############################################################
        #
        #       Network and data setup
        #
        ##############################################################
        RNN_DIMS = 100
        NUM_ACTIONS = num_actions

        tensor5 = T.TensorType('float32', [False, True, True, True, True])
        self.x = T.tensor4('features')
        self.reward = T.tensor3('targets', dtype='float32')
        self.state = T.matrix('states', dtype='float32')

        self.hidden_states = [] # holds hidden states in np array form

        
        #data_X & data_Y supplied in init function now...

        if data_X_np is None or data_y_np is None:
            print 'you did not supply data at init'
            data_X_np = np.float32(np.random.normal(size=(1280, 1,1, width, height)))
            data_y_np = np.float32(np.random.normal(size=(1280, 1,1,1)))
        #data_states_np = np.float32(np.ones((1280, 1, 100)))
        state_shape = (data_X_np.shape[0],rnn_dims)
        self.data_states_np = np.float32(np.zeros(state_shape))


        self.datastream = IterableDataset(dict(features=data_X_np,
                                            targets=data_y_np,
                                            states=self.data_states_np)).get_example_stream()
        self.datastream_test = IterableDataset(dict(features=data_X_np,
                                            targets=data_y_np,
                                            states=self.data_states_np)).get_example_stream()
        data_X = self.datastream


        # 2 conv inputs
        # we want to take our sequence of input images and convert them to convolutional
        # representations
        conv_layers = [ConvolutionalLayer(Rectifier().apply, (3, 3), 16, (2, 2), name='l1'),
                       ConvolutionalLayer(Rectifier().apply, (3, 3), 32, (2, 2), name='l2'),
                       ConvolutionalLayer(Rectifier().apply, (3, 3), 64, (2, 2), name='l3'),
                       ConvolutionalLayer(Rectifier().apply, (3, 3), 128, (2, 2), name='l4'),
                       ConvolutionalLayer(Rectifier().apply, (3, 3), 128, (2, 2), name='l5'),
                       ConvolutionalLayer(Rectifier().apply, (3, 3), 128, (2, 2), name='l6')]
        convnet = ConvolutionalSequence(conv_layers, num_channels=4,
                                        image_size=(width, height),
                                        weights_init=init.Uniform(0, 0.01),
                                        biases_init=init.Constant(0.0),
                                        tied_biases=False,
                                        border_mode='full')
        convnet.initialize()
        output_dim = np.prod(convnet.get_dim('output'))

        conv_out = convnet.apply(self.x)

        reshape_dims = (conv_out.shape[0], conv_out.shape[1]*conv_out.shape[2]*conv_out.shape[3])
        hidden_repr = conv_out.reshape(reshape_dims)
        conv2rnn = Linear(input_dim=output_dim, output_dim=RNN_DIMS, 
                            weights_init=init.Uniform(width=0.01),
                            biases_init=init.Constant(0.))
        conv2rnn.initialize()
        conv2rnn_output = conv2rnn.apply(hidden_repr)

        # RNN hidden layer
        # then we want to feed those conv representations into an RNN
        rnn = SimpleRecurrent(dim=RNN_DIMS, activation=Rectifier(), weights_init=init.Uniform(width=0.01))
        rnn.initialize()
        self.learned_state = rnn.apply(inputs=conv2rnn_output, states=self.state, iterate=False)


        # linear output from hidden layer
        # the RNN has two outputs, but only this one has a target. That is, this is "expected return"
        # which the network attempts to minimize difference between expected return and actual return
        lin_output = Linear(input_dim=RNN_DIMS, output_dim=1, 
                            weights_init=init.Uniform(width=0.01),
                            biases_init=init.Constant(0.))
        lin_output.initialize()
        self.exp_reward = lin_output.apply(self.learned_state)
        self.get_exp_reward = theano.function([self.x, self.state], self.exp_reward)

        # softmax output from hidden layer
        # this provides a softmax of action recommendations
        # the hypothesis is that adjusting the other outputs magically influences this set of outputs
        # to suggest smarter (or more realistic?) moves
        action_output = Linear(input_dim=RNN_DIMS, output_dim=NUM_ACTIONS, 
                            weights_init=init.Constant(.001), 
                            biases_init=init.Constant(0.))
        action_output.initialize()

        self.suggested_actions = Softmax().apply(action_output.apply(self.learned_state[-1]))

        ######################
        # use this to get suggested actions... it requires the state of the hidden units from the previous
        # timestep
        #####################
        self.get_suggested_actions = theano.function([self.x, self.state], [self.suggested_actions, self.learned_state])
        #####################

        #xxx = get_suggested_actions(data_X_np, empty_state)
        #yyy = get_exp_reward(data_X_np, empty_state)

    def train(self, iters, learning_rate=0.01, momentum=0.5, states=None, actions=None, 
                rewards=None, next_states=None, terminals=None):
        ''' 
            train the network, specify number of iters, learning_rate, momentum 

            important note: this is confusing: states is actually training data, next states
            is the next step for each training data sample. they are not hidden unit
            values or hidden states.
        
        '''
        self.data_states_np = np.ones(( len(states.get_value()), 1, self.data_states_np.shape[1] ), dtype=np.float32)
        self.h0 = np.zeros((len(states.get_value()), 1, 100), dtype=np.float32)
        feature_shape = states.get_value().shape
        feature_shape = (feature_shape[0],) + (1,) + feature_shape[-3:]
        self.datastream = IterableDataset(dict(features=states.get_value().reshape(feature_shape),
                                                targets=rewards.get_value().reshape((128,1,1,1)),
                                                states=self.h0)).get_example_stream()
        feature_shape = states.get_value().shape
        feature_shape = (feature_shape[0],) + (1,) + feature_shape[-3:]
        self.datastream_test = IterableDataset(dict(features=states.get_value().reshape(feature_shape),
                                                targets=rewards.get_value().reshape((128,1,1,1)),
                                                states=self.h0)).get_example_stream()
        self.cost = SquaredError().apply(self.reward, self.exp_reward*0+1)
        self.cost.name = 'MSE_with_regularization'
        self.get_cost = theano.function([self.x, self.state, self.reward], self.cost)
        cg = ComputationGraph(self.cost)
        step_rule = Momentum(learning_rate=learning_rate, momentum=momentum)
        algo = GradientDescent(cost=self.cost, params=cg.parameters, step_rule=step_rule)

        monitor = DataStreamMonitoring(variables=[self.cost], data_stream=self.datastream_test)

        main_loop = MainLoop(data_stream=self.datastream, 
                    algorithm=algo, 
                    extensions=[monitor, FinishAfter(after_n_epochs=5), Printing()])

        main_loop.run()
        return np.array(main_loop.log[main_loop.log.keys()[-1]]['MSE_with_regularization'])


