"""

# THIS file is for the architeture of RNN
"""
import lasagne
import lasagne.layers as LL
import theano.tensor as T
import math

def build_rnn_net(input_var=None, input_width=None, input_dim = None,
                            nin_units=80, h_num_units=[64,64], h_grad_clip=1.0,
                            output_width=1):
    """
    A stacked bidirectional RNN network for regression, alternating
    with dense layers and merging of the two directions, followed by
    a feature mean pooling in the time direction, with a linear
    dim-reduction layer at the start
    add dropout for generalizations
    
    Args:
        input_var (theano 3-tensor): minibatch of input sequence vectors
        input_width (int): length of input sequences
        nin_units (list): number of NIN features
        h_num_units (int list): no. of units in hidden layer in each stack
                                from bottom to top
        h_grad_clip (float): gradient clipping maximum value 
        output_width (int): size of output layer (e.g. =1 for 1D regression)
    Returns:
        output layer (Lasagne layer object)
    """
    
    # Non-linearity hyperparameter
    leaky_ratio = 0.3
    nonlin = lasagne.nonlinearities.LeakyRectify(leakiness=leaky_ratio)
    
    # Input layer
    l_in = LL.InputLayer(shape=(None, input_width, input_dim), 
                            input_var=input_var) 
    batchsize = l_in.input_var.shape[0]
    
    # NIN-layer
    #l_in_1 = LL.NINLayer(l_in, num_units=nin_units,
                       #nonlinearity=lasagne.nonlinearities.linear)
    l_in_1 = l_in
    #l_in_d = LL.DropoutLayer(l_in, p = 0.8) Do not use drop out now, for the first rnn layer is 256
    
    # currently, we do not drop features
    # RNN layers
    # dropout in the first two (total three) or three (total five) layers
    counter = -1
    drop_ends = 2
    for h in h_num_units:
        counter += 1
        # Forward layers
        l_forward_0 = LL.RecurrentLayer(l_in_1,
                                        nonlinearity=nonlin,
                                        num_units=h,
                                        W_in_to_hid=lasagne.init.Normal(0.01, 0),
                                        #W_in_to_hid=lasagne.init.He(initializer, math.sqrt(2/(1+0.15**2))),
                                        W_hid_to_hid=lasagne.init.Orthogonal(math.sqrt(2/(1+leaky_ratio**2))),
                                        backwards=False,
                                        learn_init=True,
                                        grad_clipping=h_grad_clip,
                                        #gradient_steps = 20,
                                        unroll_scan=True,
                                        precompute_input=True)
                                   
        l_forward_0a = LL.ReshapeLayer(l_forward_0, (-1, h))
        
        if(counter < drop_ends and counter % 2 != 0):
            l_forward_0a = LL.DropoutLayer(l_forward_0a, p = 0.2)
        else:
            l_forward_0a = l_forward_0a
        
        l_forward_0b = LL.DenseLayer(l_forward_0a, num_units=h,
                                     nonlinearity=nonlin)
        l_forward_0c = LL.ReshapeLayer(l_forward_0b,
                                       (batchsize, input_width, h))
        
        
        l_forward_out = l_forward_0c
        
        # Backward layers
        l_backward_0 = LL.RecurrentLayer(l_in_1,
                                         nonlinearity=nonlin,
                                         num_units=h,
                                         W_in_to_hid=lasagne.init.Normal(0.01, 0),
                                         #W_in_to_hid=lasagne.init.He(initializer, math.sqrt(2/(1+0.15**2))),
                                         W_hid_to_hid=lasagne.init.Orthogonal(math.sqrt(2/(1+leaky_ratio**2))),
                                         backwards=True,
                                         learn_init=True,
                                         grad_clipping=h_grad_clip,
                                         #gradient_steps = 20,
                                         unroll_scan=True,
                                         precompute_input=True)
                                        
        l_backward_0a = LL.ReshapeLayer(l_backward_0, (-1, h))
        
        if(counter < drop_ends and counter % 2 == 0):
            l_backward_0a = LL.DropoutLayer(l_backward_0a, p = 0.2)
        else:
            l_backward_0a = l_backward_0a
        
        l_backward_0b = LL.DenseLayer(l_backward_0a, num_units=h,
                                      nonlinearity=nonlin)
        l_backward_0c = LL.ReshapeLayer(l_backward_0b,
                                        (batchsize, input_width, h))
        
        l_backward_out = l_backward_0c
        
        l_in_1 = LL.ElemwiseSumLayer([l_forward_out, l_backward_out])
                                                                                  
    # Output layers
    network_0a = LL.DenseLayer(l_in_1, num_units=1,
                               num_leading_axes = 2, nonlinearity=nonlin)  
    
    output_net = LL.FlattenLayer(network_0a, outdim = 2)
    
    return output_net
