# -*- coding: utf-8 -*-
"""
Joint training of CoR
"""
import gc
import os
import time
#import itertools

import lasagne
import lasagne.layers as LL
from lasagne.random import set_rng #, get_rng
from lasagne.regularization import regularize_network_params, l2, l1

import numpy as np
import theano
import theano.tensor as T

import matplotlib.pyplot as plt
#from math import log
from RNN_arch import build_rnn_net
from sgcrf import SparseGaussianCRF, inv

theano.config.exception_verbosity = 'low'

feature_dim = 14
time_step = 48
Gaussian_threshold = 10.0
rnd_SEED = 2017

    
def trim(tX, th):
    tX = np.where(tX > th, th, tX)
    tX = np.where(tX < -th, -th, tX)
    return tX
    
def trans_feature(ft):
    ft[:, :, 10] -= 2011
    ft[:, :, 12] += 0.5
    ft[:, :, 13] += 0.5
    return ft
 
# preprocessing data, minus mean and devide variance
def transform(tX, tY):
    m_x = np.mean(tX, axis = (0, 1))
    std_x = np.std(tX, axis = (0, 1))
    tX = (tX - m_x) / std_x
    tX = trim(tX, Gaussian_threshold)
    
    m_y = np.mean(tY)
    std_y = np.std(tY)
    tY = (tY - m_y) / std_y
    tY = trim(tY, Gaussian_threshold)
    
    # cast into float32
    m_x = m_x.astype(np.float32)
    std_x = std_x.astype(np.float32)
    m_y = m_y.astype(np.float32)
    std_y = std_y.astype(np.float32)
    tX = tX.astype(np.float32)
    tY = tY.astype(np.float32)
    
    return tX, tY, m_x, std_x, m_y, std_y
    
def transdata(tX, tY, mx, stdx, my, stdy):
    testX = (tX - mx) / stdx
    testY = (tY - my) / stdy
    
    testX = testX.astype(np.float32)
    testY = testY.astype(np.float32)
    return testX, testY
    
def shuffle_data(tX, tY, cvX, cvY):
    num1 = tX.shape[0]
    idx1 = np.random.permutation(num1)
    tX = tX[idx1]
    tY = tY[idx1]
    
#    num2 = cvX.shape[0]
#    idx2 = np.random.permutation(num2)
#    cvX = cvX[idx2]
#    cvY = cvY[idx2]
    
    return tX, tY, cvX, cvY
    
def retrain(trainX, trainY, testX, testY, theta_mat, lambda_mat, 
                learning_rate = 5e-4,
                rate_decay = 1.0,
                init_scale = 0.2,
                scale_decay = 0.998,
                momentum = 0.0,
                minibatch_size = 64,
                num_epochs = 70,
                rng_seed = 2017,
                model_path = None,
                model_to_save = None):
                    
    if rng_seed is not None:
        print("Setting RandomState with seed=%i" % (rng_seed))
        rng = np.random.RandomState(rng_seed)
        set_rng(rng)
    
    index = T.lscalar() # Minibatch index
    x = T.tensor3('x') # Inputs 
    y = T.fmatrix('y') # Target
    
    #define and initialize RNN network
    network_0 = build_rnn_net(
                        input_var = x,
                        input_width = time_step,
                        input_dim = feature_dim,
                        nin_units = 12,
                        h_num_units = [16, 16],
                        h_grad_clip = 5.0,
                        output_width = time_step
                        )
    if not os.path.isfile(model_path):
        print("Model file does not exist!")
        return None
    init_model = np.load(model_path)
    init_params = init_model[init_model.files[0]]           
    LL.set_all_param_values([network_0], init_params)
    
    train_set_y = theano.shared(np.zeros((1, time_step), dtype=theano.config.floatX),
                                borrow=True) 
    train_set_x = theano.shared(np.zeros((1, time_step, feature_dim), dtype=theano.config.floatX),
                                borrow=True)
    
    valid_set_y = theano.shared(np.zeros((1, time_step), dtype=theano.config.floatX),
                                borrow=True)
    valid_set_x = theano.shared(np.zeros((1, time_step, feature_dim), dtype=theano.config.floatX),
                                borrow=True)
    test_set_x = theano.shared(np.zeros((1, time_step, feature_dim), dtype=theano.config.floatX),
                               borrow=True)
    
    theta = theano.shared(np.zeros((time_step, time_step), dtype = theano.config.floatX))
    lamda = theano.shared(np.zeros((time_step, time_step), dtype = theano.config.floatX))
    
    out_x = LL.BatchNormLayer(network_0)
    
    #define updates
    params = LL.get_all_params(out_x, trainable=True)
    r = lasagne.regularization.regularize_network_params(out_x, l2)
    semi_x = LL.get_output(out_x, deterministic = True)
    
    #define SGCRF in theano expressions
    S_yy = T.dot(y.T, y) / minibatch_size
    S_yx = T.dot(y.T, semi_x) / minibatch_size
    S_xx = T.dot(semi_x.T, semi_x) / minibatch_size
    
    ilamda = T.nlinalg.matrix_inverse(lamda)
    t1 = T.dot(S_yy, lamda)
    t2 = 2 * T.dot(S_yx, theta)
    t3 = T.dot(T.dot(T.dot(ilamda, theta.T), S_xx), theta)
    
    det_lamda = T.nlinalg.det(lamda)
    loss = -T.log(det_lamda) + T.nlinalg.trace(t1 + t2 + t3)
    
    eigen_lamda, _ = T.nlinalg.eig(lamda)
    train_loss = -T.sum(T.log(eigen_lamda)) + T.nlinalg.trace(t1 + t2 + t3)
    
    lamda_diag = T.nlinalg.diag(lamda)
    regularized_loss = loss + 1e-4 * r + 1e-3 * l1(theta) + 1e-3 * l1(lamda - lamda_diag)
    
    learn_rate = T.scalar('learn_rate', dtype=theano.config.floatX)
    momentum = T.scalar('momentum', dtype = theano.config.floatX)
    scale_rate = T.scalar('scale_rate', dtype=theano.config.floatX)
    
    # scale the grads of theta, lamda
    new_params = [theta, lamda]
    new_grads = T.grad(regularized_loss, new_params)
    for i in range(len(new_grads)):
        new_grads[i] *= scale_rate
    grads = T.grad(regularized_loss, params)
    params += new_params
    grads += new_grads
    clipped_grads = lasagne.updates.total_norm_constraint(grads, 5.0)
    updates = lasagne.updates.nesterov_momentum(clipped_grads, params,
                                      learning_rate=learn_rate, momentum = momentum)
    
    pred_x = LL.get_output(out_x, deterministic = True)                               
    valid_predictions = -T.dot(T.dot(ilamda, theta.T), pred_x.T).T
    valid_loss = T.mean(T.abs_(pred_x - y))
                                      
    train_model = theano.function(
        [index, learn_rate, momentum, scale_rate],
        train_loss,
        updates=updates,
        givens={
            x: train_set_x[(index*minibatch_size):
                            ((index+1)*minibatch_size)],
            y: train_set_y[(index*minibatch_size):
                            ((index+1)*minibatch_size)]  
        })
    
    
    validate_model = theano.function(
        [index],
        valid_loss,
        givens={
            x: valid_set_x[index*minibatch_size:
                            (index+1)*minibatch_size],
            y: valid_set_y[index*minibatch_size:
                            (index+1)*minibatch_size]
        })
        
    test_model = theano.function(
        [index],
        valid_predictions,
        givens = {
            x: test_set_x[(index*minibatch_size):
                            ((index+1)*minibatch_size)],
        })
    
    this_train_loss = 0.0
    this_valid_loss = 0.0
    best_valid_loss = np.inf
    best_train_loss = np.inf
    best_test_loss = np.inf
    
    eval_starts = 0
    near_convergence = 1500  # to be set
    eval_multiple = 10
    eval_num = 1000
    train_eval_scores = np.ones(eval_num)
    valid_eval_scores = np.ones(eval_num)
    test_eval_scores = np.ones(eval_num)
    cum_iterations = 0
    eval_index = 0
    
    theta.set_value(theta_mat.astype(np.float32))
    lamda.set_value(lambda_mat.astype(np.float32))
    
    batch_num = trainX.shape[0] // minibatch_size
    near_convergence = batch_num * (num_epochs - 10)
    
    for i in range(num_epochs):
        x_train, y_train, x_cv, y_cv = shuffle_data(trainX, trainY, testX, testY)
        train_batch_num = x_train.shape[0] // minibatch_size #discard last small batch
        valid_batch_num = x_cv.shape[0] // minibatch_size + 1
        start_time = time.time() 
        
        train_set_y.set_value(y_train[:])
        train_set_x.set_value(x_train)
        valid_set_y.set_value(y_cv[:])
        valid_set_x.set_value(x_cv)
        test_set_x.set_value(x_cv)
        
#        if(num_epochs % 10 == 0):
#            learning_rate *= 0.7
        
        # Iterate over minibatches in each batch
        for mini_index in xrange(train_batch_num):
            this_rate = np.float32(learning_rate*(rate_decay**cum_iterations))
            this_scale_rate = np.float32(init_scale*(scale_decay**cum_iterations))
            # adaptive momentum
            this_momentum = 0.99
				
            if cum_iterations > near_convergence:
                this_momentum = 0.90
            
            this_train_loss += train_model(mini_index, this_rate, this_momentum, this_scale_rate)
            cum_iterations += 1
            if np.isnan(this_train_loss):
                print "Training Error!!!!!!!!!"
                return
                # begin evaluation and report loss
            if (cum_iterations % eval_multiple == 0 and cum_iterations > eval_starts):
                this_train_loss = this_train_loss / eval_multiple
                this_valid_loss = np.mean([validate_model(k) for
                                    k in xrange(valid_batch_num)])
                predictions = np.concatenate([test_model(k) for
                                    k in xrange(valid_batch_num)])
                this_test_loss = np.mean(np.abs(predictions - y_cv))
                train_eval_scores[eval_index] = this_train_loss
                valid_eval_scores[eval_index] = this_valid_loss
                test_eval_scores[eval_index] = this_test_loss
                
                # Save model if best validation score
                if (this_valid_loss < best_valid_loss):  
                    best_valid_loss = this_valid_loss
                    
                if (this_test_loss < best_test_loss):
                    best_test_loss = this_test_loss
                    #np.savez(model_to_save, LL.get_all_param_values(network_0))
                    
                print("Training Loss:", this_train_loss)
                print("Validation Loss:", this_valid_loss)
                print("Test Loss:", this_test_loss)
                print("Current scale rate:", this_scale_rate)
                eval_index += 1
                this_train_loss = 0.0
                this_valid_loss = 0.0
                
        end_time = time.time()
        print("Computing time for epoch %d: %f" % (i, end_time-start_time))
        cur_train_loss = np.min(train_eval_scores)
        cur_valid_loss = np.min(valid_eval_scores)
        cur_test_loss = np.min(test_eval_scores)
        print("The best training loss in epoch!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! %f" %cur_train_loss)
        print("The best validation loss in epoch!!!!!!!!!!!!!!!!!!!!!!!!!!!!! : %f" %cur_valid_loss)
        print("The best test loss in epoch!!!!!!!!!!!!!!!!!!!!!!!!!!!!! : %f" %cur_test_loss)
        
    print("Best loss in training: %f" %best_train_loss)
    print("Best loss in cross-validation: %f" %best_valid_loss)
    print("Best loss in testing: %f" %best_test_loss)
    del train_set_x, train_set_y, valid_set_x, valid_set_y, trainX, trainY
    gc.collect()
    
def rnn_reg(feature, gt, minibatch_size = 64, model_path = None):
    """
    Input: normalized feature, model file
    Output: predictions by the trained RNN.
    """
    print("Get the output by the trained RNN model.")
    n_sample = feature.shape[0]
    batch_num = n_sample // minibatch_size + 1
    
    #define theano variables
    index = T.lscalar()
    x = T.tensor3('x')
    
    #define and initialize RNN network
    network_0 = build_rnn_net(
                        input_var = x,
                        input_width = time_step,
                        input_dim = feature_dim,
                        nin_units = 12,
                        h_num_units = [16, 16],
                        h_grad_clip = 1.0,
                        output_width = time_step
                        )
    if not os.path.isfile(model_path):
        print("Model file does not exist!")
        return None
    init_model = np.load(model_path)
    init_params = init_model[init_model.files[0]]           
    LL.set_all_param_values([network_0], init_params)
    
    #import data to theano variable
    output_x = theano.shared(np.zeros((1,1,1), dtype=theano.config.floatX),
                               borrow=True)
    output_x.set_value(feature)
    
    #define prediction model
    prediction_0 = LL.get_output(network_0, deterministic=True)
    output_fn = theano.function(
        [index],
        prediction_0,
        givens = {
            x: output_x[(index*minibatch_size):
                            ((index+1)*minibatch_size)],
        })
    
    # predicting...
    predictions = np.concatenate([output_fn(i) for i in xrange(batch_num)])
    
    err = np.mean(np.abs(predictions-gt))
    print("rnn_MAE:", err)
    
    return predictions
    
def train_sgcrf(feature, gt, n_iter):
    """
    Input: normalized feature and gt, number of iterations.
    Output: SGCRF model 
    """
    print("Train SGCRF based on the output of RNN.")
    model = SparseGaussianCRF(learning_rate = 0.1, 
                              lamL = 0.01, lamT = 0.001, n_iter = n_iter)
    model.fit(feature, gt)
    return model
    
def evaluate(feature, gt, model):
    """
    Input: normalized feature and gt of test set, trained model
    Output: ...
    """
    Y = model.predict(feature)
    err = np.mean(np.abs(Y-gt))
    print("combined MAE:", err)
    
def integrate_model(feature_train, gt_train, feature_test, gt_test, rnn_model):
    
    trainX, trainY, testX, testY = feature_train, gt_train, feature_test, gt_test   
    trainX, trainY, mx, stdx, my, stdy = transform(trainX, trainY)
    testX, testY = transdata(testX, testY, mx, stdx, my, stdy)
    trainX, trainY, testX, testY = shuffle_data(trainX, trainY, testX, testY)
    
    rnn_predictions = rnn_reg(trainX, trainY, 100, rnn_model)
    combined_model = train_sgcrf(rnn_predictions, trainY, 1000)
    rnn_test_predictions = rnn_reg(testX, testY, 100, rnn_model)
    evaluate(rnn_test_predictions, testY, combined_model)
    
    theta = combined_model.Theta
    lamda = combined_model.Lam
 
    model_to_save = rnn_model
    retrain(trainX, trainY, testX, testY, theta, lamda, 
            model_path = rnn_model, 
            model_to_save = model_to_save)
                    
       
def main(InputDirectory, model):
        train_feature_file = os.path.join(InputDirectory, "train_feature.npy")
        train_gt_file = os.path.join(InputDirectory, "train_gt.npy")
        test_feature_file = os.path.join(InputDirectory, "test_feature.npy")
        test_gt_file = os.path.join(InputDirectory, "test_gt.npy")
        
        feature_train = np.load(train_feature_file)
        gt_train = np.load(train_gt_file)
        feature_test = np.load(test_feature_file)
        gt_test = np.load(test_gt_file)
        
        feature_train = trans_feature(feature_train)
        feature_test = trans_feature(feature_test)
        integrate_model(feature_train, gt_train, feature_test, gt_test, model)

if __name__ == '__main__':
    current_dir = os.getcwd()
    par_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(par_dir, 'data')
    model_file = os.path.join(par_dir, 'model\\model.npz')
    args = {'InputDirectory' : data_dir,
    'model' : model_file
    }
    main(**args)
         
            
                
            
            
            











