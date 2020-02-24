import os, time
import argparse
import logging
import random
import string
import numpy as np
import mxnet as mx
from mxnet import gluon
from data import MovieLens
from model import GCMCLayer, BiDecoder
from utils import get_activation, parse_ctx, gluon_net_info, gluon_total_param_num, \
                  params_clip_global_norm, MetricLogger
from mxnet.gluon import Block
import dgl.function as fn
from collections import OrderedDict
import dgl.ndarray as dglnd
from dgl.heterograph import AdaptedHeteroGraph
import dgl.runtime.spmv as spmv
from dgl.kernel import tvm_enabled

def zerocopy_to_dgl_ndarray(arr):
    return dglnd.from_dlpack(arr.to_dlpack_for_read())

def zerocopy_to_dgl_ndarray_for_write(arr):
    return dglnd.from_dlpack(arr.to_dlpack_for_write())
class MyContext:
    def __init__(self):
        self.device_type =1
        self.device_id=0
def train(args):
    print(args.ctx)
    print(args)
    dataset = MovieLens(args.data_name, args.ctx, use_one_hot_fea=args.use_one_hot_fea, symm=args.gcn_agg_norm_symm,
                        test_ratio=args.data_test_ratio, valid_ratio=args.data_valid_ratio)
    print("Loading data finished ...\n")
    graph=dataset.train_enc_graph
    msg_in_unit = args.gcn_agg_units // len(dataset.possible_rating_values)
    movie_in_unit = dataset.movie_feature_shape[1]
    users_in_unit = dataset.user_feature_shape[1]
    print((movie_in_unit, users_in_unit, msg_in_unit))
    allfeat = [mx.ndarray.random.randn(movie_in_unit, msg_in_unit,dtype=np.float32) for i in dataset.possible_rating_values]
    num_u = graph.number_of_nodes('user')
    
    tvm_enabled = args.tvm != 0 

    def do_copy_reduce():
        funcs=OrderedDict()
        for i, rating in enumerate(dataset.possible_rating_values):
            rating = str(rating)
            graph.nodes['movie'].data['h%d' % i] = allfeat[i]
            #funcs[rating] = (fn.copy_u('h%d' % i, 'm'), fn.sum('m', 'h'))
            funcs['rev-%s' % rating] = (fn.copy_u('h%d' % i, 'm'), fn.sum('m', 'h'))
            # message passing
        graph.multi_update_all(funcs, "stack")
        #graph.nodes['user'].data.pop('h')
        return graph.nodes['user'].data.pop('h').reshape(num_u, -1)
        #mx.nd.save()
    #time.sleep(20)
    #(943, 75) (1682, 75) (72000,) (72000,) (943, 75)
    def do_copy_reduce_bwd():
        X = zerocopy_to_dgl_ndarray(allfeat[0])
        import dgl
        out = dgl.ndarray.empty((users_in_unit, msg_in_unit))
        grad_out = zerocopy_to_dgl_ndarray(mx.ndarray.random.randn(users_in_unit, msg_in_unit,dtype=np.float32))
        outgrad_x=mx.ndarray.zeros((movie_in_unit, msg_in_unit),dtype=np.float32)
        grad_x = zerocopy_to_dgl_ndarray_for_write(outgrad_x)
        etid = graph.get_etype_id('rev-1')
        stid, dtid = getattr(graph,"_graph").metagraph.find_edge(etid)
        gidx=AdaptedHeteroGraph(graph, stid, dtid, etid).get_immutable_gidx(MyContext())
        print(type(grad_out))
        dgl.kernel.backward_copy_reduce("sum", gidx, 0, X, out, grad_out, grad_x)
        print(outgrad_x.shape)
        return outgrad_x

    dot_lhs = mx.ndarray.random.randn(users_in_unit, 75,dtype=np.float32)
    dot_rhs = mx.ndarray.random.randn(movie_in_unit, 75,dtype=np.float32)
    dot_out = mx.ndarray.random.randn(72000,dtype=np.float32)
    dot_outgrad = mx.ndarray.random.randn(72000,dtype=np.float32)
    dot_lhsgrad = mx.ndarray.zeros((users_in_unit, 75),dtype=np.float32)
    dot_rhsgrad = mx.ndarray.zeros((movie_in_unit, 75),dtype=np.float32)
    def do_binary_op_dot_bwd(islhs):
        import dgl
        A = zerocopy_to_dgl_ndarray(dot_lhs)
        B = zerocopy_to_dgl_ndarray(dot_rhs)
        out = zerocopy_to_dgl_ndarray(dot_out)
        grad_out = zerocopy_to_dgl_ndarray(dot_outgrad)

        G = graph.local_var()
        etid = 0
        stid, dtid = getattr(G,"_graph").metagraph.find_edge(etid)
        gidx=AdaptedHeteroGraph(graph, stid, dtid, etid).get_immutable_gidx(MyContext())
        if islhs:
            grad_A = zerocopy_to_dgl_ndarray_for_write(dot_lhsgrad)
            dgl.kernel.backward_lhs_binary_op_reduce("none", "dot", gidx, 0, 1, A, B, out, grad_out, grad_A)
            return dot_lhsgrad
        else:
            grad_B = zerocopy_to_dgl_ndarray_for_write(dot_rhsgrad)
            dgl.kernel.backward_rhs_binary_op_reduce("none", "dot", gidx, 0, 1, A, B, out, grad_out, grad_B)
            return dot_rhsgrad
                

    workloads = {
        'copyreduce': do_copy_reduce,
        'copyreduce_bwd': do_copy_reduce_bwd,
        'binary_dot_bwd_lhs': lambda:do_binary_op_dot_bwd(True),
        'binary_dot_bwd_rhs': lambda:do_binary_op_dot_bwd(False),
    }
    if args.mode == "save":
        mx.nd.save(args.workload + ".mxnd",workloads[args.workload]())
    elif args.mode == "compare":
        r = workloads[args.workload]()
        loaded = mx.nd.load(args.workload + ".mxnd")[0]
        print(loaded)
        print(r.shape, loaded.shape)
        #print(mx.test_utils.almost_equal(r.asnumpy(),loaded.asnumpy()))
        for idx, row in enumerate(r):
            lrow = loaded[idx]
            for j in range(len(row)):
                r=row[j].asscalar()
                l=lrow[j].asscalar()
                if abs((r-l)/ (r+l + 1e-11)) > 1e-3:
                    print(idx, j, r, l)
    else:
        for i in range(3):
            do_copy_reduce()
        t0 = time.time()
        for i in range(200):
            do_copy_reduce()
        print(time.time()-t0)
        print("DONE")


def config():
    parser = argparse.ArgumentParser(description='Run the baseline method.')

    parser.add_argument('--tvm', default=0, type=int)
    parser.add_argument('--workload', default="copyreduce", type=str)
    parser.add_argument('--mode', default="benchmark", type=str)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--ctx', dest='ctx', default='gpu0', type=str,
                        help='Running Context. E.g `--ctx gpu` or `--ctx gpu0,gpu1` or `--ctx cpu`')
    parser.add_argument('--save_dir', type=str, help='The saving directory')
    parser.add_argument('--save_id', type=int, help='The saving log id')
    parser.add_argument('--silent', action='store_true')

    parser.add_argument('--data_name', default='ml-1m', type=str,
                        help='The dataset name: ml-100k, ml-1m, ml-10m')
    parser.add_argument('--data_test_ratio', type=float, default=0.1) ## for ml-100k the test ration is 0.2
    parser.add_argument('--data_valid_ratio', type=float, default=0.1)
    parser.add_argument('--use_one_hot_fea', action='store_true', default=False)

    #parser.add_argument('--model_remove_rating', type=bool, default=False)
    parser.add_argument('--model_activation', type=str, default="leaky")

    parser.add_argument('--gcn_dropout', type=float, default=0.7)
    parser.add_argument('--gcn_agg_norm_symm', type=bool, default=True)
    parser.add_argument('--gcn_agg_units', type=int, default=500)
    parser.add_argument('--gcn_agg_accum', type=str, default="sum")
    parser.add_argument('--gcn_out_units', type=int, default=75)

    parser.add_argument('--gen_r_num_basis_func', type=int, default=2)

    # parser.add_argument('--train_rating_batch_size', type=int, default=10000)
    parser.add_argument('--train_max_iter', type=int, default=2000)
    parser.add_argument('--train_log_interval', type=int, default=1)
    parser.add_argument('--train_valid_interval', type=int, default=1)
    parser.add_argument('--train_optimizer', type=str, default="adam")
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--train_min_lr', type=float, default=0.001)
    parser.add_argument('--train_lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--train_decay_patience', type=int, default=50)
    parser.add_argument('--train_early_stopping_patience', type=int, default=100)
    parser.add_argument('--share_param', default=False, action='store_true')

    args = parser.parse_args()
    args.ctx = parse_ctx(args.ctx)[0]


    ### configure save_fir to save all the info
    if args.save_dir is None:
        args.save_dir = args.data_name+"_" + ''.join(random.sample(string.ascii_uppercase + string.digits, k=2))
    if args.save_id is None:
        args.save_id = np.random.randint(20)
    args.save_dir = os.path.join("log", args.save_dir)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    return args


if __name__ == '__main__':
    args = config()
    np.random.seed(args.seed)
    mx.random.seed(args.seed, args.ctx)
    train(args)