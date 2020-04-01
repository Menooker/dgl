import tvm
import topi
#import numpy
import tvm.relay
import dgl

dtype = "float32"

# using Intel AVX2(Advanced Vector Extensions) ISA for SIMD
# To get the best performance, please change the following line
# to llvm -mcpu=core-avx2, or specific type of CPU you use
target = 'llvm -mcpu=skylake-avx512'#'llvm -mcpu=core-avx2'
#avx2-> len(ymm)=256 -> 8 floats
#avx512-> len(zmm)=512 -> 16 floats
simd_size = 16 
ctx = tvm.context(target, 0)

def gen_csr_iterate(irb: tvm.ir_builder.IRBuilder, indices, indptr, parallel, functor, **buffers):
        """define ir for csrmm"""
        indices_ptr = irb.buffer_ptr(indices)
        indptr_ptr = irb.buffer_ptr(indptr)
        buffer_ptrs = dict()
        for (k,v) in buffers.items():
            buffer_ptrs[k] = irb.buffer_ptr(v)

        M = topi.util.simplify(indptr.shape[0]-1)
        with irb.for_range(0, M, name='row', for_type="parallel" if parallel else "serial") as row:
            row_start = indptr_ptr[row]
            row_end = indptr_ptr[row+1]
            row_len = row_end - row_start
            with irb.for_range(0, row_len, name='idx') as idx:
                eid = topi.util.simplify(idx+row_start)
                dst = topi.util.simplify(indices_ptr[eid])
                functor(irb, row, dst, eid, **buffer_ptrs)
        #return irb.get()

def gen_zero_out_tensor(irb, tensor):
    outptr = irb.buffer_ptr(tensor)
    out_len = 1
    for i in tensor.shape:
        out_len*=i
    with irb.for_range(0, out_len, name="i") as i:
        outptr[i] = 0.

def gen_vectorized_for_loop(irb, length, blkSize, fcompute):
    with irb.for_range(0, tvm.floordiv(length, blkSize), name='v.outer') as outer: #for_type="vectorize"
        with irb.for_range(0, blkSize, name='v.inner', for_type="vectorize") as inner: #
            fcompute(outer * blkSize + inner)
    with irb.for_range(0, tvm.floormod(length, blkSize), name='v') as i:
            fcompute(length - tvm.floormod(length, blkSize) + i)
    '''with irb.for_range(0, length, name='v') as i:
            fcompute(i)'''

def gen_copy_reduce_sum(isfwd):
    indptrN = tvm.var('indptrN')
    indicesN = tvm.var('indicesN')
    outN = tvm.var('outN')
    inN = tvm.var('inN')
    x_len = tvm.var('x_len')
    indices = tvm.placeholder((indicesN,), name='indices', dtype=tvm.int32)
    indptr = tvm.placeholder((indptrN,), name='indptr', dtype=tvm.int32)
    inbuf = tvm.placeholder((inN, x_len), name='inbuf', dtype=tvm.float32)
    #outbuf = tvm.placeholder((outN, x_len), name='outbuf')
    def gen(ins, outs):
        irb = tvm.ir_builder.create()
        outptr = irb.buffer_ptr(outs[0])
        gen_zero_out_tensor(irb, outs[0])
        block_size = 32
        x_len_s = topi.util.simplify(x_len)
        '''with irb.for_range(0, tvm.floordiv(x_len_s + (block_size - 1), block_size), for_type="parallel" ,name='blkIdx') as blkIdx:
            def workload(irb, src, dst, eid, inptr):
                with irb.for_range(0, blkIdx * block_size, name='i') as i: #for_type="vectorize"
                    with irb.if_scope(irb.likely(blkIdx * block_size + i < x_len_s)) :
                        if isfwd:
                            outptr[dst * x_len_s + blkIdx * block_size + i] += inptr[src * x_len_s + blkIdx * block_size + i]
                        else:
                            outptr[src * x_len_s + blkIdx * block_size + i] += inptr[dst * x_len_s + blkIdx * block_size + i]
            gen_csr_iterate(irb, ins[0], ins[1], False, workload, inptr = ins[2])'''
        def for_each_edge(irb, src, dst, eid, inptr):
            def assign(idx):
                if isfwd:
                    outptr[dst * x_len_s + idx] += inptr[src * x_len_s + idx]
                else:
                    outptr[src * x_len_s + idx] += inptr[dst * x_len_s + idx]
            gen_vectorized_for_loop(irb, x_len_s, simd_size, assign)               
        gen_csr_iterate(irb, ins[0], ins[1], not isfwd, for_each_edge, inptr = ins[2])
        '''def workload(irb, src, dst, eid, inptr):
            blkSize=16
            #with irb.for_range(0, tvm.floordiv(x_len_s, blkSize), name='x_len.outer') as outer: #for_type="vectorize"
            with irb.for_range(0, x_len_s, name='x_len.inner') as inner: #
                    if isfwd:
                        outptr[dst * x_len_s + inner] += inptr[src * x_len_s + inner]
                    else:
                        outptr[src * x_len_s + inner] += inptr[dst * x_len_s + inner]
        gen_csr_iterate(irb, ins[0], ins[1], True, workload, inptr = ins[2])'''
        return irb.get()
    C = tvm.extern((outN, x_len),[indices, indptr, inbuf], gen, dtype=tvm.float32, name = "C")
    return C,indices,indptr,inbuf


def get_copy_reduce_sum(isfwd):
    C,indices,indptr,inbuf = gen_copy_reduce_sum(isfwd)
    # Default schedule
    s = tvm.create_schedule(C.op)
    #print(tvm.lower(s, [indices,indptr,inbuf, C], simple_mode=True))
    func = tvm.build(s, [indices,indptr,inbuf, C], target=target, name='copy_reduce_sum_' + "fwd" if isfwd else "bwd")
    def call(*args):
        targs=[tvm.nd.from_dlpack(arg.to_dlpack()) if isinstance(arg, dgl.ndarray.NDArray) else arg for arg in args]
        return func(*targs)
    return call

def gen_binary_op_dot_bwd_lhs(islhs):
    indptrN = tvm.var('indptrN')
    indicesN = tvm.var('indicesN')
    rhsDataN = tvm.var('rhsDataN')
    gradoutDataN = tvm.var('gradoutDataN')
    lhsgradoutDataN = tvm.var('lhsgradoutDataN')
    #xLen = tvm.var('xLen')
    xLen = 1
    #fix-me: we eliminated x_len dimension here
    dataLen = tvm.var('dataLen')
    indices = tvm.placeholder((indicesN,), name='indices', dtype=tvm.int32)
    indptr = tvm.placeholder((indptrN,), name='indptr', dtype=tvm.int32)
    rhsData = tvm.placeholder((rhsDataN, dataLen), name='rhsData', dtype=tvm.float32)
    gradoutData = tvm.placeholder((gradoutDataN, ), name='gradoutData', dtype=tvm.float32)
    outMapping = tvm.placeholder((gradoutDataN, ), name='outMapping', dtype=tvm.int32)
    #lhsgradoutData = tvm.placeholder((lhsgradoutDataN, xLen, dataLen), name='lhsgradoutData', dtype=tvm.float32)

    def gen_func(ins, outs):
        irb = tvm.ir_builder.create()
        gen_zero_out_tensor(irb, outs[0])
        indices, indptr, rhsData, gradoutData, outMapping = ins[0], ins[1], ins[2], ins[3], ins[4]
        #with irb.for_range(0, xLen, name='i') as i:
        def for_each_edge(irb, src, dst, eid, rhsDataPtr, gradoutDataPtr, lhsgradoutDataPtr, outMappingPtr):
            lhsIdx = topi.util.simplify(src * dataLen)
            outIdx = topi.util.simplify(outMappingPtr[eid])
            rhsIdx = topi.util.simplify(dst * dataLen)
            grad = gradoutDataPtr[outIdx]
            def fcompute(j):
                    if islhs:
                        lhsgradoutDataPtr[lhsIdx + j] += grad * rhsDataPtr[rhsIdx +j]
                    else:
                        lhsgradoutDataPtr[rhsIdx + j] += grad * rhsDataPtr[lhsIdx +j]
            gen_vectorized_for_loop(irb, dataLen, simd_size, fcompute)
        gen_csr_iterate(irb, indices, indptr, islhs, for_each_edge, rhsDataPtr = rhsData, gradoutDataPtr = gradoutData, lhsgradoutDataPtr= outs[0], outMappingPtr = outMapping )
        return irb.get()
    #outbuf = tvm.placeholder((outN, x_len), name='outbuf')

    
    C = tvm.extern((lhsgradoutDataN, dataLen),[indices, indptr, rhsData, gradoutData, outMapping],
        gen_func,
        dtype=tvm.float32, name = "lhsgradoutData"
    )
    return C,indices,indptr,rhsData, gradoutData, outMapping

def get_binary_op_dot_bwd(islhs):
    C,indices,indptr,rhsData, gradoutData, outMapping = gen_binary_op_dot_bwd_lhs(islhs)
    # Default schedule
    s = tvm.create_schedule(C.op)
    #print(tvm.lower(s, [indices,indptr,rhsData, gradoutData, outMapping, C], simple_mode=True))
    func = tvm.build(s, [indices,indptr,rhsData, gradoutData, outMapping, C], target=target, name='binary_op_dot_bwd_' + "lhs" if islhs else "rhs")
    def call(*args):
        targs=[tvm.nd.from_dlpack(arg.to_dlpack()) if isinstance(arg, dgl.ndarray.NDArray) else arg for arg in args]
        return func(*targs)
    return call

copy_reduce_sum = get_copy_reduce_sum(True)
copy_reduce_sum_bwd = get_copy_reduce_sum(False)
binary_op_dot_bwd_lhs = get_binary_op_dot_bwd(True)
binary_op_dot_bwd_rhs = get_binary_op_dot_bwd(False)
