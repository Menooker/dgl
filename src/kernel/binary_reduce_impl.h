/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/binary_reduce_impl.h
 * \brief Implementations of binary reduce operations.
 */
#ifndef DGL_KERNEL_BINARY_REDUCE_IMPL_H_
#define DGL_KERNEL_BINARY_REDUCE_IMPL_H_

#include <minigun/minigun.h>
#include <dgl/runtime/device_api.h>

#include <algorithm>
#include <string>

#ifdef __CUDACC__
#include "../runtime/cuda/cuda_common.h"
#endif
#include "./binary_reduce.h"
#include "./binary_reduce_impl_decl.h"
#include "./csr_interface.h"
#include "./utils.h"

namespace dgl {
namespace kernel {

///////////////////////////////////////////////////////////////////////////////
// BinaryReduce device-agnostic implementation
///////////////////////////////////////////////////////////////////////////////

template <int XPU, typename Idx, typename DType, typename Reducer>
GData<Idx, DType> AllocGData(const std::string& op,
    const DLContext& ctx, int64_t x_len,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping,
    runtime::NDArray lhs_data, runtime::NDArray rhs_data,
    runtime::NDArray out_mapping, runtime::NDArray out_data) {
  // GData
  GData<Idx, DType> gdata;
  gdata.x_length = x_len;
  gdata.lhs_data = static_cast<DType*>(lhs_data->data);
  gdata.rhs_data = static_cast<DType*>(rhs_data->data);
  gdata.out_data = static_cast<DType*>(out_data->data);
  if (!utils::IsNoneArray(lhs_mapping)) {
    gdata.lhs_mapping = static_cast<Idx*>(lhs_mapping->data);
  }
  if (!utils::IsNoneArray(rhs_mapping)) {
    gdata.rhs_mapping = static_cast<Idx*>(rhs_mapping->data);
  }
  if (!utils::IsNoneArray(out_mapping)) {
    gdata.out_mapping = static_cast<Idx*>(out_mapping->data);
  }

  // for dot operation: vector [dot] vector
  if (op == binary_op::kDot) {
    // get size of vector
    gdata.data_len = lhs_data->shape[lhs_data->ndim - 1];
  } else {
    gdata.data_len = 1;
  }

  // fill out data with zero values
  utils::Fill<XPU>(ctx, gdata.out_data, utils::NElements(out_data), Zero<Reducer>::value);
  return gdata;
}

inline int floor_div(int a, int b)
{
  return (a+b-1)/b;
}

__attribute__((optimize("unroll-loops")))
inline void CopyReduceFloat(const CSRWrapper& graph,
    runtime::NDArray lhs_data,
    runtime::NDArray out_data,
    runtime::NDArray lhs_mapping,
    runtime::NDArray out_mapping,
    int32_t x_len,
    bool reverse)
{
      float* mylhs_data = static_cast<float*>(lhs_data->data);
      float* myout_data = static_cast<float*>(out_data->data);
      int32_t* mylhs_mapping = utils::IsNoneArray(lhs_mapping)? nullptr:static_cast<int32_t*>(lhs_mapping->data);
      int32_t* myout_mapping =utils::IsNoneArray(out_mapping) ? nullptr: static_cast<int32_t*>(out_mapping->data);
      assert(!mylhs_mapping);
      assert(!myout_mapping);
      memset(myout_data, 0, utils::NElements(out_data)* sizeof(float));
      auto outcsr = graph.GetOutCSRMatrix();
      // row_offsets ->indptr
      typedef int32_t Idx;
      int32_t N = outcsr.indptr->shape[0] - 1;
      Idx* row_offsets = static_cast<Idx*>(outcsr.indptr->data);
      Idx* column_indices = static_cast<Idx*>(outcsr.indices->data);
      const int BLOCK_SIZE = 32;
      const int total=floor_div(x_len, BLOCK_SIZE);
      
      #pragma omp parallel for
      for(int t=0; t<total;t++)
      { 
        //Idx start = 0;
        //Idx end = x_len;
        for (Idx vid = 0; vid < N; ++vid) {
            Idx start = BLOCK_SIZE*t;
            Idx end = (t==total-1) ? x_len : BLOCK_SIZE*(t+1);
            Idx src = vid;
            const Idx row_start = row_offsets[src];
            const Idx row_end = row_offsets[src + 1];
            for (Idx eid = row_start; eid < row_end; ++eid) {
              const Idx dst = column_indices[eid];
              //Idx srcIdx = mylhs_mapping ?  mylhs_mapping[src]:src;
              //Idx outIdx = myout_mapping ? myout_mapping[dst]:dst;
              Idx srcIdx = reverse ? dst :src;
              Idx outIdx = reverse ? src :dst;
              for(Idx i=start;i<end;i++)
              {
                //#pragma omp atomic
                myout_data[outIdx * x_len + i] += mylhs_data[srcIdx * x_len + i];
              }
            }
        }
      }
}

template <int XPU>
void BinaryReduceImpl(
    const std::string& reducer,
    const std::string& op,
    const CSRWrapper& graph,
    binary_op::Target lhs, binary_op::Target rhs,
    runtime::NDArray lhs_data, runtime::NDArray rhs_data,
    runtime::NDArray out_data,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping,
    runtime::NDArray out_mapping) {
  using runtime::NDArray;
  using minigun::Csr;
  // device
#ifdef __CUDACC__
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
#endif
  const int64_t x_len = utils::ComputeXLength(out_data);

  // advance config
  minigun::advance::RuntimeConfig rtcfg;
  rtcfg.ctx = out_data->ctx;
#ifdef __CUDACC__
  rtcfg.stream = thr_entry->stream;
  const int nt = utils::FindNumThreads(x_len, 64);
  rtcfg.data_num_threads = nt;
  // XXX(minjie): hard-code to let each thread compute two elements to increase
  //              instruction level parallelism
  rtcfg.data_num_blocks = (x_len + (nt * 2) - 1) / (nt * 2);
#endif
  if (reducer == binary_op::kReduceMean) {
    // TODO(minjie): divide
    LOG(FATAL) << "reduce mean is not supported.";
  }
  const DLDataType& dtype = out_data->dtype;
  const auto bits = graph.NumBits();
  if (XPU == kDLCPU) {
    if (bits== 32 && reducer == binary_op::kReduceSum 
      && op == binary_op::kUseLhs && dtype.code==kDLFloat
      && dtype.bits==32 && lhs == binary_op::Target::kSrc){
        CopyReduceFloat(graph, lhs_data, out_data, lhs_mapping, out_mapping, x_len, false);
        return;
    }
  }
  DGL_DTYPE_SWITCH(dtype, DType, {
    DGL_IDX_TYPE_SWITCH(bits, Idx, {
      REDUCER_SWITCH(reducer, XPU, DType, Reducer, {
        auto gdata = AllocGData<XPU, Idx, DType, Reducer>(op,
            rtcfg.ctx, x_len, lhs_mapping, rhs_mapping,
            lhs_data, rhs_data, out_mapping, out_data);
        OP_TARGET_SWITCH(op, lhs, rhs, DType, BinaryOp, LeftTarget, RightTarget, {
          CallBinaryReduce<XPU, Idx, DType, LeftTarget,
            RightTarget, BinaryOp, Reducer>(rtcfg, graph, &gdata);
        });
      });
    });
  });
}

///////////////////////////////////////////////////////////////////////////////
// BackwardBinaryReduce device-agnostic implementation
///////////////////////////////////////////////////////////////////////////////

template <int XPU, typename Idx, typename DType>
BackwardGData<Idx, DType> AllocBackwardGData(
    const std::string& op, const DLContext& ctx, int64_t x_len,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping, runtime::NDArray out_mapping,
    runtime::NDArray lhs_data, runtime::NDArray rhs_data, runtime::NDArray out_data,
    runtime::NDArray grad_out_data,
    runtime::NDArray grad_lhs_data, runtime::NDArray grad_rhs_data) {
  // GData
  BackwardGData<Idx, DType> gdata;
  gdata.x_length = x_len;
  gdata.lhs_data = static_cast<DType*>(lhs_data->data);
  gdata.rhs_data = static_cast<DType*>(rhs_data->data);
  gdata.out_data = static_cast<DType*>(out_data->data);
  gdata.grad_out_data = static_cast<DType*>(grad_out_data->data);
  if (!utils::IsNoneArray(grad_lhs_data)) {
    gdata.grad_lhs_data = static_cast<DType*>(grad_lhs_data->data);
    // fill out data with zero values
    utils::Fill<XPU>(ctx, gdata.grad_lhs_data, utils::NElements(grad_lhs_data),
                static_cast<DType>(0));
  }
  if (!utils::IsNoneArray(grad_rhs_data)) {
    gdata.grad_rhs_data = static_cast<DType*>(grad_rhs_data->data);
    // fill out data with zero values
    utils::Fill<XPU>(ctx, gdata.grad_rhs_data, utils::NElements(grad_rhs_data),
                static_cast<DType>(0));
  }
  if (!utils::IsNoneArray(lhs_mapping)) {
    gdata.lhs_mapping = static_cast<Idx*>(lhs_mapping->data);
  }
  if (!utils::IsNoneArray(rhs_mapping)) {
    gdata.rhs_mapping = static_cast<Idx*>(rhs_mapping->data);
  }
  if (!utils::IsNoneArray(out_mapping)) {
    gdata.out_mapping = static_cast<Idx*>(out_mapping->data);
  }

  // for dot operation: vector [dot] vector
  if (op == binary_op::kDot) {
    // get size of vector
    gdata.data_len = lhs_data->shape[lhs_data->ndim - 1];
  } else {
    gdata.data_len = 1;
  }
  return gdata;
}

// Left=Src Right=Dst
__attribute__((optimize("unroll-loops")))
inline void DotLeftBwdFloat(const CSRWrapper& graph,
    runtime::NDArray rhs_data,
    runtime::NDArray gradout_data,
    runtime::NDArray lhsgradout_data,
    runtime::NDArray lhs_mapping,
    runtime::NDArray rhs_mapping,
    int32_t x_len,
    int32_t data_len,
    bool reverse) {
    
      //the rhs_data feature is [x_len * data_len] matrix
      float* myrhs_data = static_cast<float*>(rhs_data->data);
      float* mygradout_data = static_cast<float*>(gradout_data->data);
      float* mylhsgradout_data = static_cast<float*>(lhsgradout_data->data);
      int32_t* mylhs_mapping = utils::IsNoneArray(lhs_mapping)? nullptr:static_cast<int32_t*>(lhs_mapping->data);
      int32_t* myrhs_mapping = utils::IsNoneArray(rhs_mapping)? nullptr:static_cast<int32_t*>(rhs_mapping->data);
      assert(!mylhs_mapping);
      assert(!myrhs_mapping);
      memset(mylhsgradout_data, 0, utils::NElements(lhsgradout_data)* sizeof(float));
      auto outcsr = graph.GetOutCSRMatrix();
      int32_t* myout_mapping = static_cast<int32_t*>(outcsr.data->data);
      // row_offsets ->indptr
      typedef int32_t Idx;
      int32_t N = outcsr.indptr->shape[0] - 1;
      Idx* row_offsets = static_cast<Idx*>(outcsr.indptr->data);
      Idx* column_indices = static_cast<Idx*>(outcsr.indices->data);
      
      #pragma omp parallel for
      for(Idx i=0;i<x_len;i++){
        for (Idx vid = 0; vid < N; ++vid) {
            Idx src = vid;
            const Idx row_start = row_offsets[src];
            const Idx row_end = row_offsets[src + 1];
            for (Idx eid = row_start; eid < row_end; ++eid) {
              const Idx dst = column_indices[eid];
              const Idx outIdx = myout_mapping[eid];
              //Idx srcIdx = mylhs_mapping ?  mylhs_mapping[src]:src;
              //Idx outIdx = myout_mapping ? myout_mapping[dst]:dst;
              Idx rIdx = reverse ? src:dst;
              Idx lIdx = reverse ? dst:src;
              float* out = mylhsgradout_data + lIdx * (x_len * data_len);
              float* grads = mygradout_data + outIdx * (x_len);
              float* rhs = myrhs_data + rIdx * (x_len * data_len);
              {
                float outgrad=grads[i];
                for (Idx j=0;j<data_len;j++)
                  out[i * data_len + j] += outgrad * rhs[i * data_len + j];
              }
            }
        }
      }
      

}

template <int XPU>
void BackwardBinaryReduceImpl(
    const std::string& reducer,
    const std::string& op,
    const CSRWrapper& graph,
    binary_op::Target lhs, binary_op::Target rhs,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping, runtime::NDArray out_mapping,
    runtime::NDArray lhs_data, runtime::NDArray rhs_data, runtime::NDArray out_data,
    runtime::NDArray grad_out_data,
    runtime::NDArray grad_lhs_data, runtime::NDArray grad_rhs_data) {
  using runtime::NDArray;
  using minigun::Csr;
#ifdef __CUDACC__
  // device
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
#endif
  // Graph
  const int64_t x_len = utils::ComputeXLength(out_data);
  // advance config
  minigun::advance::RuntimeConfig rtcfg;
  rtcfg.ctx = out_data->ctx;
#ifdef __CUDACC__
  rtcfg.stream = thr_entry->stream;
  const int nt = utils::FindNumThreads(x_len, 64);
  rtcfg.data_num_threads = nt;
  // XXX(minjie): hard-code to let each thread compute two elements to increase
  //              instruction level parallelism
  rtcfg.data_num_blocks = (x_len + (nt * 2) - 1) / (nt * 2);
#endif

  const DLDataType& dtype = out_data->dtype;
  const bool req_lhs = !utils::IsNoneArray(grad_lhs_data);
  const bool req_rhs = !utils::IsNoneArray(grad_rhs_data);
  const auto bits = graph.NumBits();

  if (reducer == binary_op::kReduceMean) {
    // TODO(minjie): divide
    LOG(FATAL) << "reduce mean is not supported.";
  }
  if (XPU == kDLCPU && bits== 32) {
    if (reducer == binary_op::kReduceSum 
      && op == binary_op::kUseLhs && dtype.code==kDLFloat
      && dtype.bits==32 && lhs == binary_op::Target::kSrc){
        CopyReduceFloat(graph, grad_out_data, grad_lhs_data, out_mapping, lhs_mapping, x_len, true);
        return;
    }
    if (reducer == binary_op::kReduceNone 
      && op == binary_op::kDot && dtype.code==kDLFloat
      && dtype.bits==32 && lhs == binary_op::Target::kSrc
      && rhs == binary_op::Target::kDst){
        int32_t data_len = lhs_data->shape[lhs_data->ndim - 1];
        if (req_lhs) {
          DotLeftBwdFloat(graph, rhs_data, grad_out_data, grad_lhs_data, lhs_mapping, rhs_mapping, 
            x_len, data_len, false);
          return;
        } else {
          DotLeftBwdFloat(graph, lhs_data, grad_out_data, grad_rhs_data, rhs_mapping, lhs_mapping, 
            x_len, data_len, true);
          return;
        }
        //CopyReduceFloat(graph, grad_out_data, grad_lhs_data, out_mapping, lhs_mapping, x_len, true);
        //return;
    }
  }

  DGL_DTYPE_SWITCH(dtype, DType, {
    DGL_IDX_TYPE_SWITCH(bits, Idx, {
      auto gdata = AllocBackwardGData<XPU, Idx, DType>(op,
          rtcfg.ctx, x_len, lhs_mapping, rhs_mapping, out_mapping,
          lhs_data, rhs_data, out_data, grad_out_data,
          grad_lhs_data, grad_rhs_data);
      BACKWARD_MODE_SWITCH(req_lhs, req_rhs, Mode, {
        REDUCER_SWITCH(reducer, XPU, DType, Reducer, {
          OP_TARGET_SWITCH(op, lhs, rhs, DType, BinaryOp, LeftTarget, RightTarget, {
            CallBackwardBinaryReduce<XPU, Mode, Idx, DType, LeftTarget,
              RightTarget, BinaryOp, Reducer>(rtcfg, graph, &gdata);
          });
        });
      });
    });
  });
}

///////////////////////////////////////////////////////////////////////////////
// BinaryReduceBcast device-agnostic implementation
///////////////////////////////////////////////////////////////////////////////

template <int XPU, int NDim, typename Idx, typename DType, typename Reducer>
BcastGData<NDim, Idx, DType> AllocBcastGData(
    const DLContext& ctx, const BcastInfo& info,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping,
    runtime::NDArray lhs_data, runtime::NDArray rhs_data,
    runtime::NDArray out_mapping, runtime::NDArray out_data) {
  // GData
  BcastGData<NDim, Idx, DType> gdata;
  // dim, shape and stride
  gdata.ndim = info.lhs_shape.size();
  std::copy(info.lhs_shape.begin(), info.lhs_shape.end(), gdata.lhs_shape);
  std::copy(info.lhs_stride.begin(), info.lhs_stride.end(), gdata.lhs_stride);
  std::copy(info.rhs_shape.begin(), info.rhs_shape.end(), gdata.rhs_shape);
  std::copy(info.rhs_stride.begin(), info.rhs_stride.end(), gdata.rhs_stride);
  std::copy(info.out_shape.begin(), info.out_shape.end(), gdata.out_shape);
  std::copy(info.out_stride.begin(), info.out_stride.end(), gdata.out_stride);
  gdata.lhs_len = utils::Prod(info.lhs_shape);
  gdata.rhs_len = utils::Prod(info.rhs_shape);
  gdata.out_len = utils::Prod(info.out_shape);
  // data
  gdata.lhs_data = static_cast<DType*>(lhs_data->data);
  gdata.rhs_data = static_cast<DType*>(rhs_data->data);
  gdata.out_data = static_cast<DType*>(out_data->data);
  if (!utils::IsNoneArray(lhs_mapping)) {
    gdata.lhs_mapping = static_cast<Idx*>(lhs_mapping->data);
  }
  if (!utils::IsNoneArray(rhs_mapping)) {
    gdata.rhs_mapping = static_cast<Idx*>(rhs_mapping->data);
  }
  if (!utils::IsNoneArray(out_mapping)) {
    gdata.out_mapping = static_cast<Idx*>(out_mapping->data);
  }
  gdata.data_len = info.data_len;

  // fill out data with zero values
  utils::Fill<XPU>(ctx, gdata.out_data, utils::NElements(out_data), Zero<Reducer>::value);
  return gdata;
}

template <int XPU>
void BinaryReduceBcastImpl(
    const BcastInfo& info,
    const std::string& reducer,
    const std::string& op,
    const CSRWrapper& graph,
    binary_op::Target lhs,
    binary_op::Target rhs,
    runtime::NDArray lhs_data,
    runtime::NDArray rhs_data,
    runtime::NDArray out_data,
    runtime::NDArray lhs_mapping,
    runtime::NDArray rhs_mapping,
    runtime::NDArray out_mapping) {
  using runtime::NDArray;
  using minigun::Csr;
#ifdef __CUDACC__
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
#endif
  // advance config
  minigun::advance::RuntimeConfig rtcfg;
  rtcfg.ctx = out_data->ctx;
#ifdef __CUDACC__
  rtcfg.stream = thr_entry->stream;
  const int64_t x_len = utils::ComputeXLength(out_data);
  const int nt = utils::FindNumThreads(x_len, 64);
  rtcfg.data_num_threads = nt;
  // XXX(minjie): hard-code to let each thread compute two elements to increase
  //              instruction level parallelism
  rtcfg.data_num_blocks = (x_len + (nt * 2) - 1) / (nt * 2);
#endif

  const DLDataType& dtype = out_data->dtype;
  const int bcast_ndim = info.out_shape.size();
  const auto bits = graph.NumBits();

  if (reducer == binary_op::kReduceMean) {
    // TODO(minjie): divide
    LOG(FATAL) << "reduce mean is not supported.";
  }
  DGL_DTYPE_SWITCH(dtype, DType, {
    DGL_IDX_TYPE_SWITCH(bits, Idx, {
      REDUCER_SWITCH(reducer, XPU, DType, Reducer, {
        BCAST_NDIM_SWITCH(bcast_ndim, NDim, {
          auto gdata = AllocBcastGData<XPU, NDim, Idx, DType, Reducer>(
              rtcfg.ctx, info, lhs_mapping, rhs_mapping,
              lhs_data, rhs_data, out_mapping, out_data);
          OP_TARGET_SWITCH(op, lhs, rhs, DType, BinaryOp, LeftTarget, RightTarget, {
            CallBinaryReduceBcast<XPU, NDim, Idx, DType, LeftTarget,
              RightTarget, BinaryOp, Reducer>(rtcfg, graph, &gdata);
          });
        });
      });
    });
  });
}

///////////////////////////////////////////////////////////////////////////////
// BackwardBinaryReduceBcast device-agnostic implementation
///////////////////////////////////////////////////////////////////////////////

template <int XPU, int NDim, typename Idx, typename DType>
BackwardBcastGData<NDim, Idx, DType> AllocBackwardBcastGData(
    const DLContext& ctx, const BcastInfo& info,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping, runtime::NDArray out_mapping,
    runtime::NDArray lhs, runtime::NDArray rhs, runtime::NDArray out, runtime::NDArray grad_out,
    runtime::NDArray grad_lhs, runtime::NDArray grad_rhs) {
  // GData
  BackwardBcastGData<NDim, Idx, DType> gdata;
  // dim, shape and stride
  gdata.ndim = info.lhs_shape.size();
  gdata.lhs_len = utils::Prod(info.lhs_shape);
  gdata.rhs_len = utils::Prod(info.rhs_shape);
  gdata.out_len = utils::Prod(info.out_shape);
  std::copy(info.lhs_shape.begin(), info.lhs_shape.end(), gdata.lhs_shape);
  std::copy(info.lhs_stride.begin(), info.lhs_stride.end(), gdata.lhs_stride);
  std::copy(info.rhs_shape.begin(), info.rhs_shape.end(), gdata.rhs_shape);
  std::copy(info.rhs_stride.begin(), info.rhs_stride.end(), gdata.rhs_stride);
  std::copy(info.out_shape.begin(), info.out_shape.end(), gdata.out_shape);
  std::copy(info.out_stride.begin(), info.out_stride.end(), gdata.out_stride);
  // mappings
  if (!utils::IsNoneArray(lhs_mapping)) {
    gdata.lhs_mapping = static_cast<Idx*>(lhs_mapping->data);
  }
  if (!utils::IsNoneArray(rhs_mapping)) {
    gdata.rhs_mapping = static_cast<Idx*>(rhs_mapping->data);
  }
  if (!utils::IsNoneArray(out_mapping)) {
    gdata.out_mapping = static_cast<Idx*>(out_mapping->data);
  }
  gdata.data_len = info.data_len;

  // data
  gdata.lhs_data = static_cast<DType*>(lhs->data);
  gdata.rhs_data = static_cast<DType*>(rhs->data);
  gdata.out_data = static_cast<DType*>(out->data);
  gdata.grad_out_data = static_cast<DType*>(grad_out->data);
  if (!utils::IsNoneArray(grad_lhs)) {
    gdata.grad_lhs_data = static_cast<DType*>(grad_lhs->data);
    // fill out data with zero values
    utils::Fill<XPU>(ctx, gdata.grad_lhs_data, utils::NElements(grad_lhs),
                static_cast<DType>(0));
  }
  if (!utils::IsNoneArray(grad_rhs)) {
    gdata.grad_rhs_data = static_cast<DType*>(grad_rhs->data);
    // fill out data with zero values
    utils::Fill<XPU>(ctx, gdata.grad_rhs_data, utils::NElements(grad_rhs),
                static_cast<DType>(0));
  }
  return gdata;
}

template <int XPU>
void BackwardBinaryReduceBcastImpl(
    const BcastInfo& info,
    const std::string& reducer,
    const std::string& op,
    const CSRWrapper& graph,
    binary_op::Target lhs_tgt, binary_op::Target rhs_tgt,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping, runtime::NDArray out_mapping,
    runtime::NDArray lhs, runtime::NDArray rhs, runtime::NDArray out, runtime::NDArray grad_out,
    runtime::NDArray grad_lhs, runtime::NDArray grad_rhs) {
  using runtime::NDArray;
  using minigun::Csr;
#ifdef __CUDACC__
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
#endif
  // advance config
  minigun::advance::RuntimeConfig rtcfg;
  rtcfg.ctx = out->ctx;
#ifdef __CUDACC__
  rtcfg.stream = thr_entry->stream;
  const int64_t x_len = utils::ComputeXLength(out);
  const int nt = utils::FindNumThreads(x_len, 64);
  rtcfg.data_num_threads = nt;
  // XXX(minjie): hard-code to let each thread compute two elements to increase
  //              instruction level parallelism
  rtcfg.data_num_blocks = (x_len + (nt * 2) - 1) / (nt * 2);
#endif

  const DLDataType& dtype = out->dtype;
  const int bcast_ndim = info.out_shape.size();
  const bool req_lhs = !utils::IsNoneArray(grad_lhs);
  const bool req_rhs = !utils::IsNoneArray(grad_rhs);
  const auto bits = graph.NumBits();

  if (reducer == binary_op::kReduceMean) {
    // TODO(minjie): divide
    LOG(FATAL) << "reduce mean is not supported.";
  }
  DGL_DTYPE_SWITCH(dtype, DType, {
    DGL_IDX_TYPE_SWITCH(bits, Idx, {
      BCAST_NDIM_SWITCH(bcast_ndim, NDim, {
        auto gdata = AllocBackwardBcastGData<XPU, NDim, Idx, DType>(
            rtcfg.ctx, info,
            lhs_mapping, rhs_mapping, out_mapping,
            lhs, rhs, out, grad_out,
            grad_lhs, grad_rhs);
        BACKWARD_MODE_SWITCH(req_lhs, req_rhs, Mode, {
          REDUCER_SWITCH(reducer, XPU, DType, Reducer, {
            OP_TARGET_SWITCH(op, lhs_tgt, rhs_tgt, DType, BinaryOp, LeftTarget, RightTarget, {
              CallBackwardBinaryReduceBcast<XPU, Mode, NDim, Idx, DType,
                LeftTarget, RightTarget, BinaryOp, Reducer>(rtcfg, graph, &gdata);
            });
          });
        });
      });
    });
  });
}

}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_BINARY_REDUCE_IMPL_H_
