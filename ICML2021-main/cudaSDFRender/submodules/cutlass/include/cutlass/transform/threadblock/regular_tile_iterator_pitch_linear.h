/***************************************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Templates implementing loading of tiles from pitch-linear rank=2 tensors. 

    This iterator uses masks to guard out-of-bounds accesses and visits the last "residue" tile
    first, with the objective of minimizing predicate mask updates during steady-state operation.

    A precomputed "Params" object minimizes the amount of state that must be stored in registers,
    and integer addition is used to advance the pointer through memory.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"

#include "regular_tile_iterator.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Regular tile iterator specialized for pitch-linear
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int Alignment
>
class RegularTileIterator<Shape_, Element_, layout::PitchLinear, AdvanceRank, ThreadMap_, Alignment> {
public:

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::PitchLinear;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Fragment = Array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

  static_assert(kAdvanceRank == 0 || kAdvanceRank == 1, 
    "Advance rank may only be along the contiguous or strided dimensions.");

private:

  //
  // Types
  //
  
  using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess, kAlignment>;

  //
  // Data members
  //

  /// Pointer to memory
  uint8_t *pointer_;

  /// Stride quantity
  Index stride_;

  /// Amount to increment pointer along strided dimension
  Index increment_strided_;

  /// Amount to advance pointer between tiles
  Index increment_advance_;

public:

  CUTLASS_DEVICE
  RegularTileIterator(): pointer_(nullptr), increment_strided_(0), increment_advance_(0) { }

  CUTLASS_DEVICE
  RegularTileIterator(
    TensorRef const &ref, 
    int thread_idx
  ): 
    pointer_(reinterpret_cast<uint8_t *>(ref.data()) + (ref.offset(ThreadMap::initial_offset(thread_idx)) * sizeof_bits<Element>::value / 8)) {
    
    stride_ = ref.stride()[0];
    increment_strided_ = (ref.stride()[0] * sizeof_bits<Element>::value) * ThreadMap::Delta::kStrided / 8;
    
    increment_advance_ = 
      (kAdvanceRank == 0 ? 
        Shape::kContiguous * sizeof_bits<Element>::value / 8 : 
        Shape::kStrided * (ref.stride()[0] * sizeof_bits<Element>::value / 8));
  }

  /// Loads a fragment
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);
    uint8_t const *byte_pointer = pointer_ + pointer_offset * sizeof_bits<Element>::value / 8;

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

      AccessType const *access_ptr = reinterpret_cast<AccessType const *>(byte_pointer);

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

        int idx = c + s * ThreadMap::Iterations::kContiguous;
        frag_ptr[idx] = access_ptr[c * ThreadMap::Delta::kContiguous];
      }

      if (s + 1 < ThreadMap::Iterations::kStrided) {
        byte_pointer += increment_strided_;
      }
    }
  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag, TensorCoord const & tile_offset) {
    load_with_pointer_offset(
      frag, 
      tile_offset.contiguous() * Shape::kContiguous / ThreadMap::kElementsPerAccess + 
        tile_offset.strided() * Shape::kStrided * stride_
    );
  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {

    AccessType const *frag_ptr = reinterpret_cast<AccessType const*>(&frag);
    uint8_t *byte_pointer = pointer_ + pointer_offset * sizeof_bits<Element>::value / 8;

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

      AccessType *access_ptr = reinterpret_cast<AccessType *>(byte_pointer);

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

        int idx = c + s * ThreadMap::Iterations::kContiguous;
        access_ptr[c * ThreadMap::Delta::kContiguous] = frag_ptr[idx];
      }

      if (s + 1 < ThreadMap::Iterations::kStrided) {
        byte_pointer += increment_strided_;
      }
    }
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag, TensorCoord const & tile_offset) {
    store_with_pointer_offset(
      frag,
      tile_offset.contiguous() * Shape::kContiguous + tile_offset.strided() * Shape::kStrided * stride_
    );
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {
    pointer_ += increment_advance_;
    return *this;
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  RegularTileIterator &operator--() {
    pointer_ -= increment_advance_;
    return *this;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    pointer_ += pointer_offset;
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    int offset = sizeof_bits<Element>::value *
        (coord.contiguous() * Shape::kContiguous + coord.strided() * Shape::kStrided * stride_) / 8;
    add_pointer_offset(offset);
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Regular tile iterator specialized for pitch-linear
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int Alignment
>
class RegularTileIterator<Shape_, Element_, layout::RowMajor, AdvanceRank, ThreadMap_, Alignment> {
public:

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajor;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Fragment = Array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

  using Underlying = RegularTileIterator<
    layout::PitchLinearShape<Shape::kColumn, Shape::kRow>,
    Element,
    layout::PitchLinear,
    (kAdvanceRank == 0 ? 1 : 0),
    ThreadMap,
    kAlignment
  >;

  static_assert(kAdvanceRank == 0 || kAdvanceRank == 1, 
    "Advance rank may only be along the row or column dimensions.");

private:

  Underlying iterator_;

public:

  CUTLASS_DEVICE
  RegularTileIterator() { }

  CUTLASS_DEVICE
  RegularTileIterator(
    TensorRef const &ref, 
    int thread_idx
  ):
    iterator_({ref.data(), ref.stride()}, thread_idx) {

  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag, TensorCoord const & tile_offset) {
    iterator_.load_with_pointer_offset(frag, {tile_offset.column(), tile_offset.row()});
  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) {
    iterator_.load_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag, TensorCoord const & tile_offset) {
    iterator_.store_with_pointer_offset(frag, {tile_offset.column(), tile_offset.row()});
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    iterator_.store_with_pointer_offset(frag, 0);
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  RegularTileIterator &operator--() {
    --iterator_;
    return *this;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.column(), coord.row()});
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Regular tile iterator specialized for pitch-linear
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int Alignment
>
class RegularTileIterator<Shape_, Element_, layout::ColumnMajor, AdvanceRank, ThreadMap_, Alignment> {
public:

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajor;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Fragment = Array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

  using Underlying = RegularTileIterator<
    layout::PitchLinearShape<Shape::kRow, Shape::kColumn>,
    Element,
    layout::PitchLinear,
    (kAdvanceRank == 0 ? 0 : 1),
    ThreadMap
  >;

  static_assert(kAdvanceRank == 0 || kAdvanceRank == 1, 
    "Advance rank may only be along the row or column dimensions.");

private:

  Underlying iterator_;

public:

  CUTLASS_DEVICE
  RegularTileIterator() { }

  CUTLASS_DEVICE
  RegularTileIterator(
    TensorRef const &ref, 
    int thread_idx
  ):
    iterator_({ref.data(), ref.stride()}, thread_idx) {

  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag, TensorCoord const & tile_offset) {
    iterator_.load_with_pointer_offset(frag, {tile_offset.row(), tile_offset.column()});
  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) {
    iterator_.load_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag, TensorCoord const & tile_offset) {
    iterator_.store_with_pointer_offset(frag, {tile_offset.row(), tile_offset.column()});
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    iterator_.store_with_pointer_offset(frag, 0);
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  RegularTileIterator &operator--() {
    --iterator_;
    return *this;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.row(), coord.column()});
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace transform
} // namespace cutlass

