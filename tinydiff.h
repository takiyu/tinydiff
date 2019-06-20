#ifndef TINYDIFF_H_ONCE
#define TINYDIFF_H_ONCE

#include <cassert>
#include <cmath>
#include <exception>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

namespace tinydiff {

class NdArray;
using InitShape = std::initializer_list<int>;
using Shape = std::vector<int>;
using Index = std::vector<int>;
using SliceIndex = std::vector<std::pair<int, int>>;
class Variable;
using Variables = std::vector<Variable>;
class Function;

// =============================================================================
// ======================= Nested Float Initializer List =======================
// =============================================================================
template <std::size_t D>
struct FloatListHelper {
    using type = std::initializer_list<typename FloatListHelper<D - 1>::type>;
};

template <>
struct FloatListHelper<0> {
    using type = std::initializer_list<float>;
};

template <std::size_t D>
using FloatList = typename FloatListHelper<D>::type;

// =============================================================================
// ================================== NdArray ==================================
// =============================================================================
class NdArray {
public:
    NdArray();
    NdArray(const NdArray&);
    NdArray(NdArray&&);
    NdArray& operator=(const NdArray&);
    NdArray& operator=(NdArray&&);
    ~NdArray();

    NdArray(FloatList<0> init_list);
    NdArray(FloatList<1> init_list);
    NdArray(FloatList<2> init_list);
    NdArray(FloatList<3> init_list);
    NdArray(FloatList<4> init_list);
    NdArray(FloatList<5> init_list);
    NdArray(FloatList<6> init_list);
    NdArray(FloatList<7> init_list);
    NdArray(FloatList<8> init_list);
    NdArray(FloatList<9> init_list);

    NdArray(const InitShape& shape);
    NdArray(const Shape& shape);
    NdArray(const Shape& shape, float fill_v);

    static NdArray Empty(const Shape& shape);
    static NdArray Zeros(const Shape& shape);
    static NdArray Ones(const Shape& shape);
    template <typename... S>
    static NdArray Empty(S... shape);
    template <typename... S>
    static NdArray Zeros(S... shape);
    template <typename... S>
    static NdArray Ones(S... shape);

    static NdArray Arange(float stop);
    static NdArray Arange(float start, float stop, float step = 1.f);

    static void Seed();
    static void Seed(uint32_t seed);
    static NdArray Uniform(float low = 0.f, float high = 1.f,
                           const Shape& shape = {1});
    static NdArray Uniform(const Shape& shape);
    static NdArray Normal(float loc = 0.f, float scale = 1.f,
                          const Shape& shape = {1});
    static NdArray Normal(const Shape& shape);

    bool empty() const;
    size_t size() const;
    const Shape& shape() const;
    float* data();
    const float* data() const;

    float* begin();
    float* end();
    const float* begin() const;
    const float* end() const;

    operator float() const;

    float& operator[](int i);
    const float& operator[](int i) const;

    float& operator[](const Index& index);
    const float& operator[](const Index& index) const;
    template <typename... I>
    float& operator()(I... index);
    template <typename... I>
    const float& operator()(I... index) const;

    NdArray reshape(const Shape& shape) const;
    template <typename... S>
    NdArray reshape(S... shape) const;

    NdArray slice(const SliceIndex& slice_index) const;
    template <typename... I>
    NdArray slice(std::initializer_list<I>... slice_index) const;  // {i, j}...

    NdArray dot(const NdArray& other) const;
    NdArray cross(const NdArray& other) const;

    class Substance;

private:
    std::shared_ptr<Substance> m_sub;
    NdArray(std::shared_ptr<Substance> sub);

    static std::random_device s_rand_seed;
    static std::mt19937 s_rand_engine;
};

// --------------------------------- Operators ---------------------------------
std::ostream& operator<<(std::ostream& os, const NdArray& x);
std::ostream& operator<<(std::ostream& os, const Shape& shape);
NdArray operator+(const NdArray& lhs, const NdArray& rhs);
NdArray operator-(const NdArray& lhs, const NdArray& rhs);
NdArray operator*(const NdArray& lhs, const NdArray& rhs);
NdArray operator/(const NdArray& lhs, const NdArray& rhs);
NdArray operator+(const NdArray& lhs, const float& rhs);
NdArray operator-(const NdArray& lhs, const float& rhs);
NdArray operator*(const NdArray& lhs, const float& rhs);
NdArray operator/(const NdArray& lhs, const float& rhs);
NdArray operator+(const float& lhs, const NdArray& rhs);
NdArray operator-(const float& lhs, const NdArray& rhs);
NdArray operator*(const float& lhs, const NdArray& rhs);
NdArray operator/(const float& lhs, const NdArray& rhs);
NdArray operator+(const NdArray& x);
NdArray operator-(const NdArray& x);

// =============================================================================
// ================================== Variable =================================
// =============================================================================
class Variable {
public:
    Variable();
    Variable(float v);
    Variable(const Variable&);
    Variable(Variable&&);
    Variable& operator=(const Variable&);
    Variable& operator=(Variable&&);
    ~Variable();

    float data() const;
    float grad() const;
    void backward();

    void setCreator(Function f);
    Function getCreator() const;
    void addGrad(float grad);

    class Substance;

private:
    std::shared_ptr<Substance> m_sub;
    Variable(std::shared_ptr<Substance> sub);
};

// --------------------------------- Operators ---------------------------------
std::ostream& operator<<(std::ostream& os, Variable& x);
Variable operator+(const Variable& lhs, const Variable& rhs);
Variable operator*(const Variable& lhs, const Variable& rhs);

// =============================================================================
// ================================== Function =================================
// =============================================================================
class Function {
public:
    Function();
    Function(const Function&);
    Function(Function&&);
    Function& operator=(const Function&);
    Function& operator=(Function&&);
    virtual ~Function();

    void clear();

    // Build computational graph with forwarding
    Variables operator()(const Variables& x);

    std::vector<float> forward(const std::vector<float>& x);
    std::vector<float> backward(const std::vector<float>& x,
                                const std::vector<float>& y,
                                const std::vector<float>& gy);

    Variables getInputs() const;
    Variables getOutputs() const;

    void setRank(size_t rank);
    size_t getRank() const;

    class Substance;

protected:
    std::shared_ptr<Substance> m_sub;
    Function(std::shared_ptr<Substance> sub);
};

// =============================================================================
// ============================ Implemented Function ===========================
// =============================================================================

namespace F {

Variable add(Variable lhs, Variable rhs);
Variable mul(Variable lhs, Variable rhs);
Variable exp(Variable x);

}  // namespace F

// *****************************************************************************
// *****************************************************************************
// **************************** Begin of Definitions ***************************
// *****************************************************************************
// *****************************************************************************
#ifdef TINYDIFF_IMPLEMENTATION

// --------------------------- Utilities for NdArray ---------------------------
template <typename T>
T clamp(const T& v, const T& lower, const T& upper) {
    return std::min(std::max(v, lower), upper);
}

template <typename FList>
std::list<int> CheckFListShapeImpl(const FList& init_list) {
    if (init_list.size() == 0) {
        return {};
    }
    // Check all children have same shape
    auto itr = init_list.begin();
    auto shape = CheckFListShapeImpl(*itr);
    for (size_t i = 0; i < init_list.size(); i++, itr++) {
        if (shape != CheckFListShapeImpl(*itr)) {
            throw std::runtime_error("Initializing shape is invalid");
        }
    }
    // Return total shape of children
    shape.push_front(static_cast<int>(init_list.size()));
    return shape;
}

template <>
std::list<int> CheckFListShapeImpl(const FloatList<0>& init_list) {
    return {static_cast<int>(init_list.size())};
}

template <typename FList>
Shape CheckFListShape(const FList& init_list) {
    // Check and get the shape of nested initializer.
    const std::list<int>& shape = CheckFListShapeImpl(init_list);
    // Cast to vector
    return Shape(shape.begin(), shape.end());
}

template <typename FList>
void CopyFListElemsImpl(const FList& init_list, float*& data) {
    // Copy sequentially
    for (auto itr = init_list.begin(); itr != init_list.end(); itr++) {
        CopyFListElemsImpl(*itr, data);
    }
}

template <>
void CopyFListElemsImpl(const FloatList<0>& init_list, float*& data) {
    // Copy sequentially
    for (auto&& v : init_list) {
        *(data++) = v;
    }
}

template <typename FList>
void CopyFListElems(const FList& init_list, float* data) {
    // Pass to impl (create pointer instance)
    CopyFListElemsImpl(init_list, data);
}

template <typename D, typename R>
NdArray CreateRandomArray(const Shape& shape, D&& dist, R&& rand_engine) {
    // Create empty array
    NdArray ret(shape);
    // Fill by random value
    float* data = ret.data();
    for (size_t i = 0; i < ret.size(); i++) {
        *(data++) = static_cast<float>(dist(rand_engine));
    }
    return ret;
}

static std::vector<int> ComputeChildSizes(const Shape& shape) {
    const size_t n_shape = shape.size();
    if (n_shape == 0) {
        return {};
    }
    // Compute child sizes from back (the number of children for each dimension)
    std::vector<int> child_sizes(n_shape, 1);
    int size = 1;
    for (size_t depth = n_shape - 1; 0 < depth; depth--) {
        child_sizes[depth] = size;
        size *= shape[depth];
    }
    child_sizes[0] = size;
    return child_sizes;
}

static void CopySliceImpl(const float*& src_data, float*& dst_data,
                          const Shape& src_shape, const SliceIndex& slice_index,
                          const std::vector<int>& child_sizes, size_t depth) {
    if (depth < src_shape.size()) {
        const auto& si = slice_index[depth];
        // Add previous offset
        src_data += child_sizes[depth] * si.first;
        // Copy
        for (int i = si.first; i < si.second; i++) {
            // Recursive call
            CopySliceImpl(src_data, dst_data, src_shape, slice_index,
                          child_sizes, depth + 1);
        }
        // Add post offset
        src_data += child_sizes[depth] * (src_shape[depth] - si.second);
    } else {
        // Copy
        *(dst_data++) = *(src_data++);
    }
}

static NdArray CopySlice(const NdArray& src, const Shape& slice_shape,
                         const SliceIndex& slice_index) {
    const Shape& src_shape = src.shape();

    // Pre-compute child sizes (index offsets)
    const std::vector<int>& child_sizes = ComputeChildSizes(src_shape);

    // Create slice instance
    NdArray ret(slice_shape);

    // Start to copy
    const float* src_data = src.data();
    float* dst_data = ret.data();
    CopySliceImpl(src_data, dst_data, src_shape, slice_index, child_sizes, 0);

    return ret;
}

static std::pair<int, int> CvtToSliceIndexItem(std::initializer_list<int> l) {
    if (l.size() != 2) {
        throw std::runtime_error("Invalid slice index format");
    }
    return {*l.begin(), *(l.begin() + 1)};
}

static Shape CheckBroadcastable(const Shape& l_shape, const Shape& r_shape) {
    // We assuming left array has deeper shape than right one.
    if (l_shape.size() < r_shape.size()) {
        return CheckBroadcastable(r_shape, l_shape);  // Swap
    }
    // `l_shape.size()` is maximum depth.

    // Compute broadcasted shape
    Shape shape(l_shape.size());
    size_t r_offset = l_shape.size() - r_shape.size();
    for (size_t i = 0; i < l_shape.size(); i++) {
        if (i < r_offset) {
            shape[i] = l_shape[i];
        } else {
            const int l = l_shape[i];
            const int r = r_shape[i - r_offset];
            if (l == r) {
                shape[i] = l;  // no broadcast
            } else if (l == 1) {
                shape[i] = r;  // left broadcast
            } else if (r == 1) {
                shape[i] = l;  // right broadcast
            } else {
                std::stringstream ss;
                ss << "Non operatable shape";
                ss << " (" << l_shape << " vs " << r_shape << ")";
                throw std::runtime_error(ss.str());
            }
        }
    }
    return shape;
}

static Shape PadShape(const Shape& shape, size_t size) {
    if (size < shape.size()) {
        throw std::runtime_error("Invalid shape to pad");
    }
    const size_t n_pad = size - shape.size();
    Shape ret_shape;
    ret_shape.reserve(size);
    ret_shape.resize(n_pad, 1);                                     // Fill by 1
    ret_shape.insert(ret_shape.end(), shape.begin(), shape.end());  // Concat
    return ret_shape;
}

template <typename F>
static void ApplyOpBroadcastImpl(float* ret_data, const float* l_data,
                                 const float* r_data, const Shape& ret_shape,
                                 const Shape& l_shape, const Shape& r_shape,
                                 const std::vector<int>& ret_child_sizes,
                                 const std::vector<int>& l_child_sizes,
                                 const std::vector<int>& r_child_sizes,
                                 size_t depth, size_t const depth_offset,
                                 F op) {
    if (depth < ret_shape.size() - depth_offset) {
        // Fetch shapes
        const int l_s = l_shape[depth];
        const int r_s = r_shape[depth];
        // Decide pointer steps by broadcast patterns.
        const int ret_step = ret_child_sizes[depth];
        const int l_step = (l_s == r_s || r_s == 1) ? l_child_sizes[depth] : 0;
        const int r_step = (l_s == r_s || l_s == 1) ? r_child_sizes[depth] : 0;
        // Applying loop
        const int n_loop = std::max(l_s, r_s);
        for (int i = 0; i < n_loop; i++) {
            // Apply recursively
            ApplyOpBroadcastImpl(ret_data, l_data, r_data, ret_shape, l_shape,
                                 r_shape, ret_child_sizes, l_child_sizes,
                                 r_child_sizes, depth + 1, depth_offset, op);
            // Next pointer
            ret_data += ret_step;
            l_data += l_step;
            r_data += r_step;
        }
    } else {
        // Apply operator
        op(ret_data, l_data, r_data);
    }
}

template <typename F>
static NdArray ApplyOpBroadcast(const NdArray& lhs, const NdArray& rhs,
                                const Shape& ret_shape,
                                const size_t depth_offset, F op) {
    // Pre-compute padded shape
    const Shape& l_shape_pad = PadShape(lhs.shape(), ret_shape.size());
    const Shape& r_shape_pad = PadShape(rhs.shape(), ret_shape.size());

    // Pre-compute child sizes
    const std::vector<int>& ret_child_sizes = ComputeChildSizes(ret_shape);
    const std::vector<int>& l_child_sizes = ComputeChildSizes(l_shape_pad);
    const std::vector<int>& r_child_sizes = ComputeChildSizes(r_shape_pad);

    // Apply with broadcast
    NdArray ret(ret_shape);
    ApplyOpBroadcastImpl(ret.data(), lhs.data(), rhs.data(), ret_shape,
                         l_shape_pad, r_shape_pad, ret_child_sizes,
                         l_child_sizes, r_child_sizes, 0, depth_offset, op);
    return ret;
}

template <typename F>
static NdArray ApplyElemWiseOp(const NdArray& lhs, const NdArray& rhs, F op) {
    if (lhs.shape() == rhs.shape()) {
        // Apply without broadcast because of same size for speed up.
        NdArray ret(lhs.shape());
        float* ret_data = ret.data();
        const float* l_data = lhs.data();
        const float* r_data = rhs.data();
        // Simply apply all
        for (size_t i = 0; i < ret.size(); i++) {
            *(ret_data++) = op(*(l_data++), *(r_data++));
        }
        return ret;
    } else {
        // Check it is possible to broadcast
        const Shape& ret_shape = CheckBroadcastable(lhs.shape(), rhs.shape());
        // Wrap operator `float(float, float)` for pointer.
        auto wrapped_op = [&](float* o, const float* l, const float* r) {
            *o = op(*l, *r);
        };
        // Apply broadcast
        return ApplyOpBroadcast(lhs, rhs, ret_shape, 0, wrapped_op);
    }
}

template <typename F>
static NdArray ApplyElemWiseOp(const NdArray& lhs, const float& rhs, F op) {
    // Broadcast right float
    NdArray ret(lhs.shape());
    const float* l_data = lhs.data();
    float* ret_data = ret.data();
    // Simply apply all
    for (size_t i = 0; i < ret.size(); i++) {
        *(ret_data++) = op(*(l_data++), rhs);
    }
    return ret;
}

template <typename F>
static NdArray ApplyElemWiseOp(const float& lhs, const NdArray& rhs, F op) {
    // Broadcast left float
    NdArray ret(rhs.shape());
    const float* r_data = rhs.data();
    float* ret_data = ret.data();
    // Simply apply all
    for (size_t i = 0; i < ret.size(); i++) {
        *(ret_data++) = op(lhs, *(r_data++));
    }
    return ret;
}

static void OutputArrayLine(std::ostream& os, const float*& data, size_t size) {
    os << "[";  // Begin of a line
    for (size_t i = 0; i < size; i++) {
        os << *(data++);  // Output an element
        if (i == size - 1) {
            os << "]";  // End of a line
        } else {
            os << ", ";  // Splitter of an element
        }
    }
}

static void OutputArrayMultiDim(std::ostream& os, const float*& data,
                                const Shape& shape, size_t depth) {
    for (size_t i = 0; i < static_cast<size_t>(shape[depth]); i++) {
        // Heading
        if (i == 0) {
            os << "[";  // begin of array
        } else {
            for (size_t d = 0; d < depth + 1; d++) {  // array indent
                os << " ";
            }
        }

        // Output internal array
        if (depth == shape.size() - 2) {
            OutputArrayLine(os, data, static_cast<size_t>(shape[depth + 1]));
        } else {
            OutputArrayMultiDim(os, data, shape, depth + 1);
        }

        // Tailing
        if (i == static_cast<size_t>(shape[depth]) - 1) {
            os << "]";  // End of array
        } else {
            os << "," << std::endl;  // Splitter of array
        }
    }
}

static NdArray DotNdArray1d(const NdArray& lhs, const NdArray& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error("Invalid size for inner product of 1D");
    }
    // Inner product of vectors
    const float* l_data = lhs.data();
    const float* r_data = rhs.data();
    float sum = 0.f;
    for (size_t i = 0; i < lhs.size(); i++) {
        sum += l_data[i] * r_data[i];
    }
    return {sum};
}

static NdArray DotNdArray2d(const NdArray& lhs, const NdArray& rhs) {
    const Shape& l_shape = lhs.shape();  // 2 == size
    const Shape& r_shape = rhs.shape();  // 2 == size
    if (l_shape[1] != r_shape[0]) {
        throw std::runtime_error("Invalid size for inner product of 2D");
    }
    // Inner product of 2D matrix
    const int n_row = l_shape[0];
    const int n_col = r_shape[1];
    const int n_contract = l_shape[1];
    const int& n_l_col = n_contract;
    const int& n_r_col = n_col;
    NdArray ret({n_row, n_col});
    float* ret_data = ret.data();
    const float* l_data = lhs.data();
    const float* r_data = rhs.data();
    for (int row_idx = 0; row_idx < n_row; row_idx++) {
        for (int col_idx = 0; col_idx < n_col; col_idx++) {
            float sum = 0.f;
            for (int i = 0; i < n_contract; i++) {
                sum += l_data[row_idx * n_l_col + i] *
                       r_data[i * n_r_col + col_idx];
            }
            *(ret_data++) = sum;
        }
    }
    return ret;
}

static NdArray DotNdArrayNdMd(const NdArray& lhs, const NdArray& rhs) {
    const Shape& l_shape = lhs.shape();  // 1 <= size
    const Shape& r_shape = rhs.shape();  // 2 <= size

    // The last axis of left and the second-to-last axis of right must be same.
    if (l_shape[l_shape.size() - 1] != r_shape[r_shape.size() - 2]) {
        throw std::runtime_error("Invalid shape for dot product");
    }

    // Result shape
    Shape ret_shape(l_shape.begin(), l_shape.end() - 1);
    ret_shape.insert(ret_shape.end(), r_shape.begin(), r_shape.end() - 2);
    ret_shape.push_back(r_shape.back());

    // Compute child sizes
    //   [2, 3, (4)] [5, 6, (4), 7] -> [2, 3, 5, 6, 7]
    const int ret_child_size_2 = std::accumulate(
            ret_shape.begin() + static_cast<long>(l_shape.size()) - 1,
            ret_shape.end(), 1, std::multiplies<int>());  // [(5), (6), 4, (7)]
    const int ret_child_size_1 = ret_shape.back();        // [5, 6, 4, (7)]
    const int l_child_size = l_shape.back();              // [2, 3, (4)]
    const int r_child_size_2 =
            r_shape.end()[-1] * r_shape.end()[-2];  // [5, 6, (4), (7)]
    const int r_child_size_1 = ret_child_size_1;    // [2, 3, (4)]

    // Basic matrix product
    NdArray ret(ret_shape);
    const int n_ret = static_cast<int>(ret.size());
    const int n_contract = l_child_size;
    float* ret_data = ret.data();
    const float* l_data = lhs.data();
    const float* r_data = rhs.data();
    for (int ret_idx = 0; ret_idx < n_ret; ret_idx++) {
        // Compute left/right index from `ret_idx`
        const int l_idx = (ret_idx / ret_child_size_2) * l_child_size;
        const int r_idx_2 = (ret_idx % ret_child_size_2) / ret_child_size_1;
        const int r_idx_1 = ret_idx % ret_child_size_1;
        const int r_idx = (r_idx_2 * r_child_size_2) + r_idx_1;
        // Sum up
        float sum = 0.f;
        for (int i = 0; i < n_contract; i++) {
            sum += l_data[l_idx + i] * r_data[r_idx + (i * r_child_size_1)];
        }
        // Register
        ret_data[ret_idx] = sum;
    }
    return ret;
}

static void CrossNdArray1d1dShape33(float* ret_data, const float* l_data,
                                    const float* r_data) {
    // lhs.shape() == {3} && rhs.shape == {3}
    *(ret_data++) = l_data[1] * r_data[2] - l_data[2] * r_data[1];
    *(ret_data++) = l_data[2] * r_data[0] - l_data[0] * r_data[2];
    *(ret_data++) = l_data[0] * r_data[1] - l_data[1] * r_data[0];
}

static void CrossNdArray1d1dShape32(float* ret_data, const float* l_data,
                                    const float* r_data) {
    // lhs.shape() == {3} && rhs.shape == {2}
    *(ret_data++) = -l_data[2] * r_data[1];
    *(ret_data++) = l_data[2] * r_data[0];
    *ret_data = l_data[0] * r_data[1] - l_data[1] * r_data[0];
}

static void CrossNdArray1d1dShape23(float* ret_data, const float* l_data,
                                    const float* r_data) {
    // lhs.shape() == {3} && rhs.shape == {3}
    *(ret_data++) = l_data[1] * r_data[2];
    *(ret_data++) = -l_data[0] * r_data[2];
    *ret_data = l_data[0] * r_data[1] - l_data[1] * r_data[0];
}

static void CrossNdArray1d1dShape22(float* ret_data, const float* l_data,
                                    const float* r_data) {
    // lhs.shape() == {2} && rhs.shape == {2}
    *ret_data = l_data[0] * r_data[1] - l_data[1] * r_data[0];
}

template <typename F>
static NdArray CrossNdArrayNdMd(const NdArray& lhs, const NdArray& rhs,
                                size_t last_size, F op) {
    const Shape& l_shape = lhs.shape();
    const Shape& r_shape = rhs.shape();
    Shape ret_shape = CheckBroadcastable({l_shape.begin(), l_shape.end() - 1},
                                         {r_shape.begin(), r_shape.end() - 1});
    ret_shape.push_back(last_size);
    // Apply broadcast
    return ApplyOpBroadcast(lhs, rhs, ret_shape, 1, op);
}

// -------------------------- Utilities for Variable ---------------------------
template <typename T>
void CheckSize(const std::vector<T>& x, size_t n) {
    if (x.size() != n) {
        throw std::runtime_error("Invalid vector size");
    }
}

template <typename K, typename V>
V PopLast(std::map<K, V>& m) {
    auto&& last_itr = std::prev(m.end());
    V last = last_itr->second;
    m.erase(last_itr);
    return last;
}

static std::vector<float> CvtFromVariables(const Variables& src) {
    std::vector<float> ret;
    ret.reserve(src.size());
    for (auto&& s_elem : src) {
        ret.emplace_back(s_elem.data());
    }
    return ret;
}

static Variables CvtToVariables(const std::vector<float>& src) {
    Variables ret;
    ret.reserve(src.size());
    for (auto&& s_elem : src) {
        ret.emplace_back(Variable(s_elem));
    }
    return ret;
}

static std::vector<float> GetGrads(const Variables& src) {
    std::vector<float> ret;
    ret.reserve(src.size());
    for (auto&& s_elem : src) {
        ret.emplace_back(s_elem.grad());
    }
    return ret;
}

// =============================================================================
// ============================ NdArray Definition =============================
// =============================================================================
NdArray::NdArray() : m_sub(std::make_shared<Substance>()) {}

NdArray::NdArray(std::shared_ptr<Substance> sub) : m_sub(sub) {}

NdArray::NdArray(const NdArray& lhs) = default;  // shallow copy

NdArray::NdArray(NdArray&&) = default;  // move

NdArray& NdArray::operator=(const NdArray& lhs) = default;  // shallow copy

NdArray& NdArray::operator=(NdArray&&) = default;

NdArray::~NdArray() = default;

// --------------------------------- Substance ---------------------------------
class NdArray::Substance {
public:
    Substance() {}
    Substance(size_t size_, const Shape& shape_)
        : size(size_),
          shape(shape_),
          v(new float[size_], std::default_delete<float[]>()) {}
    size_t size = 0;
    Shape shape = {0};
    std::shared_ptr<float> v;  // C++17: Replace with `shared_ptr<float[]>`.
};

// ------------------------------- Static Member -------------------------------
std::random_device NdArray::s_rand_seed;
std::mt19937 NdArray::s_rand_engine(s_rand_seed());

// -------------------- Constructors with Float Initializers -------------------
NdArray::NdArray(FloatList<0> init_list) : NdArray(CheckFListShape(init_list)) {
    // Fill after empty initialization
    CopyFListElems(init_list, m_sub->v.get());
}

NdArray::NdArray(FloatList<1> init_list) : NdArray(CheckFListShape(init_list)) {
    CopyFListElems(init_list, m_sub->v.get());
}

NdArray::NdArray(FloatList<2> init_list) : NdArray(CheckFListShape(init_list)) {
    CopyFListElems(init_list, m_sub->v.get());
}

NdArray::NdArray(FloatList<3> init_list) : NdArray(CheckFListShape(init_list)) {
    CopyFListElems(init_list, m_sub->v.get());
}

NdArray::NdArray(FloatList<4> init_list) : NdArray(CheckFListShape(init_list)) {
    CopyFListElems(init_list, m_sub->v.get());
}

NdArray::NdArray(FloatList<5> init_list) : NdArray(CheckFListShape(init_list)) {
    CopyFListElems(init_list, m_sub->v.get());
}

NdArray::NdArray(FloatList<6> init_list) : NdArray(CheckFListShape(init_list)) {
    CopyFListElems(init_list, m_sub->v.get());
}

NdArray::NdArray(FloatList<7> init_list) : NdArray(CheckFListShape(init_list)) {
    CopyFListElems(init_list, m_sub->v.get());
}

NdArray::NdArray(FloatList<9> init_list) : NdArray(CheckFListShape(init_list)) {
    CopyFListElems(init_list, m_sub->v.get());
}

// -------------------------- Constructors with Shape --------------------------
NdArray::NdArray(const InitShape& shape) : NdArray(Shape(shape)) {
    // Just pass initializer list to `Shape` (== std::vector<int>).
}

NdArray::NdArray(const Shape& shape) {
    // Compute total size
    size_t size = 1;
    for (auto&& s : shape) {
        if (s < 0) {
            throw std::runtime_error("Invalid shape format (neg)");
        }
        size *= static_cast<size_t>(s);
    }
    // Create substance
    m_sub = std::make_shared<Substance>(size, shape);
}

NdArray::NdArray(const Shape& shape, float fill_v) : NdArray(shape) {
    // Fill after empty initialization
    std::fill_n(m_sub->v.get(), m_sub->size, fill_v);
}

// ------------------------------- Static Methods ------------------------------
NdArray NdArray::Empty(const Shape& shape) {
    return NdArray(shape);
}

NdArray NdArray::Zeros(const Shape& shape) {
    return NdArray(shape, 0.f);
}

NdArray NdArray::Ones(const Shape& shape) {
    return NdArray(shape, 1.f);
}

template <typename... S>
NdArray NdArray::Empty(S... shape) {
    return Empty({shape...});  // Unpack
}

template <typename... S>
NdArray NdArray::Zeros(S... shape) {
    return Zeros({shape...});  // Unpack
}

template <typename... S>
NdArray NdArray::Ones(S... shape) {
    return Ones({shape...});  // Unpack
}

NdArray NdArray::Arange(float stop) {
    return Arange(0.f, stop, 1.f);
}

NdArray NdArray::Arange(float start, float stop, float step) {
    const size_t n = static_cast<size_t>(std::ceil((stop - start) / step));
    // Create empty array
    NdArray ret({static_cast<int>(n)});
    // Fill by step
    float* data = ret.data();
    for (size_t i = 0; i < n; i++) {
        data[i] = start + step * static_cast<float>(i);
    }
    return ret;
}

void NdArray::Seed() {
    s_rand_engine = std::mt19937(s_rand_seed());
}

void NdArray::Seed(uint32_t seed) {
    s_rand_engine = std::mt19937(seed);
}

NdArray NdArray::Uniform(float low, float high, const Shape& shape) {
    // Create uniform distribution
    std::uniform_real_distribution<> dist(low, high);
    // Create random array
    return CreateRandomArray(shape, dist, s_rand_engine);
}

NdArray NdArray::Uniform(const Shape& shape) {
    return Uniform(0.f, 1.f, shape);
}

NdArray NdArray::Normal(float loc, float scale, const Shape& shape) {
    // Create normal distribution
    std::normal_distribution<> dist(loc, scale);
    // Create random array
    return CreateRandomArray(shape, dist, s_rand_engine);
}

NdArray NdArray::Normal(const Shape& shape) {
    return Normal(0.f, 1.f, shape);
}

// ------------------------------- Basic Methods -------------------------------
bool NdArray::empty() const {
    return m_sub->size == 0;
}

size_t NdArray::size() const {
    return m_sub->size;
}

const Shape& NdArray::shape() const {
    return m_sub->shape;
}

float* NdArray::data() {
    return m_sub->v.get();
}

const float* NdArray::data() const {
    return m_sub->v.get();
}

// ----------------------------- Begin/End Methods -----------------------------
float* NdArray::begin() {
    return m_sub->v.get();
}

float* NdArray::end() {
    return m_sub->v.get() + m_sub->size;
}

const float* NdArray::begin() const {
    return m_sub->v.get();
}

const float* NdArray::end() const {
    return m_sub->v.get() + m_sub->size;
}

// ------------------------------- Cast Operator -------------------------------
NdArray::operator float() const {
    if (m_sub->size != 1) {
        throw std::runtime_error("Only size-1 arrays can be casted to float");
    }
    return *(m_sub->v.get());
}

// ------------------------------- Index Methods -------------------------------
float& NdArray::operator[](int i) {
    // Use the same implementation of constant method.
    return const_cast<float&>(static_cast<const NdArray&>(*this)[i]);
}

const float& NdArray::operator[](int i) const {
    // Make the index positive
    const int p_idx = (0 <= i) ? i : static_cast<int>(m_sub->size) + i;
    // Direct access with range check
    return *(m_sub->v.get() + p_idx);
}

float& NdArray::operator[](const Index& index) {
    // Use the same implementation of constant method.
    return const_cast<float&>(static_cast<const NdArray&>(*this)[index]);
}

const float& NdArray::operator[](const Index& index) const {
    const auto& shape = m_sub->shape;
    if (index.size() != shape.size()) {
        throw std::runtime_error("Invalid index size");
    }
    // Compute flatten index
    int i = 0;
    for (size_t d = 0; d < index.size(); d++) {
        // Compute `i = i * shape + index` recurrently
        i *= shape[d];
        // Make the index positive
        const int p_idx = (0 <= index[d]) ? index[d] : shape[d] + index[d];
        i += p_idx;
    }
    // Direct access
    return *(m_sub->v.get() + i);
}

template <typename... I>
float& NdArray::operator()(I... index) {
    // Use the same implementation of constant method.
    return const_cast<float&>(static_cast<const NdArray&>(*this)(index...));
}

template <typename... I>
const float& NdArray::operator()(I... index) const {
    // Pass to operator[]
    return (*this)[{index...}];
}

// ------------------------------- Reshape Method ------------------------------
NdArray NdArray::reshape(const Shape& shape) const {
    // Check shape validity
    size_t unknown_idx = shape.size();
    size_t size = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        if (shape[i] < 0) {
            if (unknown_idx != shape.size()) {
                throw std::runtime_error("Invalid shape format (multi-neg)");
            } else {
                unknown_idx = i;
            }
        } else {
            size *= static_cast<size_t>(shape[i]);
        }
    }
    Shape new_shape = shape;
    if (unknown_idx == shape.size()) {
        if (m_sub->size != size) {
            std::stringstream ss;
            ss << "Invalid reshape (" << m_sub->size << "->" << size << ")";
            throw std::runtime_error(ss.str());
        }
    } else {
        if (m_sub->size % size != 0) {
            throw std::runtime_error("Invalid reshape (-1)");
        }
        new_shape[unknown_idx] = static_cast<int>(m_sub->size / size);
    }

    // Create reshaped array
    NdArray ret;
    ret.m_sub->size = m_sub->size;            // Same size
    ret.m_sub->shape = std::move(new_shape);  // New shape
    ret.m_sub->v = m_sub->v;                  // Shared elements
    return ret;
}

template <typename... S>
NdArray NdArray::reshape(S... shape) const {
    // Pass to `reshape(Shape)`
    return reshape({shape...});
}

// -------------------------------- Slice Method -------------------------------
NdArray NdArray::slice(const SliceIndex& slice_index) const {
    const Shape& shape = m_sub->shape;

    // Compute slice shape and new positive index
    Shape slice_shape;
    SliceIndex new_index;
    for (size_t i = 0; i < shape.size(); i++) {
        const auto& si = slice_index[i];
        if (slice_index.size() <= i) {
            // All
            slice_shape.push_back(shape[i]);
            new_index.push_back({0, shape[i]});
        } else {
            // Make index positive
            int s = (0 <= si.first) ? si.first : shape[i] + si.first;
            int e = (0 <= si.second) ? si.second : shape[i] + si.second;
            // Clamp
            s = clamp(s, 0, shape[i] - 1);  // Start must be in range.
            e = clamp(e, 0, shape[i]);      // End can be next of the last.
            // Register
            slice_shape.push_back(e - s);
            new_index.push_back({s, e});
        }
    }

    // Copy to slice array
    return CopySlice((*this), slice_shape, new_index);
}

template <typename... I>
NdArray NdArray::slice(std::initializer_list<I>... slice_index) const {
    // Cast `initializer_list` to `pair`, and pass to 'slice(SliceIndex)'
    return slice(SliceIndex{CvtToSliceIndexItem(slice_index)...});
}

// --------------------------------- Dot Method --------------------------------
NdArray NdArray::dot(const NdArray& other) const {
    const NdArray& lhs = *this;
    const NdArray& rhs = other;
    const Shape& l_shape = lhs.shape();
    const Shape& r_shape = rhs.shape();
    if (lhs.size() == 0 || rhs.size() == 0) {
        // Empty array
        throw std::runtime_error("Dot product of empty array");
    } else if (lhs.size() == 1) {
        // Simple multiply (left)
        return static_cast<float>(lhs) * rhs;
    } else if (rhs.size() == 1) {
        // Simple multiply (right)
        return lhs * static_cast<float>(rhs);
    } else if (l_shape.size() == 1 && r_shape.size() == 1) {
        // Inner product of vector (1D, 1D)
        return DotNdArray1d(lhs, rhs);
    } else if (l_shape.size() == 2 && r_shape.size() == 2) {
        // Inner product of 2D matrix (2D, 2D)
        // Special version of NDMD. This is for faster calculation.
        return DotNdArray2d(lhs, rhs);
    } else if (l_shape.size() == 2 && r_shape.size() == 1) {
        // Inner product of 2D matrix and vector (2D, 1D)
        // Special version of ND1D. This is for faster calculation.
        const int n_elem = l_shape[0];
        return DotNdArray2d(lhs, rhs.reshape(r_shape[0], 1)).reshape(n_elem);
    } else if (r_shape.size() == 1) {
        // Broadcast right 1D array
        const Shape shape(l_shape.begin(), l_shape.end() - 1);
        return DotNdArrayNdMd(lhs, rhs.reshape(r_shape[0], 1)).reshape(shape);
    } else {
        // Basic matrix product
        return DotNdArrayNdMd(lhs, rhs);
    }
}

// -------------------------------- Cross Method -------------------------------
NdArray NdArray::cross(const NdArray& other) const {
    const NdArray& lhs = *this;
    const NdArray& rhs = other;
    if (lhs.size() == 0 || rhs.size() == 0) {
        // Empty array
        throw std::runtime_error("Cross product of empty array");
    }
    const Shape& l_shape = lhs.shape();
    const Shape& r_shape = rhs.shape();
    const int l_back = l_shape.back();
    const int r_back = r_shape.back();
    if (l_shape.size() == 1 && r_shape.size() == 1) {
        // 1D cross
        if (l_back == 3 && r_back == 3) {  // [3].cross([3]) -> [3]
            NdArray ret({3});
            CrossNdArray1d1dShape33(ret.data(), lhs.data(), rhs.data());
            return ret;
        } else if (l_back == 3 && r_back == 2) {  // [3].cross([2]) -> [3]
            NdArray ret({3});
            CrossNdArray1d1dShape32(ret.data(), lhs.data(), rhs.data());
            return ret;
        } else if (l_back == 2 && r_back == 3) {  // [2].cross([3]) -> [3]
            NdArray ret({3});
            CrossNdArray1d1dShape23(ret.data(), lhs.data(), rhs.data());
            return ret;
        } else if (l_back == 2 && r_back == 2) {  // [2].cross([2]) -> [1]
            NdArray ret({1});
            CrossNdArray1d1dShape22(ret.data(), lhs.data(), rhs.data());
            return ret;
        }
    } else {
        // ND cross
        if (l_back == 3 && r_back == 3) {  // [3].cross([3]) -> [3]
            return CrossNdArrayNdMd(lhs, rhs, 3, CrossNdArray1d1dShape33);
        } else if (l_back == 3 && r_back == 2) {  // [2].cross([3]) -> [3]
            return CrossNdArrayNdMd(lhs, rhs, 3, CrossNdArray1d1dShape32);
        } else if (l_back == 2 && r_back == 3) {  // [3].cross([2]) -> [3]
            return CrossNdArrayNdMd(lhs, rhs, 3, CrossNdArray1d1dShape23);
        } else if (l_back == 2 && r_back == 2) {  // [2].cross([2]) -> [1]
            auto&& ret = CrossNdArrayNdMd(lhs, rhs, 1, CrossNdArray1d1dShape22);
            const Shape& ret_shape = ret.shape();  // Remove last dim
            return ret.reshape(Shape{ret_shape.begin(), ret_shape.end() - 1});
        }
    }
    throw std::runtime_error(
            "incompatible dimensions for cross product"
            " (dimension must be 2 or 3)");
}

// ---------------------- Template Method Specializations ----------------------
// Assuming up to 10 dimensions.
// For `Empty(S... shape)`
template NdArray NdArray::Empty(int);
template NdArray NdArray::Empty(int, int);
template NdArray NdArray::Empty(int, int, int);
template NdArray NdArray::Empty(int, int, int, int);
template NdArray NdArray::Empty(int, int, int, int, int);
template NdArray NdArray::Empty(int, int, int, int, int, int);
template NdArray NdArray::Empty(int, int, int, int, int, int, int);
template NdArray NdArray::Empty(int, int, int, int, int, int, int, int);
template NdArray NdArray::Empty(int, int, int, int, int, int, int, int, int);
template NdArray NdArray::Empty(int, int, int, int, int, int, int, int, int,
                                int);
// For `Zeros(S... shape)`
template NdArray NdArray::Zeros(int);
template NdArray NdArray::Zeros(int, int);
template NdArray NdArray::Zeros(int, int, int);
template NdArray NdArray::Zeros(int, int, int, int);
template NdArray NdArray::Zeros(int, int, int, int, int);
template NdArray NdArray::Zeros(int, int, int, int, int, int);
template NdArray NdArray::Zeros(int, int, int, int, int, int, int);
template NdArray NdArray::Zeros(int, int, int, int, int, int, int, int);
template NdArray NdArray::Zeros(int, int, int, int, int, int, int, int, int);
template NdArray NdArray::Zeros(int, int, int, int, int, int, int, int, int,
                                int);
// For `Ones(S... shape)`
template NdArray NdArray::Ones(int);
template NdArray NdArray::Ones(int, int);
template NdArray NdArray::Ones(int, int, int);
template NdArray NdArray::Ones(int, int, int, int);
template NdArray NdArray::Ones(int, int, int, int, int);
template NdArray NdArray::Ones(int, int, int, int, int, int);
template NdArray NdArray::Ones(int, int, int, int, int, int, int);
template NdArray NdArray::Ones(int, int, int, int, int, int, int, int);
template NdArray NdArray::Ones(int, int, int, int, int, int, int, int, int);
template NdArray NdArray::Ones(int, int, int, int, int, int, int, int, int,
                               int);
// For `float& operator()(I... index)`
template float& NdArray::operator()(int);
template float& NdArray::operator()(int, int);
template float& NdArray::operator()(int, int, int);
template float& NdArray::operator()(int, int, int, int);
template float& NdArray::operator()(int, int, int, int, int);
template float& NdArray::operator()(int, int, int, int, int, int);
template float& NdArray::operator()(int, int, int, int, int, int, int);
template float& NdArray::operator()(int, int, int, int, int, int, int, int);
template float& NdArray::operator()(int, int, int, int, int, int, int, int,
                                    int);
template float& NdArray::operator()(int, int, int, int, int, int, int, int, int,
                                    int);
// For `const float& operator()(I... index) const`
template const float& NdArray::operator()(int) const;
template const float& NdArray::operator()(int, int) const;
template const float& NdArray::operator()(int, int, int) const;
template const float& NdArray::operator()(int, int, int, int) const;
template const float& NdArray::operator()(int, int, int, int, int) const;
template const float& NdArray::operator()(int, int, int, int, int, int) const;
template const float& NdArray::operator()(int, int, int, int, int, int,
                                          int) const;
template const float& NdArray::operator()(int, int, int, int, int, int, int,
                                          int) const;
template const float& NdArray::operator()(int, int, int, int, int, int, int,
                                          int, int) const;
template const float& NdArray::operator()(int, int, int, int, int, int, int,
                                          int, int, int) const;
// For `NdArray reshape(S... shape) const`
template NdArray NdArray::reshape(int) const;
template NdArray NdArray::reshape(int, int) const;
template NdArray NdArray::reshape(int, int, int) const;
template NdArray NdArray::reshape(int, int, int, int) const;
template NdArray NdArray::reshape(int, int, int, int, int) const;
template NdArray NdArray::reshape(int, int, int, int, int, int) const;
template NdArray NdArray::reshape(int, int, int, int, int, int, int) const;
template NdArray NdArray::reshape(int, int, int, int, int, int, int, int) const;
template NdArray NdArray::reshape(int, int, int, int, int, int, int, int,
                                  int) const;
template NdArray NdArray::reshape(int, int, int, int, int, int, int, int, int,
                                  int) const;
template NdArray NdArray::reshape(int, int, int, int, int, int, int, int, int,
                                  int, int) const;
// For `NdArray slice(std::initializer_list<I>... slice_index) const`
using ISII = std::initializer_list<int>;  // Initializer of Slice Index Item
template NdArray NdArray::slice(ISII) const;
template NdArray NdArray::slice(ISII, ISII) const;
template NdArray NdArray::slice(ISII, ISII, ISII) const;
template NdArray NdArray::slice(ISII, ISII, ISII, ISII) const;
template NdArray NdArray::slice(ISII, ISII, ISII, ISII, ISII) const;
template NdArray NdArray::slice(ISII, ISII, ISII, ISII, ISII, ISII) const;
template NdArray NdArray::slice(ISII, ISII, ISII, ISII, ISII, ISII, ISII) const;
template NdArray NdArray::slice(ISII, ISII, ISII, ISII, ISII, ISII, ISII,
                                ISII) const;
template NdArray NdArray::slice(ISII, ISII, ISII, ISII, ISII, ISII, ISII, ISII,
                                ISII) const;
template NdArray NdArray::slice(ISII, ISII, ISII, ISII, ISII, ISII, ISII, ISII,
                                ISII, ISII) const;
template NdArray NdArray::slice(ISII, ISII, ISII, ISII, ISII, ISII, ISII, ISII,
                                ISII, ISII, ISII) const;

// --------------------------------- Operators ---------------------------------
std::ostream& operator<<(std::ostream& os, const NdArray& x) {
    const size_t size = x.size();
    const Shape& shape = x.shape();
    const float* data = x.data();

    if (size == 0 || shape.size() == 0) {
        // Empty
        os << "[]";
    } else if (shape.size() == 1) {
        // 1-dim
        OutputArrayLine(os, data, size);
    } else {
        // Multi-dim
        OutputArrayMultiDim(os, data, shape, 0);
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const Shape& shape) {
    os << "[";
    for (size_t i = 0; i < shape.size(); i++) {
        os << shape[i];
        if (i < shape.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

NdArray operator+(const NdArray& lhs, const NdArray& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::plus<float>());
}

NdArray operator-(const NdArray& lhs, const NdArray& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::minus<float>());
}

NdArray operator*(const NdArray& lhs, const NdArray& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::multiplies<float>());
}

NdArray operator/(const NdArray& lhs, const NdArray& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::divides<float>());
}

NdArray operator+(const NdArray& lhs, const float& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::plus<float>());
}

NdArray operator-(const NdArray& lhs, const float& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::minus<float>());
}

NdArray operator*(const NdArray& lhs, const float& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::multiplies<float>());
}

NdArray operator/(const NdArray& lhs, const float& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::divides<float>());
}

NdArray operator+(const float& lhs, const NdArray& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::plus<float>());
}

NdArray operator-(const float& lhs, const NdArray& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::minus<float>());
}

NdArray operator*(const float& lhs, const NdArray& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::multiplies<float>());
}

NdArray operator/(const float& lhs, const NdArray& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::divides<float>());
}

NdArray operator+(const NdArray& x) {
    return x;
}

NdArray operator-(const NdArray& x) {
    NdArray ret(x.shape());
    const float* x_data = x.data();
    float* ret_data = ret.data();
    for (size_t i = 0; i < ret.size(); i++) {
        *(ret_data++) = -*(x_data++);  // Copy negative
    }
    return ret;
}

// =============================================================================
// ============================ Variable Definition ============================
// =============================================================================
Variable::Variable() : m_sub(std::make_shared<Substance>()) {}

Variable::Variable(std::shared_ptr<Substance> sub) : m_sub(sub) {}

Variable::Variable(float v) : m_sub(std::make_shared<Substance>(v)) {}

Variable::Variable(const Variable& lhs) = default;  // shallow copy

Variable::Variable(Variable&&) = default;  // move

Variable& Variable::operator=(const Variable& lhs) = default;  // shallow copy

Variable& Variable::operator=(Variable&&) = default;

Variable::~Variable() = default;

// --------------------------------- Substance ---------------------------------
class Variable::Substance {
public:
    Substance() {}
    Substance(float v_) : v(v_) {}
    float v;
    float grad = 0.f;
    Function creator;
};

// ---------------------------------- Methods ----------------------------------
float Variable::data() const {
    return m_sub->v;
}

float Variable::grad() const {
    return m_sub->grad;
}

void Variable::backward() {
    // Set the last gradients 'one'
    for (auto&& output : m_sub->creator.getOutputs()) {
        output.addGrad(1.f);
    }

    // Ordered storage to resolve
    std::map<size_t, Function> cand_funcs;
    // Register the last node
    cand_funcs[m_sub->creator.getRank()] = m_sub->creator;
    // Resolving loop
    while (!cand_funcs.empty()) {
        // Get highest rank function
        auto&& last_func = PopLast(cand_funcs);
        // Ignore no connected function (already resolved)
        if (last_func.getRank() == 0) {
            continue;
        }
        // Call backward
        auto&& inputs = last_func.getInputs();
        auto&& outputs = last_func.getOutputs();
        auto&& inputs_data = CvtFromVariables(inputs);
        auto&& outputs_data = CvtFromVariables(outputs);
        auto&& out_grads = GetGrads(outputs);
        auto&& in_grads = last_func.backward(
                std::move(inputs_data), std::move(outputs_data),
                std::move(out_grads));  // with free
        assert(inputs.size() == in_grads.size());
        // Accumulate gradients
        for (size_t i = 0; i < inputs.size(); i++) {
            inputs[i].addGrad(in_grads[i]);
        }
        // Track chain
        for (size_t i = 0; i < inputs.size(); i++) {
            auto&& func = inputs[i].getCreator();
            if (0 < func.getRank()) {
                cand_funcs[func.getRank()] = func;
            }
        }
        // Remove chain (set rank 0)
        last_func.clear();
    }
}

void Variable::setCreator(Function f) {
    m_sub->creator = f;
    f.setRank(f.getRank() + 1);  // Increase rank
}

Function Variable::getCreator() const {
    return m_sub->creator;
}

void Variable::addGrad(float grad) {
    m_sub->grad += grad;
}

// --------------------------------- Operators ---------------------------------
std::ostream& operator<<(std::ostream& os, Variable& x) {
    return os << x.data();
}

Variable operator+(const Variable& lhs, const Variable& rhs) {
    return F::add(lhs, rhs);
}

Variable operator*(const Variable& lhs, const Variable& rhs) {
    return F::mul(lhs, rhs);
}

// =============================================================================
// ============================ Function Definition ============================
// =============================================================================
Function::Function() : m_sub(std::make_shared<Substance>()) {}

Function::Function(std::shared_ptr<Substance> sub) : m_sub(sub) {}

Function::Function(const Function& lhs) = default;  // shallow copy

Function::Function(Function&&) = default;

Function& Function::operator=(const Function& lhs) = default;  // shallow copy

Function& Function::operator=(Function&&) = default;

Function::~Function() = default;

// --------------------------------- Substance ---------------------------------
class Function::Substance {
public:
    virtual ~Substance() {}
    virtual std::vector<float> forward(const std::vector<float>& x) {
        (void)x;
        throw std::runtime_error("Invalid use of tinydiff::Function");
    }
    virtual std::vector<float> backward(const std::vector<float>& x,
                                        const std::vector<float>& y,
                                        const std::vector<float>& gy) {
        (void)x, (void)y, (void)gy;
        throw std::runtime_error("Invalid use of tinydiff::Function");
    }

    size_t rank = 0;
    Variables inputs;
    Variables outputs;
};

// ---------------------------------- Methods ----------------------------------
void Function::clear() {
    m_sub = std::make_shared<Substance>();
}

Variables Function::operator()(const Variables& x) {
    // Forward (with variable conversion)
    auto&& x_data = CvtFromVariables(x);
    auto&& y_data = forward(std::move(x_data));    // with free
    auto&& y = CvtToVariables(std::move(y_data));  // with free
    // Retain input/output variables
    m_sub->inputs = x;
    m_sub->outputs = y;
    // Set rank of this function
    m_sub->rank = 0;
    for (auto&& x_elem : x) {
        size_t rank = x_elem.getCreator().getRank();
        m_sub->rank = std::max(m_sub->rank, rank);
    }

    // Build chain
    for (auto&& y_elem : y) {
        y_elem.setCreator(m_sub);
    }

    return std::move(y);
}

std::vector<float> Function::forward(const std::vector<float>& x) {
    return m_sub->forward(x);
}

std::vector<float> Function::backward(const std::vector<float>& x,
                                      const std::vector<float>& y,
                                      const std::vector<float>& gy) {
    return m_sub->backward(x, y, gy);
}

Variables Function::getInputs() const {
    return m_sub->inputs;
}

Variables Function::getOutputs() const {
    return m_sub->outputs;
}

void Function::setRank(size_t rank) {
    m_sub->rank = rank;
}

size_t Function::getRank() const {
    return m_sub->rank;
}

// =============================================================================
// =========================== Implemented Functions ===========================
// =============================================================================

namespace F {

class AddSub : public Function::Substance {
public:
    virtual ~AddSub() {}
    virtual std::vector<float> forward(const std::vector<float>& x) {
        CheckSize(x, 2);
        return {x[0] + x[1]};
    }
    virtual std::vector<float> backward(const std::vector<float>& x,
                                        const std::vector<float>& y,
                                        const std::vector<float>& gy) {
        CheckSize(x, 2);
        CheckSize(y, 1);
        CheckSize(gy, 1);
        return {gy[0], gy[0]};
    }
};

class MulSub : public Function::Substance {
public:
    virtual ~MulSub() {}
    virtual std::vector<float> forward(const std::vector<float>& x) {
        CheckSize(x, 2);
        return {x[0] * x[1]};
    }
    virtual std::vector<float> backward(const std::vector<float>& x,
                                        const std::vector<float>& y,
                                        const std::vector<float>& gy) {
        CheckSize(x, 2);
        CheckSize(y, 1);
        CheckSize(gy, 1);
        return {gy[0] * x[1], gy[0] * x[0]};
    }
};

class ExpSub : public Function::Substance {
public:
    virtual ~ExpSub() {}
    virtual std::vector<float> forward(const std::vector<float>& x) {
        CheckSize(x, 1);
        return {std::exp(x[0])};
    }
    virtual std::vector<float> backward(const std::vector<float>& x,
                                        const std::vector<float>& y,
                                        const std::vector<float>& gy) {
        CheckSize(x, 1);
        CheckSize(y, 1);
        CheckSize(gy, 1);
        return {gy[0] * y[0]};
    }
};

// ------------------------------- Helper Class --------------------------------
// Helper to replace default substance with implemented one
template <typename S>
class FunctionImplHelper : public Function {
public:
    FunctionImplHelper() : Function(std::make_shared<S>()) {}
    virtual ~FunctionImplHelper() {}
};

// ------------------------- Alias for Function Classes ------------------------
using Add = FunctionImplHelper<AddSub>;
using Mul = FunctionImplHelper<MulSub>;
using Exp = FunctionImplHelper<ExpSub>;

// ----------------------------- Function Wrapping -----------------------------
Variable add(Variable lhs, Variable rhs) {
    return Add()({lhs, rhs})[0];
}

Variable mul(Variable lhs, Variable rhs) {
    return Mul()({lhs, rhs})[0];
}

Variable exp(Variable x) {
    return Exp()({x})[0];
}

}  // namespace F

#endif  // TINYDIFF_IMPLEMENTATION

}  // namespace tinydiff

#endif /* end of include guard */
