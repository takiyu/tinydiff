#ifndef TINYDIFF_H_ONCE
#define TINYDIFF_H_ONCE

#ifndef TINYDIFF_NO_INCLUDE
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>
#include <vector>
#endif

#ifndef TINYDIFF_NO_NAMESPACE
namespace tinydiff {
#endif  // TINYDIFF_NO_NAMESPACE

// #############################################################################
// ############################ Begin of Declaration ###########################
// #############################################################################
#ifndef TINYDIFF_NO_DECLARATION

// Forward Declaration of TinyDiff
class NdArray;
using NdArrays = std::vector<NdArray>;
class Variable;
using Variables = std::vector<Variable>;
class Function;

// Forward Declaration of TinyNdArray
class NdArray;
using InitShape = std::initializer_list<int>;
using Shape = std::vector<int>;
using Index = std::vector<int>;
using SliceIndex = std::vector<std::pair<int, int>>;
using Axis = std::vector<int>;
template <bool C>
using Float = std::conditional_t<C, const float, float>;

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
    template <bool C>
    class IterBase;
    using Iter = IterBase<false>;
    using ConstIter = IterBase<true>;

    NdArray();
    NdArray(const NdArray&);
    NdArray(NdArray&&) noexcept;
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

    static int GetNumWorkers();
    static void SetNumWorkers(int n_workers);  // -1: Hardware Concurrency
    static int GetBatchScale();
    static void SetBatchScale(int batch_scale);

    // Profiling methods (Defining of `TINYNDARRAY_PROFILE_MEMORY` is needed)
    static size_t GetNumInstance();
    static size_t GetTotalMemory();

    uintptr_t id() const;
    bool empty() const;
    size_t size() const;
    const Shape& shape() const;
    size_t ndim() const;
    Iter data();
    ConstIter data() const;
    void fill(float v);
    NdArray copy() const;
    void resize(const Shape& shape);

    Iter begin();
    Iter end();
    ConstIter begin() const;
    ConstIter end() const;

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
    NdArray flatten() const;  // with copy
    NdArray ravel() const;    // without copy

    NdArray slice(const SliceIndex& slice_index) const;
    template <typename... I>
    NdArray slice(std::initializer_list<I>... slice_index) const;  // {i, j}...

    NdArray dot(const NdArray& other) const;
    NdArray cross(const NdArray& other) const;

    NdArray sum(const Axis& axes = {}, bool keepdims = false) const;
    NdArray mean(const Axis& axes = {}, bool keepdims = false) const;
    NdArray min(const Axis& axes = {}, bool keepdims = false) const;
    NdArray max(const Axis& axes = {}, bool keepdims = false) const;

    // Parameters
    static constexpr int DOT_CACHE_SCALE = 10;  // depends on situation and CPU
    static constexpr int DEFAULT_N_WORKERS = -1;
    static constexpr int DEFAULT_BATCH_SCALE = 4;

    class Substance;

private:
    std::shared_ptr<Substance> m_sub;
    NdArray(std::shared_ptr<Substance> sub);

    static std::random_device s_rand_seed;
    static std::mt19937 s_rand_engine;

    static int s_n_workers;
    static int s_batch_scale;
};

// --------------------------------- Operators ---------------------------------
// Print
std::ostream& operator<<(std::ostream& os, const NdArray& x);
std::ostream& operator<<(std::ostream& os, const Shape& shape);
// Single
NdArray operator+(const NdArray& x);
NdArray operator-(const NdArray& x);
// Arithmetic (NdArray, NdArray)
NdArray operator+(const NdArray& lhs, const NdArray& rhs);
NdArray operator-(const NdArray& lhs, const NdArray& rhs);
NdArray operator*(const NdArray& lhs, const NdArray& rhs);
NdArray operator/(const NdArray& lhs, const NdArray& rhs);
// Arithmetic (NdArray, float)
NdArray operator+(const NdArray& lhs, float rhs);
NdArray operator-(const NdArray& lhs, float rhs);
NdArray operator*(const NdArray& lhs, float rhs);
NdArray operator/(const NdArray& lhs, float rhs);
// Arithmetic (float, NdArray)
NdArray operator+(float lhs, const NdArray& rhs);
NdArray operator-(float lhs, const NdArray& rhs);
NdArray operator*(float lhs, const NdArray& rhs);
NdArray operator/(float lhs, const NdArray& rhs);
// Comparison (NdArray, NdArray)
NdArray operator==(const NdArray& lhs, const NdArray& rhs);
NdArray operator!=(const NdArray& lhs, const NdArray& rhs);
NdArray operator>(const NdArray& lhs, const NdArray& rhs);
NdArray operator>=(const NdArray& lhs, const NdArray& rhs);
NdArray operator<(const NdArray& lhs, const NdArray& rhs);
NdArray operator<=(const NdArray& lhs, const NdArray& rhs);
// Comparison (NdArray, float)
NdArray operator==(const NdArray& lhs, float rhs);
NdArray operator!=(const NdArray& lhs, float rhs);
NdArray operator>(const NdArray& lhs, float rhs);
NdArray operator>=(const NdArray& lhs, float rhs);
NdArray operator<(const NdArray& lhs, float rhs);
NdArray operator<=(const NdArray& lhs, float rhs);
// Comparison (float, NdArray)
NdArray operator==(float lhs, const NdArray& rhs);
NdArray operator!=(float lhs, const NdArray& rhs);
NdArray operator>(float lhs, const NdArray& rhs);
NdArray operator>=(float lhs, const NdArray& rhs);
NdArray operator<(float lhs, const NdArray& rhs);
NdArray operator<=(float lhs, const NdArray& rhs);
// ----------------------------- In-place Operators ----------------------------
// Single
NdArray operator+(NdArray&& x);
NdArray operator-(NdArray&& x);
// Arithmetic (NdArray, NdArray)
NdArray operator+(NdArray&& lhs, NdArray&& rhs);
NdArray operator+(const NdArray& lhs, NdArray&& rhs);
NdArray operator+(NdArray&& lhs, const NdArray& rhs);
NdArray operator-(NdArray&& lhs, NdArray&& rhs);
NdArray operator-(const NdArray& lhs, NdArray&& rhs);
NdArray operator-(NdArray&& lhs, const NdArray& rhs);
NdArray operator*(NdArray&& lhs, NdArray&& rhs);
NdArray operator*(const NdArray& lhs, NdArray&& rhs);
NdArray operator*(NdArray&& lhs, const NdArray& rhs);
NdArray operator/(NdArray&& lhs, NdArray&& rhs);
NdArray operator/(const NdArray& lhs, NdArray&& rhs);
NdArray operator/(NdArray&& lhs, const NdArray& rhs);
// Arithmetic (NdArray, float)
NdArray operator+(NdArray&& lhs, float rhs);
NdArray operator-(NdArray&& lhs, float rhs);
NdArray operator*(NdArray&& lhs, float rhs);
NdArray operator/(NdArray&& lhs, float rhs);
// Arithmetic (float, NdArray)
NdArray operator+(float lhs, NdArray&& rhs);
NdArray operator-(float lhs, NdArray&& rhs);
NdArray operator*(float lhs, NdArray&& rhs);
NdArray operator/(float lhs, NdArray&& rhs);
// Comparison (NdArray, NdArray)
NdArray operator==(NdArray&& lhs, NdArray&& rhs);
NdArray operator==(const NdArray& lhs, NdArray&& rhs);
NdArray operator==(NdArray&& lhs, const NdArray& rhs);
NdArray operator!=(NdArray&& lhs, NdArray&& rhs);
NdArray operator!=(const NdArray& lhs, NdArray&& rhs);
NdArray operator!=(NdArray&& lhs, const NdArray& rhs);
NdArray operator>(NdArray&& lhs, NdArray&& rhs);
NdArray operator>(const NdArray& lhs, NdArray&& rhs);
NdArray operator>(NdArray&& lhs, const NdArray& rhs);
NdArray operator>=(NdArray&& lhs, NdArray&& rhs);
NdArray operator>=(const NdArray& lhs, NdArray&& rhs);
NdArray operator>=(NdArray&& lhs, const NdArray& rhs);
NdArray operator<(NdArray&& lhs, NdArray&& rhs);
NdArray operator<(const NdArray& lhs, NdArray&& rhs);
NdArray operator<(NdArray&& lhs, const NdArray& rhs);
NdArray operator<=(NdArray&& lhs, NdArray&& rhs);
NdArray operator<=(const NdArray& lhs, NdArray&& rhs);
NdArray operator<=(NdArray&& lhs, const NdArray& rhs);
// Comparison (NdArray, float)
NdArray operator==(NdArray&& lhs, float rhs);
NdArray operator!=(NdArray&& lhs, float rhs);
NdArray operator>(NdArray&& lhs, float rhs);
NdArray operator>=(NdArray&& lhs, float rhs);
NdArray operator<(NdArray&& lhs, float rhs);
NdArray operator<=(NdArray&& lhs, float rhs);
// Comparison (float, NdArray)
NdArray operator==(float lhs, NdArray&& rhs);
NdArray operator!=(float lhs, NdArray&& rhs);
NdArray operator>(float lhs, NdArray&& rhs);
NdArray operator>=(float lhs, NdArray&& rhs);
NdArray operator<(float lhs, NdArray&& rhs);
NdArray operator<=(float lhs, NdArray&& rhs);
// Compound Assignment (NdArray, NdArray)
NdArray operator+=(NdArray& lhs, const NdArray& rhs);
NdArray operator+=(NdArray&& lhs, const NdArray& rhs);
NdArray operator-=(NdArray& lhs, const NdArray& rhs);
NdArray operator-=(NdArray&& lhs, const NdArray& rhs);
NdArray operator*=(NdArray& lhs, const NdArray& rhs);
NdArray operator*=(NdArray&& lhs, const NdArray& rhs);
NdArray operator/=(NdArray& lhs, const NdArray& rhs);
NdArray operator/=(NdArray&& lhs, const NdArray& rhs);
// Compound Assignment (NdArray, float)
NdArray operator+=(NdArray& lhs, float rhs);
NdArray operator+=(NdArray&& lhs, float rhs);
NdArray operator-=(NdArray& lhs, float rhs);
NdArray operator-=(NdArray&& lhs, float rhs);
NdArray operator*=(NdArray& lhs, float rhs);
NdArray operator*=(NdArray&& lhs, float rhs);
NdArray operator/=(NdArray& lhs, float rhs);
NdArray operator/=(NdArray&& lhs, float rhs);

// ---------------------------- Operator Functions -----------------------------
// Single operators
NdArray Positive(const NdArray& lhs);
NdArray Negative(const NdArray& lhs);
// Arithmetic operators (NdArray, NdArray)
NdArray Add(const NdArray& lhs, const NdArray& rhs);
NdArray Subtract(const NdArray& lhs, const NdArray& rhs);
NdArray Multiply(const NdArray& lhs, const NdArray& rhs);
NdArray Divide(const NdArray& lhs, const NdArray& rhs);
// Arithmetic operators (NdArray, float)
NdArray Add(const NdArray& lhs, float rhs);
NdArray Subtract(const NdArray& lhs, float rhs);
NdArray Multiply(const NdArray& lhs, float rhs);
NdArray Divide(const NdArray& lhs, float rhs);
// Arithmetic operators (float, NdArray)
NdArray Add(float lhs, const NdArray& rhs);
NdArray Subtract(float lhs, const NdArray& rhs);
NdArray Multiply(float lhs, const NdArray& rhs);
NdArray Divide(float lhs, const NdArray& rhs);
// Comparison operators (NdArray, NdArray)
NdArray Equal(const NdArray& lhs, const NdArray& rhs);
NdArray NotEqual(const NdArray& lhs, const NdArray& rhs);
NdArray Greater(const NdArray& lhs, const NdArray& rhs);       // >
NdArray GreaterEqual(const NdArray& lhs, const NdArray& rhs);  // >=
NdArray Less(const NdArray& lhs, const NdArray& rhs);          // <
NdArray LessEqual(const NdArray& lhs, const NdArray& rhs);     // <=
// Comparison operators (NdArray, float)
NdArray Equal(const NdArray& lhs, float rhs);
NdArray NotEqual(const NdArray& lhs, float rhs);
NdArray Greater(const NdArray& lhs, float rhs);
NdArray GreaterEqual(const NdArray& lhs, float rhs);
NdArray Less(const NdArray& lhs, float rhs);
NdArray LessEqual(const NdArray& lhs, float rhs);
// Comparison operators (float, NdArray)
NdArray Equal(float lhs, const NdArray& rhs);
NdArray NotEqual(float lhs, const NdArray& rhs);
NdArray Greater(float lhs, const NdArray& rhs);
NdArray GreaterEqual(float lhs, const NdArray& rhs);
NdArray Less(float lhs, const NdArray& rhs);
NdArray LessEqual(float lhs, const NdArray& rhs);
// Matrix operators
NdArray Dot(const NdArray& lhs, const NdArray& rhs);
NdArray Matmul(const NdArray& lhs, const NdArray& rhs);
NdArray Cross(const NdArray& lhs, const NdArray& rhs);
// Basic math operators
NdArray Abs(const NdArray& x);
NdArray Sign(const NdArray& x);
NdArray Ceil(const NdArray& x);
NdArray Floor(const NdArray& x);
NdArray Clip(const NdArray& x, float x_min, float x_max);
NdArray Sqrt(const NdArray& x);
NdArray Exp(const NdArray& x);
NdArray Log(const NdArray& x);
NdArray Square(const NdArray& x);
NdArray Power(const NdArray& x, const NdArray& y);
NdArray Power(const NdArray& x, float y);
NdArray Power(float x, const NdArray& y);
// Trigonometric functions
NdArray Sin(const NdArray& x);
NdArray Cos(const NdArray& x);
NdArray Tan(const NdArray& x);
// Inverse trigonometric functions
NdArray ArcSin(const NdArray& x);
NdArray ArcCos(const NdArray& x);
NdArray ArcTan(const NdArray& x);
NdArray ArcTan2(const NdArray& y, const NdArray& x);
NdArray ArcTan2(const NdArray& y, float x);
NdArray ArcTan2(float y, const NdArray& x);
// Axis functions
NdArray Sum(const NdArray& x, const Axis& axes = {}, bool keepdims = false);
NdArray Mean(const NdArray& x, const Axis& axes = {}, bool keepdims = false);
NdArray Min(const NdArray& x, const Axis& axes = {}, bool keepdims = false);
NdArray Max(const NdArray& x, const Axis& axes = {}, bool keepdims = false);
// Logistic functions
bool All(const NdArray& x);
bool Any(const NdArray& x);
NdArray All(const NdArray& x, const Axis& axes, bool keepdims = false);
NdArray Any(const NdArray& x, const Axis& axes, bool keepdims = false);
NdArray Where(const NdArray& cond, const NdArray& x, const NdArray& y);
NdArray Where(const NdArray& cond, const NdArray& x, float y);
NdArray Where(const NdArray& cond, float x, const NdArray& y);
NdArray Where(const NdArray& cond, float x, float y);
// Shape functions
NdArray Reshape(const NdArray& x, const Shape& shape);
NdArray Squeeze(const NdArray& x, const Axis& axes = {});
NdArray ExpandDims(const NdArray& x, int axis);
// Grouping functions
NdArray Stack(const std::vector<NdArray>& xs, int axis = 0);
NdArray Concatenate(const std::vector<NdArray>& xs, int axis = 0);
std::vector<NdArray> Split(const NdArray& x, int n_section, int axis = 0);
std::vector<NdArray> Split(const NdArray& x, const Index& idxs, int axis = 0);
std::vector<NdArray> Separate(const NdArray& x, int axis = 0);
// Change view
NdArray Transpose(const NdArray& x);
NdArray Swapaxes(const NdArray& x, int axis1, int axis2);
NdArray BroadcastTo(const NdArray& x, const Shape& shape);
NdArray SumTo(const NdArray& x, const Shape& shape);
// Inverse
NdArray Inv(const NdArray& x);
// ------------------------ In-place Operator Functions ------------------------
// Single operators
NdArray Positive(NdArray&& lhs);
NdArray Negative(NdArray&& lhs);
// Arithmetic operators (NdArray, NdArray)
NdArray Add(NdArray&& lhs, NdArray&& rhs);
NdArray Add(const NdArray& lhs, NdArray&& rhs);
NdArray Add(NdArray&& lhs, const NdArray& rhs);
NdArray Subtract(NdArray&& lhs, NdArray&& rhs);
NdArray Subtract(const NdArray& lhs, NdArray&& rhs);
NdArray Subtract(NdArray&& lhs, const NdArray& rhs);
NdArray Multiply(NdArray&& lhs, NdArray&& rhs);
NdArray Multiply(const NdArray& lhs, NdArray&& rhs);
NdArray Multiply(NdArray&& lhs, const NdArray& rhs);
NdArray Divide(NdArray&& lhs, NdArray&& rhs);
NdArray Divide(const NdArray& lhs, NdArray&& rhs);
NdArray Divide(NdArray&& lhs, const NdArray& rhs);
// Arithmetic operators (NdArray, float)
NdArray Add(NdArray&& lhs, float rhs);
NdArray Subtract(NdArray&& lhs, float rhs);
NdArray Multiply(NdArray&& lhs, float rhs);
NdArray Divide(NdArray&& lhs, float rhs);
// Arithmetic operators (float, NdArray)
NdArray Add(float lhs, NdArray&& rhs);
NdArray Subtract(float lhs, NdArray&& rhs);
NdArray Multiply(float lhs, NdArray&& rhs);
NdArray Divide(float lhs, NdArray&& rhs);
// Comparison operators (NdArray, NdArray)
NdArray Equal(NdArray&& lhs, NdArray&& rhs);
NdArray Equal(const NdArray& lhs, NdArray&& rhs);
NdArray Equal(NdArray&& lhs, const NdArray& rhs);
NdArray NotEqual(NdArray&& lhs, NdArray&& rhs);
NdArray NotEqual(const NdArray& lhs, NdArray&& rhs);
NdArray NotEqual(NdArray&& lhs, const NdArray& rhs);
NdArray Greater(NdArray&& lhs, NdArray&& rhs);
NdArray Greater(const NdArray& lhs, NdArray&& rhs);
NdArray Greater(NdArray&& lhs, const NdArray& rhs);
NdArray GreaterEqual(NdArray&& lhs, NdArray&& rhs);
NdArray GreaterEqual(const NdArray& lhs, NdArray&& rhs);
NdArray GreaterEqual(NdArray&& lhs, const NdArray& rhs);
NdArray Less(NdArray&& lhs, NdArray&& rhs);
NdArray Less(const NdArray& lhs, NdArray&& rhs);
NdArray Less(NdArray&& lhs, const NdArray& rhs);
NdArray LessEqual(NdArray&& lhs, NdArray&& rhs);
NdArray LessEqual(const NdArray& lhs, NdArray&& rhs);
NdArray LessEqual(NdArray&& lhs, const NdArray& rhs);
// Comparison operators (NdArray, float)
NdArray Equal(NdArray&& lhs, float rhs);
NdArray NotEqual(NdArray&& lhs, float rhs);
NdArray Greater(NdArray&& lhs, float rhs);
NdArray GreaterEqual(NdArray&& lhs, float rhs);
NdArray Less(NdArray&& lhs, float rhs);
NdArray LessEqual(NdArray&& lhs, float rhs);
// Comparison operators (float, NdArray)
NdArray Equal(float lhs, NdArray&& rhs);
NdArray NotEqual(float lhs, NdArray&& rhs);
NdArray Greater(float lhs, NdArray&& rhs);
NdArray GreaterEqual(float lhs, NdArray&& rhs);
NdArray Less(float lhs, NdArray&& rhs);
NdArray LessEqual(float lhs, NdArray&& rhs);
// Basic math operators
NdArray Abs(NdArray&& x);
NdArray Sign(NdArray&& x);
NdArray Ceil(NdArray&& x);
NdArray Floor(NdArray&& x);
NdArray Clip(NdArray&& x, float x_min, float x_max);
NdArray Sqrt(NdArray&& x);
NdArray Exp(NdArray&& x);
NdArray Log(NdArray&& x);
NdArray Square(NdArray&& x);
NdArray Power(NdArray&& x, NdArray&& y);
NdArray Power(const NdArray& x, NdArray&& y);
NdArray Power(NdArray&& x, const NdArray& y);
NdArray Power(NdArray&& x, float y);
NdArray Power(float x, NdArray&& y);
// Trigonometric functions
NdArray Sin(NdArray&& x);
NdArray Cos(NdArray&& x);
NdArray Tan(NdArray&& x);
// Inverse trigonometric functions
NdArray ArcSin(NdArray&& x);
NdArray ArcCos(NdArray&& x);
NdArray ArcTan(NdArray&& x);
NdArray ArcTan2(NdArray&& y, NdArray&& x);
NdArray ArcTan2(const NdArray& y, NdArray&& x);
NdArray ArcTan2(NdArray&& y, const NdArray& x);
NdArray ArcTan2(NdArray&& y, float x);
NdArray ArcTan2(float y, NdArray&& x);
// Logistic functions
NdArray Where(NdArray&& cond, const NdArray& x, const NdArray& y);
NdArray Where(NdArray&& cond, const NdArray& x, float y);
NdArray Where(NdArray&& cond, float x, const NdArray& y);
NdArray Where(NdArray&& cond, float x, float y);
// Inverse
NdArray Inv(NdArray&& x);

// --------------------------------- Iterator ----------------------------------
template <bool C>
class NdArray::IterBase {
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = Float<C>;
    using difference_type = std::ptrdiff_t;
    using pointer = Float<C>*;
    using reference = Float<C>&;

    IterBase(Float<C>* p_);
    virtual ~IterBase();
    Float<C>& operator*() const;
    Float<C>& operator[](int i) const;
    IterBase& operator++();
    IterBase& operator--();
    IterBase operator++(int);
    IterBase operator--(int);
    IterBase operator+(int i) const;
    IterBase operator-(int i) const;
    IterBase& operator+=(int i);
    IterBase& operator-=(int i);
    bool operator==(const IterBase& other) const;
    bool operator!=(const IterBase& other) const;
    operator ConstIter() const;

private:
    Float<C>* p;
};

// --------------------- Iterator Template Implementation ----------------------
template <bool C>
NdArray::IterBase<C>::IterBase(Float<C>* p_) : p(p_) {}

template <bool C>
NdArray::IterBase<C>::~IterBase() {}

template <bool C>
Float<C>& NdArray::IterBase<C>::operator*() const {
    return *p;
}

template <bool C>
Float<C>& NdArray::IterBase<C>::operator[](int i) const {
    return p[i];
}

template <bool C>
NdArray::IterBase<C>& NdArray::IterBase<C>::operator++() {
    p++;
    return *this;
}

template <bool C>
NdArray::IterBase<C>& NdArray::IterBase<C>::operator--() {
    p--;
    return *this;
}

template <bool C>
NdArray::IterBase<C> NdArray::IterBase<C>::operator++(int) {
    IterBase tmp = *this;
    p++;
    return tmp;
}

template <bool C>
NdArray::IterBase<C> NdArray::IterBase<C>::operator--(int) {
    IterBase tmp = *this;
    p--;
    return tmp;
}

template <bool C>
NdArray::IterBase<C> NdArray::IterBase<C>::operator+(int i) const {
    return {p + i};
}

template <bool C>
NdArray::IterBase<C> NdArray::IterBase<C>::operator-(int i) const {
    return {p - i};
}

template <bool C>
NdArray::IterBase<C>& NdArray::IterBase<C>::operator+=(int i) {
    p += i;
    return *this;
}

template <bool C>
NdArray::IterBase<C>& NdArray::IterBase<C>::operator-=(int i) {
    p -= i;
    return *this;
}

template <bool C>
bool NdArray::IterBase<C>::operator==(const IterBase& other) const {
    return p == other.p;
}

template <bool C>
bool NdArray::IterBase<C>::operator!=(const IterBase& other) const {
    return p != other.p;
}

template <bool C>
NdArray::IterBase<C>::operator NdArray::ConstIter() const {
    return NdArray::ConstIter{p};
}

// =============================================================================
// ================================== Variable =================================
// =============================================================================
class Variable {
public:
    Variable();
    Variable(const Variable&);
    Variable(Variable&&) noexcept;
    Variable& operator=(const Variable&);
    Variable& operator=(Variable&&);
    ~Variable();

    Variable(const NdArray& v);

    Variable(FloatList<0> init_list);
    Variable(FloatList<1> init_list);
    Variable(FloatList<2> init_list);
    Variable(FloatList<3> init_list);
    Variable(FloatList<4> init_list);
    Variable(FloatList<5> init_list);
    Variable(FloatList<6> init_list);
    Variable(FloatList<7> init_list);
    Variable(FloatList<8> init_list);
    Variable(FloatList<9> init_list);

    uintptr_t id() const;
    bool empty() const;
    size_t size() const;
    const Shape& shape() const;
    size_t ndim() const;
    Variable copyUnretained() const;

    NdArray& data();
    const NdArray& data() const;
    NdArray& grad();
    const NdArray& grad() const;

    void backward(bool clear_grads = true);

    void setCreator(Function f);
    Function getCreator() const;
    void clearData();
    void clearGrad();
    void addGrad(NdArray&& grad);

    class Subsetance;

private:
    std::shared_ptr<Subsetance> m_sub;
    Variable(std::shared_ptr<Subsetance> sub);
};

// --------------------------------- Operators ---------------------------------
// Print
std::ostream& operator<<(std::ostream& os, const Variable& x);
// Single
Variable operator+(const Variable& x);
Variable operator-(const Variable& x);
// Arithmetic (Variable, Variable)
Variable operator+(const Variable& lhs, const Variable& rhs);
Variable operator-(const Variable& lhs, const Variable& rhs);
Variable operator*(const Variable& lhs, const Variable& rhs);
Variable operator/(const Variable& lhs, const Variable& rhs);
// Arithmetic (Variable, float)
Variable operator+(const Variable& lhs, float rhs);
Variable operator-(const Variable& lhs, float rhs);
Variable operator*(const Variable& lhs, float rhs);
Variable operator/(const Variable& lhs, float rhs);
// Arithmetic (float, Variable)
Variable operator+(float lhs, const Variable& rhs);
Variable operator-(float lhs, const Variable& rhs);
Variable operator*(float lhs, const Variable& rhs);
Variable operator/(float lhs, const Variable& rhs);
// Compound Assignment (Variable, Variable)
Variable operator+=(Variable& lhs, const Variable& rhs);
Variable operator-=(Variable& lhs, const Variable& rhs);
Variable operator*=(Variable& lhs, const Variable& rhs);
Variable operator/=(Variable& lhs, const Variable& rhs);
// Compound Assignment (Variable, float)
Variable operator+=(Variable& lhs, float rhs);
Variable operator-=(Variable& lhs, float rhs);
Variable operator*=(Variable& lhs, float rhs);
Variable operator/=(Variable& lhs, float rhs);

// =============================================================================
// ================================== Function =================================
// =============================================================================
class Function {
public:
    Function();
    Function(const Function&);
    Function(Function&&) noexcept;
    Function& operator=(const Function&);
    Function& operator=(Function&&);
    virtual ~Function();

    // Clear all member variables (including input and output variables)
    void clear();

    // Build computational graph with forwarding
    Variables operator()(const Variables& x);

    NdArrays forward(const NdArrays& x);
    NdArrays backward(const NdArrays& x, const NdArrays& y, const NdArrays& gy);

    Variables getInputs() const;
    Variables getOutputs() const;

    void setRank(size_t rank);
    size_t getRank() const;

    class Subsetance;

protected:
    std::shared_ptr<Subsetance> m_sub;
    Function(std::shared_ptr<Subsetance> sub);
};

// =============================================================================
// ============================ Implemented Function ===========================
// =============================================================================

namespace F {

// Single
Variable Positive(const Variable& x);
Variable Negative(const Variable& x);
// Arithmetic (Variable, Variable)
Variable Add(const Variable& lhs, const Variable& rhs);
Variable Subtract(const Variable& lhs, const Variable& rhs);
Variable Multiply(const Variable& lhs, const Variable& rhs);
Variable Divide(const Variable& lhs, const Variable& rhs);
// Arithmetic (Variable, float)
Variable Add(const Variable& lhs, float rhs);
Variable Subtract(const Variable& lhs, float rhs);
Variable Multiply(const Variable& lhs, float rhs);
Variable Divide(const Variable& lhs, float rhs);
// Arithmetic (float, Variable)
Variable Add(float lhs, const Variable& rhs);
Variable Subtract(float lhs, const Variable& rhs);
Variable Multiply(float lhs, const Variable& rhs);
Variable Divide(float lhs, const Variable& rhs);
// Matrix operators
Variable Matmul(const Variable& lhs, const Variable& rhs);
// Basic math operators
Variable Abs(const Variable& x);
Variable Sign(const Variable& x);   // No backward
Variable Ceil(const Variable& x);   // No backward
Variable Floor(const Variable& x);  // No backward
Variable Clip(const Variable& x, float x_min, float x_max);
Variable Sqrt(const Variable& x);
Variable Exp(const Variable& x);
Variable Log(const Variable& x);
Variable Square(const Variable& x);
Variable Power(const Variable& x, const Variable& y);
Variable Power(const Variable& x, float y);
Variable Power(float x, const Variable& y);
// Trigonometric functions
Variable Sin(const Variable& x);
Variable Cos(const Variable& x);
Variable Tan(const Variable& x);
// Inverse trigonometric functions
Variable ArcSin(const Variable& x);
Variable ArcCos(const Variable& x);
Variable ArcTan(const Variable& x);
Variable ArcTan2(const Variable& y, const Variable& x);
Variable ArcTan2(const Variable& y, float x);
Variable ArcTan2(float y, const Variable& x);
// Axis functions
Variable Sum(const Variable& x, const Axis& axes = {}, bool keepdims = false);
Variable Mean(const Variable& x, const Axis& axes = {}, bool keepdims = false);
Variable Min(const Variable& x, const Axis& axes = {}, bool keepdims = false);
Variable Max(const Variable& x, const Axis& axes = {}, bool keepdims = false);
// Logistic functions
Variable Where(const NdArray& cond, const Variable& x0, const Variable& x1);
Variable Where(const NdArray& cond, const Variable& x0, float x1);
Variable Where(const NdArray& cond, float x0, const Variable& x1);
// Shape functions
Variable Reshape(const Variable& x, const Shape& shape);
Variable Squeeze(const Variable& x, const Axis& axes = {});
Variable ExpandDims(const Variable& x, int axis);
// Grouping functions
Variable Stack(const std::vector<Variable>& xs, int axis = 0);
Variable Concatenate(const std::vector<Variable>& xs, int axis = 0);
std::vector<Variable> Split(const Variable& x, int n_section, int axis = 0);
std::vector<Variable> Split(const Variable& x, const Index& idxs, int axis = 0);
std::vector<Variable> Separate(const Variable& x, int axis = 0);
// Change view
Variable Transpose(const Variable& x);
Variable Swapaxes(const Variable& x, int axis1, int axis2);
Variable BroadcastTo(const Variable& x, const Shape& shape);
Variable SumTo(const Variable& x, const Shape& shape);
// Inverse
Variable Inv(const Variable& x);

}  // namespace F

#endif  // TINYDIFF_NO_DECLARATION
// #############################################################################
// ############################# End of Declaration ############################
// #############################################################################

// #############################################################################
// ############################ Begin of Definitions ###########################
// #############################################################################
#ifdef TINYDIFF_IMPLEMENTATION

// -----------------------------------------------------------------------------
// --------------------------- Utilities for NdArray ---------------------------
// -----------------------------------------------------------------------------
inline float SignOp(float x) {
    if (0.f < x) {
        return 1.f;
    } else if (x < 0.f) {
        return -1.f;
    } else {
        return 0.f;
    }
}

inline float SquareOp(float x) {
    return x * x;
}

template <typename T>
inline T ClipOp(const T& v, const T& lower, const T& upper) {
    return std::min(std::max(v, lower), upper);
}

template <typename F>
inline auto ReverseOp(F op) {
    return [op](float a, float b) { return op(b, a); };  // Swap left and right
}

static int ResolveAxis(int axis, size_t ndim, const std::string& name) {
    // Resolve negative
    const int ndim_i = static_cast<int>(ndim);
    if (axis < 0) {
        axis = ndim_i + axis;
    }
    // Check range
    if (axis < 0 || ndim_i <= axis) {
        std::stringstream ss;
        ss << "Invalid axes for " << name;
        ss << " (" << ndim << "vs" << axis << ")";
        throw std::runtime_error(ss.str());
    }
    return axis;
}

static Axis ResolveAxis(const Axis& axes, size_t ndim, const std::string& name,
                        bool sort = false, bool sort_order_normal = true) {
    // Resolve for each
    Axis ret_axes;
    ret_axes.reserve(axes.size());
    for (auto&& axis : axes) {
        ret_axes.push_back(ResolveAxis(axis, ndim, name));
    }
    // Sort axes
    if (sort) {
        if (sort_order_normal) {
            // Normal order
            std::sort(ret_axes.begin(), ret_axes.end());
        } else {
            // Inverse order
            std::sort(ret_axes.begin(), ret_axes.end(), std::greater<int>());
        }
    }
    return ret_axes;
}

static void GetParallelParams(int size, int& n_workers, int& n_batch,
                              int& batch_size) {
    // Fetch the number of workers
    n_workers = NdArray::GetNumWorkers();
    if (n_workers <= 0) {
        n_workers = static_cast<int>(std::thread::hardware_concurrency());
    }
    // Compute batch size and it number
    n_batch = n_workers * NdArray::GetBatchScale();
    batch_size = size / n_batch + (size % n_batch ? 1 : 0);
    n_workers = std::min(n_workers, batch_size);
}

template <typename F>
void RunParallel(int size, F op) {
    // Decide parallelization parameters
    int n_workers = -1, n_batch = -1, batch_size = -1;
    GetParallelParams(size, n_workers, n_batch, batch_size);

    if (n_workers <= 1) {
        // Single execution
        for (int i = 0; i < size; i++) {
            // Operation
            op(i);
        }
    } else {
        // Parallel execution
        std::atomic<int> next_batch(0);
        std::vector<std::thread> workers(static_cast<size_t>(n_workers));
        for (auto&& worker : workers) {
            worker = std::thread([ =, &next_batch ]() noexcept {
                int batch_cnt = 0;
                while ((batch_cnt = next_batch++) < n_batch) {
                    for (int i = 0; i < batch_size; i++) {
                        const int idx = batch_size * batch_cnt + i;
                        if (size <= idx) {
                            break;
                        }
                        // Operation
                        op(idx);
                    }
                }
            });
        }
        for (auto&& worker : workers) {
            worker.join();
        }
    }
}

template <typename F, typename R>
float RunParallelWithReduce(int size, F op, R reduce, float init_v) {
    // Decide parallelization parameters
    int n_workers = -1, n_batch = -1, batch_size = -1;
    GetParallelParams(size, n_workers, n_batch, batch_size);

    if (n_workers <= 1) {
        // Single execution
        float v = init_v;
        for (int i = 0; i < size; i++) {
            // Operation with reduction
            v = reduce(v, op(i));
        }
        return v;
    } else {
        // Parallel execution
        std::atomic<int> next_batch(0);
        std::vector<std::thread> workers(static_cast<size_t>(n_workers));
        std::vector<float> results(workers.size());
        for (size_t t = 0; t < workers.size(); t++) {
            workers[t] = std::thread([ =, &next_batch, &results ]() noexcept {
                int batch_cnt = 0;
                float v = init_v;
                while ((batch_cnt = next_batch++) < n_batch) {
                    for (int i = 0; i < batch_size; i++) {
                        const int idx = batch_size * batch_cnt + i;
                        if (size <= idx) {
                            break;
                        }
                        // Operation with reduction
                        v = reduce(v, op(idx));
                    }
                }
                results[t] = v;
            });
        }
        for (auto&& worker : workers) {
            worker.join();
        }
        return std::accumulate(results.begin(), results.end(), init_v, reduce);
    }
}

template <typename Iter>
void FillN(Iter&& iter, const int n, float v) {
    // Fill in parallel
    RunParallel(n, [&](int i) { iter[i] = v; });
}

template <typename RetIter, typename SrcIter>
void Copy(RetIter&& ret_iter, SrcIter&& src_iter, const int n) {
    // Copy in parallel
    RunParallel(n, [&](int i) { ret_iter[i] = src_iter[i]; });
}

template <typename F>
inline void ApplyOpSimple(NdArray& ret, F op) {
    auto&& ret_data = ret.data();
    // Simply apply all
    RunParallel(static_cast<int>(ret.size()),
                [&](int i) { ret_data[i] = op(); });
}

template <typename F>
inline void ApplyOpSimple(NdArray& ret, const NdArray& src, F op) {
    auto&& ret_data = ret.data();
    auto&& src_data = src.data();
    // Simply apply all
    RunParallel(static_cast<int>(ret.size()),
                [&](int i) { ret_data[i] = op(src_data[i]); });
}

template <typename F>
inline void ApplyOpSimple(NdArray& ret, const NdArray& lhs, const NdArray& rhs,
                          F op) {
    auto&& ret_data = ret.data();
    auto&& l_data = lhs.data();
    auto&& r_data = rhs.data();
    // Simply apply all
    RunParallel(static_cast<int>(ret.size()),
                [&](int i) { ret_data[i] = op(l_data[i], r_data[i]); });
}

template <typename F>
inline void ApplyOpSimple(NdArray& ret, const NdArray& lhs, const float rhs,
                          F op) {
    auto&& ret_data = ret.data();
    auto&& l_data = lhs.data();
    // Simply apply all
    RunParallel(static_cast<int>(ret.size()),
                [&](int i) { ret_data[i] = op(l_data[i], rhs); });
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

// --------------- Utilities for NdArray (Float initializer list) --------------
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
inline std::list<int> CheckFListShapeImpl(const FloatList<0>& init_list) {
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

// ---------------------- Utilities for NdArray (Random) -----------------------
template <typename D, typename R>
NdArray CreateRandomArray(const Shape& shape, D&& dist, R&& rand_engine) {
    // Create empty array
    NdArray ret(shape);
    // Fill by random value
    ApplyOpSimple(ret, [&]() { return static_cast<float>(dist(rand_engine)); });
    return ret;
}

// ----------------------- Utilities for NdArray (Slice) -----------------------
static void CopySliceImpl(const NdArray::ConstIter& src_data,
                          const NdArray::Iter& ret_data, const Shape& ret_shape,
                          const std::vector<int>& prev_offsets,
                          const std::vector<int>& post_offsets,
                          const int src_step_top, const int ret_step_top) {
    const size_t n_depth = ret_shape.size();

    // Run in parallel (Only top level)
    RunParallel(ret_shape[0], [&](int ret_top) {
        const int ret_idx_base = ret_top * ret_step_top;
        const int src_idx_base = ret_top * src_step_top + prev_offsets[0];

        // Create stacks and counter
        std::vector<int> ret_cnts(n_depth);
        size_t depth = 1;  // Start from 1
        int src_idx = 0;

        for (int ret_idx = 0; ret_idx < ret_step_top; ret_idx++) {
            // Go down
            for (; depth < n_depth; depth++) {
                src_idx += prev_offsets[depth];  // Forward prev offset
            }

            // Operate
            ret_data[ret_idx_base + ret_idx] = src_data[src_idx_base + src_idx];
            src_idx += 1;  // Forward normally

            // Go up and count (Down to 1)
            for (; 1 < depth; depth--) {
                const size_t prev_d = depth - 1;
                ret_cnts[prev_d]++;  // Count up
                if (ret_cnts[prev_d] < ret_shape[prev_d]) {
                    break;  // Continue normally
                }
                // Go upper depth
                ret_cnts[prev_d] = 0;             // Clear count
                src_idx += post_offsets[prev_d];  // Forward post offset
            }
        }
    });
}

template <bool IsPrev>
std::vector<int> ComputeSliceOffset(const std::vector<int>& child_sizes,
                                    const SliceIndex& slice_index,
                                    const Shape& src_shape) {
    std::vector<int> offsets;
    offsets.reserve(child_sizes.size());
    for (size_t depth = 0; depth < child_sizes.size(); depth++) {
        const auto& si = slice_index[depth];
        const int len = (IsPrev ? si.first : src_shape[depth] - si.second);
        offsets.push_back(child_sizes[depth] * len);
    }
    return offsets;
}

static NdArray CopySlice(const NdArray& src, const Shape& ret_shape,
                         const SliceIndex& slice_index) {
    const Shape& src_shape = src.shape();

    // Pre-compute child sizes
    const std::vector<int>& child_sizes = ComputeChildSizes(src_shape);
    // Pre-compute offsets
    const std::vector<int>& prev_offsets =
            ComputeSliceOffset<true>(child_sizes, slice_index, src_shape);
    const std::vector<int>& post_offsets =
            ComputeSliceOffset<false>(child_sizes, slice_index, src_shape);

    // Pre-compute top steps for parallel
    const int ret_step_top = std::accumulate(
            ret_shape.begin() + 1, ret_shape.end(), 1, std::multiplies<int>());
    const int src_step_top = child_sizes[0];

    // Create slice instance
    NdArray ret(ret_shape);

    // Start to copy
    auto&& src_data = src.data();
    auto&& ret_data = ret.data();
    CopySliceImpl(src_data, ret_data, ret_shape, prev_offsets, post_offsets,
                  src_step_top, ret_step_top);

    return ret;
}

inline std::pair<int, int> CvtToSliceIndexItem(std::initializer_list<int> l) {
    if (l.size() != 2) {
        throw std::runtime_error("Invalid slice index format");
    }
    return {*l.begin(), *(l.begin() + 1)};
}

// ------------------ Utilities for NdArray (Broadcast common) -----------------
static Shape CheckBroadcastable(const Shape& l_shape, const Shape& r_shape) {
    // We assuming left array has deeper shape than right one.
    if (l_shape.size() < r_shape.size()) {
        return CheckBroadcastable(r_shape, l_shape);  // Swap
    }
    // `l_shape.size()` is maximum depth.

    // Check empty
    if (r_shape.size() == 0 || (r_shape.size() == 1 && r_shape[0] == 0)) {
        throw std::runtime_error("Broadcast of empty array");
    }

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

static size_t ReduceShapesBroadcast(Shape& ret_shape, Shape& l_shape,
                                    Shape& r_shape, const size_t depth_offset) {
    // Require `ret_shape.size() == l_shape.size() == r_shape.size()`

    // Remove meaningless dimensions.
    Shape ret_shape_cleaned, l_shape_cleaned, r_shape_cleaned;
    int size_pool = 1;
    size_t depth = 0;
    for (; depth < ret_shape.size() - depth_offset; depth++) {
        if (l_shape[depth] == r_shape[depth]) {
            // Store
            size_pool *= l_shape[depth];
        } else {
            // Pop
            if (size_pool != 1) {
                ret_shape_cleaned.push_back(size_pool);
                l_shape_cleaned.push_back(size_pool);
                r_shape_cleaned.push_back(size_pool);
                size_pool = 1;
            }
            // Through current dimension
            ret_shape_cleaned.push_back(ret_shape[depth]);
            l_shape_cleaned.push_back(l_shape[depth]);
            r_shape_cleaned.push_back(r_shape[depth]);
        }
    }
    // Pop
    if (size_pool != 1 || ret_shape_cleaned.size() == 0) {
        ret_shape_cleaned.push_back(size_pool);
        l_shape_cleaned.push_back(size_pool);
        r_shape_cleaned.push_back(size_pool);
    }
    // Store actual depth count
    const size_t n_depth = ret_shape_cleaned.size();
    // Pass through included in `depth_offset`.
    for (; depth < ret_shape.size(); depth++) {
        ret_shape_cleaned.push_back(ret_shape[depth]);
        l_shape_cleaned.push_back(l_shape[depth]);
        r_shape_cleaned.push_back(r_shape[depth]);
    }
    // Return
    ret_shape = std::move(ret_shape_cleaned);
    l_shape = std::move(l_shape_cleaned);
    r_shape = std::move(r_shape_cleaned);
    return n_depth;
}

template <typename F>
void ApplyOpBroadcastImpl(const NdArray::Iter& ret_data,
                          const NdArray::ConstIter& l_data,
                          const NdArray::ConstIter& r_data,
                          const Shape& ret_shape, const int ret_size,
                          const std::vector<int>& l_steps,
                          const std::vector<int>& r_steps,
                          const size_t start_depth, const size_t n_depth,
                          const int ret_step, F op) {
    // Create stacks and counter
    std::vector<int> ret_cnts(n_depth);
    std::vector<int> l_idx_stack(n_depth), r_idx_stack(n_depth);
    size_t depth = start_depth;
    int l_idx = 0;
    int r_idx = 0;

    for (int ret_idx = 0; ret_idx < ret_size; ret_idx += ret_step) {
        // Go down
        for (; depth < n_depth; depth++) {
            l_idx_stack[depth] = l_idx;  // Push stack
            r_idx_stack[depth] = r_idx;
        }

        // Operate
        op(ret_data + ret_idx, l_data + l_idx, r_data + r_idx);

        // Go up and count
        for (; start_depth < depth; depth--) {
            const size_t prev_d = depth - 1;
            ret_cnts[prev_d]++;        // Count up
            l_idx += l_steps[prev_d];  // Forward index
            r_idx += r_steps[prev_d];
            if (ret_cnts[prev_d] < ret_shape[prev_d]) {
                break;  // Continue normally
            }
            // Go upper depth
            ret_cnts[prev_d] = 0;         // Clear count
            l_idx = l_idx_stack[prev_d];  // Pop stack
            r_idx = r_idx_stack[prev_d];
        }
    }
}

template <typename F>
void ApplyOpBroadcast(NdArray& ret, const NdArray& lhs, const NdArray& rhs,
                      const size_t depth_offset, const int ret_step, F op) {
    Shape ret_shape = ret.shape();

    // Pre-compute padded shapes
    Shape l_shape = PadShape(lhs.shape(), ret_shape.size());
    Shape r_shape = PadShape(rhs.shape(), ret_shape.size());

    // Pre-compute reduced shapes
    const size_t n_depth =
            ReduceShapesBroadcast(ret_shape, l_shape, r_shape, depth_offset);

    // Pre-compute child sizes
    const std::vector<int>& ret_child_sizes = ComputeChildSizes(ret_shape);
    const std::vector<int>& l_child_sizes = ComputeChildSizes(l_shape);
    const std::vector<int>& r_child_sizes = ComputeChildSizes(r_shape);

    // Pre-compute steps
    std::vector<int> l_steps, r_steps;
    l_steps.reserve(n_depth);
    r_steps.reserve(n_depth);
    for (size_t depth = 0; depth < n_depth; depth++) {
        const int& l_s = l_shape[depth];
        const int& r_s = r_shape[depth];
        const int l_step = (l_s == r_s || r_s == 1) ? l_child_sizes[depth] : 0;
        const int r_step = (l_s == r_s || l_s == 1) ? r_child_sizes[depth] : 0;
        l_steps.push_back(l_step);
        r_steps.push_back(r_step);
    }

#if 1  // Run in parallel
    RunParallel(ret_shape[0], [&](int i) {
        const int ret_size = static_cast<int>(ret.size()) / ret_shape[0];
        ApplyOpBroadcastImpl(ret.data() + ret_child_sizes[0] * i,
                             lhs.data() + l_steps[0] * i,
                             rhs.data() + r_steps[0] * i, ret_shape, ret_size,
                             l_steps, r_steps, 1, n_depth, ret_step, op);
    });
#else  // Run sequentially
    ApplyOpBroadcastImpl(ret.data(), lhs.data(), rhs.data(), ret_shape,
                         static_cast<int>(ret.size()), l_steps, r_steps, 0,
                         n_depth, ret_step, op);
#endif
}

template <typename F>
inline auto WrapOpForIter(F op) {
    return [op](const NdArray::Iter& o, const NdArray::ConstIter& l,
                const NdArray::ConstIter& r) {
        *o = op(*l, *r);  // wrap pointer operation for iterator's one
    };
}

// ------------------ Utilities for NdArray (Single operator) ------------------
template <typename F>
NdArray ApplySingleOp(const NdArray& x, F op) {
    NdArray ret(x.shape());
    ApplyOpSimple(ret, x, op);
    return ret;
}

template <typename F>
NdArray ApplySingleOpInplace(NdArray&& x, F op) {
    ApplyOpSimple(x, x, op);
    return std::move(x);
}

// ------------------- Utilities for NdArray (Dual operator) -------------------
template <typename F>
NdArray ApplyDualOp(const NdArray& lhs, const NdArray& rhs, F op) {
    if (lhs.shape() == rhs.shape()) {
        // Apply without broadcast because of same size for speed up.
        NdArray ret(lhs.shape());
        // Simply apply all
        ApplyOpSimple(ret, lhs, rhs, op);
        return ret;
    } else {
        // Check it is possible to broadcast
        const Shape& ret_shape = CheckBroadcastable(lhs.shape(), rhs.shape());
        // Apply broadcast
        NdArray ret(ret_shape);
        ApplyOpBroadcast(ret, lhs, rhs, 0, 1, WrapOpForIter(op));
        return ret;
    }
}

template <typename F>
NdArray ApplyDualOp(const NdArray& lhs, const float rhs, F op) {
    // Broadcast right float
    NdArray ret(lhs.shape());
    // Simply apply all
    ApplyOpSimple(ret, lhs, rhs, op);
    return ret;
}

template <typename F>
inline NdArray ApplyDualOp(const float lhs, const NdArray& rhs, F op) {
    // Swap left and right
    return ApplyDualOp(rhs, lhs, ReverseOp(op));
}

// -------------- Utilities for NdArray (Dual operator in-place) ---------------
template <typename F>
NdArray ApplyDualOpInplace(NdArray&& lhs, NdArray&& rhs, F op,
                           const bool allow_new = true) {
    if (lhs.shape() == rhs.shape()) {
        // Apply without broadcast because of same size for speed up.
        ApplyOpSimple(lhs, lhs, rhs, op);  // Use left as result
        return std::move(lhs);
    } else {
        // Check it is possible to broadcast
        const Shape& ret_shape = CheckBroadcastable(lhs.shape(), rhs.shape());
        // Select result with shape matching
        NdArray&& ret =
                (ret_shape == lhs.shape()) ? lhs :                  // left
                        (ret_shape == rhs.shape()) ? rhs :          // right
                                (allow_new) ? NdArray(ret_shape) :  // new
                                        throw std::runtime_error(
                                                "Invalid shape for in-place"
                                                " operation");
        // Apply broadcast
        ApplyOpBroadcast(ret, lhs, rhs, 0, 1, WrapOpForIter(op));
        return std::move(ret);
    }
}

template <typename F>
NdArray ApplyDualOpInplace(NdArray&& lhs, const NdArray& rhs, F op,
                           const bool allow_new = true) {
    if (lhs.shape() == rhs.shape()) {
        // Apply without broadcast because of same size for speed up.
        ApplyOpSimple(lhs, lhs, rhs, op);
        return std::move(lhs);
    } else {
        // Check it is possible to broadcast
        const Shape& ret_shape = CheckBroadcastable(lhs.shape(), rhs.shape());
        // Select result with shape matching
        NdArray&& ret =
                (ret_shape == lhs.shape()) ? lhs :          // left
                        (allow_new) ? NdArray(ret_shape) :  // new
                                throw std::runtime_error(
                                        "Invalid shape for in-place operation");
        // Apply broadcast (result matrix is lhs)
        ApplyOpBroadcast(ret, lhs, rhs, 0, 1, WrapOpForIter(op));
        return std::move(ret);
    }
}

template <typename F>
inline NdArray ApplyDualOpInplace(const NdArray& lhs, NdArray&& rhs, F op,
                                  const bool allow_new = true) {
    // Swap left and right
    return ApplyDualOpInplace(std::move(rhs), lhs, ReverseOp(op), allow_new);
}

template <typename F>
NdArray ApplyDualOpInplace(NdArray&& lhs, float rhs, F op) {
    // Broadcast right float
    // Simply apply all
    ApplyOpSimple(lhs, lhs, rhs, op);
    return std::move(lhs);
}

template <typename F>
inline NdArray ApplyDualOpInplace(float lhs, NdArray&& rhs, F op) {
    // Swap left and right
    return ApplyDualOpInplace(std::move(rhs), lhs, ReverseOp(op));
}

// ----------------------- Utilities for NdArray (Where) -----------------------
static float WhereOpLeft(const float& c, const float& x) {
    if (c != static_cast<float>(false)) {
        return x;  // True
    }
    return 0.f;
}

static float WhereOpRight(const float& c, const float& y) {
    if (c == static_cast<float>(false)) {
        return y;  // False
    }
    return 0.f;
}

static NdArray ApplyWhereOp(const NdArray& cond, const NdArray& x,
                            const NdArray& y) {
    auto tmp = ApplyDualOp(cond, x, WhereOpLeft);
    return ApplyDualOp(cond, y, WhereOpRight) + tmp;
}

static NdArray ApplyWhereOp(const NdArray& cond, const NdArray& x, float y) {
    auto tmp = ApplyDualOp(cond, x, WhereOpLeft);
    return ApplyDualOp(cond, y, WhereOpRight) + tmp;
}

static NdArray ApplyWhereOp(const NdArray& cond, float x, const NdArray& y) {
    auto tmp = ApplyDualOp(cond, x, WhereOpLeft);
    return ApplyDualOp(cond, y, WhereOpRight) + tmp;
}

static NdArray ApplyWhereOp(const NdArray& cond, float x, float y) {
    auto tmp = ApplyDualOp(cond, x, WhereOpLeft);
    return ApplyDualOp(cond, y, WhereOpRight) + tmp;
}

// ----------------------- Utilities for NdArray (Where) -----------------------
static NdArray ApplyWhereOpInplace(NdArray&& cond, const NdArray& x,
                                   const NdArray& y) {
    auto tmp = ApplyDualOp(cond, x, WhereOpLeft);
    return ApplyDualOp(std::move(cond), y, WhereOpRight) + tmp;
}

static NdArray ApplyWhereOpInplace(NdArray&& cond, const NdArray& x, float y) {
    auto tmp = ApplyDualOp(cond, x, WhereOpLeft);
    return ApplyDualOp(std::move(cond), y, WhereOpRight) + tmp;
}

static NdArray ApplyWhereOpInplace(NdArray&& cond, float x, const NdArray& y) {
    auto tmp = ApplyDualOp(cond, x, WhereOpLeft);
    return ApplyDualOp(std::move(cond), y, WhereOpRight) + tmp;
}

static NdArray ApplyWhereOpInplace(NdArray&& cond, float x, float y) {
    auto tmp = ApplyDualOp(cond, x, WhereOpLeft);
    return ApplyDualOp(std::move(cond), y, WhereOpRight) + tmp;
}

// ------------------- Utilities for NdArray (Axis reduction) ------------------
static Shape CheckReductable(const Shape& shape, const Axis& axes,
                             bool keepdims) {
    // Mark reduction axes
    std::vector<char> mark(shape.size(), false);
    const int n_shape = static_cast<int>(shape.size());
    for (auto&& axis : axes) {
        if (0 <= axis && axis < n_shape) {
            mark[static_cast<size_t>(axis)] = true;
        } else {
            throw std::runtime_error("Invalid axes for reduction");
        }
    }

    if (keepdims) {
        // Pick up unmarked dimension
        Shape ret_shape_pad;
        for (size_t i = 0; i < mark.size(); i++) {
            if (mark[i]) {
                ret_shape_pad.push_back(1);
            } else {
                ret_shape_pad.push_back(shape[i]);
            }
        }
        return ret_shape_pad;
    } else {
        // No necessary
        return {};
    }
}

static auto ComputeReduceSizes(const Shape& src_shape, const size_t axis) {
    // Compute result shape
    Shape ret_shape;
    for (size_t dim = 0; dim < src_shape.size(); dim++) {
        if (dim != axis) {
            ret_shape.push_back(src_shape[dim]);
        }
    }
    if (ret_shape.empty()) {  // For all reduction
        ret_shape.push_back(1);
    }

    // Compute sizes
    int n_upper = 1, n_lower = 1, n_reduce = 0;
    for (size_t dim = 0; dim < src_shape.size(); dim++) {
        // Sizes
        if (dim < axis) {
            n_upper *= src_shape[dim];
        } else if (axis < dim) {
            n_lower *= src_shape[dim];
        } else {
            n_reduce = src_shape[dim];
        }
    }

    // Return
    return std::make_tuple(std::move(ret_shape), n_upper, n_lower, n_reduce);
}

template <typename F>
float ReduceAxisAll(const NdArray::ConstIter& data, const size_t size,
                    const float init_v, F reduce_op) {
    auto&& op = [&](int i) { return data[i]; };
    const float ret = RunParallelWithReduce(static_cast<int>(size), op,
                                            reduce_op, init_v);
    return ret;
}

template <typename F>
NdArray ReduceAxisOne(const NdArray& src, const size_t axis, const float init_v,
                      F reduce_op) {
    const Shape& src_shape = src.shape();

    // Compute sizes
    auto reduce_sizes = ComputeReduceSizes(src_shape, axis);
    const Shape& ret_shape = std::get<0>(reduce_sizes);
    const int n_upper = std::get<1>(reduce_sizes);
    const int n_lower = std::get<2>(reduce_sizes);
    const int n_reduce = std::get<3>(reduce_sizes);

    // Create result array with fill
    NdArray ret(ret_shape, init_v);

    auto&& src_data = src.data();
    auto&& ret_data = ret.data();

    // Reduce function
    auto&& reduce = [=](int u_idx) {
        const int ret_idx_base = u_idx * n_lower;
        const int src_idx_base0 = u_idx * n_reduce * n_lower;
        for (int redu_idx = 0; redu_idx < n_reduce; redu_idx++) {
            const int src_idx_base1 = src_idx_base0 + redu_idx * n_lower;
            for (int l_idx = 0; l_idx < n_lower; l_idx++) {
                // Reduce
                float& r = ret_data[ret_idx_base + l_idx];
                const float s = src_data[src_idx_base1 + l_idx];
                r = reduce_op(r, s);
            }
        }
    };

    // TODO: Run parallel for `axis == 0` (means `n_upper == 1`)

#if 1  // Run in parallel
    RunParallel(n_upper, reduce);
#else  // Run sequentially
    for (int u_idx = 0; u_idx < n_upper; u_idx++) {
        reduce(u_idx);
    }
#endif
    return ret;
}

template <typename F>
NdArray ReduceAxis(const NdArray& src, const Axis& axes_raw, bool keepdims,
                   const float init_v, F reduce_op) {
    if (axes_raw.size() == 0) {
        // No Axis -> Reduce all
        float ret_v = ReduceAxisAll(src.data(), src.size(), init_v, reduce_op);
        NdArray ret = {ret_v};
        // Reshape for keepdims
        if (keepdims) {
            Shape ret_shape(src.shape().size(), 1);
            ret = ret.reshape(ret_shape);
        }
        return ret;
    } else {
        // Resolve axes (sort: on)
        const Axis& axes = ResolveAxis(axes_raw, src.ndim(), "Reduce", true);

        // Check it is possible to reduce.
        Shape src_shape = src.shape();
        const Shape& ret_shape_pad = CheckReductable(src_shape, axes, keepdims);

        // Reduce axes one by one
        NdArray ret;
        for (size_t i = 0; i < axes.size(); i++) {
            // From back
            const size_t axis = static_cast<size_t>(axes[axes.size() - i - 1]);
            // Reduce
            if (i == 0) {
                ret = ReduceAxisOne(src, axis, init_v, reduce_op);
            } else {
                ret = ReduceAxisOne(ret, axis, init_v, reduce_op);
            }
        }

        // Reshape for keepdims
        if (keepdims) {
            ret = ret.reshape(ret_shape_pad);
        }
        return ret;
    }
}

template <typename F>
NdArray ReduceAxisNoEmpty(const NdArray& src, const Axis& axes, bool keepdims,
                          const float init_v, F reduce_op) {
    // Check empty
    if (src.size() == 0) {
        throw std::runtime_error("zero-size array to reduction operation");
    }
    // Call normally
    return ReduceAxis(src, axes, keepdims, init_v, reduce_op);
}

// ----------------------- Utilities for NdArray (Print) -----------------------
static void OutputArrayLine(std::ostream& os, const NdArray::ConstIter& data,
                            const int size) {
    os << "[";  // Begin of a line
    for (int i = 0; i < size; i++) {
        os << data[i];  // Output an element
        if (i == size - 1) {
            os << "]";  // End of a line
        } else {
            os << ", ";  // Splitter of an element
        }
    }
}

static void OutputArrayMultiDim(std::ostream& os,
                                const NdArray::ConstIter& data,
                                const Shape& shape,
                                const std::vector<int>& child_sizes,
                                size_t depth) {
    for (int i = 0; i < shape[depth]; i++) {
        // Heading
        if (i == 0) {
            os << "[";  // begin of array
        } else {
            for (size_t d = 0; d < depth + 1; d++) {  // array indent
                os << " ";
            }
        }

        // Output internal array
        const int& child_size = child_sizes[depth];
        if (depth == shape.size() - 2) {
            OutputArrayLine(os, data + child_size * i, shape[depth + 1]);
        } else {
            OutputArrayMultiDim(os, data + child_size * i, shape, child_sizes,
                                depth + 1);
        }

        // Tailing
        if (i == shape[depth] - 1) {
            os << "]";  // End of array
        } else {
            os << "," << std::endl;  // Splitter of array
        }
    }
}

static void OutputNdArray(std::ostream& os, const NdArray& x) {
    const int size = static_cast<int>(x.size());
    const Shape& shape = x.shape();
    const std::vector<int>& child_sizes = ComputeChildSizes(shape);

    if (size == 0 || shape.size() == 0) {
        // Empty
        os << "[]";
    } else if (shape.size() == 1) {
        // 1-dim
        OutputArrayLine(os, x.data(), size);
    } else {
        // Multi-dim
        OutputArrayMultiDim(os, x.data(), shape, child_sizes, 0);
    }
}

static void OutputShape(std::ostream& os, const Shape& shape) {
    os << "[";
    for (size_t i = 0; i < shape.size(); i++) {
        os << shape[i];
        if (i < shape.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
}

// -------------------- Utilities for NdArray (Dot product) --------------------
static NdArray DotNdArray1d(const NdArray& lhs, const NdArray& rhs) {
    const size_t size = lhs.size();
    if (size != rhs.size()) {
        throw std::runtime_error("Invalid size for inner product of 1D");
    }
    // Inner product of vectors
    auto&& l_data = lhs.data();
    auto&& r_data = rhs.data();
    auto&& op = [&](int i) { return l_data[i] * r_data[i]; };
    // Run in parallel
    const float sum = RunParallelWithReduce(static_cast<int>(size), op,
                                            std::plus<float>(), 0.f);
    return {sum};
}

static void DotNdArray1d2dImplColMajor(const NdArray::Iter& ret_data,
                                       const NdArray::ConstIter& l_data,
                                       const NdArray::ConstIter& r_data,
                                       const int n_col, const int n_contract) {
    // Zero initialization (no parallel)
    std::fill_n(ret_data, n_col, 0.f);
    // Col-major dot product
    int r_idx = 0;
    for (int l_idx = 0; l_idx < n_contract; l_idx++) {
        for (int col_cnt = 0; col_cnt < n_col; col_cnt++) {
            ret_data[col_cnt] += l_data[l_idx] * r_data[r_idx];
            r_idx++;
        }
    }
}

static void DotNdArray1d2dImplRowMajor(const NdArray::Iter& ret_data,
                                       const NdArray::ConstIter& l_data,
                                       const NdArray::ConstIter& r_data,
                                       const int n_col, const int n_contract) {
    // Row-major dot product
    for (int col_cnt = 0; col_cnt < n_col; col_cnt++) {
        float sum = 0.f;
        int r_idx = col_cnt;
        for (int l_idx = 0; l_idx < n_contract; l_idx++) {
            sum += l_data[l_idx] * r_data[r_idx];
            r_idx += n_col;
        }
        ret_data[col_cnt] = sum;
    }
}

static auto SelectDot1d2dOp(const Shape& l_shape, const Shape& r_shape) {
    // Debug macros
#if defined(TINYNDARRAY_FORCE_DOT_COLMAJOR)
    (void)l_shape;
    (void)r_shape;
    return DotNdArray1d2dImplColMajor;  // Col
#elif defined(TINYNDARRAY_FORCE_DOT_RAWMAJOR)
    (void)l_shape;
    (void)r_shape;
    return DotNdArray1d2dImplRowMajor;  // Row
#endif

    // Decide which major is better
    const int left = l_shape.end()[-1];
    const int right = r_shape.end()[-2] * r_shape.end()[-1];
    if (left * NdArray::DOT_CACHE_SCALE < right) {
        return DotNdArray1d2dImplColMajor;  // Col
    } else {
        return DotNdArray1d2dImplRowMajor;  // Row
    }
}

template <typename F1d2d>
void DotNdArrayNdMdImpl(const NdArray::Iter& ret_data,
                        const NdArray::ConstIter& l_data,
                        const NdArray::ConstIter& r_data, const int n_l,
                        const int n_r, const int ret_step, const int l_step,
                        const int r_step, F1d2d op_1d2d) {
    const int& n_contract = l_step;
    const int& n_col = ret_step;
    const int& ret_idx_base = n_r;
#if 1  // Run in parallel
    if (n_l < n_r) {
        RunParallel(n_r, [&](int r_cnt) {  // Right-hand side loop
            const int ret_step_base = ret_idx_base * ret_step;
            const int r_idx = r_cnt * r_step;
            int l_idx = 0;
            int ret_idx = r_cnt * ret_step;
            for (int l_cnt = 0; l_cnt < n_l; l_cnt++) {
                op_1d2d(ret_data + ret_idx, l_data + l_idx, r_data + r_idx,
                        n_col, n_contract);
                l_idx += l_step;
                ret_idx += ret_step_base;
            }
        });
    } else {
        RunParallel(n_l, [&](int l_cnt) {  // Left-hand side loop
            const int l_idx = l_cnt * l_step;
            int r_idx = 0;
            int ret_idx = l_cnt * ret_idx_base * ret_step;
            for (int r_cnt = 0; r_cnt < n_r; r_cnt++) {
                op_1d2d(ret_data + ret_idx, l_data + l_idx, r_data + r_idx,
                        n_col, n_contract);
                r_idx += r_step;
                ret_idx += ret_step;
            }
        });
    }
#else  // Run sequentially
    int l_idx = 0;
    int ret_idx0 = 0;
    for (int l_cnt = 0; l_cnt < n_l; l_cnt++) {
        int r_idx = 0;
        int ret_idx = ret_idx0 * ret_step;
        for (int r_cnt = 0; r_cnt < n_r; r_cnt++) {
            op_1d2d(ret_data + ret_idx, l_data + l_idx, r_data + r_idx, n_col,
                    n_contract);
            r_idx += r_step;
            ret_idx += ret_step;
        }
        l_idx += l_step;
        ret_idx0 += ret_idx_base;
    }
#endif
}

static NdArray DotNdArrayNdMd(const NdArray& lhs, const NdArray& rhs) {
    const Shape& l_shape = lhs.shape();  // 1 <= l.size
    const Shape& r_shape = rhs.shape();  // 2 <= r.size

    // The last axis of left and the second-to-last axis of right must be same.
    const int n_contract = l_shape.end()[-1];
    if (n_contract != r_shape.end()[-2]) {
        throw std::runtime_error("Invalid shape for dot product");
    }

    // Result shape
    Shape ret_shape(l_shape.begin(), l_shape.end() - 1);
    ret_shape.insert(ret_shape.end(), r_shape.begin(), r_shape.end() - 2);
    ret_shape.push_back(r_shape.back());
    // Result array
    NdArray ret(ret_shape);

    // Compute 2D shapes and steps
    //   [2, 3, (4)] [5, 6, (4), 7] -> [2, 3, 5, 6, 7]
    const int ret_step = r_shape.end()[-1];    // [2, 3, 5, 6, <7>]
    const int l_step = n_contract;             // [2, 3, <4>]
    const int r_step = n_contract * ret_step;  // [5, 6, <4>, <7>]

    const int n_l = static_cast<int>(lhs.size()) / l_step;
    const int n_r = static_cast<int>(rhs.size()) / r_step;  // [<5>, <6>, 4, 7]

    // Dot product
    DotNdArrayNdMdImpl(ret.data(), lhs.data(), rhs.data(), n_l, n_r, ret_step,
                       l_step, r_step, SelectDot1d2dOp(l_shape, r_shape));

    return ret;
}

static NdArray DotNdArray(const NdArray& lhs, const NdArray& rhs) {
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
    } else if (r_shape.size() == 1) {
        // Broadcast right 1D array
        const Shape shape(l_shape.begin(), l_shape.end() - 1);
        return DotNdArrayNdMd(lhs, rhs.reshape(r_shape[0], 1)).reshape(shape);
    } else {
        // Basic matrix product
        return DotNdArrayNdMd(lhs, rhs);
    }
}

// ------------------- Utilities for NdArray (Mutmul product) ------------------
static Shape CheckMatmulable(const Shape& l_shape, const Shape& r_shape) {
    // Note: 2D at least
    // Check dot productable
    if (l_shape.end()[-1] != r_shape.end()[-2]) {
        throw std::runtime_error("Unable to apply matmul (contracting dim)");
    }
    // Check broadcastable
    Shape ret_shape = CheckBroadcastable({l_shape.begin(), l_shape.end() - 2},
                                         {r_shape.begin(), r_shape.end() - 2});
    // Add last 2D
    ret_shape.push_back(l_shape.end()[-2]);
    ret_shape.push_back(r_shape.end()[-1]);
    return ret_shape;
}

static NdArray MatmulNdArrayImpl(const NdArray& lhs, const NdArray& rhs) {
    Shape l_shape = lhs.shape();
    Shape r_shape = rhs.shape();

    // Extends 2D to 3D
    const bool l_extend_2d = (l_shape.size() == 2);
    const bool r_extend_2d = (r_shape.size() == 2);
    if (l_extend_2d) {  // Third dimension is broadcasted.
        l_shape = Shape{1, l_shape[0], l_shape[1]};
    }
    if (r_extend_2d) {
        r_shape = Shape{1, r_shape[0], r_shape[1]};
    }

    // Check shape
    const Shape& ret_shape = CheckMatmulable(l_shape, r_shape);

    // Operators
    const int n_row = l_shape.end()[-2];
    const int n_contract = l_shape.end()[-1];
    const int n_col = r_shape.end()[-1];
    auto&& op_1d2d = SelectDot1d2dOp(l_shape, r_shape);
    auto&& op_2d2d = [&](const NdArray::Iter& o, const NdArray::ConstIter& l,
                         const NdArray::ConstIter& r) {
        for (int row_idx = 0; row_idx < n_row; row_idx++) {
            op_1d2d(o + row_idx * n_col, l + row_idx * n_contract, r, n_col,
                    n_contract);
        }
    };

    // Apply broadcast
    NdArray ret(ret_shape);
    const int ret_step = n_row * n_col;
    ApplyOpBroadcast(ret, lhs, rhs, 2, ret_step, op_2d2d);

    // Shrink 3D to 2D
    if (l_extend_2d && r_extend_2d) {
        return ret.reshape({ret_shape[1], ret_shape[2]});
    }

    return ret;
}

static NdArray MatmulNdArray(const NdArray& lhs, const NdArray& rhs) {
    if (lhs.size() == 0 || rhs.size() == 0) {
        // Empty array
        throw std::runtime_error("Matmul product of empty array");
    }

    if (lhs.ndim() == 1 || rhs.ndim() == 1) {
        // Use dot for 1-dim arrays
        return Dot(lhs, rhs);
    } else {
        // Run matmul for higher dimensions
        return MatmulNdArrayImpl(lhs, rhs);
    }
}

// ------------------- Utilities for NdArray (Cross product) -------------------
static void CrossNdArray1d1dShape33(const NdArray::Iter& ret_data,
                                    const NdArray::ConstIter& l_data,
                                    const NdArray::ConstIter& r_data) {
    // lhs.shape() == {3} && rhs.shape == {3}
    ret_data[0] = l_data[1] * r_data[2] - l_data[2] * r_data[1];
    ret_data[1] = l_data[2] * r_data[0] - l_data[0] * r_data[2];
    ret_data[2] = l_data[0] * r_data[1] - l_data[1] * r_data[0];
}

static void CrossNdArray1d1dShape32(const NdArray::Iter& ret_data,
                                    const NdArray::ConstIter& l_data,
                                    const NdArray::ConstIter& r_data) {
    // lhs.shape() == {3} && rhs.shape == {2}
    ret_data[0] = -l_data[2] * r_data[1];
    ret_data[1] = l_data[2] * r_data[0];
    ret_data[2] = l_data[0] * r_data[1] - l_data[1] * r_data[0];
}

static void CrossNdArray1d1dShape23(const NdArray::Iter& ret_data,
                                    const NdArray::ConstIter& l_data,
                                    const NdArray::ConstIter& r_data) {
    // lhs.shape() == {3} && rhs.shape == {3}
    ret_data[0] = l_data[1] * r_data[2];
    ret_data[1] = -l_data[0] * r_data[2];
    ret_data[2] = l_data[0] * r_data[1] - l_data[1] * r_data[0];
}

static void CrossNdArray1d1dShape22(const NdArray::Iter& ret_data,
                                    const NdArray::ConstIter& l_data,
                                    const NdArray::ConstIter& r_data) {
    // lhs.shape() == {2} && rhs.shape == {2}
    ret_data[0] = l_data[0] * r_data[1] - l_data[1] * r_data[0];
}

template <typename F>
NdArray CrossNdArrayNdMd(const NdArray& lhs, const NdArray& rhs,
                         const int ret_step, F op) {
    const Shape& l_shape = lhs.shape();
    const Shape& r_shape = rhs.shape();
    Shape ret_shape = CheckBroadcastable({l_shape.begin(), l_shape.end() - 1},
                                         {r_shape.begin(), r_shape.end() - 1});
    ret_shape.push_back(ret_step);
    // Apply broadcast
    NdArray ret(ret_shape);
    ApplyOpBroadcast(ret, lhs, rhs, 1, ret_step, op);
    return ret;
}

static NdArray CrossNdArray(const NdArray& lhs, const NdArray& rhs) {
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
    throw std::runtime_error("incompatible dimensions for cross product"
                             " (dimension must be 2 or 3)");
}

// ----------------------- Utilities for NdArray (Shape) -----------------------
static NdArray SqueezeNdArray(const NdArray& x, const Axis& axes_raw) {
    Shape ret_shape;
    if (axes_raw.empty()) {
        // Extract non zero dimensions
        for (auto&& s : x.shape()) {
            if (s != 1) {
                ret_shape.push_back(s);
            }
        }
        // Escape zero shape
        if (ret_shape.empty()) {
            ret_shape.push_back(1);
        }
    } else {
        // Resolve axes (sort: on (inverse))
        const Axis& axes =
                ResolveAxis(axes_raw, x.ndim(), "Squeeze", true, false);
        // Copy all dim
        ret_shape = x.shape();
        // Remove passed axes from the tail
        for (auto&& axis : axes) {
            auto&& it = ret_shape.begin() + axis;
            if (*it != 1) {
                throw std::runtime_error("Invalid axis to squeeze (not 1)");
            }
            ret_shape.erase(it);
        }
    }
    return x.reshape(ret_shape);
}

static NdArray ExpandDimsNdArray(const NdArray& x, int axis) {
    // Resolve axes (axis can be same as `n_dim`)
    axis = ResolveAxis(axis, x.ndim() + 1, "Expand dims");
    // Insert
    Shape ret_shape = x.shape();
    ret_shape.insert(ret_shape.begin() + axis, 1);
    // Reshape
    return x.reshape(ret_shape);
}

// ----------------------- Utilities for NdArray (Stack) -----------------------
static Shape CheckStackable(const std::vector<NdArray>& xs) {
    // Check empty
    if (xs.empty()) {
        throw std::runtime_error("Need at least one array to stack");
    }
    // Check same shape
    const Shape& shape = xs[0].shape();
    for (size_t i = 1; i < xs.size(); i++) {
        if (shape != xs[i].shape()) {
            throw std::runtime_error("all input arrays must have the same "
                                     "shape");
        }
    }
    return shape;
}

static auto ComputeStackSizes(size_t n_src, const Shape& src_shape, int axis) {
    const auto& src_s_iter0 = src_shape.begin();
    const auto& src_s_iter1 = src_shape.begin() + axis;
    const auto& src_s_iter2 = src_shape.end();
    const auto& mul = std::multiplies<int>();

    // Sizes
    const int n_upper = std::accumulate(src_s_iter0, src_s_iter1, 1, mul);
    const int n_lower = std::accumulate(src_s_iter1, src_s_iter2, 1, mul);
    const int n_stack = static_cast<int>(n_src);

    // Create result shape
    Shape ret_shape;
    ret_shape.insert(ret_shape.end(), src_s_iter0, src_s_iter1);  // Upper
    ret_shape.push_back(n_stack);  // Stacking dimension
    ret_shape.insert(ret_shape.end(), src_s_iter1, src_s_iter2);  // Lower

    return std::make_tuple(std::move(ret_shape), n_upper, n_lower, n_stack);
}

static NdArray StackNdArray(const std::vector<NdArray>& xs, int axis) {
    // Resolve axes (axis can be same as the size)
    axis = ResolveAxis(axis, xs.size() + 1, "Stack");
    // Check it is possible to stack
    const Shape& src_shape = CheckStackable(xs);

    // Compute stack sizes
    auto stack_sizes = ComputeStackSizes(xs.size(), src_shape, axis);
    const Shape& ret_shape = std::get<0>(stack_sizes);
    const int n_upper = std::get<1>(stack_sizes);
    const int n_lower = std::get<2>(stack_sizes);
    const int n_stack = std::get<3>(stack_sizes);

    // Create result array
    NdArray ret(ret_shape);
    auto ret_data = ret.data();

    // Core function to copy
    auto copy_lower_func = [&](const int stack_idx, const int ret_idx_base0,
                               const int src_idx_base) {
        const int ret_idx_base1 = (ret_idx_base0 + stack_idx) * n_lower;
        auto src_data = xs[static_cast<size_t>(stack_idx)].data();
        for (int l_idx = 0; l_idx < n_lower; l_idx++) {
            ret_data[ret_idx_base1 + l_idx] = src_data[src_idx_base + l_idx];
        }
    };

    // Switch by upper size
    if (n_upper == 1) {
        // Run in parallel of stack direction
        RunParallel(n_stack,
                    [&](int stack_idx) { copy_lower_func(stack_idx, 0, 0); });
    } else {
        // Run in parallel of high dimension's direction
        RunParallel(n_upper, [&](int u_idx) {
            const int ret_idx_base0 = u_idx * n_stack;
            const int src_idx_base = u_idx * n_lower;
            for (int stack_idx = 0; stack_idx < n_stack; stack_idx++) {
                copy_lower_func(stack_idx, ret_idx_base0, src_idx_base);
            }
        });
    }

    return ret;
}

// -------------------- Utilities for NdArray (Concatenate) --------------------
static void CheckConcatenatable(const std::vector<NdArray>& xs, int axis) {
    // Check empty
    if (xs.empty()) {
        throw std::runtime_error("Need at least one array to concatenate");
    }
    // Check same shape except axis dimension
    const Shape& fst_shape = xs[0].shape();
    const size_t axis_l = static_cast<size_t>(axis);
    const std::string error_msg = "all the input array dimensions except for "
                                  "the concatenation axis must match exactly";
    for (size_t i = 1; i < xs.size(); i++) {
        const Shape& cur_shape = xs[i].shape();
        // Check the size of shapes
        if (fst_shape.size() != cur_shape.size()) {
            throw std::runtime_error(error_msg);
        }
        // Check dimensions except axis
        for (size_t j = 0; j < fst_shape.size(); j++) {
            if (j == axis_l) {
                continue;
            }
            if (fst_shape[j] != cur_shape[j]) {
                throw std::runtime_error(error_msg);
            }
        }
    }
}

static auto ComputeConcatSizes(const std::vector<NdArray>& xs, int axis) {
    const Shape& fst_shape = xs[0].shape();
    const auto& src_s_iter0 = fst_shape.begin();
    const auto& src_s_iter1 = fst_shape.begin() + axis;
    const auto& src_s_iter2 = fst_shape.begin() + axis + 1;
    const auto& src_s_iter3 = fst_shape.end();
    const auto& mul = std::multiplies<int>();

    // Upper common size
    const int n_upper = std::accumulate(src_s_iter0, src_s_iter1, 1, mul);
    // Lower size depends on each sources
    std::vector<int> n_lowers;
    for (auto&& x : xs) {
        n_lowers.push_back(static_cast<int>(x.size()) / n_upper);
    }
    // Result indices of concatenation
    std::vector<int> concat_offsets;
    int n_lower_accum = 0;
    for (auto&& n_lower : n_lowers) {
        concat_offsets.push_back(n_lower_accum);
        n_lower_accum += n_lower;
    }
    // Step of concatenating dimension
    const int concat_step = n_lower_accum;

    // Concatenating dimensions
    int concat_dim = 0;
    const size_t axis_l = static_cast<size_t>(axis);
    for (auto&& x : xs) {
        concat_dim += x.shape()[axis_l];
    }

    // Create result shape
    Shape ret_shape;
    ret_shape.insert(ret_shape.end(), src_s_iter0, src_s_iter1);  // Upper
    ret_shape.push_back(concat_dim);  // Concatenating dimension
    ret_shape.insert(ret_shape.end(), src_s_iter2, src_s_iter3);  // Lower

    return std::make_tuple(std::move(ret_shape), n_upper, std::move(n_lowers),
                           xs.size(), std::move(concat_offsets), concat_step);
}

static NdArray ConcatenateNdArray(const std::vector<NdArray>& xs, int axis) {
    // Resolve axes
    axis = ResolveAxis(axis, xs[0].ndim(), "Concatenate");
    // Check it is possible to concatenate
    CheckConcatenatable(xs, axis);

    // Compute concat sizes
    auto concat_sizes = ComputeConcatSizes(xs, axis);
    const Shape& ret_shape = std::get<0>(concat_sizes);
    const int n_upper = std::get<1>(concat_sizes);
    const std::vector<int>& n_lowers = std::get<2>(concat_sizes);
    const size_t n_concat = std::get<3>(concat_sizes);
    const std::vector<int>& concat_offsets = std::get<4>(concat_sizes);
    const int concat_step = std::get<5>(concat_sizes);

    // Create result array
    NdArray ret(ret_shape);
    auto ret_data = ret.data();

    // Core function to copy
    auto copy_lower_func = [&](const size_t concat_idx, const int ret_idx_base0,
                               const int src_idx_base) {
        const int ret_idx_base1 = ret_idx_base0 + concat_offsets[concat_idx];
        auto src_data = xs[concat_idx].data();
        for (int l_idx = 0; l_idx < n_lowers[concat_idx]; l_idx++) {
            ret_data[ret_idx_base1 + l_idx] = src_data[src_idx_base + l_idx];
        }
    };

    // Switch by upper size
    if (n_upper == 1) {
        // Run in parallel of stack direction
        RunParallel(static_cast<int>(n_concat), [&](int concat_idx) {
            copy_lower_func(static_cast<size_t>(concat_idx), 0, 0);
        });
    } else {
        // Run in parallel of high dimension's direction
        RunParallel(n_upper, [&](int u_idx) {
            const int ret_idx_base0 = u_idx * concat_step;
            for (size_t concat_idx = 0; concat_idx < n_concat; concat_idx++) {
                const int src_idx_base = u_idx * n_lowers[concat_idx];
                copy_lower_func(concat_idx, ret_idx_base0, src_idx_base);
            }
        });
    }

    return ret;
}

// ----------------------- Utilities for NdArray (Split) -----------------------
static std::vector<NdArray> SplitNdArrayImpl(const NdArray& x,
                                             const Index& idxs, int axis) {
    // Note: Axis must be checked previously.

    const size_t axis_l = static_cast<size_t>(axis);
    const Shape& x_shape = x.shape();
    const int idx_end = static_cast<int>(x_shape[axis_l]);

    // Create base of slice index
    SliceIndex slice_idx;
    for (auto&& dim : x_shape) {
        slice_idx.push_back({0, dim});
    }

    // Result arrays
    std::vector<NdArray> rets;

    // Resolve for each index
    int prev_idx = 0;
    for (auto&& curr_idx_raw : idxs) {
        // Upper limit of index
        const int curr_idx = std::min(curr_idx_raw, idx_end);
        // Create one of slice index
        slice_idx[axis_l] = std::make_pair(prev_idx, curr_idx);  // Overwrite
        rets.push_back(x.slice(slice_idx));  // Register slice
        // Go to next index
        prev_idx = curr_idx;
    }

    // Rest of index
    slice_idx[axis_l] = std::make_pair(prev_idx, idx_end);  // Overwrite
    rets.push_back(x.slice(slice_idx));                     // Register slice

    return rets;
}

static std::vector<NdArray> SplitNdArray(const NdArray& x, int n_section,
                                         int axis) {
    // Resolve axes
    axis = ResolveAxis(axis, x.ndim(), "Split");
    // Check splitting size
    const int dim_size = x.shape()[static_cast<size_t>(axis)];
    if (dim_size % n_section != 0) {
        throw std::runtime_error("Invalid section number in an equal division");
    }
    const int section_size = dim_size / n_section;

    // Create splitting indices
    Index idxs;
    for (int sec_i = 1; sec_i < n_section; sec_i++) {  // no first one
        idxs.push_back(section_size * sec_i);
    }

    // Split by indices
    return SplitNdArrayImpl(x, idxs, axis);
}

static std::vector<NdArray> SplitNdArray(const NdArray& x, const Index& idxs,
                                         int axis) {
    // Resolve axes
    axis = ResolveAxis(axis, x.ndim(), "Split");
    // Split by indices
    return SplitNdArrayImpl(x, idxs, axis);
}

static std::vector<NdArray> SeparateNdArray(const NdArray& x, int axis) {
    // Resolve axes
    axis = ResolveAxis(axis, x.ndim(), "Separate");
    // Get splitting size (== dim size)
    const int dim_size = x.shape()[static_cast<size_t>(axis)];

    // Create splitting indices
    Index idxs;
    for (int sec_i = 1; sec_i < dim_size; sec_i++) {  // no first one
        idxs.push_back(sec_i);
    }

    // Split by indices
    std::vector<NdArray> res = SplitNdArrayImpl(x, idxs, axis);

    // Squeeze
    for (size_t i = 0; i < res.size(); i++) {
        res[i] = Squeeze(res[i], {axis});
    }

    return res;
}

// -------------------- Utilities for NdArray (Change View) --------------------
template <typename ViewF>
static void ChangeNdArrayView(const NdArray::Iter& ret_data,
                              const NdArray::ConstIter& src_data, size_t size,
                              ViewF view_func) {
    // Copy under view conversion function
    RunParallel(static_cast<int>(size), [&](int src_idx) {
        // Get result index
        const int ret_idx = view_func(src_idx);
        // Copy
        ret_data[ret_idx] = src_data[src_idx];
    });
}

static NdArray TransposeNdArray(const NdArray& src) {
    const Shape& src_shape = src.shape();

    // Create result shape
    Shape ret_shape;
    std::reverse_copy(src_shape.begin(), src_shape.end(),
                      back_inserter(ret_shape));
    // Create result array
    NdArray ret(ret_shape);

    // Pre-compute child sizes
    const std::vector<int> src_child_sizes = ComputeChildSizes(src_shape);
    const std::vector<int> ret_child_sizes = ComputeChildSizes(ret_shape);

    // Apply view change
    const size_t ndim = src_shape.size();
    ChangeNdArrayView(ret.data(), src.data(), src.size(), [&](int src_idx) {
        // Decompose
        auto src_idxs = std::make_unique<int[]>(ndim);
        for (size_t d = 0; d < ndim; d++) {
            src_idxs[d] = src_idx / src_child_sizes[d];
            src_idx -= src_idxs[d] * src_child_sizes[d];
        }
        // Compose (with inverting indices)
        int ret_idx = 0;
        for (size_t d = 0; d < ndim; d++) {
            ret_idx += src_idxs[ndim - d - 1] * ret_child_sizes[d];
        }
        return ret_idx;
    });
    return ret;
}

static NdArray SwapaxesNdArray(const NdArray& src, int axis1, int axis2) {
    const Shape& src_shape = src.shape();
    const size_t ndim = src_shape.size();

    // Resolve axis
    const size_t axis1_l =
            static_cast<size_t>(ResolveAxis(axis1, src.ndim(), "Swapaxes"));
    const size_t axis2_l =
            static_cast<size_t>(ResolveAxis(axis2, src.ndim(), "Swapaxes"));

    // Create result shape
    Shape ret_shape = src_shape;
    std::swap(ret_shape[axis1_l], ret_shape[axis2_l]);
    // Create result array
    NdArray ret(ret_shape);

    // Pre-compute child sizes
    const std::vector<int> src_child_sizes = ComputeChildSizes(src_shape);
    const std::vector<int> ret_child_sizes = ComputeChildSizes(ret_shape);

    // Apply view change
    ChangeNdArrayView(ret.data(), src.data(), src.size(), [&](int src_idx) {
        // Decompose
        auto idxs = std::make_unique<int[]>(ndim);
        for (size_t d = 0; d < ndim; d++) {
            idxs[d] = src_idx / src_child_sizes[d];
            src_idx -= idxs[d] * src_child_sizes[d];
        }
        // Swap axes
        std::swap(idxs[axis1_l], idxs[axis2_l]);
        // Compose
        int ret_idx = 0;
        for (size_t d = 0; d < ndim; d++) {
            ret_idx += idxs[d] * ret_child_sizes[d];
        }
        return ret_idx;
    });
    return ret;
}

static NdArray BroadcastToNdArray(const NdArray& x, const Shape& shape) {
    NdArray ret(shape);
    return ApplyDualOpInplace(
            std::move(ret), x, [](float, float r) { return r; }, false);
}

static NdArray SumToNdArray(const NdArray& x, const Shape& shape) {
    const Shape& x_shape = x.shape();
    // No need
    if (x_shape == shape) {
        return x;
    }
    // Impossible
    if (x_shape.size() < shape.size()) {
        return x;
    }

    // Create reduction axis
    Axis axis;
    const size_t lead = x_shape.size() - shape.size();
    for (size_t i = 0; i < lead; i++) {  // lead_axis
        axis.push_back(static_cast<int>(i));
    }
    for (size_t i = 0; i < shape.size(); i++) {  // axis
        if (shape[i] == 1) {
            axis.push_back(static_cast<int>(i + lead));
        }
    }

    // Reduce
    NdArray ret = x.sum(axis);

    return ret;
}

// ---------------------- Utilities for NdArray (Inverse) ----------------------
static int CheckInversable(const Shape& shape) {
    if (shape.size() < 2) {
        throw std::runtime_error("For matrix inverse, require at least 2-dim");
    }
    const int size = shape.back();
    if (size != shape.end()[-2]) {
        throw std::runtime_error("For matrix inverse, last 2 dimensions of the"
                                 " array must be square");
    }
    return size;
}

static void InvertNdArray2d(const NdArray::Iter& ret_data,
                            const NdArray::ConstIter& src_data, int order) {
    const int order_2 = order * 2;
    const size_t tmp_size = static_cast<size_t>(order_2 * order_2);
    std::unique_ptr<float[]> tmp(new float[tmp_size]);
    float* tmp_data = tmp.get();

    for (int row = 0; row < order; row++) {
        for (int col = 0; col < order; col++) {
            tmp_data[row * order_2 + col] = src_data[row * order + col];
        }
    }
    for (int row = 0; row < order; row++) {
        for (int col = order; col < order_2; col++) {
            tmp_data[row * order_2 + col] = (row == col - order) ? 1.f : 0.f;
        }
    }
    for (int row = 0; row < order; row++) {
        float t = tmp_data[row * order_2 + row];
        if (std::abs(t) == 0.f) {
            t = 0.00001f;  // Escape zero (TODO: More pricise method)
        }
        for (int col = row; col < order_2; col++) {
            tmp_data[row * order_2 + col] /= t;
        }
        for (int col = 0; col < order; col++) {
            if (row == col) {
                continue;
            }
            t = tmp_data[col * order_2 + row];
            for (int k = 0; k < order_2; k++) {
                tmp_data[col * order_2 + k] -= t * tmp_data[row * order_2 + k];
            }
        }
    }

    int ret_idx = 0;
    for (int row = 0; row < order; row++) {
        for (int col = order; col < order_2; col++) {
            ret_data[ret_idx++] = tmp_data[row * order_2 + col];
        }
    }
}

static void InvertNdArrayNd(const NdArray::Iter& ret_data,
                            const NdArray::ConstIter& src_data,
                            const Shape& src_shape, size_t src_size) {
    // Check it is possible to invert
    const int order = CheckInversable(src_shape);
    // Compute inverse for each lower 2 dimension.
    const int one_size = order * order;
    const int n = static_cast<int>(src_size) / one_size;
    RunParallel(n, [&](int i) {
        const int idx = one_size * i;
        InvertNdArray2d(ret_data + idx, src_data + idx, order);
    });
}

static NdArray InvertNdArray(const NdArray& src) {
    // Create new array
    NdArray ret(src.shape());
    // Compute inverse
    InvertNdArrayNd(ret.data(), src.data(), src.shape(), src.size());
    return ret;
}

static NdArray InvertNdArrayInplace(NdArray&& src) {
    // Compute inverse
    InvertNdArrayNd(src.data(), src.data(), src.shape(), src.size());
    return std::move(src);
}

// --------------------- Utilities for NdArray (Profiling) ---------------------
#ifdef TINYNDARRAY_PROFILE_MEMORY
class MemProfiler {
public:
    MemProfiler() {}
    ~MemProfiler() {}

    static void Register(std::shared_ptr<float> v, size_t size) {
        // Compare with previously registered value
        if (s_mem_sizes.count(v) && s_mem_sizes[v] != size) {
            throw std::runtime_error("Invalid register for MemProfiler.");
        }
        // Register (overwrite)
        s_mem_sizes[v] = size;
    }

    static void Unregister(const std::shared_ptr<float>& v) {
        // If the pointer is owned by only two (there is no copy of the
        // pointer), unregister
        if (v.use_count() <= 2) {
            s_mem_sizes.erase(v);
        }
    }

    static size_t GetNumInstance() {
        // First update the instances
        Update();
        // Return
        return s_mem_sizes.size();
    }

    static size_t GetTotalMemory() {
        // First update the instances
        Update();
        // Sum up memory sizes
        size_t sum_mem = 0;
        for (auto&& key_val : s_mem_sizes) {
            sum_mem += key_val.second;
        }
        return sum_mem;
    }

    static void Update() {
        // Remove entries which have only 1 reference.
        for (auto it = s_mem_sizes.begin(); it != s_mem_sizes.end();) {
            if (it->first.use_count() <= 1) {
                it = s_mem_sizes.erase(it);
            } else {
                ++it;
            }
        }
    }

private:
    static std::map<std::shared_ptr<float>, size_t> s_mem_sizes;
};

std::map<std::shared_ptr<float>, size_t> MemProfiler::s_mem_sizes;

#endif  // TINYNDARRAY_PROFILE_MEMORY

// =============================================================================
// ============================ NdArray Definition =============================
// =============================================================================
NdArray::NdArray() : m_sub(std::make_shared<Substance>()) {}

NdArray::NdArray(std::shared_ptr<Substance> sub) : m_sub(sub) {}

NdArray::NdArray(const NdArray& lhs) = default;  // shallow copy

NdArray::NdArray(NdArray&& lhs) noexcept
    : m_sub(std::move(lhs.m_sub)) {}  // move

NdArray& NdArray::operator=(const NdArray& lhs) = default;  // shallow copy

NdArray& NdArray::operator=(NdArray&& lhs) {  // move
    m_sub = std::move(lhs.m_sub);
    return *this;
}

NdArray::~NdArray() = default;

// --------------------------------- Substance ---------------------------------
class NdArray::Substance {
public:
    Substance(size_t size_ = 0, const Shape& shape_ = {0})
        : size(size_),
          shape(shape_),
          v(new float[size_], std::default_delete<float[]>()) {
#ifdef TINYNDARRAY_PROFILE_MEMORY
        MemProfiler::Register(v, size);
#endif  // TINYNDARRAY_PROFILE_MEMORY
    }
    ~Substance() {
#ifdef TINYNDARRAY_PROFILE_MEMORY
        MemProfiler::Unregister(v);
#endif  // TINYNDARRAY_PROFILE_MEMORY
    }

    size_t size;
    Shape shape;
    std::shared_ptr<float> v;  // C++17: Replace with `shared_ptr<float[]>`.
};

// ------------------------------- Static Member -------------------------------
std::random_device NdArray::s_rand_seed;
std::mt19937 NdArray::s_rand_engine(s_rand_seed());
int NdArray::s_n_workers = NdArray::DEFAULT_N_WORKERS;
int NdArray::s_batch_scale = NdArray::DEFAULT_BATCH_SCALE;

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

NdArray::NdArray(FloatList<8> init_list) : NdArray(CheckFListShape(init_list)) {
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
    fill(fill_v);
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
    // Create empty array
    const int n = static_cast<int>(std::ceil((stop - start) / step));
    NdArray ret({n});
    // Fill by step
    auto&& data = ret.data();
    RunParallel(n,
                [&](int i) { data[i] = start + step * static_cast<float>(i); });
    return ret;
}

// ------------------------- Static Methods for Random -------------------------
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

// ------------------------ Static Methods for Parallel ------------------------
int NdArray::GetNumWorkers() {
    return s_n_workers;
}

void NdArray::SetNumWorkers(int n_workers) {
    s_n_workers = n_workers;
}

int NdArray::GetBatchScale() {
    return s_batch_scale;
}

void NdArray::SetBatchScale(int batch_scale) {
    s_batch_scale = batch_scale;
}

// ----------------------- Static Methods for Profiling ------------------------
size_t NdArray::GetNumInstance() {
#ifdef TINYNDARRAY_PROFILE_MEMORY
    return MemProfiler::GetNumInstance();
#else
    throw std::runtime_error("Profiling is not enabled. Please defilne "
                             "`TINYNDARRAY_PROFILE_MEMORY`");
#endif  // TINYNDARRAY_PROFILE_MEMORY
}

size_t NdArray::GetTotalMemory() {
#ifdef TINYNDARRAY_PROFILE_MEMORY
    return MemProfiler::GetTotalMemory();
#else
    throw std::runtime_error("Profiling is not enabled. Please defilne "
                             "`TINYNDARRAY_PROFILE_MEMORY`");
#endif  // TINYNDARRAY_PROFILE_MEMORY
}

// ------------------------------- Basic Methods -------------------------------
uintptr_t NdArray::id() const {
    return reinterpret_cast<uintptr_t>(m_sub->v.get());  // pointer of array
}

bool NdArray::empty() const {
    return m_sub->size == 0;
}

size_t NdArray::size() const {
    return m_sub->size;
}

const Shape& NdArray::shape() const {
    return m_sub->shape;
}

size_t NdArray::ndim() const {
    return m_sub->shape.size();
}

NdArray::Iter NdArray::data() {
    return begin();
}

NdArray::ConstIter NdArray::data() const {
    return begin();
}

void NdArray::fill(float v) {
    FillN(m_sub->v.get(), static_cast<int>(m_sub->size), v);
}

NdArray NdArray::copy() const {
    // Create new array with new substance
    auto sub = std::make_shared<Substance>(m_sub->size, m_sub->shape);
    NdArray ret(sub);
    // Copy array data
    ApplyOpSimple(ret, *this, [](const float& x) { return x; });
    return ret;
}

void NdArray::resize(const Shape& shape) {
    NdArray ret(shape);
    const int ret_size = static_cast<int>(ret.size());
    const int src_size = static_cast<int>(size());
    if (ret_size <= src_size) {
        // Shrink
        Copy(ret.begin(), begin(), ret_size);
    } else {
        // Extend
        Copy(ret.begin(), begin(), src_size);
        FillN(ret.begin() + src_size, ret_size - src_size, 0.f);
    }
    // Set (Keep instance of substance)
    m_sub->size = ret.m_sub->size;
    m_sub->shape = ret.m_sub->shape;
    m_sub->v = ret.m_sub->v;
}

// ----------------------------- Begin/End Methods -----------------------------
NdArray::Iter NdArray::begin() {
    return Iter(m_sub->v.get());
}

NdArray::Iter NdArray::end() {
    return Iter(m_sub->v.get() + m_sub->size);
}

NdArray::ConstIter NdArray::begin() const {
    return ConstIter(m_sub->v.get());
}

NdArray::ConstIter NdArray::end() const {
    return ConstIter(m_sub->v.get() + m_sub->size);
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

NdArray NdArray::flatten() const {
    return reshape({-1}).copy();
}

NdArray NdArray::ravel() const {
    return reshape({-1});
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
            s = ClipOp(s, 0, shape[i]);  // must be next of the last.
            e = ClipOp(e, 0, shape[i]);
            // Register
            slice_shape.push_back(std::max(e - s, 0));  // Escape negative
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
    return DotNdArray(*this, other);
}

// -------------------------------- Cross Method -------------------------------
NdArray NdArray::cross(const NdArray& other) const {
    return CrossNdArray(*this, other);
}

// -------------------------------- Axis Method --------------------------------
NdArray NdArray::sum(const Axis& axes, bool keepdims) const {
    return Sum(*this, axes, keepdims);
}

NdArray NdArray::mean(const Axis& axes, bool keepdims) const {
    return Mean(*this, axes, keepdims);
}

NdArray NdArray::min(const Axis& axes, bool keepdims) const {
    return Min(*this, axes, keepdims);
}

NdArray NdArray::max(const Axis& axes, bool keepdims) const {
    return Max(*this, axes, keepdims);
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
// Print
std::ostream& operator<<(std::ostream& os, const NdArray& x) {
    OutputNdArray(os, x);
    return os;
}

std::ostream& operator<<(std::ostream& os, const Shape& shape) {
    OutputShape(os, shape);
    return os;
}

// Single
NdArray operator+(const NdArray& x) {
    return Positive(x);
}

NdArray operator-(const NdArray& x) {
    return Negative(x);
}

// Arithmetic (NdArray, NdArray)
NdArray operator+(const NdArray& lhs, const NdArray& rhs) {
    return Add(lhs, rhs);
}

NdArray operator-(const NdArray& lhs, const NdArray& rhs) {
    return Subtract(lhs, rhs);
}

NdArray operator*(const NdArray& lhs, const NdArray& rhs) {
    return Multiply(lhs, rhs);
}

NdArray operator/(const NdArray& lhs, const NdArray& rhs) {
    return Divide(lhs, rhs);
}

// Arithmetic (NdArray, float)
NdArray operator+(const NdArray& lhs, float rhs) {
    return Add(lhs, rhs);
}

NdArray operator-(const NdArray& lhs, float rhs) {
    return Subtract(lhs, rhs);
}

NdArray operator*(const NdArray& lhs, float rhs) {
    return Multiply(lhs, rhs);
}

NdArray operator/(const NdArray& lhs, float rhs) {
    return Divide(lhs, rhs);
}

// Arithmetic (float, NdArray)
NdArray operator+(float lhs, const NdArray& rhs) {
    return Add(lhs, rhs);
}

NdArray operator-(float lhs, const NdArray& rhs) {
    return Subtract(lhs, rhs);
}

NdArray operator*(float lhs, const NdArray& rhs) {
    return Multiply(lhs, rhs);
}

NdArray operator/(float lhs, const NdArray& rhs) {
    return Divide(lhs, rhs);
}

// Comparison (NdArray, NdArray)
NdArray operator==(const NdArray& lhs, const NdArray& rhs) {
    return Equal(lhs, rhs);
}

NdArray operator!=(const NdArray& lhs, const NdArray& rhs) {
    return NotEqual(lhs, rhs);
}

NdArray operator>(const NdArray& lhs, const NdArray& rhs) {
    return Greater(lhs, rhs);
}

NdArray operator>=(const NdArray& lhs, const NdArray& rhs) {
    return GreaterEqual(lhs, rhs);
}

NdArray operator<(const NdArray& lhs, const NdArray& rhs) {
    return Less(lhs, rhs);
}

NdArray operator<=(const NdArray& lhs, const NdArray& rhs) {
    return LessEqual(lhs, rhs);
}

// Comparison (NdArray, float)
NdArray operator==(const NdArray& lhs, float rhs) {
    return Equal(lhs, rhs);
}

NdArray operator!=(const NdArray& lhs, float rhs) {
    return NotEqual(lhs, rhs);
}

NdArray operator>(const NdArray& lhs, float rhs) {
    return Greater(lhs, rhs);
}

NdArray operator>=(const NdArray& lhs, float rhs) {
    return GreaterEqual(lhs, rhs);
}

NdArray operator<(const NdArray& lhs, float rhs) {
    return Less(lhs, rhs);
}

NdArray operator<=(const NdArray& lhs, float rhs) {
    return LessEqual(lhs, rhs);
}

// Comparison (float, NdArray)
NdArray operator==(float lhs, const NdArray& rhs) {
    return Equal(lhs, rhs);
}

NdArray operator!=(float lhs, const NdArray& rhs) {
    return NotEqual(lhs, rhs);
}

NdArray operator>(float lhs, const NdArray& rhs) {
    return Greater(lhs, rhs);
}

NdArray operator>=(float lhs, const NdArray& rhs) {
    return GreaterEqual(lhs, rhs);
}

NdArray operator<(float lhs, const NdArray& rhs) {
    return Less(lhs, rhs);
}

NdArray operator<=(float lhs, const NdArray& rhs) {
    return LessEqual(lhs, rhs);
}

// ----------------------------- In-place Operators ----------------------------
// Single
NdArray operator+(NdArray&& x) {
    return std::move(x);
}

NdArray operator-(NdArray&& x) {
    return ApplySingleOpInplace(std::move(x), [](float v) { return -v; });
}

// Arithmetic (NdArray, NdArray)
NdArray operator+(NdArray&& lhs, NdArray&& rhs) {
    return Add(std::move(lhs), std::move(rhs));
}

NdArray operator+(const NdArray& lhs, NdArray&& rhs) {
    return Add(lhs, std::move(rhs));
}

NdArray operator+(NdArray&& lhs, const NdArray& rhs) {
    return Add(std::move(lhs), rhs);
}

NdArray operator-(NdArray&& lhs, NdArray&& rhs) {
    return Subtract(std::move(lhs), std::move(rhs));
}

NdArray operator-(const NdArray& lhs, NdArray&& rhs) {
    return Subtract(lhs, std::move(rhs));
}

NdArray operator-(NdArray&& lhs, const NdArray& rhs) {
    return Subtract(std::move(lhs), rhs);
}

NdArray operator*(NdArray&& lhs, NdArray&& rhs) {
    return Multiply(std::move(lhs), std::move(rhs));
}

NdArray operator*(const NdArray& lhs, NdArray&& rhs) {
    return Multiply(lhs, std::move(rhs));
}

NdArray operator*(NdArray&& lhs, const NdArray& rhs) {
    return Multiply(std::move(lhs), rhs);
}

NdArray operator/(NdArray&& lhs, NdArray&& rhs) {
    return Divide(std::move(lhs), std::move(rhs));
}

NdArray operator/(const NdArray& lhs, NdArray&& rhs) {
    return Divide(lhs, std::move(rhs));
}

NdArray operator/(NdArray&& lhs, const NdArray& rhs) {
    return Divide(std::move(lhs), rhs);
}

// Arithmetic (NdArray, float)
NdArray operator+(NdArray&& lhs, float rhs) {
    return Add(std::move(lhs), rhs);
}

NdArray operator-(NdArray&& lhs, float rhs) {
    return Subtract(std::move(lhs), rhs);
}

NdArray operator*(NdArray&& lhs, float rhs) {
    return Multiply(std::move(lhs), rhs);
}

NdArray operator/(NdArray&& lhs, float rhs) {
    return Divide(std::move(lhs), rhs);
}

// Arithmetic (float, NdArray)
NdArray operator+(float lhs, NdArray&& rhs) {
    return Add(lhs, std::move(rhs));
}

NdArray operator-(float lhs, NdArray&& rhs) {
    return Subtract(lhs, std::move(rhs));
}

NdArray operator*(float lhs, NdArray&& rhs) {
    return Multiply(lhs, std::move(rhs));
}

NdArray operator/(float lhs, NdArray&& rhs) {
    return Divide(lhs, std::move(rhs));
}

// Comparison (NdArray, NdArray)
NdArray operator==(NdArray&& lhs, NdArray&& rhs) {
    return Equal(std::move(lhs), std::move(rhs));
}

NdArray operator==(const NdArray& lhs, NdArray&& rhs) {
    return Equal(lhs, std::move(rhs));
}

NdArray operator==(NdArray&& lhs, const NdArray& rhs) {
    return Equal(std::move(lhs), rhs);
}

NdArray operator!=(NdArray&& lhs, NdArray&& rhs) {
    return NotEqual(std::move(lhs), std::move(rhs));
}

NdArray operator!=(const NdArray& lhs, NdArray&& rhs) {
    return NotEqual(lhs, std::move(rhs));
}

NdArray operator!=(NdArray&& lhs, const NdArray& rhs) {
    return NotEqual(std::move(lhs), rhs);
}

NdArray operator>(NdArray&& lhs, NdArray&& rhs) {
    return Greater(std::move(lhs), std::move(rhs));
}

NdArray operator>(const NdArray& lhs, NdArray&& rhs) {
    return Greater(lhs, std::move(rhs));
}

NdArray operator>(NdArray&& lhs, const NdArray& rhs) {
    return Greater(std::move(lhs), rhs);
}

NdArray operator>=(NdArray&& lhs, NdArray&& rhs) {
    return GreaterEqual(std::move(lhs), std::move(rhs));
}

NdArray operator>=(const NdArray& lhs, NdArray&& rhs) {
    return GreaterEqual(lhs, std::move(rhs));
}

NdArray operator>=(NdArray&& lhs, const NdArray& rhs) {
    return GreaterEqual(std::move(lhs), rhs);
}

NdArray operator<(NdArray&& lhs, NdArray&& rhs) {
    return Less(std::move(lhs), std::move(rhs));
}

NdArray operator<(const NdArray& lhs, NdArray&& rhs) {
    return Less(lhs, std::move(rhs));
}

NdArray operator<(NdArray&& lhs, const NdArray& rhs) {
    return Less(std::move(lhs), rhs);
}

NdArray operator<=(NdArray&& lhs, NdArray&& rhs) {
    return LessEqual(std::move(lhs), std::move(rhs));
}

NdArray operator<=(const NdArray& lhs, NdArray&& rhs) {
    return LessEqual(lhs, std::move(rhs));
}

NdArray operator<=(NdArray&& lhs, const NdArray& rhs) {
    return LessEqual(std::move(lhs), rhs);
}

// Comparison (NdArray, float)
NdArray operator==(NdArray&& lhs, float rhs) {
    return Equal(std::move(lhs), rhs);
}

NdArray operator!=(NdArray&& lhs, float rhs) {
    return NotEqual(std::move(lhs), rhs);
}

NdArray operator>(NdArray&& lhs, float rhs) {
    return Greater(std::move(lhs), rhs);
}

NdArray operator>=(NdArray&& lhs, float rhs) {
    return GreaterEqual(std::move(lhs), rhs);
}

NdArray operator<(NdArray&& lhs, float rhs) {
    return Less(std::move(lhs), rhs);
}

NdArray operator<=(NdArray&& lhs, float rhs) {
    return LessEqual(std::move(lhs), rhs);
}

// Comparison (float, NdArray)
NdArray operator==(float lhs, NdArray&& rhs) {
    return Equal(lhs, std::move(rhs));
}

NdArray operator!=(float lhs, NdArray&& rhs) {
    return NotEqual(lhs, std::move(rhs));
}

NdArray operator>(float lhs, NdArray&& rhs) {
    return Greater(lhs, std::move(rhs));
}

NdArray operator>=(float lhs, NdArray&& rhs) {
    return GreaterEqual(lhs, std::move(rhs));
}

NdArray operator<(float lhs, NdArray&& rhs) {
    return Less(lhs, std::move(rhs));
}

NdArray operator<=(float lhs, NdArray&& rhs) {
    return LessEqual(lhs, std::move(rhs));
}

// Compound Assignment (NdArray, NdArray)
NdArray operator+=(NdArray& lhs, const NdArray& rhs) {
    return lhs = ApplyDualOpInplace(std::move(lhs), rhs, std::plus<float>(),
                                    false);  // force in-place
}

NdArray operator+=(NdArray&& lhs, const NdArray& rhs) {
    return ApplyDualOpInplace(std::move(lhs), rhs, std::plus<float>(),
                              false);  // force in-place
}

NdArray operator-=(NdArray& lhs, const NdArray& rhs) {
    return lhs = ApplyDualOpInplace(std::move(lhs), rhs, std::minus<float>(),
                                    false);  // force in-place
}

NdArray operator-=(NdArray&& lhs, const NdArray& rhs) {
    return ApplyDualOpInplace(std::move(lhs), rhs, std::minus<float>(),
                              false);  // force in-place
}

NdArray operator*=(NdArray& lhs, const NdArray& rhs) {
    return lhs = ApplyDualOpInplace(std::move(lhs), rhs,
                                    std::multiplies<float>(),
                                    false);  // force in-place
}

NdArray operator*=(NdArray&& lhs, const NdArray& rhs) {
    return ApplyDualOpInplace(std::move(lhs), rhs, std::multiplies<float>(),
                              false);  // force in-place
}

NdArray operator/=(NdArray& lhs, const NdArray& rhs) {
    return lhs = ApplyDualOpInplace(std::move(lhs), rhs, std::divides<float>(),
                                    false);  // force in-place
}

NdArray operator/=(NdArray&& lhs, const NdArray& rhs) {
    return ApplyDualOpInplace(std::move(lhs), rhs, std::divides<float>(),
                              false);  // force in-place
}

// Compound Assignment (NdArray, float)
NdArray operator+=(NdArray& lhs, float rhs) {
    return lhs = Add(std::move(lhs), rhs);
}

NdArray operator+=(NdArray&& lhs, float rhs) {
    return Add(std::move(lhs), rhs);
}

NdArray operator-=(NdArray& lhs, float rhs) {
    return lhs = Subtract(std::move(lhs), rhs);
}

NdArray operator-=(NdArray&& lhs, float rhs) {
    return Subtract(std::move(lhs), rhs);
}

NdArray operator*=(NdArray& lhs, float rhs) {
    return lhs = Multiply(std::move(lhs), rhs);
}

NdArray operator*=(NdArray&& lhs, float rhs) {
    return Multiply(std::move(lhs), rhs);
}

NdArray operator/=(NdArray& lhs, float rhs) {
    return lhs = Divide(std::move(lhs), rhs);
}

NdArray operator/=(NdArray&& lhs, float rhs) {
    return Divide(std::move(lhs), rhs);
}

// ---------------------------- Operator Functions -----------------------------
// Single operators
NdArray Positive(const NdArray& x) {
    return x.copy();  // Numpy behavior
}

NdArray Negative(const NdArray& x) {
    return ApplySingleOp(x, [](float v) { return -v; });
}

// Arithmetic operators (NdArray, NdArray)
NdArray Add(const NdArray& lhs, const NdArray& rhs) {
    return ApplyDualOp(lhs, rhs, std::plus<float>());
}

NdArray Subtract(const NdArray& lhs, const NdArray& rhs) {
    return ApplyDualOp(lhs, rhs, std::minus<float>());
}

NdArray Multiply(const NdArray& lhs, const NdArray& rhs) {
    return ApplyDualOp(lhs, rhs, std::multiplies<float>());
}

NdArray Divide(const NdArray& lhs, const NdArray& rhs) {
    return ApplyDualOp(lhs, rhs, std::divides<float>());
}

// Arithmetic operators (NdArray, float)
NdArray Add(const NdArray& lhs, float rhs) {
    return ApplyDualOp(lhs, rhs, std::plus<float>());
}

NdArray Subtract(const NdArray& lhs, float rhs) {
    return ApplyDualOp(lhs, rhs, std::minus<float>());
}

NdArray Multiply(const NdArray& lhs, float rhs) {
    return ApplyDualOp(lhs, rhs, std::multiplies<float>());
}

NdArray Divide(const NdArray& lhs, float rhs) {
    return ApplyDualOp(lhs, rhs, std::divides<float>());
}

// Arithmetic operators (float, NdArray)
NdArray Add(float lhs, const NdArray& rhs) {
    return ApplyDualOp(lhs, rhs, std::plus<float>());
}

NdArray Subtract(float lhs, const NdArray& rhs) {
    return ApplyDualOp(lhs, rhs, std::minus<float>());
}

NdArray Multiply(float lhs, const NdArray& rhs) {
    return ApplyDualOp(lhs, rhs, std::multiplies<float>());
}

NdArray Divide(float lhs, const NdArray& rhs) {
    return ApplyDualOp(lhs, rhs, std::divides<float>());
}

// Comparison operators (NdArray, NdArray)
NdArray Equal(const NdArray& lhs, const NdArray& rhs) {
    return ApplyDualOp(lhs, rhs, std::equal_to<float>());
}

NdArray NotEqual(const NdArray& lhs, const NdArray& rhs) {
    return ApplyDualOp(lhs, rhs, std::not_equal_to<float>());
}

NdArray Greater(const NdArray& lhs, const NdArray& rhs) {
    return ApplyDualOp(lhs, rhs, std::greater<float>());
}

NdArray GreaterEqual(const NdArray& lhs, const NdArray& rhs) {
    return ApplyDualOp(lhs, rhs, std::greater_equal<float>());
}

NdArray Less(const NdArray& lhs, const NdArray& rhs) {
    return ApplyDualOp(lhs, rhs, std::less<float>());
}

NdArray LessEqual(const NdArray& lhs, const NdArray& rhs) {
    return ApplyDualOp(lhs, rhs, std::less_equal<float>());
}

// Comparison operators (NdArray, float)
NdArray Equal(const NdArray& lhs, float rhs) {
    return ApplyDualOp(lhs, rhs, std::equal_to<float>());
}

NdArray NotEqual(const NdArray& lhs, float rhs) {
    return ApplyDualOp(lhs, rhs, std::not_equal_to<float>());
}

NdArray Greater(const NdArray& lhs, float rhs) {
    return ApplyDualOp(lhs, rhs, std::greater<float>());
}

NdArray GreaterEqual(const NdArray& lhs, float rhs) {
    return ApplyDualOp(lhs, rhs, std::greater_equal<float>());
}

NdArray Less(const NdArray& lhs, float rhs) {
    return ApplyDualOp(lhs, rhs, std::less<float>());
}

NdArray LessEqual(const NdArray& lhs, float rhs) {
    return ApplyDualOp(lhs, rhs, std::less_equal<float>());
}

// Comparison operators (float, NdArray)
NdArray Equal(float lhs, const NdArray& rhs) {
    return ApplyDualOp(lhs, rhs, std::equal_to<float>());
}

NdArray NotEqual(float lhs, const NdArray& rhs) {
    return ApplyDualOp(lhs, rhs, std::not_equal_to<float>());
}

NdArray Greater(float lhs, const NdArray& rhs) {
    return ApplyDualOp(lhs, rhs, std::greater<float>());
}

NdArray GreaterEqual(float lhs, const NdArray& rhs) {
    return ApplyDualOp(lhs, rhs, std::greater_equal<float>());
}

NdArray Less(float lhs, const NdArray& rhs) {
    return ApplyDualOp(lhs, rhs, std::less<float>());
}

NdArray LessEqual(float lhs, const NdArray& rhs) {
    return ApplyDualOp(lhs, rhs, std::less_equal<float>());
}

// Matrix operators
NdArray Dot(const NdArray& lhs, const NdArray& rhs) {
    return lhs.dot(rhs);
}

NdArray Matmul(const NdArray& lhs, const NdArray& rhs) {
    return MatmulNdArray(lhs, rhs);
}

NdArray Cross(const NdArray& lhs, const NdArray& rhs) {
    return lhs.cross(rhs);
}

// Basic math operators
NdArray Abs(const NdArray& x) {
    return ApplySingleOp(x, static_cast<float (*)(float)>(std::abs));
}

NdArray Sign(const NdArray& x) {
    return ApplySingleOp(x, SignOp);
}

NdArray Ceil(const NdArray& x) {
    return ApplySingleOp(x, static_cast<float (*)(float)>(std::ceil));
}

NdArray Floor(const NdArray& x) {
    return ApplySingleOp(x, static_cast<float (*)(float)>(std::floor));
}

NdArray Clip(const NdArray& x, float x_min, float x_max) {
    return ApplySingleOp(
            x, std::bind(ClipOp<float>, std::placeholders::_1, x_min, x_max));
}

NdArray Sqrt(const NdArray& x) {
    return ApplySingleOp(x, static_cast<float (*)(float)>(std::sqrt));
}

NdArray Exp(const NdArray& x) {
    return ApplySingleOp(x, static_cast<float (*)(float)>(std::exp));
}

NdArray Log(const NdArray& x) {
    return ApplySingleOp(x, static_cast<float (*)(float)>(std::log));
}

NdArray Square(const NdArray& x) {
    return ApplySingleOp(x, SquareOp);
}

NdArray Power(const NdArray& x, const NdArray& y) {
    return ApplyDualOp(x, y, static_cast<float (*)(float, float)>(std::pow));
}

NdArray Power(const NdArray& x, float y) {
    return ApplyDualOp(x, y, static_cast<float (*)(float, float)>(std::pow));
}

NdArray Power(float x, const NdArray& y) {
    return ApplyDualOp(x, y, static_cast<float (*)(float, float)>(std::pow));
}

// Trigonometric functions
NdArray Sin(const NdArray& x) {
    return ApplySingleOp(x, static_cast<float (*)(float)>(std::sin));
}

NdArray Cos(const NdArray& x) {
    return ApplySingleOp(x, static_cast<float (*)(float)>(std::cos));
}

NdArray Tan(const NdArray& x) {
    return ApplySingleOp(x, static_cast<float (*)(float)>(std::tan));
}

// Inverse trigonometric functions
NdArray ArcSin(const NdArray& x) {
    return ApplySingleOp(x, static_cast<float (*)(float)>(std::asin));
}

NdArray ArcCos(const NdArray& x) {
    return ApplySingleOp(x, static_cast<float (*)(float)>(std::acos));
}

NdArray ArcTan(const NdArray& x) {
    return ApplySingleOp(x, static_cast<float (*)(float)>(std::atan));
}

NdArray ArcTan2(const NdArray& y, const NdArray& x) {
    return ApplyDualOp(y, x, static_cast<float (*)(float, float)>(std::atan2));
}

NdArray ArcTan2(const NdArray& y, float x) {
    return ApplyDualOp(y, x, static_cast<float (*)(float, float)>(std::atan2));
}

NdArray ArcTan2(float y, const NdArray& x) {
    return ApplyDualOp(y, x, static_cast<float (*)(float, float)>(std::atan2));
}

// Axis functions
NdArray Sum(const NdArray& x, const Axis& axes, bool keepdims) {
    return ReduceAxis(x, axes, keepdims, 0.f, std::plus<float>());
}

NdArray Mean(const NdArray& x, const Axis& axes, bool keepdims) {
    if (x.size() == 0) {
        return {std::numeric_limits<float>::quiet_NaN()};
    }
    auto&& sum = Sum(x, axes, keepdims);
    return sum / static_cast<float>(x.size() / sum.size());
}

NdArray Min(const NdArray& x, const Axis& axes, bool keepdims) {
    return ReduceAxisNoEmpty(x, axes, keepdims,
                             std::numeric_limits<float>::max(),
                             [](float a, float b) { return std::min(a, b); });
}

NdArray Max(const NdArray& x, const Axis& axes, bool keepdims) {
    return ReduceAxisNoEmpty(x, axes, keepdims,
                             -std::numeric_limits<float>::max(),
                             [](float a, float b) { return std::max(a, b); });
}

bool All(const NdArray& x) {
    // Cast to bool
    return static_cast<float>(All(x, {}, false)) == static_cast<float>(true);
}

bool Any(const NdArray& x) {
    // Cast to bool
    return static_cast<float>(Any(x, {}, false)) == static_cast<float>(true);
}

NdArray All(const NdArray& x, const Axis& axes, bool keepdims) {
    return ReduceAxis(x, axes, keepdims, 1.f, [](float a, float b) {
        return static_cast<bool>(a) && static_cast<bool>(b);
    });
}

NdArray Any(const NdArray& x, const Axis& axes, bool keepdims) {
    return ReduceAxis(x, axes, keepdims, 0.f, [](float a, float b) {
        return static_cast<bool>(a) || static_cast<bool>(b);
    });
}

NdArray Where(const NdArray& cond, const NdArray& x, const NdArray& y) {
    return ApplyWhereOp(cond, x, y);
}

NdArray Where(const NdArray& cond, const NdArray& x, float y) {
    return ApplyWhereOp(cond, x, y);
}

NdArray Where(const NdArray& cond, float x, const NdArray& y) {
    return ApplyWhereOp(cond, x, y);
}

NdArray Where(const NdArray& cond, float x, float y) {
    return ApplyWhereOp(cond, x, y);
}

// Shape functions
NdArray Reshape(const NdArray& x, const Shape& shape) {
    return x.reshape(shape);
}

NdArray Squeeze(const NdArray& x, const Axis& axes) {
    return SqueezeNdArray(x, axes);
}

NdArray ExpandDims(const NdArray& x, int axis) {
    return ExpandDimsNdArray(x, axis);
}

// Grouping functions
NdArray Stack(const std::vector<NdArray>& xs, int axis) {
    return StackNdArray(xs, axis);
}

NdArray Concatenate(const std::vector<NdArray>& xs, int axis) {
    return ConcatenateNdArray(xs, axis);
}

std::vector<NdArray> Split(const NdArray& x, int n_section, int axis) {
    return SplitNdArray(x, n_section, axis);
}

std::vector<NdArray> Split(const NdArray& x, const Index& idxs, int axis) {
    return SplitNdArray(x, idxs, axis);
}

std::vector<NdArray> Separate(const NdArray& x, int axis) {
    return SeparateNdArray(x, axis);
}

// Change view
NdArray Transpose(const NdArray& x) {
    return TransposeNdArray(x);
}

NdArray Swapaxes(const NdArray& x, int axis1, int axis2) {
    return SwapaxesNdArray(x, axis1, axis2);
}

NdArray BroadcastTo(const NdArray& x, const Shape& shape) {
    return BroadcastToNdArray(x, shape);
}

NdArray SumTo(const NdArray& x, const Shape& shape) {
    return SumToNdArray(x, shape);
}

// Inverse
NdArray Inv(const NdArray& x) {
    return InvertNdArray(x);
}

// ------------------------ In-place Operator Functions ------------------------
// Single operators
NdArray Positive(NdArray&& x) {
    return std::move(x);
}

NdArray Negative(NdArray&& x) {
    return ApplySingleOpInplace(std::move(x), [](float v) { return -v; });
}

// Arithmetic operators (NdArray, NdArray)
NdArray Add(NdArray&& lhs, NdArray&& rhs) {
    return ApplyDualOpInplace(std::move(lhs), std::move(rhs),
                              std::plus<float>());
}

NdArray Add(const NdArray& lhs, NdArray&& rhs) {
    return ApplyDualOpInplace(lhs, std::move(rhs), std::plus<float>());
}

NdArray Add(NdArray&& lhs, const NdArray& rhs) {
    return ApplyDualOpInplace(std::move(lhs), rhs, std::plus<float>());
}

NdArray Subtract(NdArray&& lhs, NdArray&& rhs) {
    return ApplyDualOpInplace(std::move(lhs), std::move(rhs),
                              std::minus<float>());
}

NdArray Subtract(const NdArray& lhs, NdArray&& rhs) {
    return ApplyDualOpInplace(lhs, std::move(rhs), std::minus<float>());
}

NdArray Subtract(NdArray&& lhs, const NdArray& rhs) {
    return ApplyDualOpInplace(std::move(lhs), rhs, std::minus<float>());
}

NdArray Multiply(NdArray&& lhs, NdArray&& rhs) {
    return ApplyDualOpInplace(std::move(lhs), std::move(rhs),
                              std::multiplies<float>());
}

NdArray Multiply(const NdArray& lhs, NdArray&& rhs) {
    return ApplyDualOpInplace(lhs, std::move(rhs), std::multiplies<float>());
}

NdArray Multiply(NdArray&& lhs, const NdArray& rhs) {
    return ApplyDualOpInplace(std::move(lhs), rhs, std::multiplies<float>());
}

NdArray Divide(NdArray&& lhs, NdArray&& rhs) {
    return ApplyDualOpInplace(std::move(lhs), std::move(rhs),
                              std::divides<float>());
}

NdArray Divide(const NdArray& lhs, NdArray&& rhs) {
    return ApplyDualOpInplace(lhs, std::move(rhs), std::divides<float>());
}

NdArray Divide(NdArray&& lhs, const NdArray& rhs) {
    return ApplyDualOpInplace(std::move(lhs), rhs, std::divides<float>());
}

// Arithmetic operators (NdArrarhs, float)
NdArray Add(NdArray&& lhs, float rhs) {
    return ApplyDualOpInplace(std::move(lhs), rhs, std::plus<float>());
}

NdArray Subtract(NdArray&& lhs, float rhs) {
    return ApplyDualOpInplace(std::move(lhs), rhs, std::minus<float>());
}

NdArray Multiply(NdArray&& lhs, float rhs) {
    return ApplyDualOpInplace(std::move(lhs), rhs, std::multiplies<float>());
}

NdArray Divide(NdArray&& lhs, float rhs) {
    return ApplyDualOpInplace(std::move(lhs), rhs, std::divides<float>());
}

// Arithmetic operators (float, NdArray)
NdArray Add(float lhs, NdArray&& rhs) {
    return ApplyDualOpInplace(lhs, std::move(rhs), std::plus<float>());
}

NdArray Subtract(float lhs, NdArray&& rhs) {
    return ApplyDualOpInplace(lhs, std::move(rhs), std::minus<float>());
}

NdArray Multiply(float lhs, NdArray&& rhs) {
    return ApplyDualOpInplace(lhs, std::move(rhs), std::multiplies<float>());
}

NdArray Divide(float lhs, NdArray&& rhs) {
    return ApplyDualOpInplace(lhs, std::move(rhs), std::divides<float>());
}

// Comparison operators (NdArrarhs, NdArray)
NdArray Equal(NdArray&& lhs, NdArray&& rhs) {
    return ApplyDualOpInplace(std::move(lhs), std::move(rhs),
                              std::equal_to<float>());
}

NdArray Equal(const NdArray& lhs, NdArray&& rhs) {
    return ApplyDualOpInplace(lhs, std::move(rhs), std::equal_to<float>());
}

NdArray Equal(NdArray&& lhs, const NdArray& rhs) {
    return ApplyDualOpInplace(std::move(lhs), rhs, std::equal_to<float>());
}

NdArray NotEqual(NdArray&& lhs, NdArray&& rhs) {
    return ApplyDualOpInplace(std::move(lhs), std::move(rhs),
                              std::not_equal_to<float>());
}

NdArray NotEqual(const NdArray& lhs, NdArray&& rhs) {
    return ApplyDualOpInplace(lhs, std::move(rhs), std::not_equal_to<float>());
}

NdArray NotEqual(NdArray&& lhs, const NdArray& rhs) {
    return ApplyDualOpInplace(std::move(lhs), rhs, std::not_equal_to<float>());
}

NdArray Greater(NdArray&& lhs, NdArray&& rhs) {
    return ApplyDualOpInplace(std::move(lhs), std::move(rhs),
                              std::greater<float>());
}

NdArray Greater(const NdArray& lhs, NdArray&& rhs) {
    return ApplyDualOpInplace(lhs, std::move(rhs), std::greater<float>());
}

NdArray Greater(NdArray&& lhs, const NdArray& rhs) {
    return ApplyDualOpInplace(std::move(lhs), rhs, std::greater<float>());
}

NdArray GreaterEqual(NdArray&& lhs, NdArray&& rhs) {
    return ApplyDualOpInplace(std::move(lhs), std::move(rhs),
                              std::greater_equal<float>());
}

NdArray GreaterEqual(const NdArray& lhs, NdArray&& rhs) {
    return ApplyDualOpInplace(lhs, std::move(rhs), std::greater_equal<float>());
}

NdArray GreaterEqual(NdArray&& lhs, const NdArray& rhs) {
    return ApplyDualOpInplace(std::move(lhs), rhs, std::greater_equal<float>());
}

NdArray Less(NdArray&& lhs, NdArray&& rhs) {
    return ApplyDualOpInplace(std::move(lhs), std::move(rhs),
                              std::less<float>());
}

NdArray Less(const NdArray& lhs, NdArray&& rhs) {
    return ApplyDualOpInplace(lhs, std::move(rhs), std::less<float>());
}

NdArray Less(NdArray&& lhs, const NdArray& rhs) {
    return ApplyDualOpInplace(std::move(lhs), rhs, std::less<float>());
}

NdArray LessEqual(NdArray&& lhs, NdArray&& rhs) {
    return ApplyDualOpInplace(std::move(lhs), std::move(rhs),
                              std::less_equal<float>());
}

NdArray LessEqual(const NdArray& lhs, NdArray&& rhs) {
    return ApplyDualOpInplace(lhs, std::move(rhs), std::less_equal<float>());
}

NdArray LessEqual(NdArray&& lhs, const NdArray& rhs) {
    return ApplyDualOpInplace(std::move(lhs), rhs, std::less_equal<float>());
}

// Comparison operators (NdArrarhs, float)
NdArray Equal(NdArray&& lhs, float rhs) {
    return ApplyDualOpInplace(std::move(lhs), rhs, std::equal_to<float>());
}

NdArray NotEqual(NdArray&& lhs, float rhs) {
    return ApplyDualOpInplace(std::move(lhs), rhs, std::not_equal_to<float>());
}

NdArray Greater(NdArray&& lhs, float rhs) {
    return ApplyDualOpInplace(std::move(lhs), rhs, std::greater<float>());
}

NdArray GreaterEqual(NdArray&& lhs, float rhs) {
    return ApplyDualOpInplace(std::move(lhs), rhs, std::greater_equal<float>());
}

NdArray Less(NdArray&& lhs, float rhs) {
    return ApplyDualOpInplace(std::move(lhs), rhs, std::less<float>());
}

NdArray LessEqual(NdArray&& lhs, float rhs) {
    return ApplyDualOpInplace(std::move(lhs), rhs, std::less_equal<float>());
}

// Comparison operators (float, NdArray)
NdArray Equal(float lhs, NdArray&& rhs) {
    return ApplyDualOpInplace(lhs, std::move(rhs), std::equal_to<float>());
}

NdArray NotEqual(float lhs, NdArray&& rhs) {
    return ApplyDualOpInplace(lhs, std::move(rhs), std::not_equal_to<float>());
}

NdArray Greater(float lhs, NdArray&& rhs) {
    return ApplyDualOpInplace(lhs, std::move(rhs), std::greater<float>());
}

NdArray GreaterEqual(float lhs, NdArray&& rhs) {
    return ApplyDualOpInplace(lhs, std::move(rhs), std::greater_equal<float>());
}

NdArray Less(float lhs, NdArray&& rhs) {
    return ApplyDualOpInplace(lhs, std::move(rhs), std::less<float>());
}

NdArray LessEqual(float lhs, NdArray&& rhs) {
    return ApplyDualOpInplace(lhs, std::move(rhs), std::less_equal<float>());
}

// Basic math operators
NdArray Abs(NdArray&& x) {
    return ApplySingleOpInplace(std::move(x),
                                static_cast<float (*)(float)>(std::abs));
}

NdArray Sign(NdArray&& x) {
    return ApplySingleOpInplace(std::move(x), SignOp);
}

NdArray Ceil(NdArray&& x) {
    return ApplySingleOpInplace(std::move(x),
                                static_cast<float (*)(float)>(std::ceil));
}

NdArray Floor(NdArray&& x) {
    return ApplySingleOpInplace(std::move(x),
                                static_cast<float (*)(float)>(std::floor));
}

NdArray Clip(NdArray&& x, float x_min, float x_max) {
    return ApplySingleOpInplace(
            std::move(x),
            std::bind(ClipOp<float>, std::placeholders::_1, x_min, x_max));
}

NdArray Sqrt(NdArray&& x) {
    return ApplySingleOpInplace(std::move(x),
                                static_cast<float (*)(float)>(std::sqrt));
}

NdArray Exp(NdArray&& x) {
    return ApplySingleOpInplace(std::move(x),
                                static_cast<float (*)(float)>(std::exp));
}

NdArray Log(NdArray&& x) {
    return ApplySingleOpInplace(std::move(x),
                                static_cast<float (*)(float)>(std::log));
}

NdArray Square(NdArray&& x) {
    return ApplySingleOpInplace(std::move(x), SquareOp);
}

NdArray Power(NdArray&& x, NdArray&& y) {
    return ApplyDualOpInplace(std::move(x), std::move(y),
                              static_cast<float (*)(float, float)>(std::pow));
}

NdArray Power(const NdArray& x, NdArray&& y) {
    return ApplyDualOpInplace(x, std::move(y),
                              static_cast<float (*)(float, float)>(std::pow));
}

NdArray Power(NdArray&& x, const NdArray& y) {
    return ApplyDualOpInplace(std::move(x), y,
                              static_cast<float (*)(float, float)>(std::pow));
}

NdArray Power(NdArray&& x, float y) {
    return ApplyDualOpInplace(std::move(x), y,
                              static_cast<float (*)(float, float)>(std::pow));
}

NdArray Power(float x, NdArray&& y) {
    return ApplyDualOpInplace(x, std::move(y),
                              static_cast<float (*)(float, float)>(std::pow));
}

// Trigonometric functions
NdArray Sin(NdArray&& x) {
    return ApplySingleOpInplace(std::move(x),
                                static_cast<float (*)(float)>(std::sin));
}

NdArray Cos(NdArray&& x) {
    return ApplySingleOpInplace(std::move(x),
                                static_cast<float (*)(float)>(std::cos));
}

NdArray Tan(NdArray&& x) {
    return ApplySingleOpInplace(std::move(x),
                                static_cast<float (*)(float)>(std::tan));
}

// Inverse trigonometric functions
NdArray ArcSin(NdArray&& x) {
    return ApplySingleOpInplace(std::move(x),
                                static_cast<float (*)(float)>(std::asin));
}

NdArray ArcCos(NdArray&& x) {
    return ApplySingleOpInplace(std::move(x),
                                static_cast<float (*)(float)>(std::acos));
}

NdArray ArcTan(NdArray&& x) {
    return ApplySingleOpInplace(std::move(x),
                                static_cast<float (*)(float)>(std::atan));
}

NdArray ArcTan2(NdArray&& y, NdArray&& x) {
    return ApplyDualOpInplace(std::move(y), std::move(x),
                              static_cast<float (*)(float, float)>(std::atan2));
}

NdArray ArcTan2(const NdArray& y, NdArray&& x) {
    return ApplyDualOpInplace(y, std::move(x),
                              static_cast<float (*)(float, float)>(std::atan2));
}

NdArray ArcTan2(NdArray&& y, const NdArray& x) {
    return ApplyDualOpInplace(std::move(y), x,
                              static_cast<float (*)(float, float)>(std::atan2));
}

NdArray ArcTan2(NdArray&& y, float x) {
    return ApplyDualOpInplace(std::move(y), x,
                              static_cast<float (*)(float, float)>(std::atan2));
}

NdArray ArcTan2(float y, NdArray&& x) {
    return ApplyDualOpInplace(y, std::move(x),
                              static_cast<float (*)(float, float)>(std::atan2));
}

NdArray Where(NdArray&& cond, const NdArray& x, const NdArray& y) {
    return ApplyWhereOpInplace(std::move(cond), x, y);
}

NdArray Where(NdArray&& cond, const NdArray& x, float y) {
    return ApplyWhereOpInplace(std::move(cond), x, y);
}

NdArray Where(NdArray&& cond, float x, const NdArray& y) {
    return ApplyWhereOpInplace(std::move(cond), x, y);
}

NdArray Where(NdArray&& cond, float x, float y) {
    return ApplyWhereOpInplace(std::move(cond), x, y);
}

// Inverse
NdArray Inv(NdArray&& x) {
    return InvertNdArrayInplace(std::move(x));
}

// -----------------------------------------------------------------------------
// -------------------------- Utilities for Variable ---------------------------
// -----------------------------------------------------------------------------
template <typename T>
void CheckVecSize(const std::vector<T>& x, const size_t n) {
    if (x.size() != n) {
        throw std::runtime_error("Invalid vector size");
    }
}

template <typename T>
void CheckVecSize(const std::vector<T>& x0, const size_t n0,
                  const std::vector<T>& x1, const size_t n1,
                  const std::vector<T>& x2, const size_t n2) {
    CheckVecSize(x0, n0);
    CheckVecSize(x1, n1);
    CheckVecSize(x2, n2);
}

template <typename K, typename V>
V PopLast(std::multimap<K, V>& m) {
    auto&& last_itr = std::prev(m.end());
    V last = last_itr->second;
    m.erase(last_itr);
    return last;
}

template <typename DST, typename SRC, typename F>
std::vector<DST> CvtVec(const std::vector<SRC>& src, F func) {
    std::vector<DST> ret;
    ret.reserve(src.size());
    for (auto&& s_elem : src) {
        ret.emplace_back(func(s_elem));
    }
    return ret;
}

static NdArrays GetData(const Variables& src) {
    return CvtVec<NdArray>(src, [](const Variable& v) { return v.data(); });
}

static NdArrays GetGrads(const Variables& src) {
    return CvtVec<NdArray>(src, [](const Variable& v) { return v.grad(); });
}

static Variables CvtToVars(const NdArrays& src) {
    return CvtVec<Variable>(src, [](const NdArray& v) { return Variable(v); });
}

template <typename T>
static void ClearGrads(T vars) {
    for (auto&& v : vars) {
        v.clearGrad();
    }
}

static void PrintVariable(std::ostream& os, const Variable& x) {
    const static std::string HEADER = "Variable(";
    const static std::string OFFSET = "         ";

    // Pooling data string
    std::stringstream data_ss;
    data_ss << x.data();
    // Split lines
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(data_ss, line, '\n')) {
        lines.emplace_back(std::move(line));
        line = std::string();
    }

    // Print header
    os << HEADER;

    // Print data
    for (size_t i = 0; i < lines.size(); i++) {
        if (i != 0) {
            os << OFFSET;
        }
        os << lines[i];
        if (i != lines.size() - 1) {
            os << std::endl;
        }
    }
    os << ", ";

    // Gradient existence
    if (x.grad().empty()) {
        os << "grad:empty";
    } else {
        os << "grad:exits";
    }

    // Tail
    os << ")";
}

// -----------------------------------------------------------------------------
// -------------------------- Utilities for Function ---------------------------
// -----------------------------------------------------------------------------
std::vector<Function> ResolveChain(const Function& start_func) {
    std::vector<Function> ret_funcs;

    // Pool
    std::multimap<size_t, Function> m_cand_funcs;
    // Set the last function
    m_cand_funcs.emplace(start_func.getRank(), start_func);

    // Resolving loop
    while (!m_cand_funcs.empty()) {
        // Get one of highest rank function
        Function last_func = PopLast(m_cand_funcs);
        if (last_func.getRank() == 0) {
            continue;  // Already resolved
        }
        ret_funcs.push_back(last_func);

        // Track chain
        Variables inputs = last_func.getInputs();
        for (size_t i = 0; i < inputs.size(); i++) {
            auto&& func = inputs[i].getCreator();
            // When rank is zero, it is already resolved.
            if (0 < func.getRank()) {
                m_cand_funcs.emplace(func.getRank(), func);
            }
        }

        // Clear rank
        last_func.setRank(0);
    }

    return ret_funcs;
}

static Variables RetainVariables(const Variables& vs,
                                 const std::vector<size_t>& retain_idxs) {
    // Create retaining mask
    std::vector<char> mask(vs.size(), false);
    for (auto&& idx : retain_idxs) {
        mask[idx] = true;
    }

    // Copy switching shallow copy or lightweight copy
    Variables retained;
    for (size_t i = 0; i < vs.size(); i++) {
        if (mask[i]) {
            retained.push_back(vs[i]);
        } else {
            retained.push_back(vs[i].copyUnretained());
        }
    }
    return retained;
}

static Shape GedPaddedShape(const Shape& src_shape, const Shape& tgt_shape,
                            const Axis& axes_raw, bool keepdims) {
    const size_t ndim = tgt_shape.size();
    if (!(ndim == 0 || axes_raw.empty() || keepdims)) {
        // Resolve axis (sort: on) (`ResolveAxis` is in tinyndarray)
        const Axis& axes = ResolveAxis(axes_raw, ndim, "Axis backward", true);
        // Reconstruct shape
        Shape padded_shape = src_shape;
        for (auto&& axis : axes) {
            padded_shape.insert(padded_shape.begin() + axis, 1);
        }
        return padded_shape;
    } else {
        // No change
        return src_shape;
    }
}

// =============================================================================
// ============================ Variable Definition ============================
// =============================================================================
Variable::Variable() : m_sub(std::make_shared<Subsetance>()) {}

Variable::Variable(std::shared_ptr<Subsetance> sub) : m_sub(sub) {}

Variable::Variable(const Variable& lhs) = default;  // shallow copy

Variable::Variable(Variable&& lhs) noexcept
    : m_sub(std::move(lhs.m_sub)) {}  // move

Variable& Variable::operator=(const Variable& lhs) = default;  // shallow copy

Variable& Variable::operator=(Variable&& lhs) {  // move
    m_sub = std::move(lhs.m_sub);
    return *this;
}

Variable::~Variable() = default;

// --------------------------------- Subsetance
// ---------------------------------
class Variable::Subsetance {
public:
    Subsetance(const NdArray& data_ = {}) : data(data_), shape(data_.shape()) {}

    NdArray data;
    NdArray grad;
    Shape shape;
    Function creator;
};

// ------------------------------- Constructors --------------------------------
Variable::Variable(const NdArray& v) : m_sub(std::make_shared<Subsetance>(v)) {}

Variable::Variable(FloatList<0> init_list) : Variable(NdArray(init_list)) {}

Variable::Variable(FloatList<1> init_list) : Variable(NdArray(init_list)) {}

Variable::Variable(FloatList<2> init_list) : Variable(NdArray(init_list)) {}

Variable::Variable(FloatList<3> init_list) : Variable(NdArray(init_list)) {}

Variable::Variable(FloatList<4> init_list) : Variable(NdArray(init_list)) {}

Variable::Variable(FloatList<5> init_list) : Variable(NdArray(init_list)) {}

Variable::Variable(FloatList<6> init_list) : Variable(NdArray(init_list)) {}

Variable::Variable(FloatList<7> init_list) : Variable(NdArray(init_list)) {}

Variable::Variable(FloatList<8> init_list) : Variable(NdArray(init_list)) {}

Variable::Variable(FloatList<9> init_list) : Variable(NdArray(init_list)) {}

// ------------------------------- Basic methods -------------------------------
uintptr_t Variable::id() const {
    return m_sub->data.id();
}

bool Variable::empty() const {
    return m_sub->data.empty();
}

size_t Variable::size() const {
    return m_sub->data.size();
}

const Shape& Variable::shape() const {
    return m_sub->shape;
}

size_t Variable::ndim() const {
    return m_sub->shape.size();
}

Variable Variable::copyUnretained() const {
    // Copy all (NdArray is copied in shallow)
    auto sub = std::make_shared<Subsetance>(*m_sub);
    // Unretain
    sub->data = NdArray();
    return Variable(sub);
}

// ------------------------ Unique methods for Variable ------------------------
NdArray& Variable::data() {
    return m_sub->data;
}

const NdArray& Variable::data() const {
    return m_sub->data;
}

NdArray& Variable::grad() {
    return m_sub->grad;
}

const NdArray& Variable::grad() const {
    return m_sub->grad;
}

void Variable::backward(bool clear_grads) {
    Function last_creator = m_sub->creator;

    // Resolve chain
    std::vector<Function> funcs = ResolveChain(last_creator);

    // Remove previous gradients
    if (clear_grads) {
        // Clear last outputs
        ClearGrads(last_creator.getOutputs());
        // For all inputs
        for (auto&& func : funcs) {
            ClearGrads(func.getInputs());
        }
    }

    // Set the last gradients 'one'
    for (auto&& output : last_creator.getOutputs()) {
        output.addGrad({1.f});
    }

    // Run backward functions
    for (auto&& func : funcs) {
        // Call backward
        Variables inputs = func.getInputs();
        Variables outputs = func.getOutputs();
        NdArrays in_grads = func.backward(GetData(inputs), GetData(outputs),
                                          GetGrads(outputs));
        assert(inputs.size() == in_grads.size());

        // Accumulate gradients
        for (size_t i = 0; i < inputs.size(); i++) {
            inputs[i].addGrad(std::move(in_grads[i]));
        }

        // Remove all members (input, output and rank)
        func.clear();
        // Remove used gradient
        ClearGrads(outputs);
    }
}

// ------------------------------ Internal methods -----------------------------
void Variable::setCreator(Function f) {
    m_sub->creator = f;
    f.setRank(f.getRank() + 1);  // Increase rank
}

Function Variable::getCreator() const {
    return m_sub->creator;
}

void Variable::clearData() {
    m_sub->data.resize({0});
}

void Variable::clearGrad() {
    m_sub->grad.resize({0});
}

void Variable::addGrad(NdArray&& grad) {
    // Accumulate gradient for broadcasting
    //   Note: When broadcasting succeeded in forwarding operation, the
    //         broadcasted axes are not ones. Containing ones in the shapes
    //         means that the axes do not affect neither broadcasting nor any
    //         computation. Squeeze operation can omit the non-affective
    //         dimensions.
    if (m_sub->grad.empty()) {
        // Initialize its shape
        m_sub->grad.resize(m_sub->shape);
        Squeeze(m_sub->grad) += Squeeze(std::move(grad));
    } else {
        // Accumulate
        Squeeze(m_sub->grad) += Squeeze(std::move(grad));
    }
}

// --------------------------------- Operators ---------------------------------
// Print
std::ostream& operator<<(std::ostream& os, const Variable& x) {
    PrintVariable(os, x);
    return os;
}

// Single
Variable operator+(const Variable& x) {
    return F::Positive(x);
}

Variable operator-(const Variable& x) {
    return F::Negative(x);
}

// Arithmetic (Variable, Variable)
Variable operator+(const Variable& lhs, const Variable& rhs) {
    return F::Add(lhs, rhs);
}

Variable operator-(const Variable& lhs, const Variable& rhs) {
    return F::Subtract(lhs, rhs);
}

Variable operator*(const Variable& lhs, const Variable& rhs) {
    return F::Multiply(lhs, rhs);
}

Variable operator/(const Variable& lhs, const Variable& rhs) {
    return F::Divide(lhs, rhs);
}

// Arithmetic (Variable, float)
Variable operator+(const Variable& lhs, float rhs) {
    return F::Add(lhs, rhs);
}

Variable operator-(const Variable& lhs, float rhs) {
    return F::Subtract(lhs, rhs);
}

Variable operator*(const Variable& lhs, float rhs) {
    return F::Multiply(lhs, rhs);
}

Variable operator/(const Variable& lhs, float rhs) {
    return F::Divide(lhs, rhs);
}

// Arithmetic (float, Variable)
Variable operator+(float lhs, const Variable& rhs) {
    return F::Add(lhs, rhs);
}

Variable operator-(float lhs, const Variable& rhs) {
    return F::Subtract(lhs, rhs);
}

Variable operator*(float lhs, const Variable& rhs) {
    return F::Multiply(lhs, rhs);
}

Variable operator/(float lhs, const Variable& rhs) {
    return F::Divide(lhs, rhs);
}

// Compound Assignment (Variable, Variable)
Variable operator+=(Variable& lhs, const Variable& rhs) {
    return lhs = F::Add(lhs, rhs);
}

Variable operator-=(Variable& lhs, const Variable& rhs) {
    return lhs = F::Subtract(lhs, rhs);
}

Variable operator*=(Variable& lhs, const Variable& rhs) {
    return lhs = F::Multiply(lhs, rhs);
}

Variable operator/=(Variable& lhs, const Variable& rhs) {
    return lhs = F::Divide(lhs, rhs);
}

// Compound Assignment (Variable, float)
Variable operator+=(Variable& lhs, float rhs) {
    return lhs = F::Add(lhs, rhs);
}

Variable operator-=(Variable& lhs, float rhs) {
    return lhs = F::Subtract(lhs, rhs);
}

Variable operator*=(Variable& lhs, float rhs) {
    return lhs = F::Multiply(lhs, rhs);
}

Variable operator/=(Variable& lhs, float rhs) {
    return lhs = F::Divide(lhs, rhs);
}

// =============================================================================
// ============================ Function Definition ============================
// =============================================================================
Function::Function() : m_sub(std::make_shared<Subsetance>()) {}

Function::Function(std::shared_ptr<Subsetance> sub) : m_sub(sub) {}

Function::Function(const Function& lhs) = default;  // shallow copy

Function::Function(Function&& lhs) noexcept
    : m_sub(std::move(lhs.m_sub)) {}  // move

Function& Function::operator=(const Function& lhs) = default;  // shallow copy

Function& Function::operator=(Function&& lhs) {  // move
    m_sub = std::move(lhs.m_sub);
    return *this;
}

Function::~Function() = default;

// --------------------------------- Subsetance
// ---------------------------------
class Function::Subsetance {
public:
    // Alias for shorter code
    using InNd = const NdArrays&;

    Subsetance(size_t n_inp_ = 0, size_t n_out_ = 0,
               const std::vector<size_t>& retain_inp_idxs_ = {},
               const std::vector<size_t>& retain_out_idxs_ = {})
        : n_inp(n_inp_),
          n_out(n_out_),
          retain_inp_idxs(retain_inp_idxs_),
          retain_out_idxs(retain_out_idxs_) {}
    virtual ~Subsetance() {}

    virtual NdArrays forward(InNd x) {
        return x;  // Through
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) {
        (void)x, (void)y;
        return gy;  // Through
    }

    // Member variables
    size_t rank = 0;
    Variables inputs;
    Variables outputs;
    // Constant member variables
    const size_t n_inp;
    const size_t n_out;
    const std::vector<size_t> retain_inp_idxs;
    const std::vector<size_t> retain_out_idxs;
};

// ---------------------------------- Methods ----------------------------------
void Function::clear() {
    m_sub = std::make_shared<Subsetance>();
}

Variables Function::operator()(const Variables& x) {
    // Forward (with variable conversion)
    auto&& x_data = GetData(x);
    auto&& y_data = forward(std::move(x_data));  // with free
    auto&& y = CvtToVars(std::move(y_data));     // with free
    // Retain input/output variables
    m_sub->inputs = RetainVariables(x, m_sub->retain_inp_idxs);
    m_sub->outputs = RetainVariables(y, m_sub->retain_out_idxs);
    // Set rank of this function (maximum one)
    m_sub->rank = 0;
    for (auto&& x_elem : x) {
        const size_t rank_cand = x_elem.getCreator().getRank();
        m_sub->rank = std::max(m_sub->rank, rank_cand);
    }

    // Build chain
    for (auto&& y_elem : y) {
        y_elem.setCreator(m_sub);
    }

    return std::move(y);
}

NdArrays Function::forward(const NdArrays& x) {
    // Forward with size checking
    CheckVecSize(x, m_sub->n_inp);
    auto&& y = m_sub->forward(x);
    CheckVecSize(y, m_sub->n_out);
    return std::move(y);
}

NdArrays Function::backward(const NdArrays& x, const NdArrays& y,
                            const NdArrays& gy) {
    // Backward with size checking
    CheckVecSize(x, m_sub->n_inp);
    CheckVecSize(y, m_sub->n_out);
    CheckVecSize(gy, m_sub->n_out);
    auto&& gx = m_sub->backward(x, y, gy);
    CheckVecSize(gx, m_sub->n_inp);
    return std::move(gx);
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

// ---------------------------------- Single -----------------------------------
struct PositiveSubset : public Function::Subsetance {
    PositiveSubset()
        : Subsetance(1, 1, {}, {}) {}  // n_inp, n_out, retain_indices
    virtual ~PositiveSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {+x[0]};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)x, (void)y;
        return {+gy[0]};
    }
};

struct NegativeSubset : public Function::Subsetance {
    NegativeSubset() : Subsetance(1, 1, {}, {}) {}
    virtual ~NegativeSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {-x[0]};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)x, (void)y;
        return {-gy[0]};
    }
};

// ---------------------- Arithmetic (Variable, Variable) ----------------------
struct AddSubset : public Function::Subsetance {
    AddSubset() : Subsetance(2, 1, {0, 1}, {}) {}
    virtual ~AddSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {x[0] + x[1]};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        return {SumTo(gy[0], x[0].shape()), SumTo(gy[0], x[1].shape())};
    }
};

struct SubtractSubset : public Function::Subsetance {
    SubtractSubset() : Subsetance(2, 1, {0, 1}, {}) {}
    virtual ~SubtractSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {x[0] - x[1]};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        return {SumTo(gy[0], x[0].shape()), SumTo(-gy[0], x[1].shape())};
    }
};

struct MultiplySubset : public Function::Subsetance {
    MultiplySubset() : Subsetance(2, 1, {0, 1}, {}) {}
    virtual ~MultiplySubset() {}
    virtual NdArrays forward(InNd x) override {
        return {x[0] * x[1]};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        return {SumTo(gy[0] * x[1], x[0].shape()),
                SumTo(gy[0] * x[0], x[1].shape())};
    }
};

struct DivideSubset : public Function::Subsetance {
    DivideSubset() : Subsetance(2, 1, {0, 1}, {}) {}
    virtual ~DivideSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {x[0] / x[1]};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        const auto& gx0 = gy[0] / x[1];
        const auto& gx1 = -gx0 * x[0] / x[1];
        return {SumTo(gx0, x[0].shape()), SumTo(gx1, x[1].shape())};
    }
};

// ------------------------ Arithmetic (Variable, float) -----------------------
struct AddFloatSubset : public Function::Subsetance {
    AddFloatSubset(float c_) : Subsetance(1, 1, {0}, {}), c(c_) {}
    virtual ~AddFloatSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {x[0] + c};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        return {SumTo(gy[0], x[0].shape())};
    }
    const float c;
};

struct SubtractFloatSubset : public Function::Subsetance {
    SubtractFloatSubset(float c_) : Subsetance(1, 1, {0}, {}), c(c_) {}
    virtual ~SubtractFloatSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {x[0] - c};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        return {SumTo(gy[0], x[0].shape())};
    }
    const float c;
};

struct MultiplyFloatSubset : public Function::Subsetance {
    MultiplyFloatSubset(float c_) : Subsetance(1, 1, {0}, {}), c(c_) {}
    virtual ~MultiplyFloatSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {x[0] * c};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        return {SumTo(gy[0] * c, x[0].shape())};
    }
    const float c;
};

struct DivideFloatSubset : public Function::Subsetance {
    DivideFloatSubset(float c_) : Subsetance(1, 1, {0}, {}), c(c_) {}
    virtual ~DivideFloatSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {x[0] / c};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        const auto& gx0 = gy[0] / c;
        return {SumTo(gx0, x[0].shape())};
    }
    const float c;
};

// ----------------------- Arithmetic (float, Variable) ------------------------
struct SubtractFromFloatSubset : public Function::Subsetance {
    SubtractFromFloatSubset(float c_) : Subsetance(1, 1, {0}, {}), c(c_) {}
    virtual ~SubtractFromFloatSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {c - x[0]};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        return {SumTo(-gy[0], x[0].shape())};
    }
    const float c;
};

struct DivideFromFloatSubset : public Function::Subsetance {
    DivideFromFloatSubset(float c_) : Subsetance(1, 1, {0}, {}), c(c_) {}
    virtual ~DivideFromFloatSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {c / x[0]};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        const auto& gx0 = gy[0] / x[0];
        const auto& gx1 = -gx0 * c / x[0];
        return {SumTo(gx1, x[0].shape())};
    }
    const float c;
};

// ----------------------------- Matrix operators ------------------------------
struct MatmulSubset : public Function::Subsetance {
    MatmulSubset() : Subsetance(2, 1, {0, 1}, {}) {}
    virtual ~MatmulSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {Matmul(x[0], x[1])};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        const NdArray& a = x[0];
        const NdArray& b = x[1];
        const NdArray& gy0 = gy[0];
        const bool is_a_vec = (a.ndim() == 1);
        const bool is_b_vec = (b.ndim() == 1);

        // For gradient lhs
        NdArray ga;
        if (is_b_vec) {
            if (is_a_vec) {
                ga = gy0 * b;
            } else {
                // Multiply (Extend gy axis at (-1))
                ga = ExpandDims(gy0, -1) * b;
            }
        } else {
            ga = SumTo(Matmul(gy0, Swapaxes(b, -1, -2)), a.shape());
        }

        // For gradient rhs
        NdArray gb;
        if (is_a_vec) {
            if (is_b_vec) {
                gb = a * gy0;
            } else {
                // Extend gy axis at (-2)
                auto gy_shape = gy0.shape();
                if (1 < gy_shape.size()) {
                    gy_shape.insert(gy_shape.end() - 1, 1);
                }
                // Multiply (Extend a axis (-1))
                gb = ExpandDims(a, -1) * gy0.reshape(gy_shape);
            }
        } else if (is_b_vec) {
            // Multiply (Extend gy axis at (-1))
            gb = Matmul(Swapaxes(a, -1, -2), ExpandDims(gy0, -1));
            // Shrink the last dim (b is vector)
            gb = gb.reshape(-1);
        } else {
            gb = SumTo(Matmul(Swapaxes(a, -1, -2), gy0), b.shape());
        }

        return {std::move(ga), std::move(gb)};
    }
};

// --------------------------- Basic math operators ----------------------------
struct AbsSubset : public Function::Subsetance {
    AbsSubset() : Subsetance(1, 1, {0}, {}) {}
    virtual ~AbsSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {Abs(x[0])};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        return {Sign(x[0] * gy[0])};
    }
};

struct ClipSubset : public Function::Subsetance {
    ClipSubset(float x_min_, float x_max_)
        : Subsetance(1, 1, {0}, {}), x_min(x_min_), x_max(x_max_) {}
    virtual ~ClipSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {Clip(x[0], x_min, x_max)};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        NdArray cond = (x_min <= x[0]) * (x[0] <= x_max);
        return {std::move(cond) * gy[0]};
    }
    float x_min, x_max;
};

struct SqrtSubset : public Function::Subsetance {
    SqrtSubset() : Subsetance(1, 1, {}, {0}) {}
    virtual ~SqrtSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {Sqrt(x[0])};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)x;
        return {gy[0] / (y[0] * 2.f)};
    }
};

struct ExpSubset : public Function::Subsetance {
    ExpSubset() : Subsetance(1, 1, {}, {0}) {}
    virtual ~ExpSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {Exp(x[0])};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)x;
        return {gy[0] * y[0]};
    }
};

struct LogSubset : public Function::Subsetance {
    LogSubset() : Subsetance(1, 1, {0}, {}) {}
    virtual ~LogSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {Log(x[0])};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        return {gy[0] / x[0]};
    }
};

struct SquareSubset : public Function::Subsetance {
    SquareSubset() : Subsetance(1, 1, {0}, {}) {}
    virtual ~SquareSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {Square(x[0])};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        return {gy[0] * 2.f * x[0]};
    }
};

struct PowerSubset : public Function::Subsetance {
    PowerSubset() : Subsetance(2, 1, {0, 1}, {0}) {}
    virtual ~PowerSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {Power(x[0], x[1])};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        return {SumTo(x[1] * Power(x[0], x[1] - 1.f) * gy[0], x[0].shape()),
                SumTo(Log(x[0]) * y[0] * gy[0], x[1].shape())};
    }
};

struct PowerFloatSubset : public Function::Subsetance {
    PowerFloatSubset(float c_) : Subsetance(1, 1, {0}, {}), c(c_) {}
    virtual ~PowerFloatSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {Power(x[0], c)};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        return {SumTo(c * Power(x[0], c - 1.f) * gy[0], x[0].shape())};
    }
    const float c;
};

struct PowerFromFloatSubset : public Function::Subsetance {
    PowerFromFloatSubset(float c_) : Subsetance(1, 1, {0}, {0}), c(c_) {}
    virtual ~PowerFromFloatSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {Power(c, x[0])};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        return {SumTo(std::log(c) * y[0] * gy[0], x[0].shape())};
    }
    const float c;
};

// -------------------------  Trigonometric functions --------------------------
struct SinSubset : public Function::Subsetance {
    SinSubset() : Subsetance(1, 1, {0}, {}) {}
    virtual ~SinSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {Sin(x[0])};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        return {gy[0] * Cos(x[0])};
    }
};

struct CosSubset : public Function::Subsetance {
    CosSubset() : Subsetance(1, 1, {0}, {}) {}
    virtual ~CosSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {Cos(x[0])};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        return {gy[0] * -Sin(x[0])};
    }
};

struct TanSubset : public Function::Subsetance {
    TanSubset() : Subsetance(1, 1, {0}, {}) {}
    virtual ~TanSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {Tan(x[0])};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        return {gy[0] / Square(Cos(x[0]))};
    }
};

// ---------------------- Inverse trigonometric functions ----------------------
struct ArcSinSubset : public Function::Subsetance {
    ArcSinSubset() : Subsetance(1, 1, {0}, {}) {}
    virtual ~ArcSinSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {ArcSin(x[0])};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        return {gy[0] / Sqrt(1.f - Square(x[0]))};
    }
};

struct ArcCosSubset : public Function::Subsetance {
    ArcCosSubset() : Subsetance(1, 1, {0}, {}) {}
    virtual ~ArcCosSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {ArcCos(x[0])};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        return {gy[0] / -Sqrt(1.f - Square(x[0]))};
    }
};

struct ArcTanSubset : public Function::Subsetance {
    ArcTanSubset() : Subsetance(1, 1, {0}, {}) {}
    virtual ~ArcTanSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {ArcTan(x[0])};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        return {gy[0] / (1.f + Square(x[0]))};
    }
};

struct ArcTan2Subset : public Function::Subsetance {
    ArcTan2Subset() : Subsetance(2, 1, {0, 1}, {}) {}
    virtual ~ArcTan2Subset() {}
    virtual NdArrays forward(InNd x) override {
        return {ArcTan2(x[0], x[1])};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        NdArray sqnorm = Square(x[0]) + Square(x[1]);
        const NdArray gx0 = x[1] / sqnorm * gy[0];
        const NdArray gx1 = -x[0] / std::move(sqnorm) * gy[0];
        return {SumTo(gx0, x[0].shape()), SumTo(gx1, x[1].shape())};
    }
};

struct ArcTan2FloatSubset : public Function::Subsetance {
    ArcTan2FloatSubset(float c_) : Subsetance(1, 1, {0}, {}), c(c_) {}
    virtual ~ArcTan2FloatSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {ArcTan2(x[0], c)};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        return {c * gy[0] / (Square(x[0]) + (c * c))};
    }
    const float c;
};

struct ArcTan2FromFloatSubset : public Function::Subsetance {
    ArcTan2FromFloatSubset(float c_) : Subsetance(1, 1, {0}, {}), c(c_) {}
    virtual ~ArcTan2FromFloatSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {ArcTan2(c, x[0])};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        return {-c * gy[0] / ((c * c) + Square(x[0]))};
    }
    const float c;
};

// ------------------------------ Axis functions -------------------------------
struct SumSubset : public Function::Subsetance {
    SumSubset(const Axis& axes_, bool keepdims_)
        : Subsetance(1, 1, {}, {}), axes(axes_), keepdims(keepdims_) {}
    virtual ~SumSubset() {}
    virtual NdArrays forward(InNd x) override {
        x_shape = x[0].shape();
        return {Sum(x[0], axes, keepdims)};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)x, (void)y;
        const Shape& pad_shape =
                GedPaddedShape(gy[0].shape(), x_shape, axes, keepdims);
        return {BroadcastTo(gy[0].reshape(pad_shape), x_shape)};
    }
    const Axis axes;
    const bool keepdims;
    Shape x_shape;
};

struct MeanSubset : public Function::Subsetance {
    MeanSubset(const Axis& axes_, bool keepdims_)
        : Subsetance(1, 1, {}, {}), sum_subset(axes_, keepdims_) {}
    virtual ~MeanSubset() {}
    virtual NdArrays forward(InNd x) override {
        // Forward for sum up
        NdArrays rets = sum_subset.forward(x);
        // Compute multiplier
        multiplier = static_cast<float>(rets[0].size()) /
                     static_cast<float>(x[0].size());
        // Apply multiplier
        rets[0] *= multiplier;
        return rets;
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        // Backward
        NdArrays rets = sum_subset.backward(x, y, gy);
        // Apply multiplier
        rets[0] *= multiplier;
        return rets;
    }
    SumSubset sum_subset;
    float multiplier = 0.f;
};

// Forward operator of MinMaxSubset
template <bool Cond>
NdArray ForwardMinMaxSubset(const NdArray& x, const Axis& axes, bool keepdims);

template <>
NdArray ForwardMinMaxSubset<true>(const NdArray& x, const Axis& axes,
                                  bool keepdims) {
    return Min(x, axes, keepdims);  // true -> min
}

template <>
NdArray ForwardMinMaxSubset<false>(const NdArray& x, const Axis& axes,
                                   bool keepdims) {
    return Max(x, axes, keepdims);  // false -> max
}

template <bool IsMin>
struct MinMaxSubset : public Function::Subsetance {
    MinMaxSubset(const Axis& axes_, bool keepdims_)
        : Subsetance(1, 1, {0}, {0}), axes(axes_), keepdims(keepdims_) {}
    virtual ~MinMaxSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {ForwardMinMaxSubset<IsMin>(x[0], axes, keepdims)};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        const Shape& pad_shape =
                GedPaddedShape(gy[0].shape(), x[0].shape(), axes, keepdims);
        NdArray cond = (x[0] == y[0].reshape(pad_shape));
        NdArray gx = BroadcastTo(gy[0].reshape(pad_shape), cond.shape());
        return {std::move(cond) * std::move(gx)};
    }

    const Axis axes;
    const bool keepdims;
};

// ---------------------------- Logistic functions -----------------------------
struct WhereSubset : public Function::Subsetance {
    WhereSubset(const NdArray& cond_) : Subsetance(2, 1, {}, {}), cond(cond_) {}
    virtual ~WhereSubset() {}
    virtual NdArrays forward(InNd x) override {
        x0_shape = x[0].shape();
        x1_shape = x[1].shape();
        return {Where(cond, x[0], x[1])};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)x, (void)y;
        return {SumTo(Where(cond, gy[0], 0.f), x0_shape),
                SumTo(Where(cond, 0.f, gy[0]), x1_shape)};
    }
    const NdArray& cond;
    Shape x0_shape, x1_shape;
};

struct WhereRightFloatSubset : public Function::Subsetance {
    WhereRightFloatSubset(const NdArray& cond_, float x1_)
        : Subsetance(1, 1, {}, {}), cond(cond_), x1(x1_) {}
    virtual ~WhereRightFloatSubset() {}
    virtual NdArrays forward(InNd x) override {
        x0_shape = x[0].shape();
        return {Where(cond, x[0], x1)};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)x, (void)y;
        return {SumTo(Where(cond, gy[0], 0.f), x0_shape)};
    }
    const NdArray& cond;
    Shape x0_shape;
    const float x1;
};

struct WhereLeftFloatSubset : public Function::Subsetance {
    WhereLeftFloatSubset(const NdArray& cond_, float x0_)
        : Subsetance(1, 1, {}, {}), cond(cond_), x0(x0_) {}
    virtual ~WhereLeftFloatSubset() {}
    virtual NdArrays forward(InNd x) override {
        x1_shape = x[0].shape();
        return {Where(cond, x0, x[0])};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)x, (void)y;
        return {SumTo(Where(cond, 0.f, gy[0]), x1_shape)};
    }
    const NdArray& cond;
    const float x0;
    Shape x1_shape;
};

// ------------------------------ Shape functions ------------------------------
struct ReshapeSubset : public Function::Subsetance {
    ReshapeSubset(const Shape& shape)
        : Subsetance(1, 1, {}, {}), y_shape(shape) {}
    virtual ~ReshapeSubset() {}
    virtual NdArrays forward(InNd x) override {
        x_shape = x[0].shape();  // Store original shape
        return {Reshape(x[0], y_shape)};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)x, (void)y;
        return {Reshape(gy[0], x_shape)};
    }
    const Shape y_shape;
    Shape x_shape;
};

struct SqueezeSubset : public Function::Subsetance {
    SqueezeSubset(const Axis& axes_) : Subsetance(1, 1, {}, {}), axes(axes_) {}
    virtual ~SqueezeSubset() {}
    virtual NdArrays forward(InNd x) override {
        x_shape = x[0].shape();  // Store original shape
        return {Squeeze(x[0], axes)};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)x, (void)y;
        return {Reshape(gy[0], x_shape)};
    }
    const Axis axes;
    Shape x_shape;
};

struct ExpandDimsSubset : public Function::Subsetance {
    ExpandDimsSubset(int axis_) : Subsetance(1, 1, {}, {}), axis(axis_) {}
    virtual ~ExpandDimsSubset() {}
    virtual NdArrays forward(InNd x) override {
        x_shape = x[0].shape();  // Store original shape
        return {ExpandDims(x[0], axis)};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)x, (void)y;
        return {Reshape(gy[0], x_shape)};
    }
    const int axis;
    Shape x_shape;
};

// ----------------------------- Grouping functions ----------------------------
struct StackSubset : public Function::Subsetance {
    StackSubset(int axis_, size_t n_xs)
        : Subsetance(n_xs, 1, {}, {}), axis(axis_) {}
    virtual ~StackSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {Stack(x, axis)};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)x, (void)y;
        return Separate(gy[0], axis);
    }
    const int axis;
};

struct ConcatenateSubset : public Function::Subsetance {
    ConcatenateSubset(int axis_, size_t n_xs)
        : Subsetance(n_xs, 1, {}, {}), axis(axis_) {}
    virtual ~ConcatenateSubset() {}
    virtual NdArrays forward(InNd x) override {
        // Store shapes for back
        shapes.clear();
        shapes.reserve(x.size());
        for (auto&& x_elem : x) {
            shapes.push_back(x_elem.shape());
        }
        // Forward
        return {Concatenate(x, axis)};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)x, (void)y;
        // Resolve axis
        const size_t axis_l = static_cast<size_t>(
                ResolveAxis(axis, shapes[0].size(), "Concat back"));
        // Create indices to split (Note: axis is checked by forwarding)
        Index idxs;
        int next_idx = 0;
        for (size_t i = 0; i < shapes.size() - 1; i++) {
            next_idx += shapes[i][axis_l];
            idxs.push_back(next_idx);
        }
        // Backward
        return Split(gy[0], idxs, axis);
    }
    const int axis;
    std::vector<Shape> shapes;
};

template <typename Param>
struct SplitSubset : public Function::Subsetance {
    SplitSubset(const Param& param_, int axis_, size_t n_ys)
        : Subsetance(1, n_ys, {}, {}), param(param_), axis(axis_) {}
    virtual ~SplitSubset() {}
    virtual NdArrays forward(InNd x) override {
        // Forward
        NdArrays y = Split(x[0], param, axis);
        // Save shapes for backward
        y_shapes.clear();
        y_shapes.reserve(y.size());
        for (auto&& y_elem : y) {
            y_shapes.push_back(y_elem.shape());
        }
        return y;
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)x, (void)y;
        // Fill empty gradients
        NdArrays grads;
        for (size_t i = 0; i < gy.size(); i++) {
            if (gy[i].empty()) {
                grads.push_back(NdArray::Zeros(y_shapes[i]));
            } else {
                grads.push_back(gy[i]);
            }
        }
        // Backward
        return {Concatenate(grads, axis)};
    }
    Param param;
    const int axis;
    std::vector<Shape> y_shapes;
};

struct SeparateSubset : public Function::Subsetance {
    SeparateSubset(int axis_, size_t n_ys)
        : Subsetance(1, n_ys, {}, {}), axis(axis_) {}
    virtual ~SeparateSubset() {}
    virtual NdArrays forward(InNd x) override {
        // Forward
        NdArrays y = Separate(x[0], axis);
        // Save shape for backward
        y0_shape = y[0].shape();
        return y;
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)x, (void)y;
        // Fill empty gradients
        NdArrays grads;
        for (size_t i = 0; i < gy.size(); i++) {
            if (gy[i].empty()) {
                grads.push_back(NdArray::Zeros(y0_shape));
            } else {
                grads.push_back(gy[i]);
            }
        }
        // Backward
        return {Stack(grads, axis)};
    }
    const int axis;
    Shape y0_shape;
};

// -------------------------------- Change view --------------------------------
struct TransposeSubset : public Function::Subsetance {
    TransposeSubset() : Subsetance(1, 1, {}, {}) {}
    virtual ~TransposeSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {Transpose(x[0])};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)x, (void)y;
        return {Transpose(gy[0])};
    }
};

struct SwapaxesSubset : public Function::Subsetance {
    SwapaxesSubset(int axis1_, int axis2_)
        : Subsetance(1, 1, {}, {}), axis1(axis1_), axis2(axis2_) {}
    virtual ~SwapaxesSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {Swapaxes(x[0], axis1, axis2)};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)x, (void)y;
        return {Swapaxes(gy[0], axis1, axis2)};
    }
    const int axis1, axis2;
};

struct BroadcastToSubset : public Function::Subsetance {
    BroadcastToSubset(const Shape& shape_)
        : Subsetance(1, 1, {}, {}), shape(shape_) {}
    virtual ~BroadcastToSubset() {}
    virtual NdArrays forward(InNd x) override {
        x_shape = x[0].shape();
        return {BroadcastTo(x[0], shape)};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)x, (void)y;
        return {SumTo(gy[0], x_shape)};
    }
    const Shape shape;
    Shape x_shape;
};

struct SumToSubset : public Function::Subsetance {
    SumToSubset(const Shape& shape_)
        : Subsetance(1, 1, {}, {}), shape(shape_) {}
    virtual ~SumToSubset() {}
    virtual NdArrays forward(InNd x) override {
        x_shape = x[0].shape();
        return {SumTo(x[0], shape)};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)x, (void)y;
        return {BroadcastTo(gy[0], x_shape)};
    }
    const Shape shape;
    Shape x_shape;
};

// ---------------------------------- Inverse ----------------------------------
struct InvSubset : public Function::Subsetance {
    InvSubset() : Subsetance(1, 1, {}, {0}) {}
    virtual ~InvSubset() {}
    virtual NdArrays forward(InNd x) override {
        return {Inv(x[0])};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)x;
        NdArray invxT = Transpose(y[0]);
        NdArray gx = Matmul(Matmul(-invxT, gy[0]), invxT);
        return {gx};
    }
};

// ------------------------------- Helper Class --------------------------------
// Helper to replace default substance with implemented one
template <typename S>
class FuncImpl : public Function {
public:
    template <typename... Args>
    FuncImpl(Args&&... args)
        : Function(std::make_shared<S>(std::forward<Args>(args)...)) {}
    virtual ~FuncImpl() {}
};

// ----------------------------- Function Wrapping -----------------------------
// Single
Variable Positive(const Variable& x) {
    return FuncImpl<PositiveSubset>()({x})[0];
}

Variable Negative(const Variable& x) {
    return FuncImpl<NegativeSubset>()({x})[0];
}

// Arithmetic (Variable, Variable)
Variable Add(const Variable& lhs, const Variable& rhs) {
    return FuncImpl<AddSubset>()({lhs, rhs})[0];
}

Variable Subtract(const Variable& lhs, const Variable& rhs) {
    return FuncImpl<SubtractSubset>()({lhs, rhs})[0];
}

Variable Multiply(const Variable& lhs, const Variable& rhs) {
    return FuncImpl<MultiplySubset>()({lhs, rhs})[0];
}

Variable Divide(const Variable& lhs, const Variable& rhs) {
    return FuncImpl<DivideSubset>()({lhs, rhs})[0];
}

// Arithmetic (Variable, float)
Variable Add(const Variable& lhs, float rhs) {
    return FuncImpl<AddFloatSubset>(rhs)({lhs})[0];
}

Variable Subtract(const Variable& lhs, float rhs) {
    return FuncImpl<SubtractFloatSubset>(rhs)({lhs})[0];
}

Variable Multiply(const Variable& lhs, float rhs) {
    return FuncImpl<MultiplyFloatSubset>(rhs)({lhs})[0];
}

Variable Divide(const Variable& lhs, float rhs) {
    return FuncImpl<DivideFloatSubset>(rhs)({lhs})[0];
}

// Arithmetic (float, Variable)
Variable Add(float lhs, const Variable& rhs) {
    return FuncImpl<AddFloatSubset>(lhs)({rhs})[0];
}

Variable Subtract(float lhs, const Variable& rhs) {
    return FuncImpl<SubtractFromFloatSubset>(lhs)({rhs})[0];
}

Variable Multiply(float lhs, const Variable& rhs) {
    return FuncImpl<MultiplyFloatSubset>(lhs)({rhs})[0];
}

Variable Divide(float lhs, const Variable& rhs) {
    return FuncImpl<DivideFromFloatSubset>(lhs)({rhs})[0];
}

// Matrix operators
Variable Matmul(const Variable& lhs, const Variable& rhs) {
    return FuncImpl<MatmulSubset>()({lhs, rhs})[0];
}

// Basic math operators
Variable Abs(const Variable& x) {
    return FuncImpl<AbsSubset>()({x})[0];
}

Variable Sign(const Variable& x) {
    return Variable(Sign(x.data()));  // Forward only. (unchain)
}

Variable Ceil(const Variable& x) {
    return Variable(Ceil(x.data()));  // Forward only. (unchain)
}

Variable Floor(const Variable& x) {
    return Variable(Floor(x.data()));  // Forward only. (unchain)
}

Variable Clip(const Variable& x, float x_min, float x_max) {
    return FuncImpl<ClipSubset>(x_min, x_max)({x})[0];
}

Variable Sqrt(const Variable& x) {
    return FuncImpl<SqrtSubset>()({x})[0];
}

Variable Exp(const Variable& x) {
    return FuncImpl<ExpSubset>()({x})[0];
}

Variable Log(const Variable& x) {
    return FuncImpl<LogSubset>()({x})[0];
}

Variable Square(const Variable& x) {
    return FuncImpl<SquareSubset>()({x})[0];
}

Variable Power(const Variable& x, const Variable& y) {
    return FuncImpl<PowerSubset>()({x, y})[0];
}

Variable Power(const Variable& x, float y) {
    return FuncImpl<PowerFloatSubset>(y)({x})[0];
}

Variable Power(float x, const Variable& y) {
    return FuncImpl<PowerFromFloatSubset>(x)({y})[0];
}

// Trigonometric functions
Variable Sin(const Variable& x) {
    return FuncImpl<SinSubset>()({x})[0];
}

Variable Cos(const Variable& x) {
    return FuncImpl<CosSubset>()({x})[0];
}

Variable Tan(const Variable& x) {
    return FuncImpl<TanSubset>()({x})[0];
}

// Inverse trigonometric functions
Variable ArcSin(const Variable& x) {
    return FuncImpl<ArcSinSubset>()({x})[0];
}

Variable ArcCos(const Variable& x) {
    return FuncImpl<ArcCosSubset>()({x})[0];
}

Variable ArcTan(const Variable& x) {
    return FuncImpl<ArcTanSubset>()({x})[0];
}

Variable ArcTan2(const Variable& y, const Variable& x) {
    return FuncImpl<ArcTan2Subset>()({y, x})[0];
}

Variable ArcTan2(const Variable& y, float x) {
    return FuncImpl<ArcTan2FloatSubset>(x)({y})[0];
}

Variable ArcTan2(float y, const Variable& x) {
    return FuncImpl<ArcTan2FromFloatSubset>(y)({x})[0];
}

// Axis functions
Variable Sum(const Variable& x, const Axis& axes, bool keepdims) {
    return FuncImpl<SumSubset>(axes, keepdims)({x})[0];
}

Variable Mean(const Variable& x, const Axis& axes, bool keepdims) {
    return FuncImpl<MeanSubset>(axes, keepdims)({x})[0];
}

Variable Min(const Variable& x, const Axis& axes, bool keepdims) {
    return FuncImpl<MinMaxSubset<true>>(axes, keepdims)({x})[0];
}

Variable Max(const Variable& x, const Axis& axes, bool keepdims) {
    return FuncImpl<MinMaxSubset<false>>(axes, keepdims)({x})[0];
}

// Logistic functions
Variable Where(const NdArray& cond, const Variable& x0, const Variable& x1) {
    return FuncImpl<WhereSubset>(cond)({x0, x1})[0];
}

Variable Where(const NdArray& cond, const Variable& x0, float x1) {
    return FuncImpl<WhereRightFloatSubset>(cond, x1)({x0})[0];
}

Variable Where(const NdArray& cond, float x0, const Variable& x1) {
    return FuncImpl<WhereLeftFloatSubset>(cond, x0)({x1})[0];
}

// Shape functions
Variable Reshape(const Variable& x, const Shape& shape) {
    return FuncImpl<ReshapeSubset>(shape)({x})[0];
}

Variable Squeeze(const Variable& x, const Axis& axes) {
    return FuncImpl<SqueezeSubset>(axes)({x})[0];
}

Variable ExpandDims(const Variable& x, int axis) {
    return FuncImpl<ExpandDimsSubset>(axis)({x})[0];
}

// Grouping functions
Variable Stack(const std::vector<Variable>& xs, int axis) {
    return FuncImpl<StackSubset>(axis, xs.size())(xs)[0];
}

Variable Concatenate(const std::vector<Variable>& xs, int axis) {
    return FuncImpl<ConcatenateSubset>(axis, xs.size())(xs)[0];
}

std::vector<Variable> Split(const Variable& x, int n_section, int axis) {
    axis = ResolveAxis(axis, x.ndim(), "Split (n_section)");
    const size_t n_ys = static_cast<size_t>(n_section);
    return FuncImpl<SplitSubset<int>>(n_section, axis, n_ys)({x});
}

std::vector<Variable> Split(const Variable& x, const Index& idxs, int axis) {
    axis = ResolveAxis(axis, x.ndim(), "Split (index)");
    const size_t n_ys = idxs.size() + 1;
    return FuncImpl<SplitSubset<Index>>(idxs, axis, n_ys)({x});
}

std::vector<Variable> Separate(const Variable& x, int axis) {
    axis = ResolveAxis(axis, x.ndim(), "Separate");
    const size_t n_ys =
            static_cast<size_t>(x.shape()[static_cast<size_t>(axis)]);
    return FuncImpl<SeparateSubset>(axis, n_ys)({x});
}

// Change view
Variable Transpose(const Variable& x) {
    return FuncImpl<TransposeSubset>()({x})[0];
}

Variable Swapaxes(const Variable& x, int axis1, int axis2) {
    return FuncImpl<SwapaxesSubset>(axis1, axis2)({x})[0];
}

Variable BroadcastTo(const Variable& x, const Shape& shape) {
    return FuncImpl<BroadcastToSubset>(shape)({x})[0];
}

Variable SumTo(const Variable& x, const Shape& shape) {
    return FuncImpl<SumToSubset>(shape)({x})[0];
}

// Inverse
Variable Inv(const Variable& x) {
    return FuncImpl<InvSubset>()({x})[0];
}

}  // namespace F

#endif  // TINYDIFF_IMPLEMENTATION
// #############################################################################
// ############################# End of Definition ############################
// #############################################################################

#ifndef TINYDIFF_NO_NAMESPACE
}  // namespace tinydiff
#endif  // TINYDIFF_NO_NAMESPACE

#endif /* end of include guard */
