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

// Declaration of NdArray
#undef TINYNDARRAY_H_ONCE
#define TINYNDARRAY_NO_NAMESPACE
#include "./tinyndarray/tinyndarray.h"

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

// Definitions of NdArray
#undef TINYNDARRAY_H_ONCE
#define TINYNDARRAY_NO_NAMESPACE
#define TINYNDARRAY_NO_DECLARATION
#define TINYNDARRAY_IMPLEMENTATION
#include "./tinyndarray/tinyndarray.h"

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
