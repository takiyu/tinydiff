#ifndef TINYDIFF_H_ONCE
#define TINYDIFF_H_ONCE

#ifndef TINYDIFF_NO_INCLUDE
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <exception>
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

    NdArray data() const;
    NdArray grad() const;
    void backward(bool clear_grads = true);

    void setCreator(Function f);
    Function getCreator() const;
    void clearData();
    void clearGrad();
    void addGrad(const NdArray& grad);

    class Substance;

private:
    std::shared_ptr<Substance> m_sub;
    Variable(std::shared_ptr<Substance> sub);
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

    class Substance;

protected:
    std::shared_ptr<Substance> m_sub;
    Function(std::shared_ptr<Substance> sub);
};

// =============================================================================
// ============================ Implemented Function ===========================
// =============================================================================

namespace F {

// Single
Variable Pos(const Variable& x);
Variable Neg(const Variable& x);
// Arithmetic (Variable, Variable)
Variable Add(const Variable& lhs, const Variable& rhs);
Variable Sub(const Variable& lhs, const Variable& rhs);
Variable Mul(const Variable& lhs, const Variable& rhs);
Variable Div(const Variable& lhs, const Variable& rhs);
// Arithmetic (Variable, float)
Variable Add(const Variable& lhs, float rhs);
Variable Sub(const Variable& lhs, float rhs);
Variable Mul(const Variable& lhs, float rhs);
Variable Div(const Variable& lhs, float rhs);
// Arithmetic (float, Variable)
Variable Add(float lhs, const Variable& rhs);
Variable Sub(float lhs, const Variable& rhs);
Variable Mul(float lhs, const Variable& rhs);
Variable Div(float lhs, const Variable& rhs);
// Basic math operators
Variable Exp(const Variable& x);

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

static NdArray SumTo(const NdArray& x, const Shape& shape) {
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

// =============================================================================
// ============================ Variable Definition ============================
// =============================================================================
Variable::Variable() : m_sub(std::make_shared<Substance>()) {}

Variable::Variable(std::shared_ptr<Substance> sub) : m_sub(sub) {}

Variable::Variable(const Variable& lhs) = default;  // shallow copy

Variable::Variable(Variable&& lhs) noexcept
    : m_sub(std::move(lhs.m_sub)) {}  // move

Variable& Variable::operator=(const Variable& lhs) = default;  // shallow copy

Variable& Variable::operator=(Variable&& lhs) {  // move
    m_sub = std::move(lhs.m_sub);
    return *this;
}

Variable::~Variable() = default;

// --------------------------------- Substance ---------------------------------
class Variable::Substance {
public:
    Substance(const NdArray& data_ = {}) : data(data_), shape(data_.shape()) {}

    NdArray data;
    NdArray grad;
    Shape shape;
    Function creator;
};

// ------------------------------- Constructors --------------------------------
Variable::Variable(const NdArray& v) : m_sub(std::make_shared<Substance>(v)) {}

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
    auto sub = std::make_shared<Substance>(*m_sub);
    // Unretain
    sub->data = NdArray();
    return Variable(sub);
}

// ------------------------ Unique methods for Variable ------------------------
NdArray Variable::data() const {
    return m_sub->data;
}

NdArray Variable::grad() const {
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
            inputs[i].addGrad(in_grads[i]);
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

void Variable::addGrad(const NdArray& grad) {
    // Initialize its shape
    if (m_sub->grad.empty()) {
        m_sub->grad.resize(m_sub->shape);
    }
    // Accumulate gradient for broadcasting
    //   Note: When broadcasting succeeded in forwarding operation, the
    //         broadcasted axes are not ones. Containing ones in the shapes
    //         means that the axes do not affect neither broadcasting nor any
    //         computation. Squeeze operation can omit the non-affective
    //         dimensions.
    Squeeze(m_sub->grad) += Squeeze(grad);
}

// --------------------------------- Operators ---------------------------------
// Print
std::ostream& operator<<(std::ostream& os, const Variable& x) {
    PrintVariable(os, x);
    return os;
}

// Single
Variable operator+(const Variable& x) {
    return F::Pos(x);
}

Variable operator-(const Variable& x) {
    return F::Neg(x);
}

// Arithmetic (Variable, Variable)
Variable operator+(const Variable& lhs, const Variable& rhs) {
    return F::Add(lhs, rhs);
}

Variable operator-(const Variable& lhs, const Variable& rhs) {
    return F::Sub(lhs, rhs);
}

Variable operator*(const Variable& lhs, const Variable& rhs) {
    return F::Mul(lhs, rhs);
}

Variable operator/(const Variable& lhs, const Variable& rhs) {
    return F::Div(lhs, rhs);
}

// Arithmetic (Variable, float)
Variable operator+(const Variable& lhs, float rhs) {
    return F::Add(lhs, rhs);
}

Variable operator-(const Variable& lhs, float rhs) {
    return F::Sub(lhs, rhs);
}

Variable operator*(const Variable& lhs, float rhs) {
    return F::Mul(lhs, rhs);
}

Variable operator/(const Variable& lhs, float rhs) {
    return F::Div(lhs, rhs);
}

// Arithmetic (float, Variable)
Variable operator+(float lhs, const Variable& rhs) {
    return F::Add(lhs, rhs);
}

Variable operator-(float lhs, const Variable& rhs) {
    return F::Sub(lhs, rhs);
}

Variable operator*(float lhs, const Variable& rhs) {
    return F::Mul(lhs, rhs);
}

Variable operator/(float lhs, const Variable& rhs) {
    return F::Div(lhs, rhs);
}

// Compound Assignment (Variable, Variable)
Variable operator+=(Variable& lhs, const Variable& rhs) {
    return lhs = F::Add(lhs, rhs);
}

Variable operator-=(Variable& lhs, const Variable& rhs) {
    return lhs = F::Sub(lhs, rhs);
}

Variable operator*=(Variable& lhs, const Variable& rhs) {
    return lhs = F::Mul(lhs, rhs);
}

Variable operator/=(Variable& lhs, const Variable& rhs) {
    return lhs = F::Div(lhs, rhs);
}

// Compound Assignment (Variable, float)
Variable operator+=(Variable& lhs, float rhs) {
    return lhs = F::Add(lhs, rhs);
}

Variable operator-=(Variable& lhs, float rhs) {
    return lhs = F::Sub(lhs, rhs);
}

Variable operator*=(Variable& lhs, float rhs) {
    return lhs = F::Mul(lhs, rhs);
}

Variable operator/=(Variable& lhs, float rhs) {
    return lhs = F::Div(lhs, rhs);
}


// =============================================================================
// ============================ Function Definition ============================
// =============================================================================
Function::Function() : m_sub(std::make_shared<Substance>()) {}

Function::Function(std::shared_ptr<Substance> sub) : m_sub(sub) {}

Function::Function(const Function& lhs) = default;  // shallow copy

Function::Function(Function&& lhs) noexcept
    : m_sub(std::move(lhs.m_sub)) {}  // move

Function& Function::operator=(const Function& lhs) = default;  // shallow copy

Function& Function::operator=(Function&& lhs) {  // move
    m_sub = std::move(lhs.m_sub);
    return *this;
}

Function::~Function() = default;

// --------------------------------- Substance ---------------------------------
class Function::Substance {
public:
    // Alias for shorter code
    using InNd = const NdArrays&;

    Substance(size_t n_inp_ = 0, size_t n_out_ = 0,
              const std::vector<size_t>& retain_inp_idxs_ = {},
              const std::vector<size_t>& retain_out_idxs_ = {})
        : n_inp(n_inp_),
          n_out(n_out_),
          retain_inp_idxs(retain_inp_idxs_),
          retain_out_idxs(retain_out_idxs_) {}
    virtual ~Substance() {}

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
    m_sub = std::make_shared<Substance>();
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

// Single
struct PosSubst : public Function::Substance {
    PosSubst() : Substance(1, 1, {}, {}) {}  // n_inp, n_out, retain_indices
    virtual ~PosSubst() {}
    virtual NdArrays forward(InNd x) override {
        return {+x[0]};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)x, (void)y;
        return {+gy[0]};
    }
};

struct NegSubst : public Function::Substance {
    NegSubst() : Substance(1, 1, {}, {}) {}
    virtual ~NegSubst() {}
    virtual NdArrays forward(InNd x) override {
        return {-x[0]};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)x, (void)y;
        return {-gy[0]};
    }
};

// Arithmetic (Variable, Variable)
struct AddSubst : public Function::Substance {
    AddSubst() : Substance(2, 1, {0, 1}, {}) {}
    virtual ~AddSubst() {}
    virtual NdArrays forward(InNd x) override {
        return {x[0] + x[1]};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        return {SumTo(gy[0], x[0].shape()), SumTo(gy[0], x[1].shape())};
    }
};

struct SubSubst : public Function::Substance {
    SubSubst() : Substance(2, 1, {0, 1}, {}) {}
    virtual ~SubSubst() {}
    virtual NdArrays forward(InNd x) override {
        return {x[0] - x[1]};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        return {SumTo(gy[0], x[0].shape()), SumTo(-gy[0], x[1].shape())};
    }
};

struct MulSubst : public Function::Substance {
    MulSubst() : Substance(2, 1, {0, 1}, {}) {}
    virtual ~MulSubst() {}
    virtual NdArrays forward(InNd x) override {
        return {x[0] * x[1]};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        return {SumTo(gy[0] * x[1], x[0].shape()),
                SumTo(gy[0] * x[0], x[1].shape())};
    }
};

struct DivSubst : public Function::Substance {
    DivSubst() : Substance(2, 1, {0, 1}, {}) {}
    virtual ~DivSubst() {}
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

// Arithmetic (Variable, float)
struct AddFloatSubst : public Function::Substance {
    AddFloatSubst(float c_) : Substance(1, 1, {0}, {}), c(c_) {}
    virtual ~AddFloatSubst() {}
    virtual NdArrays forward(InNd x) override {
        return {x[0] + c};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        return {SumTo(gy[0], x[0].shape())};
    }
    const float c;
};

struct SubFloatSubst : public Function::Substance {
    SubFloatSubst(float c_) : Substance(1, 1, {0}, {}), c(c_) {}
    virtual ~SubFloatSubst() {}
    virtual NdArrays forward(InNd x) override {
        return {x[0] - c};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        return {SumTo(gy[0], x[0].shape())};
    }
    const float c;
};

struct MulFloatSubst : public Function::Substance {
    MulFloatSubst(float c_) : Substance(1, 1, {0}, {}), c(c_) {}
    virtual ~MulFloatSubst() {}
    virtual NdArrays forward(InNd x) override {
        return {x[0] * c};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        return {SumTo(gy[0] * c, x[0].shape())};
    }
    const float c;
};

struct DivFloatSubst : public Function::Substance {
    DivFloatSubst(float c_) : Substance(1, 1, {0}, {}), c(c_) {}
    virtual ~DivFloatSubst() {}
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

// Arithmetic (float, Variable)
struct SubFromFloatSubst : public Function::Substance {
    SubFromFloatSubst(float c_) : Substance(1, 1, {0}, {}), c(c_) {}
    virtual ~SubFromFloatSubst() {}
    virtual NdArrays forward(InNd x) override {
        return {c - x[0]};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)y;
        return {SumTo(-gy[0], x[0].shape())};
    }
    const float c;
};

struct DivFromFloatSubst : public Function::Substance {
    DivFromFloatSubst(float c_) : Substance(1, 1, {0}, {}), c(c_) {}
    virtual ~DivFromFloatSubst() {}
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

// Basic math operators
struct ExpSubst : public Function::Substance {
    ExpSubst() : Substance(1, 1, {}, {0}) {}
    virtual ~ExpSubst() {}
    virtual NdArrays forward(InNd x) override {
        return {Exp(x[0])};
    }
    virtual NdArrays backward(InNd x, InNd y, InNd gy) override {
        (void)x;
        return {gy[0] * y[0]};
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
Variable Pos(const Variable& x) {
    return FuncImpl<PosSubst>()({x})[0];
}

Variable Neg(const Variable& x) {
    return FuncImpl<NegSubst>()({x})[0];
}

// Arithmetic (Variable, Variable)
Variable Add(const Variable& lhs, const Variable& rhs) {
    return FuncImpl<AddSubst>()({lhs, rhs})[0];
}

Variable Sub(const Variable& lhs, const Variable& rhs) {
    return FuncImpl<SubSubst>()({lhs, rhs})[0];
}

Variable Mul(const Variable& lhs, const Variable& rhs) {
    return FuncImpl<MulSubst>()({lhs, rhs})[0];
}

Variable Div(const Variable& lhs, const Variable& rhs) {
    return FuncImpl<DivSubst>()({lhs, rhs})[0];
}

// Arithmetic (Variable, float)
Variable Add(const Variable& lhs, float rhs) {
    return FuncImpl<AddFloatSubst>(rhs)({lhs})[0];
}

Variable Sub(const Variable& lhs, float rhs) {
    return FuncImpl<SubFloatSubst>(rhs)({lhs})[0];
}

Variable Mul(const Variable& lhs, float rhs) {
    return FuncImpl<MulFloatSubst>(rhs)({lhs})[0];
}

Variable Div(const Variable& lhs, float rhs) {
    return FuncImpl<DivFloatSubst>(rhs)({lhs})[0];
}

// Arithmetic (float, Variable)
Variable Add(float lhs, const Variable& rhs) {
    return FuncImpl<AddFloatSubst>(lhs)({rhs})[0];
}

Variable Sub(float lhs, const Variable& rhs) {
    return FuncImpl<SubFromFloatSubst>(lhs)({rhs})[0];
}

Variable Mul(float lhs, const Variable& rhs) {
    return FuncImpl<MulFloatSubst>(lhs)({rhs})[0];
}

Variable Div(float lhs, const Variable& rhs) {
    return FuncImpl<DivFromFloatSubst>(lhs)({rhs})[0];
}

// Basic math operators
Variable Exp(const Variable& x) {
    return FuncImpl<ExpSubst>()({x})[0];
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
