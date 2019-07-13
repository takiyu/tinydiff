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

namespace tinydiff {

// #############################################################################
// ############################ Begin of Declaration ###########################
// #############################################################################
#ifndef TINYDIFF_NO_DECLARATION

// Declaration of NdArray
#define TINYNDARRAY_NO_NAMESPACE
#include "./tinyndarray/tinyndarray.h"

using NdArrays = std::vector<NdArray>;
class Variable;
using Variables = std::vector<Variable>;
class Function;

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

    Variable(float v);
    Variable(const NdArray& v);

    NdArray data() const;
    NdArray grad() const;
    void backward();

    void setCreator(Function f);
    Function getCreator() const;
    void clearGrads();
    void addGrad(const NdArray& grad);

    class Substance;

private:
    std::shared_ptr<Substance> m_sub;
    Variable(std::shared_ptr<Substance> sub);
};

// --------------------------------- Operators ---------------------------------
std::ostream& operator<<(std::ostream& os, Variable& x);
Variable operator+(const Variable& lhs, const Variable& rhs);
Variable operator-(const Variable& lhs, const Variable& rhs);
Variable operator*(const Variable& lhs, const Variable& rhs);
Variable operator/(const Variable& lhs, const Variable& rhs);

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

    // Clear all member variables
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

Variable Add(Variable lhs, Variable rhs);
Variable Sub(Variable lhs, Variable rhs);
Variable Mul(Variable lhs, Variable rhs);
Variable Div(Variable lhs, Variable rhs);
Variable Exp(Variable x);

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
V PopLast(std::map<K, V>& m) {
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

// -----------------------------------------------------------------------------
// -------------------------- Utilities for Function ---------------------------
// -----------------------------------------------------------------------------
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
    Substance() {}
    Substance(const NdArray& v_) : v(v_) {}
    NdArray v;
    NdArray grad;
    Function creator;
};

// ------------------------------- Constructors --------------------------------
Variable::Variable(float v) : m_sub(std::make_shared<Substance>(NdArray{v})) {}

Variable::Variable(const NdArray& v) : m_sub(std::make_shared<Substance>(v)) {}

// ---------------------------------- Methods ----------------------------------
NdArray Variable::data() const {
    return m_sub->v;
}

NdArray Variable::grad() const {
    return m_sub->grad;
}

void Variable::backward() {
    // Set the last gradients 'one'
    for (auto&& output : m_sub->creator.getOutputs()) {
        output.addGrad({1.f});
    }

    // Ordered storage to resolve
    std::map<size_t, Function> cand_funcs;
    // Register the last node
    cand_funcs[m_sub->creator.getRank()] = m_sub->creator;
    // Resolving loop
    while (!cand_funcs.empty()) {
        // Get highest rank function
        Function last_func = PopLast(cand_funcs);
        // Ignore no connected function (already resolved)
        if (last_func.getRank() == 0) {
            continue;
        }

        // Call backward
        Variables inputs = last_func.getInputs();
        Variables outputs = last_func.getOutputs();
        NdArrays in_grads = last_func.backward(
                GetData(inputs), GetData(outputs), GetGrads(outputs));
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
        // Remove used gradient
        for (auto&& output : outputs) {
            output.clearGrads();
        }
    }
}

void Variable::setCreator(Function f) {
    m_sub->creator = f;
    f.setRank(f.getRank() + 1);  // Increase rank
}

Function Variable::getCreator() const {
    return m_sub->creator;
}

void Variable::clearGrads() {
    m_sub->grad = NdArray();
}

void Variable::addGrad(const NdArray& grad) {
    // Initialize its shape
    if (m_sub->grad.empty()) {
        m_sub->grad = NdArray::Zeros(m_sub->v.shape());  // TODO: Omit filling
    }
    // Add
    m_sub->grad += grad;
}

// --------------------------------- Operators ---------------------------------
std::ostream& operator<<(std::ostream& os, Variable& x) {
    return os << x.data();
}

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
    virtual ~Substance() {}
    virtual NdArrays forward(const NdArrays& x) {
        (void)x;
        throw std::runtime_error("Invalid use of tinydiff::Function");
    }
    virtual NdArrays backward(const NdArrays& x, const NdArrays& y,
                              const NdArrays& gy) {
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
    auto&& x_data = GetData(x);
    auto&& y_data = forward(std::move(x_data));  // with free
    auto&& y = CvtToVars(std::move(y_data));     // with free
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

NdArrays Function::forward(const NdArrays& x) {
    return m_sub->forward(x);
}

NdArrays Function::backward(const NdArrays& x, const NdArrays& y,
                            const NdArrays& gy) {
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
    virtual NdArrays forward(const NdArrays& x) {
        CheckVecSize(x, 2);
        return {x[0] + x[1]};
    }
    virtual NdArrays backward(const NdArrays& x, const NdArrays& y,
                              const NdArrays& gy) {
        CheckVecSize(x, 2, y, 1, gy, 1);
        return {SumTo(gy[0], x[0].shape()), SumTo(gy[0], x[1].shape())};
    }
};

class SubSub : public Function::Substance {
public:
    virtual ~SubSub() {}
    virtual NdArrays forward(const NdArrays& x) {
        CheckVecSize(x, 2);
        return {x[0] - x[1]};
    }
    virtual NdArrays backward(const NdArrays& x, const NdArrays& y,
                              const NdArrays& gy) {
        CheckVecSize(x, 2, y, 1, gy, 1);
        return {SumTo(gy[0], x[0].shape()), -SumTo(gy[0], x[1].shape())};
    }
};

class MulSub : public Function::Substance {
public:
    virtual ~MulSub() {}
    virtual NdArrays forward(const NdArrays& x) {
        CheckVecSize(x, 2);
        return {x[0] * x[1]};
    }
    virtual NdArrays backward(const NdArrays& x, const NdArrays& y,
                              const NdArrays& gy) {
        CheckVecSize(x, 2, y, 1, gy, 1);
        return {SumTo(gy[0] * x[1], x[0].shape()),
                SumTo(gy[0] * x[0], x[1].shape())};
    }
};

class DivSub : public Function::Substance {
public:
    virtual ~DivSub() {}
    virtual NdArrays forward(const NdArrays& x) {
        CheckVecSize(x, 2);
        return {x[0] / x[1]};
    }
    virtual NdArrays backward(const NdArrays& x, const NdArrays& y,
                              const NdArrays& gy) {
        CheckVecSize(x, 2, y, 1, gy, 1);
        const auto& gx0 = gy[0] / x[1];
        const auto& gx1 = -gx0 * x[0] / x[1];
        return {SumTo(gx0, x[0].shape()), SumTo(gx1, x[1].shape())};
    }
};

class ExpSub : public Function::Substance {
public:
    virtual ~ExpSub() {}
    virtual NdArrays forward(const NdArrays& x) {
        CheckVecSize(x, 1);
        return {Exp(x[0])};
    }
    virtual NdArrays backward(const NdArrays& x, const NdArrays& y,
                              const NdArrays& gy) {
        CheckVecSize(x, 1, y, 1, gy, 1);
        return {gy[0] * y[0]};
    }
};

// ------------------------------- Helper Class --------------------------------
// Helper to replace default substance with implemented one
template <typename S>
class FuncImplHelper : public Function {
public:
    FuncImplHelper() : Function(std::make_shared<S>()) {}
    virtual ~FuncImplHelper() {}
};

// ----------------------------- Function Wrapping -----------------------------
Variable Add(Variable lhs, Variable rhs) {
    return FuncImplHelper<AddSub>()({lhs, rhs})[0];
}

Variable Sub(Variable lhs, Variable rhs) {
    return FuncImplHelper<SubSub>()({lhs, rhs})[0];
}

Variable Mul(Variable lhs, Variable rhs) {
    return FuncImplHelper<MulSub>()({lhs, rhs})[0];
}

Variable Div(Variable lhs, Variable rhs) {
    return FuncImplHelper<DivSub>()({lhs, rhs})[0];
}

Variable Exp(Variable x) {
    return FuncImplHelper<ExpSub>()({x})[0];
}

}  // namespace F

#endif  // TINYDIFF_IMPLEMENTATION
// #############################################################################
// ############################# End of Definition ############################
// #############################################################################

}  // namespace tinydiff

#endif /* end of include guard */
