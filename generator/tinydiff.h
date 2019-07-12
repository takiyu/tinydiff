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
    void cleargrads();
    void backward();

    void setCreator(Function f);
    Function getCreator() const;
    void addGrad(const NdArray& grad);

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
    Function(Function&&) noexcept;
    Function& operator=(const Function&);
    Function& operator=(Function&&);
    virtual ~Function();

    void clear();

    // Build computational graph with forwarding
    Variables operator()(const Variables& x);

    std::vector<NdArray> forward(const std::vector<NdArray>& x);
    std::vector<NdArray> backward(const std::vector<NdArray>& x,
                                  const std::vector<NdArray>& y,
                                  const std::vector<NdArray>& gy);

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

static std::vector<NdArray> CvtFromVariables(const Variables& src) {
    std::vector<NdArray> ret;
    ret.reserve(src.size());
    for (auto&& s_elem : src) {
        ret.emplace_back(s_elem.data());
    }
    return ret;
}

static Variables CvtToVariables(const std::vector<NdArray>& src) {
    Variables ret;
    ret.reserve(src.size());
    for (auto&& s_elem : src) {
        ret.emplace_back(Variable(s_elem));
    }
    return ret;
}

static std::vector<NdArray> GetGrads(const Variables& src) {
    std::vector<NdArray> ret;
    ret.reserve(src.size());
    for (auto&& s_elem : src) {
        ret.emplace_back(s_elem.grad());
    }
    return ret;
}

static void ClearGrads(Variables& src) {
    for (auto&& s_elem : src) {
        s_elem.cleargrads();
    }
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
    NdArray grad = {0.f};
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

void Variable::cleargrads() {
    m_sub->grad = NdArray({0.f});
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
        // Remove used gradient
        ClearGrads(outputs);
    }
}

void Variable::setCreator(Function f) {
    m_sub->creator = f;
    f.setRank(f.getRank() + 1);  // Increase rank
}

Function Variable::getCreator() const {
    return m_sub->creator;
}

void Variable::addGrad(const NdArray& grad) {
    m_sub->grad = m_sub->grad + grad;
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
    virtual std::vector<NdArray> forward(const std::vector<NdArray>& x) {
        (void)x;
        throw std::runtime_error("Invalid use of tinydiff::Function");
    }
    virtual std::vector<NdArray> backward(const std::vector<NdArray>& x,
                                          const std::vector<NdArray>& y,
                                          const std::vector<NdArray>& gy) {
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

std::vector<NdArray> Function::forward(const std::vector<NdArray>& x) {
    return m_sub->forward(x);
}

std::vector<NdArray> Function::backward(const std::vector<NdArray>& x,
                                        const std::vector<NdArray>& y,
                                        const std::vector<NdArray>& gy) {
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
    virtual std::vector<NdArray> forward(const std::vector<NdArray>& x) {
        CheckSize(x, 2);
        return {x[0] + x[1]};
    }
    virtual std::vector<NdArray> backward(const std::vector<NdArray>& x,
                                          const std::vector<NdArray>& y,
                                          const std::vector<NdArray>& gy) {
        CheckSize(x, 2);
        CheckSize(y, 1);
        CheckSize(gy, 1);
        return {gy[0], gy[0]};
    }
};

class MulSub : public Function::Substance {
public:
    virtual ~MulSub() {}
    virtual std::vector<NdArray> forward(const std::vector<NdArray>& x) {
        CheckSize(x, 2);
        return {x[0] * x[1]};
    }
    virtual std::vector<NdArray> backward(const std::vector<NdArray>& x,
                                          const std::vector<NdArray>& y,
                                          const std::vector<NdArray>& gy) {
        CheckSize(x, 2);
        CheckSize(y, 1);
        CheckSize(gy, 1);
        return {gy[0] * x[1], gy[0] * x[0]};
    }
};

class ExpSub : public Function::Substance {
public:
    virtual ~ExpSub() {}
    virtual std::vector<NdArray> forward(const std::vector<NdArray>& x) {
        CheckSize(x, 1);
        return {Exp(x[0])};
    }
    virtual std::vector<NdArray> backward(const std::vector<NdArray>& x,
                                          const std::vector<NdArray>& y,
                                          const std::vector<NdArray>& gy) {
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
// #############################################################################
// ############################# End of Definition ############################
// #############################################################################

}  // namespace tinydiff

#endif /* end of include guard */
