#ifndef TINYDIFF_H_ONCE
#define TINYDIFF_H_ONCE

#include <cassert>
#include <cmath>
#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

namespace tinydiff {

class Variable;
using Variables = std::vector<Variable>;
class Function;

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
Variable operator+(Variable& os, Variable& x);
Variable operator*(Variable& os, Variable& x);

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

// ----------------------------- Utilities -------------------------------------
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

std::vector<float> CvtFromVariables(const Variables& src) {
    std::vector<float> ret;
    ret.reserve(src.size());
    for (auto&& s_elem : src) {
        ret.emplace_back(s_elem.data());
    }
    return ret;
}

Variables CvtToVariables(const std::vector<float>& src) {
    Variables ret;
    ret.reserve(src.size());
    for (auto&& s_elem : src) {
        ret.emplace_back(Variable(s_elem));
    }
    return ret;
}

std::vector<float> GetGrads(const Variables& src) {
    std::vector<float> ret;
    ret.reserve(src.size());
    for (auto&& s_elem : src) {
        ret.emplace_back(s_elem.grad());
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
    Substance(float v) : v(v) {}
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

Variable operator+(Variable& lhs, Variable& rhs) {
    return F::add(lhs, rhs);
}

Variable operator*(Variable& lhs, Variable& rhs) {
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
    virtual std::vector<float> forward(const std::vector<float>& x) {
        throw std::runtime_error("Invalid use of tinydiff::Function");
    }
    virtual std::vector<float> backward(const std::vector<float>& x,
                                        const std::vector<float>& y,
                                        const std::vector<float>& gy) {
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

}  // namespace tinydiff

#endif /* end of include guard */
