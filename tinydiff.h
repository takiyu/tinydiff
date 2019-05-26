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

class Function;
class Variable;
using Variables = std::vector<Variable>;

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

    // Inherited by functional classes
    std::vector<float> forward(const std::vector<float>& x);
    std::vector<float> backward(const std::vector<float>& x,
                                const std::vector<float>& y,
                                const std::vector<float>& gy);

    Variables getInputs() const;
    Variables getOutputs() const;

    void setRank(size_t rank);
    size_t getRank() const;

protected:
    class Impl;
    std::shared_ptr<Impl> m_impl;
    Function(std::shared_ptr<Impl> impl);
};

// --------------------------- Implemented Functions ---------------------------
namespace F {

class Add;

}  // namespace F

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

private:
    class Impl;
    std::shared_ptr<Impl> m_impl;
    Variable(std::shared_ptr<Impl> impl);
};

// --------------------------------- Operators ---------------------------------
std::ostream& operator<<(std::ostream& os, Variable& x);

// *****************************************************************************
// *****************************************************************************
// ************************** Begin of Implementation **************************
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
    return std::move(last);
}

std::vector<float> CvtFromVariables(const Variables& src) {
    std::vector<float> ret;
    ret.reserve(src.size());
    for (auto&& s_elem : src) {
        ret.emplace_back(s_elem.data());
    }
    return std::move(ret);
}

Variables CvtToVariables(const std::vector<float>& src) {
    Variables ret;
    ret.reserve(src.size());
    for (auto&& s_elem : src) {
        ret.emplace_back(Variable(s_elem));
    }
    return std::move(ret);
}

std::vector<float> GetGrads(const Variables& src) {
    std::vector<float> ret;
    ret.reserve(src.size());
    for (auto&& s_elem : src) {
        ret.emplace_back(s_elem.grad());
    }
    return std::move(ret);
}

// =============================================================================
// =============================== Function Impl ===============================
// =============================================================================
class Function::Impl : public std::enable_shared_from_this<Function::Impl> {
public:
    Impl() {}
    Impl(const Impl& lhs) {}
    Impl(Impl&&) = delete;
    Impl& operator=(const Impl& lhs) {
        return *this;
    }
    Impl& operator=(Impl&&) = delete;
    virtual ~Impl() {}

    void clear() {
        m_rank = 0;
        m_inputs = Variables();
        m_outputs = Variables();
    }

    Variables operator()(const Variables& x) {
        // Forward (with variable conversion)
        auto&& x_data = CvtFromVariables(x);
        auto&& y_data = forward(std::move(x_data));    // with free
        auto&& y = CvtToVariables(std::move(y_data));  // with free
        // Retain input/output variables
        m_inputs = x;
        m_outputs = y;
        // Set rank of this function
        m_rank = 0;
        for (auto&& x_elem : x) {
            size_t rank = x_elem.getCreator().getRank();
            m_rank = std::max(m_rank, rank);
        }

        // Build chain
        for (auto&& y_elem : y) {
            y_elem.setCreator(Function(shared_from_this()));
        }

        return y;
    }

    virtual std::vector<float> forward(const std::vector<float>& x) {
        throw std::runtime_error("Invalid use of tinydiff::Function");
    }

    virtual std::vector<float> backward(const std::vector<float>& x,
                                        const std::vector<float>& y,
                                        const std::vector<float>& gy) {
        throw std::runtime_error("Invalid use of tinydiff::Function");
    }

    Variables getInputs() const {
        return m_inputs;
    }

    Variables getOutputs() const {
        return m_outputs;
    }

    void setRank(size_t rank) {
        m_rank = rank;
    }

    size_t getRank() const {
        return m_rank;
    }

private:
    size_t m_rank = 0;
    Variables m_inputs;
    Variables m_outputs;
};

// =============================================================================
// =============================== Variable Impl ===============================
// =============================================================================
class Variable::Impl {
public:
    Impl() {}
    Impl(float v) : m_v(v) {}
    Impl(const Impl& lhs) {}
    Impl(Impl&&) = delete;
    Impl& operator=(const Impl& lhs) {
        return *this;
    }
    Impl& operator=(Impl&&) = delete;
    ~Impl() {}

    float data() const {
        return m_v;
    }

    float grad() const {
        return m_grad;
    }

    void backward() {
        // Set the last gradients 'one'
        for (auto&& output: m_creator.getOutputs()) {
            output.addGrad(1.f);
        }

        // Ordered storage to resolve
        std::map<size_t, Function> cand_funcs;
        // Register the last node
        cand_funcs[m_creator.getRank()] = m_creator;
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
            auto&& in_grads =
                    last_func.backward(inputs_data, outputs_data, out_grads);
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

    void setCreator(Function f) {
        m_creator = f;
        f.setRank(f.getRank() + 1);  // Increase rank
    }

    Function getCreator() const {
        return m_creator;
    }

    void addGrad(float grad) {
        m_grad += grad;
    }

private:
    float m_v;
    float m_grad = 0.f;
    Function m_creator;
};

// --------------------------------- Operators ---------------------------------

std::ostream& operator<<(std::ostream& os, Variable& x) {
    return os << x.data();
}

// =============================================================================
// =========================== Function Impl Pattern ===========================
// =============================================================================
Function::Function() : m_impl(std::make_shared<Impl>()) {}

Function::Function(std::shared_ptr<Impl> impl) : m_impl(impl) {}

Function::Function(const Function& lhs) = default;  // shallow copy

Function::Function(Function&&) = default;

Function& Function::operator=(const Function& lhs) = default;  // shallow copy

Function& Function::operator=(Function&&) = default;

Function::~Function() = default;

void Function::clear() {
    m_impl->clear();
}

Variables Function::operator()(const Variables& x) {
    return (*m_impl)(x);
}

std::vector<float> Function::forward(const std::vector<float>& x) {
    return m_impl->forward(x);
}

std::vector<float> Function::backward(const std::vector<float>& x,
                                      const std::vector<float>& y,
                                      const std::vector<float>& gy) {
    return m_impl->backward(x, y, gy);
}

Variables Function::getInputs() const {
    return m_impl->getInputs();
}

Variables Function::getOutputs() const {
    return m_impl->getOutputs();
}

void Function::setRank(size_t rank) {
    m_impl->setRank(rank);
}

size_t Function::getRank() const {
    return m_impl->getRank();
}

// =============================================================================
// =========================== Variable Impl Pattern ===========================
// =============================================================================
Variable::Variable() : m_impl(std::make_shared<Impl>()) {}

Variable::Variable(std::shared_ptr<Impl> impl) : m_impl(impl) {}

Variable::Variable(float v) : m_impl(std::make_shared<Impl>(v)) {}

Variable::Variable(const Variable& lhs) = default;  // shallow copy

Variable::Variable(Variable&&) = default;  // move

Variable& Variable::operator=(const Variable& lhs) = default;  // shallow copy

Variable& Variable::operator=(Variable&&) = default;

Variable::~Variable() = default;

float Variable::data() const {
    return m_impl->data();
}

float Variable::grad() const {
    return m_impl->grad();
}

void Variable::backward() {
    return m_impl->backward();
}

void Variable::setCreator(Function f) {
    m_impl->setCreator(f);
}

Function Variable::getCreator() const {
    return m_impl->getCreator();
}

void Variable::addGrad(float grad) {
    m_impl->addGrad(grad);
}

// =============================================================================
// =========================== Implemented Functions ===========================
// =============================================================================

namespace F {

class Add : public Function {
public:
    class Impl : public Function::Impl {
    public:
        virtual ~Impl() {}
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

    Add() {
        m_impl = std::make_shared<Impl>();
    }
};

class Mul : public Function {
public:
    class Impl : public Function::Impl {
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

    Mul() {
        m_impl = std::make_shared<Impl>();
    }
};
//
// class Exp {
// public:
//     virtual Variables forward(const Variables& x) {
//         CheckSize(x, 1);
//         m_y = std::exp(x[0].data());
//         return {m_y};
//     }
//     virtual Variables backward(const Variables& x, const Variables& gy) {
//         CheckSize(x, 1);
//         CheckSize(gy, 1);
//         return {gy[0].data() * m_y};
//     }
//
// private:
//     float m_y;
// };

}  // namespace F

}  // namespace tinydiff

#endif /* end of include guard */
