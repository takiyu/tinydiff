#ifndef TINYDIFF_H_ONCE
#define TINYDIFF_H_ONCE

#include <cassert>
#include <cmath>
#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

namespace tinydiff {

class NdArray;
using Shape = std::vector<int>;
using Index = std::vector<int>;
class Variable;
using Variables = std::vector<Variable>;
class Function;

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

    size_t size() const;
    const Shape& shape() const;
    float* data();
    const float* data() const;

    NdArray reshape(const Shape& shape) const;
    template <typename... S>
    NdArray reshape(S... shape) const;

    float& operator[](int i);
    const float& operator[](int i) const;
    float& operator[](const Index& index);
    const float& operator[](const Index& index) const;

    template <typename... I>
    float& operator()(I... index);
    template <typename... I>
    const float& operator()(I... index) const;

    class Substance;

private:
    std::shared_ptr<Substance> m_sub;
    NdArray(std::shared_ptr<Substance> sub);
};

// --------------------------------- Operators ---------------------------------
std::ostream& operator<<(std::ostream& os, const NdArray& x);

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
#ifdef TINYDIFF_IMPLEMENTATION

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

// -------------------------------- Constructors -------------------------------
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
    // Fill after initialize
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
    const size_t n = static_cast<size_t>((stop - start) / step);
    NdArray ret({static_cast<int>(n)});
    float* data = ret.data();
    for (size_t i = 0; i < n; i++) {
        data[i] = start + step * static_cast<float>(i);
    }
    return ret;
}

// ------------------------------- Basic Methods -------------------------------
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
    ret.m_sub->size = m_sub->size;  // Same size
    ret.m_sub->shape = std::move(new_shape);  // New shape
    ret.m_sub->v = m_sub->v;        // Shared elements
    return ret;
}

template <typename... S>
NdArray NdArray::reshape(S... shape) const {
    // Pass to `reshape(Shape)`
    return reshape({shape...});
}

// ------------------------------- Index Methods -------------------------------
float& NdArray::operator[](int i) {
    // Use the same implementation of constant method.
    return const_cast<float&>(static_cast<const NdArray&>(*this)[i]);
}

const float& NdArray::operator[](int i) const {
    const size_t idx =
            (0 <= i) ? static_cast<size_t>(i) :
                       m_sub->size + static_cast<size_t>(i);  // Negative index
    // Direct access
    return *(m_sub->v.get() + idx);
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
        if (0 <= index[d]) {
            i += index[d];  // Positive index
        } else {
            i += shape[d] + index[d];  // Negative index
        }
    }
    return (*this)[i];
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
template NdArray NdArray::Empty(int, int, int, int, int, int, int, int, int, int);
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
template NdArray NdArray::Zeros(int, int, int, int, int, int, int, int, int, int);
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
template NdArray NdArray::Ones(int, int, int, int, int, int, int, int, int, int);
// For `NdArray reshape(S... shape)`
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
