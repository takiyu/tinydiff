#include "Catch2/single_include/catch2/catch.hpp"

#include "../tinydiff.h"

#include <iomanip>

using namespace tinydiff;

static void CheckNdArray(const NdArray& m, const std::string& str,
                         int precision = -1) {
    std::stringstream ss;
    if (0 < precision) {
        ss << std::setprecision(4);
    }
    ss << m;
    CHECK(ss.str() == str);
}

static void CheckData(const Variable& v, const std::string& str,
                      int precision = -1) {
    CheckNdArray(v.data(), str, precision);
}

static void CheckGrad(const Variable& v, const std::string& str,
                      int precision = -1) {
    CheckNdArray(v.grad(), str, precision);
}

static void ResolveAmbiguous(NdArray& x) {
    for (auto&& v : x) {
        if (std::isnan(v)) {
            v = std::abs(v);
        }
        if (v == -0.f) {
            v = 0.f;
        }
    }
}

TEST_CASE("AutoGrad") {
    // -------------------------- Basic construction ---------------------------
    SECTION("Basic construction") {
        const Variable v0;
        const Variable v1 = {1.f, 2.f};
        const Variable v2 = {{3.f, 4.f}, {5.f, 6.f}};
        const Variable v3(NdArray::Zeros(2, 4));
        CHECK(v0.data().empty());
        CheckData(v1, "[1, 2]");
        CheckData(v2,
                  "[[3, 4],\n"
                  " [5, 6]]");
        CheckData(v3,
                  "[[0, 0, 0, 0],\n"
                  " [0, 0, 0, 0]]");
    }

    // ----------------------------- Basic methods -----------------------------
    SECTION("Basic methods") {
        const Variable v0;
        const Variable v1 = {1.f, 2.f};
        const Variable v2 = {3.f, 4.f};
        auto v3 = v1;
        auto v12 = v1 - v2;
        CHECK(v0.id() != v1.id());
        CHECK(v1.id() == v3.id());
        CHECK(v0.empty());
        CHECK(!v1.empty());
        CHECK(v0.size() == 0);
        CHECK(v1.size() == 2);
        CHECK(v0.shape() == Shape{0});
        CHECK(v1.shape() == Shape{2});
        CHECK(v0.ndim() == 1);
        CHECK(v1.ndim() == 1);
        v12.backward();
        // Nothing changed after backward
        CHECK(v0.id() != v1.id());
        CHECK(v1.id() == v3.id());
        CHECK(v0.empty());
        CHECK(!v1.empty());
        CHECK(v0.size() == 0);
        CHECK(v1.size() == 2);
        CHECK(v0.shape() == Shape{0});
        CHECK(v1.shape() == Shape{2});
        CHECK(v0.ndim() == 1);
        CHECK(v1.ndim() == 1);
    }

    // -------------------------------- Single ---------------------------------
    SECTION("Single") {
        Variable v1 = {1.f, 2.f};
        Variable v2 = {{3.f, 4.f}, {5.f, 6.f}};
        // Positive
        auto v_pos = +v2 + +v1;
        v_pos.backward();
        CheckGrad(v1, "[2, 2]");
        CheckGrad(v2,
                  "[[1, 1],\n"
                  " [1, 1]]");
        CheckData(v_pos,
                  "[[4, 6],\n"
                  " [6, 8]]");
        // Negative
        auto v_neg = -(-v1) + -(-(-v2));
        v_neg.backward();
        CheckGrad(v1, "[2, 2]");
        CheckGrad(v2,
                  "[[-1, -1],\n"
                  " [-1, -1]]");
        CheckData(v_neg,
                  "[[-2, -2],\n"
                  " [-4, -4]]");
    }

    // --------------- Arithmetic functions (Variable, Variable) ---------------
    SECTION("Arithmetic (add, broadcast)") {
        Variable v1 = {1.f, 2.f};
        Variable v2 = {{3.f, 4.f}, {5.f, 6.f}};
        // Backward from left to right
        auto v12 = v1 + v2;
        v12.backward();
        CheckGrad(v1, "[2, 2]");
        CheckGrad(v2,
                  "[[1, 1],\n"
                  " [1, 1]]");
        CheckData(v12,
                  "[[4, 6],\n"
                  " [6, 8]]");
        // Backward from right to left
        auto v21 = v2 + v1;
        v21.backward();
        CheckGrad(v1, "[2, 2]");
        CheckGrad(v2,
                  "[[1, 1],\n"
                  " [1, 1]]");
        CheckData(v21,
                  "[[4, 6],\n"
                  " [6, 8]]");
        // Data check
        CheckData(v1, "[1, 2]");
        CheckData(v2,
                  "[[3, 4],\n"
                  " [5, 6]]");
    }

    SECTION("Arithmetic (sub, broadcast)") {
        Variable v1 = {1.f, 2.f};
        Variable v2 = {{3.f, 4.f}, {5.f, 6.f}};
        // Backward from left to right
        auto v12 = v1 - v2;
        v12.backward();
        CheckGrad(v1, "[2, 2]");
        CheckGrad(v2,
                  "[[-1, -1],\n"
                  " [-1, -1]]");
        CheckData(v12,
                  "[[-2, -2],\n"
                  " [-4, -4]]");
        // Backward from right to left
        auto v21 = v2 - v1;
        v21.backward();
        CheckGrad(v1, "[-2, -2]");
        CheckGrad(v2,
                  "[[1, 1],\n"
                  " [1, 1]]");
        CheckData(v21,
                  "[[2, 2],\n"
                  " [4, 4]]");
        // Data check
        CheckData(v1, "[1, 2]");
        CheckData(v2,
                  "[[3, 4],\n"
                  " [5, 6]]");
    }

    SECTION("Arithmetic (mul, broadcast)") {
        Variable v1 = {1.f, 2.f};
        Variable v2 = {{3.f, 4.f}, {5.f, 6.f}};
        // Backward from left to right
        auto v12 = v1 * v2;
        v12.backward();
        CheckGrad(v1, "[8, 10]");
        CheckGrad(v2,
                  "[[1, 2],\n"
                  " [1, 2]]");
        CheckData(v12,
                  "[[3, 8],\n"
                  " [5, 12]]");
        // Backward from right to left
        auto v21 = v2 * v1;
        v21.backward();
        CheckGrad(v1, "[8, 10]");
        CheckGrad(v2,
                  "[[1, 2],\n"
                  " [1, 2]]");
        CheckData(v21,
                  "[[3, 8],\n"
                  " [5, 12]]");
        // Data check
        CheckData(v1, "[1, 2]");
        CheckData(v2,
                  "[[3, 4],\n"
                  " [5, 6]]");
    }

    SECTION("Arithmetic (div, broadcast)") {
        Variable v1 = {1.f, 2.f};
        Variable v2 = {{3.f, 4.f}, {5.f, 6.f}};
        // Backward from left to right
        auto v12 = v1 / v2;
        v12.backward();
        CheckGrad(v1, "[0.533333, 0.416667]");
        CheckGrad(v2,
                  "[[-0.111111, -0.125],\n"
                  " [-0.04, -0.0555556]]");
        CheckData(v12,
                  "[[0.333333, 0.5],\n"
                  " [0.2, 0.333333]]");
        // Backward from right to left
        auto v21 = v2 / v1;
        v21.backward();
        CheckGrad(v1, "[-8, -2.5]");
        CheckGrad(v2,
                  "[[1, 0.5],\n"
                  " [1, 0.5]]");
        CheckData(v21,
                  "[[3, 2],\n"
                  " [5, 3]]");
        // Data check
        CheckData(v1, "[1, 2]");
        CheckData(v2,
                  "[[3, 4],\n"
                  " [5, 6]]");
    }

    // ---------------- Arithmetic functions (Variable, float) -----------------
    SECTION("Arithmetic (Variable, float)") {
        Variable v1 = {1.f, 2.f};
        // Add
        auto v_add = v1 + 2.f;
        v_add.backward();
        CheckGrad(v1, "[1, 1]");
        CheckData(v_add, "[3, 4]");
        // Sub
        auto v_sub = v1 - 2.f;
        v_sub.backward();
        CheckGrad(v1, "[1, 1]");
        CheckData(v_sub, "[-1, 0]");
        // Mul
        auto v_mul = v1 * 2.f;
        v_mul.backward();
        CheckGrad(v1, "[2, 2]");
        CheckData(v_mul, "[2, 4]");
        // Div
        auto v_div = v1 / 2.f;
        v_div.backward();
        CheckGrad(v1, "[0.5, 0.5]");
        CheckData(v_div, "[0.5, 1]");
    }

    // ---------------- Arithmetic functions (float, Variable) -----------------
    SECTION("Arithmetic (float, Variable)") {
        Variable v1 = {1.f, 2.f};
        // Add
        auto v_add = 2.f + v1;
        v_add.backward();
        CheckGrad(v1, "[1, 1]");
        CheckData(v_add, "[3, 4]");
        // Sub
        auto v_sub = 2.f - v1;
        v_sub.backward();
        CheckGrad(v1, "[-1, -1]");
        CheckData(v_sub, "[1, 0]");
        // Mul
        auto v_mul = 2.f * v1;
        v_mul.backward();
        CheckGrad(v1, "[2, 2]");
        CheckData(v_mul, "[2, 4]");
        // Div
        auto v_div = 2.f / v1;
        v_div.backward();
        CheckGrad(v1, "[-2, -0.5]");
        CheckData(v_div, "[2, 1]");
    }

    // -------------------------- Compound Assignment --------------------------
    SECTION("Compound assignment") {
        Variable v1 = {1.f, 2.f};
        // Add
        auto v_add = v1;
        v_add += 2.f;
        v_add.backward();
        CheckGrad(v1, "[1, 1]");
        CheckData(v_add, "[3, 4]");
        // Sub
        auto v_sub = v1;
        v_sub -= 2.f;
        v_sub.backward();
        CheckGrad(v1, "[1, 1]");
        CheckData(v_sub, "[-1, 0]");
        // Mul
        auto v_mul = v1;
        v_mul *= 2.f;
        v_mul.backward();
        CheckGrad(v1, "[2, 2]");
        CheckData(v_mul, "[2, 4]");
        // Div
        auto v_div = v1;
        v_div /= 2.f;
        v_div.backward();
        CheckGrad(v1, "[0.5, 0.5]");
        CheckData(v_div, "[0.5, 1]");
    }

    // -------------------------- Basic math operators -------------------------
    SECTION("Basic math operators") {
        Variable v1 = {-1.f, 0.f, 1.f, 2.f};
        // Abs
        auto v_abs = F::Abs(v1);
        v_abs.backward();
        CheckGrad(v1, "[-1, 0, 1, 1]");
        CheckData(v_abs, "[1, 0, 1, 2]");
        // Sign, Ceil, Floor cannot backward. Tested in the 2 next case.
        // Clip
        auto v_clip = F::Clip(v1, -0.5f, 1.9f) * 2.1f;
        v_clip.backward();
        CheckGrad(v1, "[0, 2.1, 2.1, 0]");
        CheckData(v_clip, "[-1.05, 0, 2.1, 3.99]");
        // Sqrt
        auto v_sqrt = F::Sqrt(v1);
        v_sqrt.backward();
        ResolveAmbiguous(v1.grad());  // -nan -> nan
        ResolveAmbiguous(v_sqrt.data());  // -nan -> nan
        CheckGrad(v1, "[nan, inf, 0.5, 0.353553]");
        CheckData(v_sqrt, "[nan, 0, 1, 1.41421]");
        // Exp
        auto v_exp = F::Exp(v1);
        v_exp.backward();
        CheckGrad(v1, "[0.367879, 1, 2.71828, 7.38906]");
        CheckData(v_exp, "[0.367879, 1, 2.71828, 7.38906]");
        // Log
        auto v_log = F::Log(v1);
        v_log.backward();
        ResolveAmbiguous(v_log.data());  // -nan -> nan
        CheckGrad(v1, "[-1, inf, 1, 0.5]");
        CheckData(v_log, "[nan, -inf, 0, 0.693147]");
        // Square
        auto v_square = F::Square(v1);
        v_square.backward();
        CheckGrad(v1, "[-2, 0, 2, 4]");
        CheckData(v_square, "[1, 0, 1, 4]");
        // Power is tested in the next case.
    }

    SECTION("Basic math operators (power)") {
        // Power
        Variable v1 = {1.5f, 2.f};
        Variable v2 = {{3.f, 4.f}, {5.f, 6.f}};
        // Backward from left to right
        auto v12 = F::Power(v1, v2);
        v12.backward();
        CheckGrad(v1, "[32.0625, 224]");
        CheckGrad(v2,
                  "[[1.36844, 11.0904],\n"
                  " [3.079, 44.3614]]");
        CheckData(v12,
                  "[[3.375, 16],\n"
                  " [7.59375, 64]]");
        // Backward from right to left
        auto v21 = F::Power(v2, v1);
        v21.backward();
        CheckGrad(v1, "[23.7026, 86.6841]");
        CheckGrad(v2,
                  "[[2.59808, 8],\n"
                  " [3.3541, 12]]");
        CheckData(v21,
                  "[[5.19615, 16],\n"
                  " [11.1803, 36]]");
        // Float
        auto v_float = F::Power(v1, 2.1f);
        v_float.backward();
        CheckGrad(v1, "[3.28035, 4.50145]");
        CheckData(v_float, "[2.3431, 4.28709]");
        // From float
        auto v_from_float = F::Power(2.1f, v1);
        v_from_float.backward();
        CheckGrad(v1, "[2.25786, 3.27194]");
        CheckData(v_from_float, "[3.04319, 4.41]");
    }

    SECTION("Basic math operators (unchained functions)") {
        Variable v1 = {-1.f, 0.f, 1.f, 2.f};
        // Sign
        v1.clearGrad();
        auto v_sign = F::Sign(v1);  // Unchain
        v_sign.backward();
        CheckGrad(v1, "[]");  // No grad
        CheckData(v_sign, "[-1, 0, 1, 1]");
        // Ceil
        v1.clearGrad();
        auto v_ceil = F::Ceil(v1 + 0.5);  // Unchain
        v_ceil.backward();
        CheckGrad(v1, "[]");  // No grad
        CheckData(v_ceil, "[-0, 1, 2, 3]");
        // Floor
        v1.clearGrad();
        auto v_floor = F::Floor(v1 + 0.5);  // Unchain
        v_floor.backward();
        CheckGrad(v1, "[]");  // No grad
        CheckData(v_floor, "[-1, 0, 1, 2]");
    }

    // ------------------------ Trigonometric functions ------------------------
    SECTION("Trigonometric functions") {
        Variable v1 = {-1.f, 0.f, 1.f, 2.f};
        // Sin
        auto v_sin = F::Sin(v1);
        v_sin.backward();
        CheckGrad(v1, "[0.540302, 1, 0.540302, -0.416147]");
        CheckData(v_sin, "[-0.841471, 0, 0.841471, 0.909297]");
        // Cos
        auto v_cos = F::Cos(v1);
        v_cos.backward();
        ResolveAmbiguous(v1.grad());  // -0 -> 0
        CheckGrad(v1, "[0.841471, 0, -0.841471, -0.909297]");
        CheckData(v_cos, "[0.540302, 1, 0.540302, -0.416147]");
        // Tan
        auto v_tan = F::Tan(v1);
        v_tan.backward();
        CheckGrad(v1, "[3.42552, 1, 3.42552, 5.7744]");
        CheckData(v_tan, "[-1.55741, 0, 1.55741, -2.18504]");
    }

    // ------------------------ Inverse trigonometric functions ------------------------
    SECTION("Inverse trigonometric functions") {
        Variable v1 = {-1.f, 0.f, 0.5, 1.f, 2.f};
        // ArcSin
        auto v_arcsin = F::ArcSin(v1);
        v_arcsin.backward();
        ResolveAmbiguous(v1.grad());  // -nan -> nan
        CheckGrad(v1, "[inf, 1, 1.1547, inf, nan]");
        CheckData(v_arcsin, "[-1.5708, 0, 0.523599, 1.5708, nan]");
        // ArcCos
        auto v_arccos = F::ArcCos(v1);
        v_arccos.backward();
        ResolveAmbiguous(v1.grad());  // -nan -> nan
        CheckGrad(v1, "[-inf, -1, -1.1547, -inf, nan]");
        CheckData(v_arccos, "[3.14159, 1.5708, 1.0472, 0, nan]");
        // ArcTan
        auto v_arctan = F::ArcTan(v1);
        v_arctan.backward();
        CheckGrad(v1, "[0.5, 1, 0.8, 0.5, 0.2]");
        CheckData(v_arctan, "[-0.785398, 0, 0.463648, 0.785398, 1.10715]");
        // ArcTan2 is tested in the next case.
    }

    SECTION("Inverse trigonometric functions (atan2)") {
        // ArcTan2
        Variable v1 = {1.5f, 2.f};
        Variable v2 = {{-1.f, 0.f}, {1.f, 2.f}};
        // Backward from left to right
        auto v12 = F::ArcTan2(v1, v2);
        v12.backward();
        CheckGrad(v1, "[0, 0.25]");
        CheckGrad(v2,
                  "[[-0.461538, -0.5],\n"
                  " [-0.461538, -0.25]]");
        CheckData(v12,
                  "[[2.1588, 1.5708],\n"
                  " [0.982794, 0.785398]]");
        // Backward from right to left
        auto v21 = F::ArcTan2(v2, v1);
        v21.backward();
        CheckGrad(v1, "[0, -0.25]");
        CheckGrad(v2,
                  "[[0.461538, 0.5],\n"
                  " [0.461538, 0.25]]");
        CheckData(v21,
                  "[[-0.588003, 0],\n"
                  " [0.588003, 0.785398]]");
        // Float
        auto v_float = F::ArcTan2(v1, 2.1f);
        v_float.backward();
        CheckGrad(v1, "[0.315315, 0.249703]");
        CheckData(v_float, "[0.62025, 0.761013]");
        // From float
        auto v_from_float = F::ArcTan2(2.1f, v1);
        v_from_float.backward();
        CheckGrad(v1, "[-0.315315, -0.249703]");
        CheckData(v_from_float, "[0.950547, 0.809784]");
    }

    // -------------------- Arithmetic functions (complex) ---------------------
    SECTION("Arithmetic (complex chain)") {
        Variable v1 = {1.f, 2.f};
        Variable v2 = {{3.f, 4.f}, {5.f, 6.f}};

        auto v3 = v1 + +v2;
        auto v4 = v3 * v2;
        auto v5 = v4 / v1;
        auto v6 = v5 - -(-v3);
        v6.backward();

        CheckGrad(v1, "[-36, -15]");
        CheckGrad(v2,
                  "[[6, 4],\n"
                  " [10, 6]]");
    }

    SECTION("Arithmetic (complex chain and shape)") {
        Variable v1(NdArray({1.f, 2.f}).reshape(2, 1));
        Variable v2(NdArray({{3.f, 4.f}, {5.f, 6.f}}).reshape(1, 1, 2, 1, 2));

        auto v3 = v1 * v2;
        auto v4 = v3 - v2;
        auto v5 = v4 + v1;
        auto v6 = v5 / v3;
        v6.backward();

        CheckGrad(v1,
                  "[[4],\n"
                  " [1]]");
        CheckGrad(v2,
                  "[[[[[-0.222222, -0.125]],\n"
                  "   [[-0.08, -0.0555556]]]]]");
    }

    // --------------------------- Backward options ----------------------------
    SECTION("Backward (no clear_grads)") {
        Variable v1 = {1.f, 2.f};
        Variable v2 = {{3.f, 4.f}, {5.f, 6.f}};

        auto v3_a = v1 + v2;
        auto v4_a = v3_a * v2;
        auto v5_a = v3_a / v4_a;
        v5_a.backward(false);  // No clear
        CheckGrad(v1, "[0, 0]");
        CheckGrad(v2,
                  "[[-0.111111, -0.0625],\n"
                  " [-0.04, -0.0277778]]");

        auto v3_b = v1 + v2;
        auto v4_b = v3_b * v2;
        auto v5_b = v3_b / v4_b;
        v5_b.backward(false);  // No clear
        CheckGrad(v1, "[0, 0]");
        CheckGrad(v2,
                  "[[-0.222222, -0.125],\n"
                  " [-0.08, -0.0555556]]");
    }
}
