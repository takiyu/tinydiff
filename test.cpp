
#include "tinydiff.h"

int main(int argc, char const* argv[]) {
    using namespace tinydiff;

    Variable a(10.f);
    Variable b(20.f);

    F::Add add1;
    F::Add add2;
    F::Mul mul;

    auto c = add1({a, b})[0];
    auto d = mul({c, b})[0];
    auto e = add2({d, a})[0];

    std::cout << a.data() << std::endl;
    std::cout << b.data() << std::endl;
    std::cout << c.data() << std::endl;
    std::cout << d.data() << std::endl;
    std::cout << e.data() << std::endl;

    d.backward();

    std::cout << a.grad() << std::endl;
    std::cout << b.grad() << std::endl;
    std::cout << c.grad() << std::endl;
    std::cout << d.grad() << std::endl;
    std::cout << e.grad() << std::endl;


    return 0;
}
