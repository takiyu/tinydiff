#include "tinydiff.h"

int main(int argc, char const* argv[]) {
    using namespace tinydiff;

    Variable a(10.f);
    Variable b(20.f);

    Variable c, d, e;
    {
        auto a2 = F::exp(a);
        c = a2 + b;
        d = c * b;
        e = d + a;
    }

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
