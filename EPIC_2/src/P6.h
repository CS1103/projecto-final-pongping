#ifndef POO2_UNIT6_ADAPTADORES_POO_V2021_2_P6_H
#define POO2_UNIT6_ADAPTADORES_POO_V2021_2_P6_H

#include <string>
#include <stack>
#include <stdexcept>


struct alumno_t {
    std::string nombre;
    std::string apellido;
    int nota;
    alumno_t() = default;
    alumno_t(std::string n, std::string a, int no)
      : nombre(std::move(n)), apellido(std::move(a)), nota(no) {}
    const std::string& get_nombre()  const { return nombre;  }
    const std::string& get_apellido()const { return apellido;}
    int               get_nota()    const { return nota;    }
};


template<typename T, typename Pred>
T find_in_stack(std::stack<T> s, Pred cmpr) {
    while (!s.empty()) {
        const T& elem = s.top();
        if (cmpr(elem)) {
            return elem;
        }
        s.pop();
    }
    throw std::runtime_error("No matching element found in stack");
}

void question_6();

#endif // POO2_UNIT6_ADAPTADORES_POO_V2021_2_P6_H
