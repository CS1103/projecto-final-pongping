#include "P6.h"
#include <iostream>
#include <limits>
using namespace std;

void question_6() {
     stack<alumno_t> s1;
    int n;
     cin >> n;
    cin.ignore( numeric_limits< streamsize>::max(), '\n');

   string nombre, apellido;
    int nota;
    while (n-- > 0) {
         getline(std::cin, nombre);
        getline(std::cin, apellido);
         cin >> nota;
         cin.ignore( numeric_limits< streamsize>::max(), '\n');
        s1.push(alumno_t{nombre, apellido, nota});
    }

   string name_tobe_found;
   getline( cin, name_tobe_found);

    try {

        alumno_t a1 = find_in_stack(
            s1,
            [=](const alumno_t& a) { return a.get_nombre() == name_tobe_found; }
        );
        cout
            << a1.get_nombre() << " "
            << a1.get_apellido() << " "
            << a1.get_nota() << "\n";
    }
    catch (const  runtime_error& e) {
        cerr << e.what() << "\n";
    }
}
