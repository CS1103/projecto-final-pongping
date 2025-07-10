#ifndef POO2_UNIT6_ADAPTADORES_POO_V2021_2_P2_H
#define POO2_UNIT6_ADAPTADORES_POO_V2021_2_P2_H

#include <stack>

template<typename T>
std::stack<T> join_stacks(std::stack<T> a, std::stack<T> b) {
    std::stack<T> aux, result;


    while (!a.empty()) {
        aux.push(a.top());
        a.pop();
    }

    while (!aux.empty()) {
        result.push(aux.top());
        aux.pop();
    }

    while (!b.empty()) {
        aux.push(b.top());
        b.pop();
    }
    while (!aux.empty()) {
        result.push(aux.top());
        aux.pop();
    }

    return result;
}

void question_2();

#endif // POO2_UNIT6_ADAPTADORES_POO_V2021_2_P2_H
