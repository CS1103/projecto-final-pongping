#include "P2.h"
#include <iostream>
using namespace std;
void question_2() {
    stack<int> stack_a, stack_b;
    int n, value;


    cin >> n;
    while (n-- > 0) {
        std::cin >> value;
        stack_a.push(value);
    }


    cin >> n;
    while (n-- > 0) {
        cin >> value;
        stack_b.push(value);
    }

    auto stack_result = join_stacks(stack_a, stack_b);

    while (!stack_result.empty()) {
        cout << stack_result.top() << " ";
        stack_result.pop();
    }
}
