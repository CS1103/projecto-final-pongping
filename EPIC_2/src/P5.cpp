#include "P5.h"
#include <stack>
#include <iostream>
using namespace std;

bool is_equation_balanced(const std::string &source) {
    stack<char> st;
    for (char c : source) {
        switch (c) {
            case '(': case '{': case '[':
                st.push(c);
            break;
            case ')':
                if (st.empty() || st.top() != '(') return false;
            st.pop();
            break;
            case '}':
                if (st.empty() || st.top() != '{') return false;
            st.pop();
            break;
            case ']':
                if (st.empty() || st.top() != '[') return false;
            st.pop();
            break;
            default:
                    break;
        }
    }
    return st.empty();
}

void question_5() {
     string text;
     getline(std::cin, text);
     cout << std::boolalpha
              << is_equation_balanced(text);
}
