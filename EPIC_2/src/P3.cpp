#include "P3.h"
#include <stack>
#include <iostream>
using namespace std;

 string process_text_by_stack(const std::string &source) {
     stack<char> st;
     string output;
    for (char c : source) {
        if (c == '*') {
            if (!st.empty()) {
                output.push_back(st.top());
                st.pop();
            }
        }
        else {
            st.push(c);
        }
    }
    return output;
}

void question_3() {
     string text;
     getline(std::cin, text);
     cout << process_text_by_stack(text);
}
