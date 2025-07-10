#include "P4.h"
#include <queue>
#include <iostream>

using namespace std;

 string process_text_by_queue(const  string &source) {
     queue<char> q;
     string output;
    for (char c : source) {
        if (c == '*') {
            if (!q.empty()) {
                output.push_back(q.front());
                q.pop();
            }
        }
        else {
            q.push(c);
        }
    }
    return output;
}

void question_4() {
     string text;
     getline( cin, text);
     cout << process_text_by_queue(text);
}
