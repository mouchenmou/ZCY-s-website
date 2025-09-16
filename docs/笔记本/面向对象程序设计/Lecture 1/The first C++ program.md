## 先来浅浅看个最简单的东西

```c
#include<iostream>
using namespace std;
int main{
    cout << "Today I ma " << "19" << " years old " << "now" << endl;
    return 0;
}
```

---

## cin 和 cout
### cin是输入，cout是输出：
```c
#include<iostream>
using namespace std;
int main(){
    int number;
    cout << "Enter a decimal number : ";
    cin >> number;
    cout << "The number you entered is " << number << "." << endl;
    return 0;
}
```

