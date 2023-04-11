# 编程练习

### 习题2.1

**编写一个C++程序，它显示您的姓名和地址。**

解答：

```C++
#include <iostream>
using namespace std;

int main()
{
    cout << "name: Richard Bian" << endl;
    cout << "address: Zhuji Shaoxing Zhejiang Province";
}
```

执行结果：

```
name: Richard Bian
address: Zhuji Shaoxing Zhejiang Province
```

### 习题2.2

**编写一个C程序，它要求用户输入一个以long为单位的距离，然后将它转换为码(一long等于220码)。**

解答：

```c++
#include <iostream>
using namespace std;

int main()
{
    double distance_long;
    cout << "Please input the distance (in long): ";
    cin >> distance_long;
    double distance_yard = 220 * distance_long;
    cout << "The distance " << distance_long << " long is " << distance_yard << " yard.";
}
```

执行结果：

```
Please input the distance (in long): 10
The distance 10 long is 2200 yard.

```

### 习题2.3

**编写一个C++程序，它使用3个用户定义的函数（包括`main()`），并生成下面的输出:**

```
Three blind mice
Three blind mice
See how they run
See how they run
```

**其中一个函数要调用两次，该函数生成前两行；另一个函数也被调用两次，并生成其余的输出。**

解答：

```C++
#include <iostream>
using namespace std;
void tbm();
void shtr();

int main()
{
    tbm();
    tbm();
    shtr();
    shtr();
}

void tbm()
{
    cout << "Three blind mice" << endl;
}

void shtr()
{
    cout << "See how they run" << endl;
}
```

执行结果：

```
Three blind mice
Three blind mice
See how they run
See how they run
```

### 习题2.4

**编写一个程序，让用户输入其年龄，然后显示该年龄包含多少个月，如下所示：**

```
Enter your age: 29
```

解答：

```C++
#include <iostream>
using namespace std;

int main()
{
    cout << "Enter your age: ";
    int age;
    cin >> age;
    int month = 12 * age;
    cout << "your age in months is " << month << endl;
}
```

执行结果：

```
Enter your age: 29
your age in months is 348
```

### 习题2.5

**编写一个程序，其中的`main()`调用一个用户定义的函数(以摄氏温度值为参数，并返回相应的华氏温度值)。该程序按下面的格式要求用户输入摄氏温度值，并显示结果：**

```
Please enter a Celsius value: 20
20 degrees Celsius is 68 degrees Fahrenheit.
```

**下面是转换公式：**
**华氏温度=1.8×摄氏温度+32.0**

解答：

```C++
#include <iostream>
using namespace std;
double transform(double);

int main()
{
    cout << "Please enter a Celsius value: ";
    double centigrade;
    cin >> centigrade;
    double Fahrenheit_degree = transform(centigrade);
    cout << centigrade << " degrees Celsius is " << Fahrenheit_degree << " degrees Fahrenheit.";
}

double transform(double centigrade)
{
    return 1.8 * centigrade + 32.0;
}
```

执行结果：

```
Please enter a Celsius value: 20
20 degrees Celsius is 68 degrees Fahrenheit.
```

### 习题2.6

**编写一个程序，其`main()`调用一个用户定义的函数(以光年值为参数，并返回对应天文单位的值) 。该程序按下面的格式要求用户输入光年值，并显示结果：**

```
Enter the number of light years: 4.2
4.2 light years = 265608 astronomical units.
```

**天文单位是从地球到太阳的平均距离（约1500000公里或9300000英里），光年是光一年走的距离 （约10万亿公里或6万亿英里）（除太阳外，最近的恒星大约离地球4.2光年）。请使用 double类型（参见程序清单2.4），转换公式为：**
**1光年=63240天文单位**

解答：

```C++
#include <iostream>
using namespace std;
double transform(double);

int main()
{
    double light_years , astronomical_units;
    cout << "Enter the number of light years: ";
    cin >> light_years;
    astronomical_units = transform(light_years);
    cout << light_years << " light years = " << astronomical_units << " astronomical units.";
}

double transform(double light_years)
{
    return 63240 * light_years;
}
```

执行结果：

```
Enter the number of light years: 4.2
4.2 light years = 265608 astronomical units.
```

### 习题2.7

**编写一个程序，要求用户输入小时数和分钟数。在`main()`函数中，将这两个值传递给一个`void`函 数，后者以下面这样的格式显示这两个值：**

```
Enter the number of hours: 9
Enter the number of minutes: 28
T1me: 9:28
```

解答：

```C++
#include <iostream>

using namespace std;

void display(int, int);

int main() {
    int hour, minute;
    cout << "Enter the number of hours: ";
    cin >> hour;
    cout << "Enter the number of minutes: ";
    cin >> minute;
    display(hour, minute);
}

void display(int hour, int minute) {
    cout << "Time: " << hour << ":" << minute;
}
```

执行结果：

```
Enter the number of hours: 9
Enter the number of minutes: 28
Time: 9:28
```

