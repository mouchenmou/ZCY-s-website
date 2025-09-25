```c++
#include<iostream>
#include<string>

struct students{
    std::string name;
    int score[3];
    double average;
};

int main(){
    students student[10];
    int i;
    int j;
    
    for (i=0;i<10;i++){
        std::cin >> student[i].name;
        double sum=0;
        for(j=0;j<3;j++){
            std::cin >> student[i].score[j];
            sum=sum+student[i].score[j];
        }
        student[i].average = sum/3;
    }
    std::cout << "no      name        score1  score2  score3  average" << std::endl;

    for(i=0;i<10;i++){
        std::cout.width(8); //下一个输入的数占8个符号位，如果
        std::cout << std::left << i+1; //输入的名字都往左对齐
        std::cout.width(12);
        std::cout << std::left << student[i].name;
        std::cout.width(8);
        std::cout << std::left << student[i].score[0];
        std::cout.width(8);
        std::cout << std::left << student[i].score[1];
        std::cout.width(8);
        std::cout << std::left << student[i].score[2];
        if(student[i].average == static_cast<int>(student[i].average)){
            std::cout.precision(0);
            std::cout << student[i].average <<std::endl;
        }
        else{
            std::cout.precision(5);
            std::cout << student[i].average <<std::endl;
        }
    }
    
    int max[3];
    for(i=0;i<3;i++){
        max[i]=student[1].score[i];
    }
    
    int min[3];
    for(i=0;i<3;i++){
        min[i]=student[1].score[i];
    }

    double average[3]={0,0,0};

    for(i=0;i<10;i++){
        for(j=0;j<3;j++){
            average[j] = average[j] + student[i].score[j];
            
            if(student[i].score[j]>max[j]){
                max[j]=student[i].score[j];
            }

            if(student[i].score[j]<min[j]){
                min[j]=student[i].score[j];
            }
        }
    }
    
    std::cout.width(20);
    std::cout <<"        average" ;
    for(i=0;i<3;i++){
        std::cout.width(8);
        std::cout.precision(1);
        std::cout << std::fixed;
        std::cout << std::left << average[i]/10;
    }
    std::cout << std::endl;
    
    std::cout.width(20);
    std::cout <<"        min";
    
    for(i=0;i<3;i++){
        std::cout.width(8);
        std::cout << std::left << min[i];
    }
    std::cout << std::endl;

    std::cout.width(20);
    std::cout <<"        max";
    for(i=0;i<3;i++){
        std::cout.width(8);
        std::cout << std::left << max[i];
    }
    std::cout << std::endl;

    return 0;

}
```

!!! info "information1"
    `std::cout.precision()` 的行为取决于当前使用的**浮点数输出模式**。
    
    1. **默认模式（ std::defaultfloat ）**：
    
        - **precision** 控制的是总有效数字。
        
        - 示例：`std::cout.precision(1); std::cout << 3.14;` 输出 `3`。
        
	2. **固定模式（ std::fixed ）**：
    
        - **precision** 控制的是小数点后的位数。
        
        - 示例：
            ```
            std::cout << std::fixed; 
            std::cout.precision(1); 
            std::cout << 3.14; //输出 3.1。
            ```
            
        
    我的代码中，输出student[i].average代码中的 `if` 语句，只使用了 `std::precision(0)`，**没有开启 `std::fixed` 模式**。所以，它使用的就是默认模式。
    
    ### 为什么 `precision(0)` 输出的是 `5` 而不是 `0`？
    这是因为 `std::cout` 在处理 `precision(0)` 时，会先检查数字是否为 **0**。
	
    - 对于 `0`，它会直接输出 `0`。
    
    - 对于非零的整数或浮点数，`std::cout.precision(0)` **并不会将它们四舍五入为 `0`。相反，它会尽可能以最简洁的方式输出非零部分，通常就是不带小数点**。
    
    所以，当 `student[i].average` 是 `5.0` 时，`precision(0)` 会被 `std::cout` 解释为“以最简洁的方式输出，不要任何有效数字以外的零”。因为 `5.0` 的有效数字就是 `5`，所以它就输出了 `5`。


!!! info "information2"
    我一开始是用double max[3]和double min[3]去定义最大值和最小值，结果输出的情况不尽人意，如下：
    
    
    ![](附件/Pasted%20image%2020250925184848.png)
    
    出现这个问题的原因如下：
    在输出 `average` 平均分时，我使用了 `std::cout << std::fixed;`。这个命令告诉 `std::cout`：“**从现在开始，所有浮点数都用固定小数点的方式输出。**”
    
    为了解决这个问题，我就用int类型来定义max[3]和min[3]，因为int类型不是浮点数，所以不会被这个代码影响，因此成功输出了以下结果：
    
    ![](附件/Pasted%20image%2020250925192146.png)