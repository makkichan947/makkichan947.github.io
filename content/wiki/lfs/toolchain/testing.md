+++
title = "临时工具链测试"
date = "2025-10-28"
description = "验证临时工具链的完整性和功能"
weight = 5
+++

# 临时工具链测试

构建完成后，必须对临时工具链进行全面测试，确保其能够正确编译和运行程序。本章将介绍各种测试方法和验证步骤。

## 🎯 测试目标

### 工具链验证内容

临时工具链测试需要验证：

1. **编译器功能**：GCC能够正确编译C/C++程序
2. **链接器功能**：能够正确链接目标文件和库
3. **库功能**：C库和运行时支持正常工作
4. **交叉编译**：能够生成目标平台的可执行代码
5. **调试支持**：调试信息和符号表正确

## 🧪 基本功能测试

### 编译器测试
```bash
# 切换到lfs用户
su - lfs

# 创建测试目录
mkdir -pv $LFS/toolchain_tests
cd $LFS/toolchain_tests

# 基本C程序测试
cat > hello.c << "EOF"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    printf("Hello, LFS Toolchain!\n");

    if (argc > 1) {
        printf("Arguments: ");
        for (int i = 1; i < argc; i++) {
            printf("%s ", argv[i]);
        }
        printf("\n");
    }

    return EXIT_SUCCESS;
}
EOF

# 编译测试
$LFS_TGT-gcc hello.c -o hello

# 检查编译结果
if [ -x hello ]; then
    echo "✓ 基本编译测试通过"
    $LFS_TGT-readelf -l hello | grep "interpreter"
else
    echo "✗ 基本编译测试失败"
    exit 1
fi
```

### 库功能测试
```bash
# 测试标准库函数
cat > lib_test.c << "EOF"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

int main() {
    printf("=== 标准库功能测试 ===\n");

    // 字符串函数
    char str1[50] = "Hello";
    char str2[50] = "World";
    strcat(str1, " ");
    strcat(str1, str2);
    printf("字符串连接: %s\n", str1);

    // 数学函数
    double x = 3.14159;
    printf("平方根: %.6f\n", sqrt(x));
    printf("正弦值: %.6f\n", sin(x));

    // 内存管理
    int *array = malloc(10 * sizeof(int));
    if (array) {
        for (int i = 0; i < 10; i++) {
            array[i] = i * i;
        }
        printf("数组内容: ");
        for (int i = 0; i < 10; i++) {
            printf("%d ", array[i]);
        }
        printf("\n");
        free(array);
        printf("✓ 内存管理测试通过\n");
    } else {
        printf("✗ 内存分配失败\n");
    }

    // 时间函数
    time_t now = time(NULL);
    printf("当前时间戳: %ld\n", now);

    printf("=== 所有测试完成 ===\n");
    return 0;
}
EOF

# 编译并测试
$LFS_TGT-gcc lib_test.c -lm -o lib_test

if [ -x lib_test ]; then
    echo "✓ 库功能测试编译成功"
else
    echo "✗ 库功能测试编译失败"
fi
```

### C++功能测试
```bash
# C++标准库测试
cat > cpp_test.cpp << "EOF"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

class TestClass {
private:
    std::string name;
    int value;

public:
    TestClass(std::string n, int v) : name(n), value(v) {}

    void display() const {
        std::cout << "对象: " << name << ", 值: " << value << std::endl;
    }

    int getValue() const { return value; }
};

int main() {
    std::cout << "=== C++功能测试 ===" << std::endl;

    // 基本输出
    std::cout << "Hello, C++!" << std::endl;

    // STL容器
    std::vector<TestClass> objects;
    objects.emplace_back("对象1", 10);
    objects.emplace_back("对象2", 20);
    objects.emplace_back("对象3", 15);

    // 排序
    std::sort(objects.begin(), objects.end(),
              [](const TestClass& a, const TestClass& b) {
                  return a.getValue() < b.getValue();
              });

    // 显示结果
    for (const auto& obj : objects) {
        obj.display();
    }

    std::cout << "=== C++测试完成 ===" << std::endl;
    return 0;
}
EOF

# 编译C++程序
$LFS_TGT-g++ cpp_test.cpp -o cpp_test

if [ -x cpp_test ]; then
    echo "✓ C++功能测试编译成功"
else
    echo "✗ C++功能测试编译失败"
fi
```

## 🔧 高级功能测试

### 链接器测试
```bash
# 测试静态链接
cat > static_lib.c << "EOF"
#include <stdio.h>

void print_message(const char *msg) {
    printf("静态库消息: %s\n", msg);
}
EOF

# 编译为目标文件
$LFS_TGT-gcc -c static_lib.c -o static_lib.o

# 创建静态库
$LFS_TGT-ar rcs libstatic.a static_lib.o

# 创建使用静态库的程序
cat > use_static.c << "EOF"
#include <stdio.h>

void print_message(const char *msg);

int main() {
    printf("=== 静态链接测试 ===\n");
    print_message("Hello from static library!");
    printf("=== 测试完成 ===\n");
    return 0;
}
EOF

# 静态链接
$LFS_TGT-gcc use_static.c -L. -lstatic -o use_static

if [ -x use_static ]; then
    echo "✓ 静态链接测试成功"
    # 检查是否包含库代码
    $LFS_TGT-nm use_static | grep "print_message"
else
    echo "✗ 静态链接测试失败"
fi
```

### 动态链接测试
```bash
# 测试动态链接
cat > dynamic_lib.c << "EOF"
#include <stdio.h>

void dynamic_function(const char *msg) {
    printf("动态库消息: %s\n", msg);
}
EOF

# 编译为位置无关代码
$LFS_TGT-gcc -fPIC -c dynamic_lib.c -o dynamic_lib.o

# 创建共享库
$LFS_TGT-gcc -shared -o libdynamic.so dynamic_lib.o

# 创建使用动态库的程序
cat > use_dynamic.c << "EOF"
#include <stdio.h>

void dynamic_function(const char *msg);

int main() {
    printf("=== 动态链接测试 ===\n");
    dynamic_function("Hello from shared library!");
    printf("=== 测试完成 ===\n");
    return 0;
}
EOF

# 动态链接编译
$LFS_TGT-gcc use_dynamic.c -L. -ldynamic -o use_dynamic

if [ -x use_dynamic ]; then
    echo "✓ 动态链接测试成功"
    # 检查动态依赖
    $LFS_TGT-readelf -d use_dynamic | grep "Shared library"
else
    echo "✗ 动态链接测试失败"
fi
```

### 交叉编译验证
```bash
# 测试交叉编译功能
cat > cross_test.c << "EOF"
#include <stdio.h>
#include <stdint.h>

int main() {
    printf("=== 交叉编译验证 ===\n");

    // 检查数据类型大小
    printf("char: %zu bytes\n", sizeof(char));
    printf("short: %zu bytes\n", sizeof(short));
    printf("int: %zu bytes\n", sizeof(int));
    printf("long: %zu bytes\n", sizeof(long));
    printf("long long: %zu bytes\n", sizeof(long long));
    printf("pointer: %zu bytes\n", sizeof(void*));

    // 检查字节序
    uint32_t test = 0x12345678;
    unsigned char *bytes = (unsigned char*)&test;
    printf("字节序: %s\n", (bytes[0] == 0x78) ? "小端" : "大端");

    // 检查编译器定义
#ifdef __GNUC__
    printf("编译器: GCC %d.%d.%d\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#endif

#ifdef __x86_64__
    printf("目标架构: x86_64\n");
#endif

    printf("=== 验证完成 ===\n");
    return 0;
}
EOF

# 交叉编译
$LFS_TGT-gcc cross_test.c -o cross_test

if [ -x cross_test ]; then
    echo "✓ 交叉编译验证成功"
    # 分析目标文件
    $LFS_TGT-readelf -h cross_test | grep "Machine\|Class\|OS/ABI"
else
    echo "✗ 交叉编译验证失败"
fi
```

## 📊 性能和稳定性测试

### 编译性能测试
```bash
# 测试编译速度
cat > perf_test.c << "EOF"
#include <stdio.h>
#include <stdlib.h>

#define SIZE 10000

int main() {
    int **matrix = malloc(SIZE * sizeof(int*));
    for (int i = 0; i < SIZE; i++) {
        matrix[i] = malloc(SIZE * sizeof(int));
        for (int j = 0; j < SIZE; j++) {
            matrix[i][j] = i + j;
        }
    }

    long long sum = 0;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            sum += matrix[i][j];
        }
        free(matrix[i]);
    }
    free(matrix);

    printf("矩阵求和结果: %lld\n", sum);
    return 0;
}
EOF

# 测试不同优化级别
echo "=== 编译性能测试 ==="
for opt in O0 O1 O2 O3; do
    echo "测试优化级别: -$opt"

    # 记录编译时间
    start_time=$(date +%s.%3N)
    $LFS_TGT-gcc -$opt perf_test.c -o perf_test_$opt
    end_time=$(date +%s.%3N)

    compile_time=$(echo "$end_time - $start_time" | bc)

    if [ -x perf_test_$opt ]; then
        file_size=$(ls -lh perf_test_$opt | awk '{print $5}')
        echo "✓ 编译成功 - 时间: ${compile_time}s, 大小: $file_size"
    else
        echo "✗ 编译失败"
    fi
done
```

### 稳定性测试
```bash
# 多进程编译测试
cat > stress_test.c << "EOF"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

#define NUM_PROCESSES 10

int main() {
    printf("=== 稳定性测试 ===\n");

    for (int i = 0; i < NUM_PROCESSES; i++) {
        pid_t pid = fork();

        if (pid == 0) {
            // 子进程：执行简单计算
            long sum = 0;
            for (long j = 0; j < 100000; j++) {
                sum += j;
            }
            printf("进程 %d 计算结果: %ld\n", i + 1, sum);
            exit(0);
        } else if (pid < 0) {
            printf("创建进程失败\n");
            return 1;
        }
    }

    // 等待所有子进程
    for (int i = 0; i < NUM_PROCESSES; i++) {
        wait(NULL);
    }

    printf("=== 稳定性测试完成 ===\n");
    return 0;
}
EOF

# 编译并测试
$LFS_TGT-gcc stress_test.c -o stress_test

if [ -x stress_test ]; then
    echo "✓ 稳定性测试编译成功"
else
    echo "✗ 稳定性测试编译失败"
fi
```

## 🔍 调试和诊断

### 编译过程分析
```bash
# 详细编译过程跟踪
cat > debug_compile.c << "EOF"
#include <stdio.h>

#define DEBUG_LEVEL 2

int main() {
#if DEBUG_LEVEL >= 1
    printf("调试级别 1: 基本信息\n");
#endif

#if DEBUG_LEVEL >= 2
    printf("调试级别 2: 详细信息\n");
#endif

    printf("程序正常运行\n");
    return 0;
}
EOF

# 显示预处理结果
echo "=== 预处理结果 ==="
$LFS_TGT-gcc -E debug_compile.c | tail -20

# 显示汇编代码
echo -e "\n=== 汇编代码 ==="
$LFS_TGT-gcc -S debug_compile.c
cat debug_compile.s

# 显示编译详细信息
echo -e "\n=== 编译详细信息 ==="
$LFS_TGT-gcc -v debug_compile.c -o debug_compile

# 分析目标文件
echo -e "\n=== 目标文件分析 ==="
$LFS_TGT-objdump -h debug_compile
$LFS_TGT-objdump -d debug_compile | head -30

# 清理
rm -f debug_compile.c debug_compile.s debug_compile
```

### 错误诊断
```bash
# 创建可能出错的程序来测试错误处理
cat > error_test.c << "EOF"
// 这个程序包含一些潜在的编译问题
#include <stdio.h>

int main() {
    // 未使用的变量
    int unused_var = 42;

    // 类型不匹配
    char *str = "Hello";
    // str[0] = 'h';  // 这会导致段错误

    printf("错误测试程序\n");
    printf("字符串: %s\n", str);

    return 0;
}
EOF

# 测试警告检测
echo "=== 警告检测测试 ==="
$LFS_TGT-gcc -Wall -Wextra error_test.c -o error_test

# 测试严格模式
echo -e "\n=== 严格模式测试 ==="
$LFS_TGT-gcc -Werror -Wall error_test.c -o error_test_strict 2>&1 || echo "预期的编译失败"

# 清理
rm -f error_test.c error_test error_test_strict
```

## 📋 完整测试套件

### 自动化测试脚本
```bash
# 创建完整的测试套件
cat > $LFS/toolchain_tests/run_all_tests.sh << 'EOF'
#!/bin/bash
# LFS工具链完整测试套件

LFS=${LFS:-/mnt/lfs}
LFS_TGT=${LFS_TGT:-x86_64-lfs-linux-gnu}

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 测试计数器
total_tests=0
passed_tests=0
failed_tests=0

# 测试函数
run_test() {
    local test_name=$1
    local test_cmd=$2

    echo -n "运行测试: $test_name... "
    total_tests=$((total_tests + 1))

    if eval "$test_cmd" >/dev/null 2>&1; then
        echo -e "${GREEN}通过${NC}"
        passed_tests=$((passed_tests + 1))
    else
        echo -e "${RED}失败${NC}"
        failed_tests=$((failed_tests + 1))
    fi
}

# 切换到测试目录
cd $LFS/toolchain_tests

echo "=== LFS工具链测试套件 ==="
echo "目标平台: $LFS_TGT"
echo "测试时间: $(date)"
echo ""

# 基本编译测试
run_test "基本C编译" "$LFS_TGT-gcc hello.c -o hello_test"
run_test "C++编译" "$LFS_TGT-g++ cpp_test.cpp -o cpp_test_exec"
run_test "库链接" "$LFS_TGT-gcc lib_test.c -lm -o lib_test_exec"

# 链接器测试
run_test "静态链接" "$LFS_TGT-gcc use_static.c -L. -lstatic -o static_exec"
run_test "动态链接" "$LFS_TGT-gcc use_dynamic.c -L. -ldynamic -o dynamic_exec"

# 交叉编译测试
run_test "交叉编译验证" "$LFS_TGT-gcc cross_test.c -o cross_exec"

# 优化测试
run_test "O0优化" "$LFS_TGT-gcc -O0 perf_test.c -o perf_O0"
run_test "O2优化" "$LFS_TGT-gcc -O2 perf_test.c -o perf_O2"
run_test "O3优化" "$LFS_TGT-gcc -O3 perf_test.c -o perf_O3"

# 稳定性测试
run_test "多进程测试" "$LFS_TGT-gcc stress_test.c -o stress_exec"

echo ""
echo "=== 测试结果汇总 ==="
echo "总测试数: $total_tests"
echo -e "通过: ${GREEN}$passed_tests${NC}"
echo -e "失败: ${RED}$failed_tests${NC}"

if [ $failed_tests -eq 0 ]; then
    echo -e "${GREEN}所有测试通过！工具链工作正常。${NC}"
    exit 0
else
    echo -e "${RED}有 $failed_tests 个测试失败，请检查工具链配置。${NC}"
    exit 1
fi
EOF

chmod +x $LFS/toolchain_tests/run_all_tests.sh
```

### 测试结果分析
```bash
# 运行完整测试套件
$LFS/toolchain_tests/run_all_tests.sh

# 生成测试报告
cat > $LFS/toolchain_tests/generate_report.sh << 'EOF'
#!/bin/bash
# 生成测试报告

LFS=${LFS:-/mnt/lfs}

echo "# LFS工具链测试报告" > $LFS/toolchain_test_report.md
echo "" >> $LFS/toolchain_test_report.md
echo "生成时间: $(date)" >> $LFS/toolchain_test_report.md
echo "" >> $LFS/toolchain_test_report.md

echo "## 系统信息" >> $LFS/toolchain_test_report.md
echo "- LFS目录: $LFS" >> $LFS/toolchain_test_report.md
echo "- 目标平台: $LFS_TGT" >> $LFS/toolchain_test_report.md
echo "- GCC版本: $($LFS_TGT-gcc --version | head -1)" >> $LFS/toolchain_test_report.md
echo "- Binutils版本: $($LFS_TGT-ld --version | head -1)" >> $LFS/toolchain_test_report.md
echo "" >> $LFS/toolchain_test_report.md

echo "## 测试文件" >> $LFS/toolchain_test_report.md
echo "\`\`\`bash" >> $LFS/toolchain_test_report.md
ls -la $LFS/toolchain_tests/ >> $LFS/toolchain_test_report.md
echo "\`\`\`" >> $LFS/toolchain_test_report.md
echo "" >> $LFS/toolchain_test_report.md

echo "## 磁盘使用情况" >> $LFS/toolchain_test_report.md
echo "\`\`\`bash" >> $LFS/toolchain_test_report.md
df -h $LFS >> $LFS/toolchain_test_report.md
echo "\`\`\`" >> $LFS/toolchain_test_report.md

echo "报告已保存到: $LFS/toolchain_test_report.md"
EOF

chmod +x $LFS/toolchain_tests/generate_report.sh
$LFS/toolchain_tests/generate_report.sh
```

## 🚨 故障排除指南

### 常见测试失败原因

1. **编译失败**
   ```bash
   # 检查环境变量
   echo $PATH
   echo $LFS_TGT

   # 验证工具存在
   which $LFS_TGT-gcc
   ls -la $LFS/tools/bin/$LFS_TGT-gcc
   ```

2. **链接失败**
   ```bash
   # 检查库文件
   find $LFS -name "libc.so*" -type f

   # 检查链接器
   $LFS_TGT-ld --version
   ```

3. **运行时失败**
   ```bash
   # 检查动态链接器
   ls -la $LFS/lib/ld-linux*

   # 验证库依赖
   $LFS_TGT-readelf -d [可执行文件]
   ```

### 调试技巧
```bash
# 启用详细输出
$LFS_TGT-gcc -v hello.c -o hello

# 显示预处理结果
$LFS_TGT-gcc -E hello.c | head -50

# 显示编译警告
$LFS_TGT-gcc -Wall -Wextra hello.c -o hello

# 分析目标文件
$LFS_TGT-objdump -x hello
$LFS_TGT-readelf -a hello
```

## 📚 相关资源

- [LFS官方文档 - 工具链测试](http://www.linuxfromscratch.org/lfs/view/stable/chapter05/chapter05.html)
- [GCC测试套件](https://gcc.gnu.org/onlinedocs/gccint/Testsuites.html)
- [Binutils测试](https://sourceware.org/binutils/binutils.pdf)

---

*最近更新: {{ .Lastmod.Format "2006-01-02" }}*