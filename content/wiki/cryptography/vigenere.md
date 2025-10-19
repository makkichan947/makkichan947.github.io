+++
date = '2025-10-10T11:36:57+08:00'
draft = false
title = '维吉尼亚密码详解'
comments = true
weight = 6
+++

# 维吉尼亚密码详解

维吉尼亚密码（Vigenère Cipher）是一种多表替换密码，由法国外交官Blaise de Vigenère于16世纪发明，是古典密码学的重要里程碑。

## 加密原理

维吉尼亚密码是对凯撒密码的扩展，使用一系列凯撒密码交替加密。

### 密钥重复

设明文为 $P = p_1 p_2 p_3 ... p_n$
密钥为 $K = k_1 k_2 k_3 ... k_m$

加密过程：
$$c_i = (p_i + k_{i \mod m}) \mod 26$$

## 加密表格

### 维吉尼亚表

维吉尼亚密码使用26×26的表格进行加密：

```
    A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
   +----------------------------------------------------
A | A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
B | B C D E F G H I J K L M N O P Q R S T U V W X Y Z A
C | C D E F G H I J K L M N O P Q R S T U V W X Y Z A B
...
```

## 密码分析

### Kasiski测试

**Kasiski测试**：通过重复片段确定密钥长度。

**算法步骤**：
1. 寻找密文中重复的三元组
2. 计算重复出现的间距
3. 求间距的最大公约数
4. 确定可能的密钥长度

### 频率分析

确定密钥长度后，对每组进行单独的频率分析。

## 编程实现

### Python实现

```python
class VigenereCipher:
    def __init__(self, key):
        self.key = key.upper()
        self.key_length = len(key)
    
    def encrypt(self, plaintext):
        plaintext = plaintext.upper()
        ciphertext = ""
        
        for i, char in enumerate(plaintext):
            if char.isalpha():
                # 计算密钥移位
                key_shift = ord(self.key[i % self.key_length]) - ord('A')
                # 加密
                encrypted = chr((ord(char) - ord('A') + key_shift) % 26 + ord('A'))
                ciphertext += encrypted
            else:
                ciphertext += char
        
        return ciphertext
    
    def decrypt(self, ciphertext):
        ciphertext = ciphertext.upper()
        plaintext = ""
        
        for i, char in enumerate(ciphertext):
            if char.isalpha():
                # 计算密钥移位
                key_shift = ord(self.key[i % self.key_length]) - ord('A')
                # 解密
                decrypted = chr((ord(char) - ord('A') - key_shift) % 26 + ord('A'))
                plaintext += decrypted
            else:
                plaintext += char
        
        return plaintext
```

## 历史意义

维吉尼亚密码标志着古典密码学的顶峰，直到19世纪被破译。

## 现代应用

主要用于教育目的，帮助理解多表替换密码的概念。

