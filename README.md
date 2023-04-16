# CA Tech Lounge 
## 課題2. Computer Vision①

## Contents of this Repository
- [CIFAR10データセット](https://www.cs.toronto.edu/~kriz/cifar.html "CIFAR10")による、10クラスの分類モデルの学習
- StreamlitによるGUI分類アプリ

## Demo

## Environment
- CIFAR10データセットによる10クラス分類モデルの学習
  
| 項目 | 値 |
|------|----|
| OS | Ubuntu 20.04.5 LTS |
| プロセッサ | Intel(R) Xeon(R) CPU @ 2.20GHz |
| メモリ | 12GB |
| Pythonバージョン | 3.９.16 |
| ライブラリ | requirements.txtを参照 |
| GPU | NVIDIA Tesla T4 (ランタイムタイプをGPUに設定) |

- Streamlitによる分類アプリ
 
| 項目 | 値 |
|------|----|
| OS | MacOS 13.2 |
| プロセッサ |2.3 GHz クアッドコアIntel Core i5 |
| メモリ | 16GB |
| Pythonバージョン | 3.10.8 |
| ライブラリ | requirements.txtを参照 |
| GPU | なし |
 

## Requirement
必要なライブラリのインストール
```bash
# 学習用
pip install -r requirements.txt
# 分類アプリ
pip install -r requirements.txt
```
 
## Usage
### CIFAR10データセットによる学習 
```bash
python3 train.py
```
