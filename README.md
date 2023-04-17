# CA Tech Lounge 
## 課題2. Computer Vision①

## Contents of this Repository
- [CIFAR10データセット](https://www.cs.toronto.edu/~kriz/cifar.html "CIFAR10")による、10クラスの分類モデルの学習スクリプト
- StreamlitによるGUI分類アプリ

## Demo
![デモ動画](https://drive.google.com/file/d/15GDpDnXOZJj8CIXB0HdVHgxyNfW5_26m/view?usp=sharing "Sample video")

## Environment
- CIFAR10データセットによる10クラス分類モデルの学習(Google Colab上で実行)
  
| 項目 | 値 |
|------|----|
| OS | Ubuntu 20.04.5 LTS |
| プロセッサ | Intel(R) Xeon(R) CPU @ 2.20GHz |
| メモリ | 12GB |
| Pythonバージョン | 3.９.16 |
| ライブラリ | requirements.txtを参照 |
| GPU | NVIDIA Tesla T4 (ランタイムタイプをGPUに設定) |

- Streamlitによる分類アプリ(ラップトップ上で実行)
 
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
pip install -r requirements.txt
```
学習済みモデルのダウンロード
**あとで書く!!!**
 
## Usage
### CIFAR10データセットによる学習 
```bash
python3 train.py
```
### StreamlitによるGUI分類アプリ
```bash
streamlit run predict_app/predict_app.py
```