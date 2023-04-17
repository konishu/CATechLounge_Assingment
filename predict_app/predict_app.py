import torch
from torchvision import transforms
from PIL import Image
import streamlit as st
import json

# 学習済みモデルのロード
model = torch.load('/Users/s_koni/work_dir/python/ML/CIFAR10_app/CIFAR10_app/BackgroundRemoval/Resnet34_model_20230413_113019.pth',
                   map_location=torch.device('cpu'))
model.eval()

# クラス名の定義
class_names = ['airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Streamlitアプリの設定
st.set_page_config(page_title='CIFAR-10 Image Classification', layout='wide')
st.title("CIFAR-10 Image Classification")
st.write(
    ":dog: 画像をアップロードして、その画像がなにかを確かめましょう。 :cat:"
)
st.write(
    "[CIFAR10データセット](https://www.cs.toronto.edu/~kriz/cifar.html 'CIFAR10')によって学習されたモデルにより、あなたの画像は10クラスに分類されます"
)
st.sidebar.subheader("クラス一覧")
classnames_list = ''
for class_name in class_names:
    classnames_list += "- " + class_name + "\n"
st.sidebar.markdown(classnames_list)

# 画像の前処理
def preprocess_image(image):
    # モノクロ画像の場合、チャンネルのサイズが一致しないため、画像をRGBモードに変換
    image = image.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image


# 画像ファイルのアップロード
uploaded_file = st.file_uploader('画像を選択してください。', type=['jpg', 'jpeg', 'png'])

# 描画
col1, col2, col3 = st.columns([3, 3, 3])
with col1:
    st.write("")

with col3:
    st.write("")
with col2:

    if uploaded_file is not None:
        # 画像の表示
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # 画像の前処理
        image_tensor = preprocess_image(image)

        # 予測結果の取得
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs.data, 1)
            score = torch.softmax(outputs, dim=1)[0][predicted[0]].item()
            result = {
                'predictions':[{
                'classification_results': [class_names[predicted[0]]],
                'score': [score]
            }]
            }

        # 予測結果の表示
        st.write('**Classification Results:**',
                 result['predictions'][0]['classification_results'][0])
        st.write('**Score:**', result['predictions'][0]['score'][0])

        print(result)