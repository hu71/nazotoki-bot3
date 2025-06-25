import os
import cv2
import numpy as np
from flask import Flask, request
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, ImageMessage, TextMessage, TextSendMessage

app = Flask(__name__)

# 環境変数やシークレットを使って管理すべきですが、仮で記述
LINE_CHANNEL_ACCESS_TOKEN = '00KCkQLhlaDFzo5+UTu+/C4A49iLmHu7bbpsfW8iamonjEJ1s88/wdm7Yrou+FazbxY7719UNGh96EUMa8QbsG Bf9K5rDWhJpq8XTxakXRuTM6HiJDSmERbIWfyfRMfscXJPcRyTL6YyGNZxqkYSAQdB04t89/1O/w1cDnyilFU='
LINE_CHANNEL_SECRET = '6c12aedc292307f95ccd67e959973761'

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# 正解画像の特徴量をあらかじめ計算しておく
answer_image_path = "static/answer.jpg"
answer_img = cv2.imread(answer_image_path, cv2.IMREAD_GRAYSCALE)
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(answer_img, None)

# 類似度のしきい値（実験しながら調整）
MATCH_THRESHOLD = 20

@app.route("/")
def index():
    return "LINE Bot with OpenCV running"

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        return 'Invalid signature', 400
    return 'OK'

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    user_id = event.source.user_id

    # 画像取得
    message_content = line_bot_api.get_message_content(event.message.id)
    os.makedirs("static/uploads", exist_ok=True)
    image_path = f"static/uploads/{user_id}.jpg"
    with open(image_path, "wb") as f:
        for chunk in message_content.iter_content():
            f.write(chunk)

    # 画像読み込みと特徴点抽出
    img2 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des2 is None:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="画像の特徴が少なすぎて判定できませんでした。もう一度撮ってみてください。"))
        return

    # マッチング
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good_matches = [m for m in matches if m.distance < 60]

    if len(good_matches) > MATCH_THRESHOLD:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="正解です！おめでとう！"))
    else:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="不正解です。もう一度考えてみてね！"))

@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    text = event.message.text
    if text == "スタート":
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="第1問: 赤い箱を見つけて、その写真を送ってね！"))
    else:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="写真を送ってください！"))

if __name__ == "__main__":
    app.run(debug=True)
