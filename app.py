import os
import cv2
import numpy as np
from flask import Flask, request
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, ImageMessage, TextMessage, TextSendMessage

app = Flask(__name__)

# LINEのトークン情報
LINE_CHANNEL_ACCESS_TOKEN = '00KCkQLhlaDFzo5+UTu+/C4A49iLmHu7bbpsfW8iamonjEJ1s88/wdm7Yrou+FazbxY7719UNGh96EUMa8QbsG Bf9K5rDWhJpq8XTxakXRuTM6HiJDSmERbIWfyfRMfscXJPcRyTL6YyGNZxqkYSAQdB04t89/1O/w1cDnyilFU='
LINE_CHANNEL_SECRET = '6c12aedc292307f95ccd67e959973761'
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# 正解画像の特徴点をあらかじめ計算
answer_img_path = "static/answer.jpg"
answer_img = cv2.imread(answer_img_path, cv2.IMREAD_GRAYSCALE)
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(answer_img, None)

@app.route("/")
def home():
    return "LINE Bot is running with OpenCV!"

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        return "Invalid signature", 400
    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    text = event.message.text.strip()
    if text == "スタート":
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="第1問: 正解だと思う写真を送ってね！"))
    else:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="写真を送ってください！"))

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    user_id = event.source.user_id
    message_content = line_bot_api.get_message_content(event.message.id)

    os.makedirs("static/uploads", exist_ok=True)
    upload_path = f"static/uploads/{user_id}.jpg"

    # 保存
    with open(upload_path, "wb") as f:
        for chunk in message_content.iter_content():
            f.write(chunk)

    # 判定
    img = cv2.imread(upload_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="画像が読み込めませんでした。もう一度送ってください。"))
        return

    kp2, des2 = orb.detectAndCompute(img, None)
    if des2 is None:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="特徴が見つかりませんでした。別の角度で送ってください。"))
        return

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good = [m for m in matches if m.distance < 60]

    if len(good) > 20:  # この値は調整可能
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="正解です！おめでとう！"))
    else:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="不正解です。もう一度挑戦してね！"))

if __name__ == "__main__":
    app.run(debug=True)
